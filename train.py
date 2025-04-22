#!/usr/bin/env python
from pathlib import Path
import re
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import urllib
from torch_geometric.nn import GCNConv
import unicodedata
from warcio.archiveiterator import ArchiveIterator
import requests
import gzip
import io
import os
import hashlib
import pickle
import random
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from bs4 import BeautifulSoup
from transformers import GraphormerModel, BertModel, AutoTokenizer
from skimage import color  # for rgb2lab conversion
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.model_selection import KFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def convert_to_data(example, tag_to_idx, device):
    tag_indices = [tag_to_idx.get(tag.lower(), 0) for tag in example['node_tag']]
    x = torch.tensor(tag_indices, dtype=torch.long, device=device).unsqueeze(1)  # Shape: [num_nodes, 1]

    edge_index = example['edge_index'].to(device)

    targets = example['targets']
    y1 = targets[:, :3].to(device)  # Color
    y2 = targets[:, 3:].to(device)  # Background color

    data = Data(x=x, edge_index=edge_index, y1=y1, y2=y2)
    return data

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)

def parse_css_color(color_str):
    """
    Given a CSS color string such as "rgb(34, 12, 64)" or "rgba(34, 12, 64, 1)",
    extract the first three numbers as floats, normalize them to [0,1] if necessary,
    and return as a numpy array.
    """
    matches = re.findall(r"[\d.]+", color_str)
    if len(matches) < 3:
        raise ValueError(f"Invalid CSS color string: {color_str}")
    rgb = np.array(list(map(float, matches[:3])))
    if rgb.max() > 1:
        rgb = rgb / 255.0
    return rgb

def get_cache_filepath(uri, cache_dir="data"):
    """
    Compute a cache file path for a given URL using its MD5 hash.
    """
    os.makedirs(cache_dir, exist_ok=True)
    uri = unicodedata.normalize('NFC', uri.strip())
    parsed = urllib.parse.urlparse(uri)
    normalized = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    uri = urllib.parse.urlunparse(normalized)
    md5 = hashlib.md5(uri.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f"{md5}.pkl")
    return cache_path

class BERTForColorPrediction(nn.Module):
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.head1 = nn.Linear(hidden_size, 3)
        self.head2 = nn.Linear(hidden_size, 3)

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        device = next(self.parameters()).device
        tokens = { k: v.to(device) for k, v in tokens.items() }
        embed = self.bert(**tokens).last_hidden_state[:, 0] # CLS token
        return self.head1(embed), self.head2(embed)



class GraphormerForColorPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Import the pretrained Graphormer model from HuggingFace
        self.graphormer = GraphormerModel(config)
        # Add a learnable linear projection.
        # We project from the hidden size to 6 dimensions (2 colors * 3 channels in CIELAB)
        hidden_size = self.graphormer.config.hidden_size
        self.linear = nn.Linear(hidden_size, 6)

    def forward(self, *args, **kwargs):
        # Assume that input data (e.g. graph structure, node features, etc.) is passed in a way
        # that GraphormerModel accepts. (For example, some implementations use batched edge indices
        # and node features.) We obtain node embeddings.
        embeddings = self.graphormer(*args, **kwargs).last_hidden_state
        # Apply linear projection to predict CIELAB colors for the masked nodes
        projected = self.linear(embeddings)
        return projected

class HTMLTreeGNN(nn.Module):
    def __init__(self, num_tag_types, tag_embed_dim, hidden_dim, target_dim):
        """
        num_tag_types: number of possible tag types.
        tag_embed_dim: dimension for tag embedding.
        hidden_dim: hidden state dimension in GNN layers.
        target_dim: output dimension for each target vector.
        """
        super(HTMLTreeGNN, self).__init__()
        # Embed the tag indices to a continuous vector.
        self.embedding = nn.Embedding(num_tag_types, tag_embed_dim)
        # Project embedded tag to hidden dimension if needed.
        # Here we choose hidden_dim to be same as tag_embed_dim, or you can add a linear here.
        
        # Two GCN layers.
        self.conv1 = GCNConv(tag_embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Two separate MLP heads: one for each target vector.
        self.head1 = nn.Linear(hidden_dim, target_dim)
        self.head2 = nn.Linear(hidden_dim, target_dim)

        
    def forward(self, data):
        """
        Expects data.x to be of shape [num_nodes, 1] (tag indices)
        and data.edge_index as the graph connectivity.
        Returns a tuple (pred1, pred2) of predicted vectors per node.
        """
        # Embed the tags.
        # Squeeze if necessary to get [num_nodes]
        x = self.embedding(data.x.squeeze())  # shape: [num_nodes, tag_embed_dim]
        
        # Apply GCN layers with ReLU and dropout.
        edge_index = data.edge_index[:, (data.edge_index < x.size(0)).all(dim=0)]
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Two heads produce the two vectors.
        pred1 = self.head1(x)  # shape: [num_nodes, target_dim]
        pred2 = self.head2(x)  # shape: [num_nodes, target_dim]
        
        return pred1, pred2

def get_warc_urls(max_docs=100):
    """
    Streams CommonCrawl's warc.paths.gz file and returns a list of full WARC URLs,
    up to a maximum of `max_docs` URLs.

    Args:
        max_docs (int): The maximum number of WARC URLs to return.

    Returns:
        List[str]: A list of full WARC file URLs.
    """
    # URL to the compressed WARC paths file for a given CommonCrawl snapshot.
    warc_paths_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-13/warc.paths.gz"
    base_url = "https://data.commoncrawl.org/"
    
    # Request the file in streaming mode
    response = requests.get(warc_paths_url, stream=True)
    response.raise_for_status()
    
    # Use the streaming response directly with gzip to avoid writing files to disk.
    # Wrap the raw response in a GzipFile and then a TextIOWrapper to decode the bytes.
    urls = []
    with gzip.GzipFile(fileobj=response.raw) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8') as reader:
            for i, line in enumerate(tqdm(reader, desc="Getting WARC URLS...", total=max_docs)):
                if i >= max_docs:
                    break
                line = line.strip()
                if line:
                    urls.append(base_url + line)
    return urls

class CommonCrawlWARCHTMLDataset(Dataset):
    """
    A PyTorch Dataset that sequentially reads HTML records from a list of
    WARC file URLs. For each WARC file, it either loads a cached version
    of the preprocessed examples (if available) or downloads and processes
    the WARC file.

    Processing for each HTML record includes:
      - Reading and decoding the HTML record.
      - Feeding the HTML into a headless browser via Selenium to render
        the page and extract computed CSS features ("color" and "background-color").
      - Building a graph from the DOM using parent–child relationships.
      - Parsing the two color features, converting them from RGB to CIELAB
        space, and packaging as targets.
    
    The final cache for each WARC file is a list of preprocessed examples.
    """
    def __init__(self, warc_urls, cache_dir="data", max_examples = 1000, max_length=640000):
        """
        Args:
            warc_urls (list): A list of WARC file URLs from CommonCrawl.
            max_docs (int): Maximum number of HTML documents (examples) to extract.
            cache_dir (str): Directory where cached files are stored.
        """
        self.warc_urls = warc_urls
        self.cache_dir = cache_dir
        self.max_examples = max_examples
        self.max_length = max_length
        self.examples = self._extract_processed_data()
    
    def _process_html(self, html, driver):
        """
        Processes a single HTML document string using Selenium.
        Returns a dictionary with:
           - node_features: placeholder tensor (num_nodes x 128)
           - edge_index: tensor (2 x num_edges) of parent-child indices
           - targets: tensor (num_nodes x 6), concatenation of two CIELAB colors
        """
        data_url = "data:text/html;charset=utf-8," + html
        driver.get(data_url)
        # Extract each node's tag and computed styles via JavaScript.
        nodes_info = driver.execute_script("""
            const ignore_tags = new Set(["HEAD", "META", "TITLE", "SCRIPT"])
            const nodes = Array.from(document.querySelectorAll('*'));
            const nodeIndexMap = new Map(nodes.map((node, idx) => [node, idx]));

            return nodes.map((node, idx) => {
                if(ignore_tags.has(node.tagName)) {
                    return
                }
                const style = window.getComputedStyle(node);
                let parent = node.parentElement;
                while(parent !== null && ignore_tags.has(parent.tagName)) {
                    parent = parent.parentElement;
                }
                const parentIndex = parent ? nodeIndexMap.get(parent) ?? -1 : -1;

                return {
                    idx: idx,
                    tagName: node.tagName,
                    class: node.getAttribute('class'),
                    color: style.getPropertyValue('color'),
                    backgroundColor: style.getPropertyValue('background-color'),
                    parentIndex: parentIndex
                };
            });
        """)
        
        # Build edge list from parent-child relationships.

        # Process each node to get the two color features, then convert to CIELAB.
        processed_targets = []
        edges = []
        tags = []
        classes = []
        for node in nodes_info:
            try:
                lab_color = color.rgb2lab(
                    parse_css_color(node['color']).reshape(1, 1, 3)
                ).reshape(3,)
                lab_bg = color.rgb2lab(
                    parse_css_color(node['backgroundColor']).reshape(1, 1, 3)
                ).reshape(3,)
                tag = node['tagName']
            except Exception:
                # If color parsing fails, skip
                continue
            processed_targets.append(np.concatenate([lab_color, lab_bg]))
            tags.append(tag)
            classes.append(node['class'].split(' ') if node['class'] else [])
            if node['parentIndex'] != -1:
                edges.append((node['parentIndex'], node['idx']))
        targets = torch.tensor(np.stack(processed_targets), dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

        # Create placeholder node features (could later be replaced with meaningful embeddings)

        return {
            "node_tag": tags,  # (num_nodes,)
            "classes": classes, # (num_nodes,)
            "edge_index": edge_index,        # (2, num_edges)
            "targets": targets               # (num_nodes, 6)
        }
    
    def _extract_processed_data(self):
        """
        For each WARC URL, check for a cached file. If it exists, load the
        processed examples. Otherwise, fetch the WARC file, process each HTML
        record, save the examples to cache, and collect them.
        """
        examples = []
        options = FirefoxOptions()
        options.add_argument("--headless")
        # HACK point Selenium at the snap driver
        service = webdriver.FirefoxService(executable_path="/snap/bin/geckodriver")
        driver = webdriver.Firefox(options=options, service=service)
        driver.set_page_load_timeout(15) # connection timeout (seconds)
        try:
            for warc_url in tqdm(self.warc_urls, desc="Processing WARC URLs"):
                if len(examples) >= self.max_examples:
                    break
                try:
                    response = requests.get(warc_url, stream=True)
                    response.raise_for_status()
                except Exception as e:
                    print(f"Failed to fetch {warc_url}: {e}")
                    continue
                # Process each record in the WARC.
                while len(examples) < self.max_examples:
                    archive = ArchiveIterator(response.raw)
                    try:
                        for record in tqdm(archive, desc="Processing WARC", leave=False):
                            if len(examples) >= self.max_examples:
                                break
                            if record.rec_type == 'response':
                                content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ""
                                if content_type and 'text/html' in content_type:
                                    target_uri = record.rec_headers.get_header('WARC-Target-URI')
                                    if not target_uri:
                                        continue
                                    content_length = record.http_headers.get_header('Content-Length')
                                    if not content_length or int(content_length) > self.max_length:
                                        continue
                                    cache_path = get_cache_filepath(target_uri.strip())
                                    if os.path.exists(cache_path):
                                        if os.stat(cache_path).st_size > 0:
                                            with open(cache_path, "rb") as f:
                                                examples.append(pickle.load(f))
                                    else:
                                        try:
                                            html = record.content_stream().read().decode('utf-8', errors='replace')
                                            processed = self._process_html(html, driver)
                                            if not processed['node_tag']:
                                                print(f'Empty document: {target_uri}')
                                                continue
                                            examples.append(processed)
                                            with open(cache_path, "wb") as f:
                                                pickle.dump(processed, f)
                                        except Exception as e:
                                            print(f'Document error: {target_uri}, {e}')
                                            # Skip problematic records.
                                            continue
                                        finally:
                                            Path(cache_path).touch(exist_ok=True)
                    except Exception as e:
                        print(e)
                        continue
                    finally:
                        archive.close()
        finally:
            driver.close()
            
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Return the preprocessed example.
        return self.examples[idx]

class CSSTextDataset(Dataset):
    def __init__(self, commoncrawl_dataset: CommonCrawlWARCHTMLDataset):
        """
        Wraps a CommonCrawlWARCHTMLDataset instance.

        Args:
            commoncrawl_dataset: An instance of CommonCrawlWARCHTMLDataset where each example is
                a dict that must include at least the following keys:
                  - "node_tag": a list of tag strings per node.
                  - "classes": a list of class information per node (each entry can be a list or string).
                  - "targets": a NumPy array or tensor of shape [num_nodes, 6] containing target color features.
                  - (other keys are ignored here)
        """
        self.examples = []
        for example in commoncrawl_dataset:
            node_tags = example["node_tag"]      # e.g. ["div", "p", ...]
            node_classes = example["classes"]      # e.g. [["header"], [""], ...] or strings
            targets = example["targets"]           # shape: [num_nodes, 6]
            num_nodes = len(node_tags)
            for i in range(num_nodes):
                tag = node_tags[i].lower()
                # Process classes: if it's a list, join non-empty strings; if already a string, use it if non-empty.
                if isinstance(node_classes[i], list):
                    cls_str = "." + ".".join(cls for cls in node_classes[i] if cls) if node_classes[i] else ""
                else:
                    cls_str = f".{node_classes[i]}" if node_classes[i] else ""
                css_selector = tag + cls_str  # e.g. "div.header.main"
                
                # Ensure targets is a tensor.
                if not torch.is_tensor(targets):
                    target_tensor = torch.tensor(targets[i], dtype=torch.float32)
                else:
                    target_tensor = targets[i].float()
                # Split targets into two color vectors.
                y1 = target_tensor[:3]
                y2 = target_tensor[3:]
                self.examples.append({"text": css_selector, "y1": y1, "y2": y2})
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_fold_graph(model, train_loader, val_loader, optimizer, criterion, num_epochs=50):
    best_val_loss = float('inf')
    best_state = None
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred1, pred2 = model(batch)
            loss1 = criterion(pred1, batch.y1)
            loss2 = criterion(pred2, batch.y2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Evaluate on the validation set:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred1, pred2 = model(batch)
                loss1 = criterion(pred1, batch.y1)
                loss2 = criterion(pred2, batch.y2)
                val_loss += (loss1 + loss2).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss {total_loss:.4f}, Val Loss {val_loss:.4f}")
    return best_state, best_val_loss

def train_graph(args):
    # Load your custom dataset
    warc_urls = get_warc_urls(10)
    custom_dataset = CommonCrawlWARCHTMLDataset(warc_urls, max_examples=args.max_docs)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Build vocabulary from tags across all examples.
    tag_set = set()
    for example in custom_dataset:
        for tag in example['node_tag']:
            tag_set.add(tag.lower())
    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}
    
    # Convert examples to PyG Data objects.
    # Do NOT try to convert the list to a np.array as the instances are ragged.
    data_list = [convert_to_data(example, tag_to_idx, device) for example in custom_dataset]
    
    # Prepare fold indices using a list of indices rather than converting data_list to a NumPy array.
    indices = list(range(len(data_list)))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []
    
    criterion = nn.MSELoss()
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nStarting fold {fold+1}")
        train_subset = [data_list[i] for i in train_idx]
        val_subset = [data_list[i] for i in val_idx]
        train_loader = GeoDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = GeoDataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize a fresh model for this fold.
        model = HTMLTreeGNN(num_tag_types=len(tag_to_idx),
                            tag_embed_dim=16,
                            hidden_dim=32,
                            target_dim=3)
        model.to(device)
        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        best_state, best_val_loss = train_fold_graph(model, train_loader, val_loader,
                                               optimizer, criterion, num_epochs=args.epochs)
        print(f"Fold {fold+1}: Best Val Loss = {best_val_loss:.4f}")
        fold_results.append((best_val_loss, best_state))
    
    # Select the best model (lowest validation loss).
    best_fold_loss, best_model_state = min(fold_results, key=lambda item: item[0])
    print(f"\nBest overall validation loss: {best_fold_loss:.4f}")
    
    # Reinitialize final model and load best weights.
    final_model = HTMLTreeGNN(num_tag_types=len(tag_to_idx),
                              tag_embed_dim=16,
                              hidden_dim=32,
                              target_dim=3)
    final_model.load_state_dict(best_model_state)
    final_model.to(device)
    
    # Final evaluation on the whole dataset.
    final_model.eval()
    total_loss = 0.0
    final_loader = GeoDataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in final_loader:
            pred1, pred2 = final_model(batch)
            loss = criterion(pred1, batch.y1) + criterion(pred2, batch.y2)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_list)
    torch.save({'state_dict': final_model.state_dict(), 'tag_to_idx': tag_to_idx }, "best_graph.pth")
    print(f"\nFinal model average loss on full data: {avg_loss:.4f}")

def train_bert(args):
    graph_dataset = CommonCrawlWARCHTMLDataset(get_warc_urls(10), max_examples=args.max_docs)
    css_dataset = CSSTextDataset(graph_dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []
    indices = list(range(len(css_dataset)))
    
    # Use Mean Squared Error loss for regression.
    criterion = nn.MSELoss()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nStarting fold {fold+1}")
        train_examples = [css_dataset[i] for i in train_idx]
        val_examples   = [css_dataset[i] for i in val_idx]
        train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_examples, batch_size=args.batch_size, shuffle=False)
        
        # Reinitialize the model each fold.
        model = BERTForColorPrediction()
        model.apply(initialize_weights)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        best_val_loss = float("inf")
        best_state = None
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                # batch["text"] is a list of strings.
                optimizer.zero_grad()
                pred1, pred2 = model(batch["text"])
                # Ensure targets are moved to the correct device.
                target1 = batch["y1"].to(pred1.device)
                target2 = batch["y2"].to(pred2.device)
                loss = criterion(pred1, target1) + criterion(pred2, target2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Evaluate on validation set.
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pred1, pred2 = model(batch["text"])
                    target1 = batch["y1"].to(pred1.device)
                    target2 = batch["y2"].to(pred2.device)
                    val_loss += (criterion(pred1, target1) + criterion(pred2, target2)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss {total_loss:.4f}, Val Loss {val_loss:.4f}")
        fold_results.append((best_val_loss, best_state))
    
    # Choose best fold.
    best_overall_loss, best_model_state = min(fold_results, key=lambda x: x[0])
    # Save best model
    final_model = BERTForColorPrediction()
    final_model.load_state_dict(best_model_state)
    torch.save(final_model.state_dict(), "best_bert.pth")
    print(f"\nBest validation loss: {best_overall_loss:.4f}. Model saved to 'best_css_selector_color_predictor.pth'.")

def inference_graph(html_path, model_path, device=None):
    # Load HTML and process into graph example
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    # Reuse dataset's _process_html via a minimal driver-less stub or export a helper
    # For simplicity, require a pickled example dict from HTML
    # Create a headless dataset instance for one URL
    dataset = CommonCrawlWARCHTMLDataset([], max_examples=0)
    options = FirefoxOptions()
    options.add_argument("--headless")
    # HACK point Selenium at the snap driver
    service = webdriver.FirefoxService(executable_path="/snap/bin/geckodriver")
    driver = webdriver.Firefox(options=options, service=service)
    try:
    # Monkey-patch to call _process_html without Selenium driver
        example = dataset._process_html(html, driver=driver)
    finally:
        driver.close()

    # Load model
    model_info = torch.load(model_path, map_location=device)
    tag_to_idx = model_info['tag_to_idx']
    model = HTMLTreeGNN(num_tag_types=len(tag_to_idx), tag_embed_dim=16, hidden_dim=32, target_dim=3)
    model.load_state_dict(model_info['state_dict'])
    model.to(device).eval()

    # Load tag_to_idx mapping
    data = convert_to_data(example, tag_to_idx, device)


    with torch.no_grad():
        pred1, pred2 = model(data)
    lab1 = pred1.cpu().numpy()
    lab2 = pred2.cpu().numpy()
    # Convert back to RGB for readability
    rgb1 = color.lab2rgb(lab1.reshape(-1,1,3)).reshape(-1,3)
    rgb2 = color.lab2rgb(lab2.reshape(-1,1,3)).reshape(-1,3)
    for i, (c1, c2) in enumerate(zip(rgb1, rgb2)):
        print(f"Node {i}: Foreground RGB: {c1}, Background RGB: {c2}")


def inference_bert(html_path, model_path, device=None):
    # Parse HTML for CSS selectors
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    selectors = []
    for node in soup.find_all():
        tag = node.name
        classes = node.get('class', [])
        cls_str = '.' + '.'.join(classes) if classes else ''
        selectors.append(tag + cls_str)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTForColorPrediction()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        pred1, pred2 = model(selectors)
    lab1 = pred1.cpu().numpy()
    lab2 = pred2.cpu().numpy()
    rgb1 = color.lab2rgb(lab1.reshape(-1,1,3)).reshape(-1,3)
    rgb2 = color.lab2rgb(lab2.reshape(-1,1,3)).reshape(-1,3)
    for sel, c1, c2 in zip(selectors, rgb1, rgb2):
        print(f"{sel}: Foreground RGB: {c1}, Background RGB: {c2}")


def main():
    parser = argparse.ArgumentParser(description="Train or infer CSS color models")
    parser.add_argument('-d', '--max-docs', type=int, default=1000,
                        help="Max HTML docs to read for training/inference.")
    parser.add_argument('-m', '--model', type=str, choices=['graph', 'bert'],
                        help="Model to train or infer: 'graph' or 'bert'.")
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=12)
    parser.add_argument('--infer', action='store_true', help="Run inference instead of training.")
    parser.add_argument('--html-file', type=str, help="Path to input HTML file for inference.")
    parser.add_argument('--model-path', type=str, help="Path to saved model weights (.pth) for inference.")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    if args.infer:
        if not args.html_file or not args.model_path:
            parser.error("--html-file and --model-path are required for inference.")
        if args.model == 'graph':
            inference_graph(args.html_file, args.model_path, device=device)
        else:
            inference_bert(args.html_file, args.model_path, device=device)
    else:
        # Existing training logic
        if args.model == 'graph':
            train_graph(args)
        else:
            train_bert(args)

if __name__ == "__main__":
    main()

