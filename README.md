# CommonCrawl Dataset builder
This is a prototype version for a work-in progress CSS generator.

What's here:
- Accessing and scraping CSS information from CommonCrawl
- Translation and preprocessing of data to suit multiple machine learning paradigms
- A few proof-of-concept machine learning architectures which are demonstrated using 5-fold cross validation
    - GCN: Trained end-to-end with HTML graphs as input. Outputs color vectors for each visible node.
    - BERT: Trained end-to-end with text as input. Outputs color vectors for each `tag.class` pair.
    - Graphormer [WIP]: Trained end-to-end with HTML graphs as input. Currently broken due to input formatting.
- To start with, I recommend using the GCN model with `./train.py -m graph`. You can find more options using `--help`

## Installation
I recommend using `uv` (`pip install uv`) to manage dependencies.
You may also be able to `pip install .`, but I have not tested.
