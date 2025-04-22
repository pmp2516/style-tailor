#!/usr/bin/env python
import argparse
import re
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from skimage.color import rgb2lab, deltaE_ciede2000

HEX_RE = re.compile(r'^#?([0-9A-Fa-f]{6})$')
RGB_RE = re.compile(r'rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\s*\)')

def hex_to_rgb_norm(hexstr):
    """Convert '#RRGGBB' → (r, g, b) floats in [0, 1]."""
    m = HEX_RE.match(hexstr)
    if not m:
        raise ValueError(f"Invalid hex: {hexstr}")
    hx = m.group(1)
    rgb = np.array([int(hx[i:i+2], 16) for i in (0,2,4)], dtype=float)
    return rgb / 255.0

def parse_css_color(css_val):
    """Parse 'rgb(...)' / 'rgba(...)' → (r,g,b,a) with r/g/b ints, a float."""
    m = RGB_RE.match(css_val)
    if not m:
        return None
    r, g, b = map(int, m.group(1,2,3))
    a = float(m.group(4)) if m.group(4) is not None else 1.0
    return (r, g, b, a)

def main():
    p = argparse.ArgumentParser(
        description="Recolor HTML via nearest‑neighbor in CIELAB (skimage)"
    )
    p.add_argument('-i','--input', required=True,
                   help="URL or file:///… path")
    p.add_argument('-p','--palette', required=True, nargs='+',
                   help="Hex codes: #112233 #445566 …")
    p.add_argument('--headless', action='store_true',
                   help="Run browser without UI")
    args = p.parse_args()

    # Build Lab palette array
    hexes = [h if h.startswith('#') else '#'+h for h in args.palette]
    rgb_pal = np.vstack([hex_to_rgb_norm(h) for h in hexes])
    lab_pal = rgb2lab(rgb_pal[np.newaxis, ...])[0]  # shape (N,3) :contentReference[oaicite:7]{index=7}

    # Launch Selenium
    opts = Options()
    if args.headless:
        opts.add_argument('--headless')
    driver = webdriver.Chrome(options=opts)
    driver.get(args.input)

    elems = driver.find_elements(By.CSS_SELECTOR, '*')

    def recolor(el):
        styles = driver.execute_script("""
          function findBg(el) {
            const s = window.getComputedStyle(el);
            // If this element has a true (non-transparent) background, use it
            if (s.backgroundColor !== 'rgba(0, 0, 0, 0)') {
              return s.backgroundColor;
            }
            // If we've reached <body> (or html), fallback to white
            if (!el.parentElement || el.tagName === 'BODY') {
              return 'rgb(255, 255, 255)';
            }
            return findBg(el.parentElement);
          }
          const s = window.getComputedStyle(arguments[0]);
          return [ s.color, findBg(arguments[0]) ];
        """, el)
        # # In your recolor(el) function, replace the JS that fetches [color, bg]:
        # styles = driver.execute_script("""
        #   function findBg(e) {
        #     const s = window.getComputedStyle(e);
        #     if (s.backgroundColor !== 'rgba(0, 0, 0, 0)') {
        #       return s.backgroundColor;
        #     }
        #     // climb to parent node until <body>, then stop
        #     return e.parentElement
        #       ? findBg(e.parentElement)
        #       : s.backgroundColor;
        #   }
        #   const s = window.getComputedStyle(arguments[0]);
        #   return [ s.color, findBg(arguments[0]) ];
        # """, el)
        # styles = driver.execute_script(
        #     "const s = window.getComputedStyle(arguments[0]);"
        #     "return [s.color, s.backgroundColor];",
        #     el
        # )
        for prop, css in zip(['color','backgroundColor'], styles):
            parsed = parse_css_color(css)
            if not parsed or parsed[3]==0:
                continue
            # normalize and convert
            rgb = np.array(parsed[:3], dtype=float) / 255.0
            lab = rgb2lab(rgb[np.newaxis, ...])[0]  # :contentReference[oaicite:8]{index=8}
            # compute ΔE to all palette entries
            diffs = deltaE_ciede2000(lab[np.newaxis, :], lab_pal)  # :contentReference[oaicite:9]{index=9}
            best = np.argmin(diffs)
            driver.execute_script(
                "arguments[0].style[arguments[1]] = arguments[2];",
                el, prop, hexes[int(best)]
            )

    list(map(recolor, elems))  # vectorized loop replacement

    print("Done. Inspect browser. Ctrl+C to exit.")
    try:
        driver.implicitly_wait(999999)
        input()
    except KeyboardInterrupt:
        pass
    finally:
        driver.quit()

if __name__ == '__main__':
    main()
