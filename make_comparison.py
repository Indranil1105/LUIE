"""Build a side-by-side comparison figure (raw | enhanced | reference)
for a few UIEB test images, so the enhancement can be judged visually.
Not part of the paper - just a convenience for looking at the results.

Run after test.py:
    python make_comparison.py --results_dir results_demo --n 4
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_demo")
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--n", type=int, default=4, help="number of rows")
    args = parser.parse_args()

    enhanced = sorted(glob.glob(os.path.join(args.results_dir, "enhanced", "*")))[:args.n]
    rows = []
    for path in enhanced:
        name = os.path.basename(path)
        raw = Image.open(os.path.join(args.data_root, "raw-890", name)).convert("RGB")
        ref = Image.open(os.path.join(args.data_root, "reference-890", name)).convert("RGB")
        enh = Image.open(path).convert("RGB")
        size = (256, 256)
        row = [img.resize(size) for img in (raw, enh, ref)]
        rows.append(np.concatenate(row, axis=1))

    figure = np.concatenate(rows, axis=0)
    out = os.path.join(args.results_dir, "comparison.png")
    Image.fromarray(figure).save(out)
    print(f"saved {out}  (columns: raw | enhanced | reference)")


if __name__ == "__main__":
    main()
