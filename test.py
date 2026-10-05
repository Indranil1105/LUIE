"""
Inference + evaluation of LUIE on the UIEB test split (paper Sec. 3.1(b), Fig. 2b,
and Sec. 4.3.1).

At inference ONLY the J-Net runs - one tiny 5-layer CNN (that is why the model
has only ~0.11M parameters, Table 4 of the paper).

For every image of the 190-image UIEB test split the script
    1. enhances the raw image with J-Net and saves it to <out_dir>/enhanced/
    2. computes the full-reference metrics (PSNR, SSIM, LPIPS) against the
       reference image
    3. computes the no-reference metrics (UIQM, UCIQE) on both the raw and the
       enhanced image
and writes everything to <out_dir>/results.csv, then prints the averages.

Run:
    python test.py --checkpoint checkpoints/luie_last.pth --data_root .
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from luie.dataset import UIEBTest, split_dataset
from luie.losses import LUIELosses
from luie.metrics import lpips_value, psnr, ssim, uciqe, uiqm
from luie.networks import LUIE


def save_image(tensor, path):
    """Save a (3, H, W) float tensor in [0, 1] as an image file."""
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- model: load the trained weights -----
    model = LUIE().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}), device: {device}")

    # the same LPIPS-AlexNet model used by the training losses
    lpips_model = LUIELosses(device).lpips

    # ----- data: the 190 UIEB test images -----
    _, test_pairs = split_dataset(args.data_root, n_train=args.n_train, seed=args.seed)
    if args.limit > 0:                        # only for quick smoke tests
        test_pairs = test_pairs[:args.limit]
    test_set = UIEBTest(test_pairs)
    print(f"test images: {len(test_set)}")

    out_dir = os.path.join(args.out_dir, "enhanced")
    os.makedirs(out_dir, exist_ok=True)

    # ----- enhance + measure every test image -----
    rows = []
    with torch.no_grad():
        for i, (raw, ref, name) in enumerate(test_set):
            # inference: only the J-Net (paper Fig. 2b)
            enhanced = model.j_net(raw[None].to(device))[0].clamp(0, 1).cpu()
            save_image(enhanced, os.path.join(out_dir, name))

            raw_np = raw.permute(1, 2, 0).numpy()
            ref_np = ref.permute(1, 2, 0).numpy()
            enh_np = enhanced.permute(1, 2, 0).numpy()

            rows.append({
                "name": name,
                "psnr": psnr(enh_np, ref_np),
                "ssim": ssim(enh_np, ref_np),
                "lpips": lpips_value(lpips_model, enh_np, ref_np),
                "uiqm_raw": uiqm(raw_np),
                "uiqm_enh": uiqm(enh_np),
                "uciqe_raw": uciqe(raw_np),
                "uciqe_enh": uciqe(enh_np),
            })
            if (i + 1) % 20 == 0 or i + 1 == len(test_set):
                print(f"processed {i + 1}/{len(test_set)} images")

    # ----- write per-image results and print the averages -----
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    means = {k: float(np.mean([r[k] for r in rows]))
             for k in rows[0] if k != "name"}
    print("\n=============== UIEB test results ===============")
    print(f"{'metric':<10}{'raw':>12}{'enhanced':>12}")
    print(f"{'PSNR':<10}{'-':>12}{means['psnr']:>12.4f}")
    print(f"{'SSIM':<10}{'-':>12}{means['ssim']:>12.4f}")
    print(f"{'LPIPS':<10}{'-':>12}{means['lpips']:>12.4f}")
    print(f"{'UIQM':<10}{means['uiqm_raw']:>12.4f}{means['uiqm_enh']:>12.4f}")
    print(f"{'UCIQE':<10}{means['uciqe_raw']:>12.4f}{means['uciqe_enh']:>12.4f}")
    print(f"\nper-image results -> {os.path.join(args.out_dir, 'results.csv')}")
    print(f"enhanced images   -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test / evaluate LUIE on UIEB")
    parser.add_argument("--checkpoint", default="checkpoints/luie_last.pth",
                        help="trained model checkpoint")
    parser.add_argument("--data_root", default=".",
                        help="folder containing raw-890/ and reference-890/")
    parser.add_argument("--out_dir", default="results", help="where to save results")
    parser.add_argument("--n_train", type=int, default=700,
                        help="number of UIEB training images (must match training)")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only on the first N test images (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (must match training)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
