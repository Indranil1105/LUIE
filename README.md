# LUIE — Learnable Physical Model-Guided Underwater Image Enhancement

A simple, readable re-implementation of the paper

> **LUIE: Learnable physical model-guided underwater image enhancement with
> bi-directional unsupervised domain adaptation**
> Jingyi Pan, Zeyu Duan, Jianghua Duan, Zhe Wang — *Neurocomputing 602 (2024) 128286*

trained and evaluated **only on the UIEB dataset** (the `raw-890/` and
`reference-890/` folders in this repository).

---

## 1. What the paper does (in one page)

Underwater images follow the physical formation model

```
I(x) = J(x)·t(x) + A·(1 − t(x))                       (Eq. 1)
```

where `I` is the captured image, `J` the clear **scene radiance** (what we want),
`A` the uniform **background light** (the color cast), and `t` the
**transmission**. Previous methods hard-code `t = e^(−βd)`, but the attenuation
coefficient `β` is unknown, so LUIE instead **learns** the mapping from depth to
transmission:

```
I(x) = J(x)·f_t(d(x)) + A(x)·(1 − f_t(d(x)))          (Eq. 4)
```

Four tiny CNNs (5 conv layers each, ~0.11 M parameters in total at inference)
decompose an image (Fig. 2 of the paper):

| Network | Input → Output | Role |
|---------|----------------|------|
| **B-Net** | image → `A(x)` | background light (color cast) |
| **J-Net** | image → `J(x)` | scene radiance = **the enhanced image** |
| **D-Net** | image → `d(x)` | depth map |
| **T-Net** | `d(x)` → `t(x)` | *learnable* transmission (replaces `e^(−βd)`) |

**Training (Sec. 3.3–3.4).** Two domains are decomposed, and:

1. **Intra-domain reconstruction** (`L_rec`, Eq. 8): rebuilding each input with
   its own components via Eq. 4 must reproduce the input (LPIPS loss).
2. **Inter-domain adaptation** (`L_dec`, Eq. 5 + 9): the background lights of
   the two domains are **swapped** to build two mixed images
   (`X_m1 = J_syn·t_syn + A_real·(1−t_syn)`, etc.). The mixed images are then
   decomposed again and must return the original components. This makes
   J-Net/D-Net domain-invariant and closes the domain gap.
3. **Color constancy** (`L_cc`, Eq. 10): gray-world assumption on `J` pushes the
   mean R/G/B values of the scene radiance towards each other.
4. **Synthetic supervision** (`L_sup`, Eq. 6): LPIPS against the ground-truth
   depth / radiance / background of the labeled synthetic domain.

```
L_total = L_sup + α·L_rec + β·L_dec + γ·L_cc            (Eq. 11)
          α = 1.0, β = 1.0, γ = 0.1
```

**Inference (Fig. 2b).** Only **J-Net** runs — one tiny CNN, which is why the
model has ~0.11 M parameters and the fastest inference in the paper's Table 4.

---

## 2. Repository layout — where each paper part lives

> **New here?** Read `CODE_EXPLANATION.md` — it walks through the code
> file-by-file and shows exactly which equation of the paper is implemented
> where, in plain language.

| File | Paper section | Content |
|------|---------------|---------|
| `luie/networks.py` | Sec. 3.2, Fig. 2c | B-Net / J-Net / D-Net / T-Net (5-layer CNNs) + Eq. 1/4 reconstruction |
| `luie/losses.py`   | Sec. 3.4, Eq. 6–11 | `L_sup`, `L_rec`, `L_dec`, `L_cc`, all LPIPS-AlexNet |
| `luie/dataset.py`  | Sec. 4.1 | UIEB 700/190 split, two-domain construction |
| `luie/metrics.py`  | Sec. 4.3.1 | PSNR, SSIM, LPIPS, UIQM, UCIQE |
| `train.py`         | Sec. 3.3, 4.2 | the full bi-directional training loop |
| `test.py`          | Fig. 2b, 4.3.1 | J-Net-only inference + evaluation on the UIEB test split |

---

## 3. How this repo adapts the paper to "UIEB only"

The paper trains with **two** domains: a *labeled synthetic* dataset
(Underwater3k, built in Blender) and *unlabeled real* images (UIEB). This repo
ships only UIEB, so two faithful adjustments are made (both documented in the
code):

1. **Two real sub-domains instead of synthetic-vs-real.** The 700 UIEB training
   images are split by their color cast into `green` (mean G > mean B, 319
   images) and `blue` (mean G ≤ mean B, 381 images). The bi-directional
   background-light swap of Eq. 5 runs between them. The mechanism is exactly
   the paper's — only the domains change, because no synthetic data is used.
2. **`L_sup` = 0.** Without Underwater3k there is no synthetic ground truth, so
   the supervision loss (Eq. 6) is inactive (it is still implemented in
   `luie/losses.py` for completeness).

Everything else follows the paper: training is **unsupervised** (the 700
reference images are *never* used for training), the 190 test images are
evaluated against their references, and all settings match Sec. 4.2
(Adam, lr 2e-4, batch size 1, 200 epochs, α/β/γ = 1.0/1.0/0.1, LPIPS-AlexNet).

---

## 4. How to run

```bash
pip install -r requirements.txt

# 1) train (paper settings: 200 epochs; ~10 min/epoch on this laptop's CPU,
#    much faster on any CUDA GPU)
python train.py --data_root . --epochs 200

# 2) enhance + evaluate on the 190-image UIEB test split
python test.py --checkpoint checkpoints/luie_last.pth --data_root .
```

Outputs:

* `checkpoints/luie_last.pth` — trained weights (all four networks)
* `checkpoints/losses.csv` — loss curve (rec / dec / cc / total)
* `results/enhanced/` — the enhanced test images
* `results/results.csv` — per-image metrics; the console prints the averages
  (PSNR/SSIM/LPIPS of the enhanced vs reference images, and UIQM/UCIQE of raw
  vs enhanced images)

Useful options: `--limit N` (small smoke runs), `--epochs`, `--size`
(training resolution, default 256), `--seed` (keep it the same for training and
testing — it controls the 700/190 split).

To eyeball the results, build a side-by-side figure
(raw | enhanced | reference):

```bash
python make_comparison.py --results_dir results --n 4
```

`calibrate_metrics.py` is a small helper that documents how the UIQM/UCIQE
implementations were sanity-checked against the values the paper reports for
the raw UIEB images.

---

## 5. Notes & honest limitations

* **The 700/190 split** follows the paper's protocol, but the paper does not
  publish its exact split list, so a fixed random seed (42) is used. Your
  PSNR/SSIM will therefore not match Table 1 digit-for-digit.
* **UIQM / UCIQE absolute values** differ between implementations found in the
  literature (the paper's own port reports e.g. UIQM ≈ 4.0 and UCIQE ≈ 28.8 for
  raw UIEB images). This repo uses the canonical published formulas; comparisons
  *within* this repo (raw vs enhanced, or against other models evaluated with
  this script) are consistent. `calibrate_metrics.py` documents this check.
* **Uranker** (the NIMA-based metric in the paper) is not included — it needs a
  separate pre-trained NIMA model.
* Expected end results after full training on UIEB (paper reports, for
  reference): PSNR ≈ 22.9, SSIM ≈ 0.92 on the UIEB test split — the paper
  reaches these with its synthetic+real training setup; training with UIEB
  alone may give somewhat lower full-reference scores.
* **The included demo checkpoint** (`checkpoints_demo/`, evaluated in
  `results_demo/`) comes from a short 2-epoch run on 120 images just to prove
  the pipeline works on this CPU-only laptop. Its enhanced colors are still
  unbalanced (magenta cast) — train with the full command above (200 epochs on
  all 700 images) before judging the visual quality.
