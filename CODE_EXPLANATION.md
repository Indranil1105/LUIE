# How the LUIE code works — a guided tour from the paper to the code

This document explains, part by part, how the paper

> **LUIE: Learnable physical model-guided underwater image enhancement with
> bi-directional unsupervised domain adaptation** (Pan et al., Neurocomputing
> 2024)

was turned into the code in this folder, and **which line of code implements
which part of the paper**. Everything is written in plain language on purpose.

---

## 1. The one-minute big picture

The paper's idea in one sentence:

> Split a degraded underwater image `I` into three physical parts — the clear
> scene `J`, the water's color cast `A`, and the transmission `t` — using the
> equation `I = J·t + A·(1−t)`, learn all three parts with tiny CNNs, and output
> `J` (the clear image) at test time.

```
                        TRAINING (train.py)
   green-cast image ──► [B-Net, J-Net, D-Net → T-Net] ──► A, J, depth, t
   blue-cast  image ──► [B-Net, J-Net, D-Net → T-Net] ──► A, J, depth, t
            │                                        │
            │   rebuild with I = J·t + A·(1−t)  ──► must match input   (L_rec)
            │   swap the two A's, rebuild 2 mixed images,
            │   decompose them again ──► must match original parts    (L_dec)
            │   mean R,G,B of both J's ──► must be equal              (L_cc)
            ▼
                     backprop + Adam (one step)

                        TESTING (test.py)
   new image ──► [J-Net ONLY] ──► enhanced image ──► metrics vs reference
```

Two facts make this method special:

* **Only 5 conv layers per network** (~0.11 M parameters in total at
  inference) — the whole model is smaller than almost every competitor.
* **No ground truth is needed for training.** The losses only compare the
  model's own pieces with each other (self-supervised), which is why the
  reference images of UIEB are only used at test time.

---

## 2. Master map: paper → code

| Paper part | Where in the paper | Where in the code |
|---|---|---|
| Physical model `I = J·t + A·(1−t)` | Eq. (1), Sec. 1 | `luie/networks.py` → `LUIE.reconstruct` (line 82) |
| Learnable transmission `t = f_t(d)` instead of `t = e^(−βd)` | Eq. (3), Sec. 3.2 | `luie/networks.py` → `LUIE.decompose` (line 66): `t = self.t_net(depth)` |
| Learnable model `I = J·f_t(d) + A·(1−f_t(d))` | Eq. (4) | same two lines combined: `decompose` + `reconstruct` |
| B-Net / J-Net / D-Net / T-Net, 5 conv layers, 64 ch, 3×3, InstanceNorm+ReLU, sigmoid | Sec. 3.2 "Network structure", Fig. 2c | `luie/networks.py` → `FiveLayerCNN` (line 24) and `LUIE.__init__` (line 59) |
| Bi-directional intra-domain reconstruction | Sec. 3.3, Eq. (8) | `train.py` lines 90–93 + `luie/losses.py` → `rec` (line 54) |
| Inter-domain mixed images `X_m1, X_m2` (swap background light) | Sec. 3.3, Eq. (5) | `train.py` lines 96–101 |
| Inter-domain decomposition loss `L_dec` | Sec. 3.3, Eq. (9) | `train.py` lines 106–107 + `luie/losses.py` → `dec` (line 58) |
| Color constancy (gray world) | Sec. 3.4, Eq. (10) | `luie/losses.py` → `cc` (line 67) |
| Synthetic supervision `L_sup` | Sec. 3.4, Eq. (6) | `luie/losses.py` → `sup` (line 48) — implemented but inactive (no synthetic data, see Sec. 8 of this file) |
| Total loss `L_sup + α·L_rec + β·L_dec + γ·L_cc` | Eq. (11), Sec. 4.2 | `train.py` line 113 |
| Adam, lr 2e-4, batch 1, 200 epochs, α=β=1.0, γ=0.1 | Sec. 4.2 | `train.py` defaults + `main()` argparse (lines 137–153) |
| UIEB: 700 train (no GT) / 190 test | Sec. 4.1 | `luie/dataset.py` → `split_dataset` (line 54) |
| Inference uses **only** J-Net | Fig. 2b, Sec. 3.1 | `test.py` line 68: `model.j_net(...)` |
| PSNR / SSIM / LPIPS / UIQM / UCIQE | Sec. 4.3.1 | `luie/metrics.py` (whole file) |

The next sections walk through every file in the same order the paper does.

---

## 3. `luie/networks.py` — the physical model and its four networks
*(paper Sec. 3.2 "Learnable underwater image formation model" + "Network structure", Fig. 2c)*

### 3.1 One network template for all four nets — `FiveLayerCNN` (line 24)

The paper says every sub-network has the same structure: *"five convolutional
layers and four instance normalization layers, with ReLU activation functions
applied after each layer. Furthermore, a sigmoid function is used in the final
layer"*. That is literally what the class builds:

```python
nn.Conv2d(in_channels, 64, 3, padding=1)  ┐
nn.InstanceNorm2d(64)                     │ convs 1–4: conv → norm → ReLU
nn.ReLU(inplace=True)                     ┘   (repeated 4 times)
nn.Conv2d(64, out_channels, 3, padding=1)     conv 5: a single conv ...
nn.Sigmoid()                                  ... then squash into [0, 1]
```

Two details worth understanding:

* **Why a sigmoid at the end?** All four outputs are *physical quantities*
  bounded between 0 and 1: colors are in [0,1], depth is relative, transmission
  must be in [0,1]. The sigmoid guarantees this for free — no clipping needed.
* **Why ReLU only after the first four convs?** If a ReLU came right before the
  sigmoid, every output value would be ≥ 0 before squashing, so the sigmoid
  could only output ≥ 0.5 — transmission could never be darker than 0.5, which
  is physically wrong. (This matches the paper's *"five conv layers, four
  instance norms"* counting.)

**Parameter check (why this is the paper's "0.11M"):** one 3→3 network has
`3·64·9 + 3·(64·64·9) + 64·3·9` ≈ 114,000 weights ≈ **0.11M** — exactly the
number reported in Table 4 of the paper for inference.

### 3.2 The four networks — `LUIE.__init__` (line 59)

```python
self.b_net = FiveLayerCNN(3, 3)   # B-Net: image        -> background light A
self.j_net = FiveLayerCNN(3, 3)   # J-Net: image        -> scene radiance J
self.d_net = FiveLayerCNN(3, 1)   # D-Net: image        -> depth d     (1 channel)
self.t_net = FiveLayerCNN(1, 1)   # T-Net: depth        -> transmission t (1 channel)
```

* B-Net and J-Net read a color image (3 channels in, 3 channels out).
* D-Net predicts depth as a single-channel map (depth has no color).
* T-Net reads the **depth map**, not the image — this is the paper's key
  trick. Old methods hard-code `t = e^(−β·d)` and must guess the attenuation
  `β`; here T-Net **learns** the depth→transmission mapping from data
  (paper Eq. 3). That is the "learnable physical model" of the title.

### 3.3 `decompose()` (line 66) — running all four nets, paper Eq. (3)

```python
A = self.b_net(image)      # background light (color cast),      (B,3,H,W)
J = self.j_net(image)      # scene radiance = the clear image,   (B,3,H,W)
depth = self.d_net(image)  # depth map,                          (B,1,H,W)
t = self.t_net(depth)      # transmission t = f_t(depth),        (B,1,H,W)
return A, J, depth, t
```

### 3.4 `reconstruct()` (line 82) — paper Eq. (1) = Eq. (4)

```python
return J * t + A * (1.0 - t)
```

One line, element-wise over every pixel: where transmission is high (clear
water) you mostly see `J`; where it is low (murky water) you mostly see the
haze color `A`. Because `t` came from the **learned** T-Net instead of a fixed
`e^(−βd)`, this line *is* the paper's Eq. (4).

---

## 4. `luie/dataset.py` — the UIEB data and the two domains
*(paper Sec. 4.1 "Datasets", UIEB paragraph)*

The paper: *"we partition the UIEB dataset into 700 training images (excluding
the use of ground truth) and 190 testing images."*

* `list_pairs` (line 41) — walks through `raw-890/` and matches every raw image
  with the file of the same name in `reference-890/`. All 890 names pair up.
* `split_dataset` (line 54) — shuffles the 890 pairs with a fixed seed (42) and
  returns the first **700** as training pairs and the remaining **190** as test
  pairs. The paper's exact split list is not public, so a seeded random split
  is the honest equivalent.
* `load_image` (line 34) — reads a PNG/JPG, converts to RGB, scales pixels from
  0–255 to 0.0–1.0, and returns a `(3, H, W)` float tensor (PyTorch's layout).
* `color_cast` (line 65) — computes the mean R/G/B of a training image. This is
  used only for the **two-domain split** explained in Sec. 8.
* `UIEBTrain` (line 71) — the training dataset of ONE domain: it keeps only the
  raws whose cast matches (`"green"` if mean G > mean B, else `"blue"`) and
  resizes each image to 256×256 (`__getitem__`, line 89). Notice it receives
  the pairs but **throws the reference path away** — training never looks at
  ground truth, exactly like the paper.
* `UIEBTest` (line 96) — the test dataset: returns `(raw, reference, name)`
  at **original resolution** (no resize), because metrics should be measured on
  real-size images.

---

## 5. `luie/losses.py` — the four loss functions
*(paper Sec. 3.4 "Objective function", Eqs. 6–11)*

One class holds everything: `LUIELosses` (line 26).

### 5.1 The shared measuring stick — `__init__` (line 29)

```python
self.lpips = lpips.LPIPS(net="alex").to(device)
```

Every loss in the paper is built on **LPIPS with the pre-trained AlexNet**
(the paper: *"we employ the LPIPS with layers from the pre-trained AlexNet"*).
LPIPS is a "perceptual distance": 0 means the two images *look* identical,
bigger means they look more different (it compares features inside AlexNet, so
it cares about textures and structures more than raw pixel values). Its weights
are frozen (`requires_grad = False`) — it is a measuring stick, not a part being
trained.

The helper `_perceptual(a, b)` (line 37) wraps one LPIPS call and contains one
small trick: depth and transmission maps have **1 channel**, but LPIPS expects
**3-channel color**, so single-channel maps are copied three times
(`a.repeat(1, 3, 1, 1)`). Without this, every depth-map comparison would crash.

### 5.2 `sup(...)` (line 48) — paper Eq. (6), the synthetic supervision

```python
L_d = LPIPS(pred_depth, gt_depth)      # depth of a synthetic image vs its true depth
L_j = LPIPS(pred_J,     gt_J)          # scene radiance vs true clear image
L_b = LPIPS(pred_A,     gt_A)          # background light vs true background
```

The paper trains on a labeled synthetic dataset (Underwater3k, made in Blender)
where the true depth / radiance / background are known, so these three LPIPS
terms directly teach the networks what each component should look like.
**In this repo it is implemented but never called** — we have no Underwater3k
(see Sec. 8).

### 5.3 `rec(image, reconstruction)` (line 54) — paper Eq. (8), intra-domain

One LPIPS between an input image and the reconstruction built from its own
components (Eq. 4). In `train.py` this is evaluated for BOTH domains and added:

```python
L_rec = LPIPS(J_g·t_g + A_g·(1−t_g), img_g) + LPIPS(J_b·t_b + A_b·(1−t_b), img_b)
```

*Why it matters:* this is the equation that ties the four networks together.
If J, A, and t were arbitrary, the rebuild would not look like the input.
Forcing the rebuild to match the input is what makes the decomposition
*physically consistent* — the paper calls this "intra-domain decomposition and
reconstruction".

### 5.4 `dec(mixed, original)` (line 58) — paper Eq. (9), inter-domain

This is the heart of the "bi-directional domain adaptation". It receives two
lists of 3 components each — the re-decomposition of one mixed image, and the
values each component *should* have kept — and returns the sum of three LPIPS:

```python
loss = LPIPS(A_new, A_kept) + LPIPS(J_new, J_kept) + LPIPS(d_new, d_kept)
```

Calling it twice (once per mixed image) gives the six LPIPS terms of Eq. (9).
What it enforces, in words:

> "If I take the clear scene from image 1 and the water color from image 2 and
> mix them, then decomposing that mixture must give back exactly the same
> scene, depth, and water color."

That can only hold if J-Net and D-Net learned **domain-invariant** features
(they respond to the scene, not to the water color) — which is precisely how
the paper closes the gap between its two domains.

### 5.5 `cc(J)` (line 67) — paper Eq. (10), gray-world color constancy

```python
mean_rgb = J.mean(dim=(2, 3))            # one mean value per R, G, B channel
r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
return |r−g| + |r−b| + |g−b|             # averaged over the batch
```

The gray-world assumption says a natural photo's three channel means should be
roughly equal — a leftover color cast makes one mean dominate. The loss simply
punishes any difference between the three channel means of the scene radiance
`J`, for both domains. This is what pulls the enhanced colors back to neutral.
(The paper sets its weight small: γ = 0.1 — a gentle push, so the loss does not
wash out true colors like red coral.)

### 5.6 Putting them together — Eq. (11)

Done in `train.py` line 113:

```python
loss = alpha * L_rec + beta * L_dec + gamma * L_cc      # + L_sup (Eq. 6) if synthetic data exists
```

with α = 1.0, β = 1.0, γ = 0.1 — the exact weights of Sec. 4.2.

---

## 6. `train.py` — the training loop, one step at a time
*(paper Sec. 3.3 + Sec. 4.2, Fig. 2a)*

### 6.1 Setup (lines 42–73)

* **Data (48–60):** the two domain datasets (`green`, `blue`) become two
  DataLoaders with **batch size = 1** (the paper's setting) and shuffling.
  `endless()` (line 35) wraps a loader so it never runs out — when the shorter
  domain is exhausted it simply starts over, so both domains feed the loop
  every step.
* **Model + losses + optimizer (63–68):** `LUIE()`, the `LUIELosses` above, and
  **Adam with lr = 2e-4** — the paper's Sec. 4.2 settings.
* **Logging (70–73):** `losses.csv` gets one row per log point so you can plot
  the training curve afterwards.

`steps_per_epoch = max(len(green), len(blue))` — one epoch visits the bigger
domain exactly once.

### 6.2 One training step, with real tensor shapes

Each of the six numbered blocks in the loop (lines 86–113) maps to a piece of
the paper. Sizes shown for one 256×256 image:

```python
# 1) decompose BOTH domains   (Sec. 3.3 "Intra-domain Decomposition")
A_g, J_g, d_g, t_g = model.decompose(img_g)     # img_g: (1,3,256,256)
A_b, J_b, d_b, t_b = model.decompose(img_b)     # A,J: (1,3,H,W); d,t: (1,1,H,W)

# 2) intra-domain reconstruction  (Eq. 4) + LPIPS  (Eq. 8)
rec_g = model.reconstruct(J_g, t_g, A_g)        # = J_g·t_g + A_g·(1−t_g)
L_rec = losses.rec(img_g, rec_g) + losses.rec(img_b, rec_b)

# 3) inter-domain mixed images — SWAP the background light   (Eq. 5)
m1 = model.reconstruct(J_g, t_g, A_b)   # green scene + blue water color
m2 = model.reconstruct(J_b, t_b, A_g)   # blue scene + green water color

#    ... and push them through the SAME four networks again
A1, J1, d1, _ = model.decompose(m1)
A2, J2, d2, _ = model.decompose(m2)

# 4) inter-domain decomposition loss   (Eq. 9)
#    m1 must return the green scene (J_g, d_g) and the swapped-in A_b
#    m2 must return the blue  scene (J_b, d_b) and the swapped-in A_g
L_dec  = losses.dec((A1, J1, d1), (A_b, J_g, d_g))
L_dec += losses.dec((A2, J2, d2), (A_g, J_b, d_b))

# 5) color constancy on both scene radiances   (Eq. 10)
L_cc = losses.cc(J_g) + losses.cc(J_b)

# 6) total loss   (Eq. 11) — L_sup stays 0 because we have no synthetic data
loss = args.alpha * L_rec + args.beta * L_dec + args.gamma * L_cc
```

Then the ordinary PyTorch ritual (lines 115–117): `zero_grad → backward →
step`. Every LPIPS call is differentiable, so gradients flow from all three
losses back into all four networks — including through the reconstruction
equation `J·t + A·(1−t)`, which is what makes the *physics* trainable.

### 6.3 Checkpoints (lines 129–132)

Every `--save_every` epochs (and at the end) the full state of all four
networks is stored in one dictionary:

```python
torch.save({"model": model.state_dict(), "epoch": epoch}, "checkpoints/luie_last.pth")
```

`test.py` later loads exactly this dictionary with `model.load_state_dict`.

---

## 7. `test.py` — inference and evaluation
*(paper Fig. 2b "inference pipeline" + Sec. 4.3.1)*

### 7.1 Inference is ONE network call (line 68)

```python
enhanced = model.j_net(raw[None].to(device))[0].clamp(0, 1).cpu()
```

This is the point the paper is proud of (Fig. 2b): at test time **only J-Net
runs**. B-Net, D-Net, T-Net, all the reconstruction, all the domain swapping —
that machinery existed only to *train* J-Net well. A new underwater image goes
in, the clear version comes out, `clamp(0,1)` just guards against float noise
at the boundaries. The image keeps its **original resolution** because the
networks are pure convolutions (no fully-connected layer that fixes the size).

### 7.2 Measuring quality (lines 75–84)

For each of the 190 test images the script fills one row of a table:

* **Full-reference (need the reference image):** `psnr`, `ssim`,
  `lpips_value` — how close the enhanced image is to the human-made reference.
  Higher PSNR/SSIM is better; lower LPIPS is better.
* **No-reference (no reference needed):** `uiqm`, `uciqe` — measured on the raw
  image *and* on the enhanced one, so you can see the improvement, not just the
  absolute value.

Results go to `results/results.csv` (per image) and the averages are printed as
the small table you saw at the end of your GPU run.

### 7.3 The device fix (line 38–49 of `luie/metrics.py`)

`lpips_value` now reads the device from the model itself
(`next(model.parameters()).device`) and moves both input tensors there — this
is the one-line bug you hit on the GPU machine, already fixed in this folder.

---

## 8. `luie/metrics.py` — the five metrics in plain words
*(paper Sec. 4.3.1)*

**Full-reference** (compare enhanced image vs reference image):

* `psnr` (line 28) — "how many pixels differ, in decibels". Higher = closer to
  the reference. Standard skimage implementation, data range fixed to 1.0
  because our images live in [0, 1].
* `ssim` (line 33) — "do structures (edges, objects) look the same?". Computed
  per RGB channel and averaged (`channel_axis=2`). Higher = better.
* `lpips_value` (line 38) — the same AlexNet-LPIPS used in training, but now as
  a *score*: lower = perceptually more similar to the reference. Inputs must be
  scaled from [0,1] to [−1,1] because that is the range LPIPS expects.

**No-reference** (judge the image on its own — these are the "UIQA" metrics of
the paper):

* `uiqm` (line 99) = `0.0282·UICM + 0.2925·UISM + 3.1565·UIConM`
  (Panetta et al.). Three sub-scores, each built on an "EME" block statistic
  (`_eme`, line 56) that measures local contrast in 8×8 blocks:
  * `uicm` (78): chroma health — measured on the red−green and
    yellow−blue opponent channels with an α-trimmed mean (outliers ignored);
    negative values mean a color cast.
  * `uism` (86): sharpness — EME of the Sobel edge map.
  * `uiconm` (94): contrast — EME of the log image, averaged over RGB.
  Blocks whose minimum is 0 (pure black) or that are flat contribute 0, the
  same guard as the original MATLAB code.
* `uciqe` (line 104) = `0.4680·std(chroma) + 0.2745·contrast(L) +
  0.2576·mean(saturation)` (Yang & Sowmya), mixing CIELAB and HSV. Note the
  `a − 128, b − 128` line: OpenCV shifts the Lab a/b channels by +128 and the
  canonical chroma is measured around 0.

**Why your absolute UIQM/UCIQE numbers differ from Table 1 of the paper:**
these two metrics have many slightly different ports in the literature
(scaling of Lab, log base, trimming details...). The paper's own port reports
e.g. UIQM ≈ 4.0 for raw UIEB, this repo reports ≈ 17 — *different scales*.
What is comparable: raw vs enhanced inside the same implementation, i.e. your
UIQM jump 17.12 → 32.21 and UCIQE 4.08 → 5.76 are genuine improvements.
PSNR / SSIM / LPIPS, on the other hand, are standardized and directly
comparable.

---

## 9. What was adapted because we only have UIEB (and why it is still the paper)

| Paper | This repo | Why |
|---|---|---|
| Source domain = labeled synthetic Underwater3k | Source "domain" = green-cast half of the 700 training raws | Underwater3k is not public/available. The adaptation *mechanism* (swap A, rebuild, re-decompose, Eq. 5+9) is kept exactly; only the two domains change. |
| Target domain = real UIEB | Target "domain" = blue-cast half of the same 700 raws | Same images, same unsupervised setting. |
| `L_sup` active (synthetic GT exists) | `L_sup` implemented but = 0 | There is no synthetic ground truth to supervise with. |
| UIEB 700/190 split (their private list) | Same sizes, fixed seed 42 | Their exact split list is not published. |
| Train at 256×256 (implied) | `--size 256` default | Not stated in the paper; 256 is the common choice for batch-1 UIE training. |

Everything else — the four networks, the formation model, all equations, all
loss weights, the optimizer settings, the J-Net-only inference, the metrics —
is implemented as written in the paper.

---

## 10. Your final results vs the paper (honest reading)

Your 200-epoch GPU run on the UIEB test split:

| Metric | raw | enhanced (yours) | paper's LUIE (Table 1/7) |
|---|---|---|---|
| PSNR ↑ | – | **17.57** | 22.87 |
| SSIM ↑ | – | **0.7753** | 0.9167 |
| LPIPS ↓ | – | **0.3013** | 0.1241 |
| UIQM ↑ | 17.12 | **32.21** | 5.246 (different port/scale!) |
| UCIQE ↑ | 4.08 | **5.76** | 31.33 (different port/scale!) |

* **PSNR/SSIM/LPIPS:** your model clearly improves over the raw inputs
  (raw PSNR on UIEB ≈ 17.18 in the paper's Table 1 — you are above it), but it
  is below the paper's 22.87. The main reason is expected: the paper trains
  with the labeled synthetic dataset (its `L_sup` teaches J-Net what a *clear*
  image looks like), which this UIEB-only re-implementation does not have. The
  different 700/190 split also shifts the numbers by a bit.
* **UIQM/UCIQE:** the raw→enhanced improvements (+15.1 UIQM, +1.7 UCIQE) show
  the enhancement works; the absolute values are on this repo's metric scale
  and cannot be compared against Table 1 directly (different ports).

---

## 11. Cheat sheet — where to tweak what

| I want to... | Change |
|---|---|
| Train longer / shorter | `--epochs N` in `train.py` |
| Use a different training resolution | `--size N` (default 256) |
| Re-balance the losses | `--alpha / --beta / --gamma` (paper: 1.0 / 1.0 / 0.1) |
| Try a different train/test split | `--seed N` (use the SAME seed in `test.py`) |
| Feed your own images (no reference needed) | put them in a folder, or call `model.j_net` — see `test.py` line 68 |
| See the loss curve | plot `checkpoints/losses.csv` |
| See visual results | `python make_comparison.py --results_dir results --n 4` |
| Evaluate other people's methods with the same metrics | import `luie/metrics.py` — every function takes a plain `(H, W, 3)` numpy image in [0, 1] |
| Re-enable synthetic supervision | you need a dataset with (image, depth, radiance, background) GT; then add `losses.sup(...)` into `train.py` line 113's total |

---

## 12. Reading order suggestion

If you want to *really* understand the code, read it in this order
(each file is < 160 lines):

1. `luie/networks.py` — 10 minutes. You now know the model.
2. `train.py` lines 86–113 only — 10 minutes. You now know the training.
3. `luie/losses.py` — 10 minutes. You now know *why* it learns.
4. `test.py` — 5 minutes. You now know how results are produced.
5. `luie/dataset.py`, `luie/metrics.py` — the plumbing.

*End of the guided tour — the code is intentionally short; the paper is in
every line.*
