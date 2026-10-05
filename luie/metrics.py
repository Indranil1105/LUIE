"""
Evaluation metrics used in the paper (Sec. 4.3.1).

    Full-reference : PSNR, SSIM, LPIPS (AlexNet backbone, as in the paper)
    No-reference   : UIQM (Panetta et al.) and UCIQE (Yang & Sowmya)
                     Uranker is not included because it needs the heavy NIMA
                     quality-prediction model.

Every function takes a float numpy image with shape (H, W, 3) in [0, 1].

NOTE on absolute values: UIQM and UCIQE have many slightly different
implementations in the literature, so their absolute values are not directly
comparable to Table 1 of the paper (which used the authors' own port).
Comparisons inside this repo (raw vs enhanced, method vs method) are consistent.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EPS = 1e-8


# ---------------------------------------------------------------------------
# Full-reference metrics
# ---------------------------------------------------------------------------

def psnr(img, ref):
    """Peak Signal-to-Noise Ratio between the enhanced image and the reference."""
    return peak_signal_noise_ratio(ref, img, data_range=1.0)


def ssim(img, ref):
    """Structural Similarity, averaged over the RGB channels."""
    return structural_similarity(ref, img, channel_axis=2, data_range=1.0)


def lpips_value(model, img, ref):
    """LPIPS with the pre-trained AlexNet (same model as the training losses)."""
    import torch

    device = next(model.parameters()).device   # the device the model lives on

    def to_tensor(x):   # (H, W, 3) in [0, 1] -> (1, 3, H, W) in [-1, 1]
        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None]
        return x * 2.0 - 1.0

    with torch.no_grad():
        return float(model(to_tensor(img).to(device), to_tensor(ref).to(device)))


# ---------------------------------------------------------------------------
# No-reference metrics
# ---------------------------------------------------------------------------

def _eme(block_map, block=8):
    """Enhancement Measure (EME) over 8x8 blocks:
       EME = mean of 20*log10(max/min); blocks that contain a zero or are flat
       contribute 0 (same guard as the original MATLAB implementation)."""
    h, w = block_map.shape
    h, w = h - h % block, w - w % block
    blocks = block_map[:h, :w].reshape(h // block, block, w // block, block)
    emax = blocks.max(axis=(1, 3))
    emin = blocks.min(axis=(1, 3))
    ok = (emin > 0) & (emax > emin)
    vals = np.zeros_like(emax)
    vals[ok] = 20.0 * np.log10(emax[ok] / emin[ok])
    return float(vals.mean())


def _trimmed_mean(x, alpha=0.01):
    """Asymmetric alpha-trimmed mean (used by UICM)."""
    x = np.sort(x.ravel())
    k = int(x.size * alpha)
    return x[k:x.size - k].mean() if x.size - 2 * k > 0 else x.mean()


def uicm(img):
    """Chroma measure: -sqrt(T(rg)^2 + T(yb)^2) on the RG / YB opponent channels."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return -np.sqrt(_trimmed_mean(rg) ** 2 + _trimmed_mean(yb) ** 2)


def uism(img):
    """Sharpness measure: EME of the Sobel edge map of the grayscale image."""
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return _eme(np.sqrt(sx ** 2 + sy ** 2))


def uiconm(img):
    """Contrast measure: EME of the log image, averaged over the RGB channels."""
    return float(np.mean([_eme(np.log(img[..., c] + 1.0)) for c in range(3)]))


def uiqm(img):
    """UIQM = 0.0282*UICM + 0.2925*UISM + 3.1565*UIConM (Panetta et al.)."""
    return 0.0282 * uicm(img) + 0.2925 * uism(img) + 3.1565 * uiconm(img)


def uciqe(img):
    """UCIQE = c1*std(chroma) + c2*contrast(L) + c3*mean(saturation)
    (Yang & Sowmya). Chroma uses the Lab a/b channels centered at 0."""
    img8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    light, a, b = cv2.split(
        cv2.cvtColor(img8, cv2.COLOR_RGB2LAB).astype(np.float32))
    a, b = a - 128.0, b - 128.0           # undo the +128 offset of OpenCV Lab
    chroma = np.sqrt(a ** 2 + b ** 2)
    saturation = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)[..., 1] / 255.0

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * chroma.std()
                 + c2 * light.std() / (light.mean() + EPS)
                 + c3 * saturation.mean())
