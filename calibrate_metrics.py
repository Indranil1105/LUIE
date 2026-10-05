"""Calibrate UIQM/UCIQE implementation variants against the values the paper
reports for the raw UIEB images (Table 1, 'Input' row on UIEB):
    UIQM = 4.025   UCIQE = 28.82
"""
import glob

import cv2
import numpy as np
from PIL import Image

files = sorted(glob.glob("raw-890/*"))[:60]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


# ---------------- UIQM pieces ----------------
def trimmed(x, alpha=0.01):
    x = np.sort(x.ravel())
    k = int(x.size * alpha)
    return x[k:x.size - k].mean()


def uicm(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return -np.sqrt(trimmed(r - g) ** 2 + trimmed(0.5 * (r + g) - b) ** 2)


def eme(block_map, block=8, log=np.log10, guard="skip"):
    h, w = block_map.shape
    h, w = h - h % block, w - w % block
    blocks = block_map[:h, :w].reshape(h // block, block, w // block, block)
    mx, mn = blocks.max(axis=(1, 3)), blocks.min(axis=(1, 3))
    vals = np.zeros_like(mx)
    if guard == "skip":            # original MATLAB: zero/flat blocks count 0
        ok = (mn > 0) & (mx > mn)
        vals[ok] = 20.0 * log(mx[ok] / mn[ok])
    else:                          # eps division (my first version)
        vals = 20.0 * log((mx + 1e-8) / (mn + 1e-8))
    return vals.mean()


def uism(img, log):
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    return eme(np.sqrt(sx ** 2 + sy ** 2), log=log)


def uiconm(img, log):
    return np.mean([eme(np.log(img[..., c] + 1.0), log=log) for c in range(3)])


def uiqm(img, log):
    return 0.0282 * uicm(img) + 0.2925 * uism(img, log) + 3.1565 * uiconm(img, log)


# ---------------- UCIQE variants ----------------
def uciqe_v1(img):  # canonical: Lab a,b centered at 0 (MATLAB convention)
    i8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    l, a, b = cv2.split(cv2.cvtColor(i8, cv2.COLOR_RGB2LAB).astype(np.float32))
    a, b = a - 128, b - 128        # cv2 offsets a,b by +128
    chroma = np.sqrt(a ** 2 + b ** 2)
    s = cv2.cvtColor(i8, cv2.COLOR_RGB2HSV)[..., 1] / 255.0
    return 0.4680 * chroma.std() + 0.2745 * l.std() / l.mean() + 0.2576 * s.mean()


def uciqe_v2(img):  # chroma std = sqrt(std_a^2 + std_b^2), cv2 a,b in [0,255]
    i8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    l, a, b = cv2.split(cv2.cvtColor(i8, cv2.COLOR_RGB2LAB).astype(np.float32))
    s = cv2.cvtColor(i8, cv2.COLOR_RGB2HSV)[..., 1] / 255.0
    sc = np.sqrt(a.std() ** 2 + b.std() ** 2)
    return 0.4680 * sc + 0.2745 * l.std() / l.mean() + 0.2576 * s.mean()


def uciqe_v3(img):  # std over the whole flattened Lab image
    i8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(i8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    s = cv2.cvtColor(i8, cv2.COLOR_RGB2HSV)[..., 1] / 255.0
    return 0.4680 * lab.std() + 0.2745 * l.std() / l.mean() + 0.2576 * s.mean()


imgs = [load(p) for p in files]
print("n =", len(imgs))
print("UICM               :", round(float(np.mean([uicm(i) for i in imgs])), 3))
print("UIQM natural log   :", round(float(np.mean([uiqm(i, np.log) for i in imgs])), 3))
print("UIQM log10         :", round(float(np.mean([uiqm(i, np.log10) for i in imgs])), 3))
print("paper input UIQM   : 4.025")
print("UCIQE v1 (centered):", round(float(np.mean([uciqe_v1(i) for i in imgs])), 3))
print("UCIQE v2 (std_ab)  :", round(float(np.mean([uciqe_v2(i) for i in imgs])), 3))
print("UCIQE v3 (flat lab):", round(float(np.mean([uciqe_v3(i) for i in imgs])), 3))
print("paper input UCIQE  : 28.82")
