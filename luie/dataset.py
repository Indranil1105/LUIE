"""
UIEB dataset handling (paper Sec. 4.1).

The paper splits the UIEB dataset (890 raw images + 890 reference images) into
    700 training images   (ground truth NOT used for training)
    190 test images       (reference images used ONLY for evaluation)

Because this repo trains with UIEB only (the paper additionally uses a labeled
synthetic dataset called Underwater3k as the "source" domain), the two domains
needed by the bi-directional adaptation (paper Sec. 3.3) are built from the
UIEB training images themselves, by their color cast:
    domain "green" : images with a green color cast (mean G > mean B)
    domain "blue"  : images with a blue  color cast (mean G <= mean B)

The paper swaps the background light between its synthetic and real domains;
here we swap it between these two real sub-domains, which keeps exactly the
same mechanism working without any synthetic data.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

RAW_DIR = "raw-890"        # folder with the raw underwater images
REF_DIR = "reference-890"  # folder with the reference (ground truth) images
IMG_SIZE = 256             # training resolution (H = W)


def load_image(path):
    """Load an image as a float tensor in [0, 1], shape (3, H, W)."""
    img = Image.open(path).convert("RGB")
    img = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def list_pairs(data_root):
    """Return all (raw_path, ref_path) pairs that share the same file name."""
    raw_dir = os.path.join(data_root, RAW_DIR)
    ref_dir = os.path.join(data_root, REF_DIR)
    pairs = []
    for name in sorted(os.listdir(raw_dir)):
        raw_path = os.path.join(raw_dir, name)
        ref_path = os.path.join(ref_dir, name)
        if os.path.isfile(raw_path) and os.path.isfile(ref_path):
            pairs.append((raw_path, ref_path))
    return pairs


def split_dataset(data_root, n_train=700, seed=42):
    """Reproduce the paper's 700 / 190 train-test split of UIEB.

    The exact split of the paper is not public, so we use a fixed random seed.
    """
    pairs = list_pairs(data_root)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n_train], pairs[n_train:]


def color_cast(raw_path):
    """Mean R, G, B value of an image (used to detect the color cast)."""
    img = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.float32) / 255.0
    return img.reshape(-1, 3).mean(axis=0)   # [R, G, B]


class UIEBTrain(Dataset):
    """Training images of ONE domain ("green" or "blue"), resized to 256x256.

    Only the raw images are used here - the reference images are never
    touched during training (unsupervised, as in the paper).
    """

    def __init__(self, pairs, domain, size=IMG_SIZE):
        self.size = size
        self.files = []
        for raw_path, _ in pairs:
            _, g, b = color_cast(raw_path)
            if (domain == "green" and g > b) or (domain == "blue" and g <= b):
                self.files.append(raw_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = load_image(self.files[idx])
        img = F.interpolate(img[None], size=(self.size, self.size),
                            mode="bilinear", align_corners=False)[0]
        return img


class UIEBTest(Dataset):
    """Test pairs (raw, reference) kept at their ORIGINAL resolution."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, ref_path = self.pairs[idx]
        return load_image(raw_path), load_image(ref_path), os.path.basename(raw_path)
