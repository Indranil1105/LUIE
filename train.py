"""
Training script of LUIE on UIEB (paper Sec. 3.3, 3.4 and 4.2).

Settings from the paper:
    Adam optimizer, learning rate = 2e-4, batch size = 1, 200 epochs
    loss weights: alpha = 1.0 (reconstruction), beta = 1.0 (decomposition),
                  gamma = 0.1 (color constancy)
    LPIPS with the pre-trained AlexNet for every perceptual loss
    (only J-Net is used at inference time, Fig. 2b)

UIEB-only adaptation (see README for details):
    * the synthetic supervision loss L_sup (Eq. 6) is skipped because the
      synthetic Underwater3k dataset is not available;
    * the bi-directional adaptation runs between the green-cast and blue-cast
      sub-domains of the 700 UIEB training images (the paper swaps the
      background light between its synthetic and real domains - same
      mechanism). The reference images are never used for training.

Run:
    python train.py --data_root . --epochs 200
"""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from luie.dataset import UIEBTrain, split_dataset
from luie.losses import LUIELosses
from luie.networks import LUIE


def endless(loader):
    """Yield batches forever, re-shuffling the loader on every pass."""
    while True:
        for batch in loader:
            yield batch


def train(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ----- data: the two sub-domains of the 700 UIEB training images -----
    train_pairs, _ = split_dataset(args.data_root, n_train=args.n_train, seed=args.seed)
    green_set = UIEBTrain(train_pairs, domain="green", size=args.size)
    blue_set = UIEBTrain(train_pairs, domain="blue", size=args.size)
    if args.limit > 0:                        # only for quick smoke tests
        green_set.files = green_set.files[:args.limit]
        blue_set.files = blue_set.files[:args.limit]
    print(f"training images: {len(train_pairs)} (green cast: {len(green_set)}, "
          f"blue cast: {len(blue_set)})")

    loader_g = DataLoader(green_set, batch_size=1, shuffle=True,
                          num_workers=args.workers)
    loader_b = DataLoader(blue_set, batch_size=1, shuffle=True,
                          num_workers=args.workers)

    # ----- model, losses, optimizer -----
    model = LUIE().to(device)
    j_params = sum(p.numel() for p in model.j_net.parameters())
    print(f"J-Net parameters (used at inference): {j_params / 1e6:.2f} M")

    losses = LUIELosses(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, "losses.csv"), "w", newline="")
    logger = csv.writer(log)
    logger.writerow(["epoch", "step", "rec", "dec", "cc", "total"])

    # ----- training loop -----
    it_g = endless(loader_g)
    it_b = endless(loader_b)
    steps_per_epoch = max(len(loader_g), len(loader_b))

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step in range(1, steps_per_epoch + 1):
            img_g = next(it_g).to(device)     # one green-cast image
            img_b = next(it_b).to(device)     # one blue-cast image

            # 1) decompose both domains with B-Net, J-Net, D-Net, T-Net
            A_g, J_g, d_g, t_g = model.decompose(img_g)
            A_b, J_b, d_b, t_b = model.decompose(img_b)

            # 2) intra-domain reconstruction with Eq. (4), then loss Eq. (8)
            rec_g = model.reconstruct(J_g, t_g, A_g)
            rec_b = model.reconstruct(J_b, t_b, A_b)
            L_rec = losses.rec(img_g, rec_g) + losses.rec(img_b, rec_b)

            # 3) inter-domain mixed images: swap the background light, Eq. (5)
            m1 = model.reconstruct(J_g, t_g, A_b)   # green scene radiance + blue background
            m2 = model.reconstruct(J_b, t_b, A_g)   # blue scene radiance + green background

            # ... and decompose the mixed images again
            A1, J1, d1, _ = model.decompose(m1)
            A2, J2, d2, _ = model.decompose(m2)

            # 4) inter-domain decomposition loss, Eq. (9):
            #    m1 must give back (J_g, d_g) and the swapped-in background A_b
            #    m2 must give back (J_b, d_b) and the swapped-in background A_g
            L_dec = losses.dec((A1, J1, d1), (A_b, J_g, d_g))
            L_dec = L_dec + losses.dec((A2, J2, d2), (A_g, J_b, d_b))

            # 5) color constancy on both scene radiances, Eq. (10)
            L_cc = losses.cc(J_g) + losses.cc(J_b)

            # 6) total loss, Eq. (11). L_sup = 0 because we have no synthetic data.
            loss = args.alpha * L_rec + args.beta * L_dec + args.gamma * L_cc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 1 or step % args.log_every == 0:
                print(f"epoch {epoch:3d}  step {step:4d}/{steps_per_epoch}  "
                      f"rec {L_rec.item():.4f}  dec {L_dec.item():.4f}  "
                      f"cc {L_cc.item():.4f}  total {loss.item():.4f}")
                logger.writerow([epoch, step, round(L_rec.item(), 4),
                                 round(L_dec.item(), 4), round(L_cc.item(), 4),
                                 round(loss.item(), 4)])
                log.flush()

        # ----- save checkpoints -----
        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = os.path.join(args.out_dir, "luie_last.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch}, path)
            print(f"saved checkpoint -> {path}")

    log.close()


def main():
    parser = argparse.ArgumentParser(description="Train LUIE on UIEB")
    parser.add_argument("--data_root", default=".", help="folder containing raw-890/ and reference-890/")
    parser.add_argument("--out_dir", default="checkpoints", help="where to save checkpoints and the loss log")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs (paper: 200)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (paper: 2e-4)")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight of the reconstruction loss (paper: 1.0)")
    parser.add_argument("--beta", type=float, default=1.0, help="weight of the decomposition loss (paper: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.1, help="weight of the color constancy loss (paper: 0.1)")
    parser.add_argument("--size", type=int, default=256, help="training resolution")
    parser.add_argument("--n_train", type=int, default=700, help="number of UIEB training images (paper: 700)")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--save_every", type=int, default=20, help="save the checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=50, help="print the losses every N steps")
    parser.add_argument("--limit", type=int, default=0, help="use only N images per domain (0 = all, for smoke tests)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()



## python train.py --data_root . --epochs 200
## python test.py --checkpoint checkpoints/luie_last.pth --data_root .