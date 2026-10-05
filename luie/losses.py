"""
Loss functions of LUIE (paper Sec. 3.4, Eqs. 6-11).

Every perceptual loss uses LPIPS with the pre-trained AlexNet, exactly as
stated in the paper ("we employ the LPIPS with layers from the pre-trained
AlexNet").

    L_sup  (Eq. 6)  : supervision of the decomposed SYNTHETIC components
                      (depth / scene radiance / background) with their ground
                      truth. It needs the synthetic Underwater3k dataset, so it
                      is NOT used in this UIEB-only re-implementation.
    L_rec  (Eq. 8)  : LPIPS between the input image and its reconstruction
                      built with Eq. (4). Keeps the decomposition consistent.
    L_dec  (Eq. 9)  : LPIPS between the re-decomposed components of the two
                      inter-domain mixed images (Eq. 5) and the original
                      components. Reduces the gap between the two domains.
    L_cc   (Eq. 10) : gray-world color constancy on the scene radiance J.
    L_total(Eq. 11) : L_sup + alpha*L_rec + beta*L_dec + gamma*L_cc
                      with alpha = 1.0, beta = 1.0, gamma = 0.1.
"""

import lpips
import torch


class LUIELosses:
    """All four loss functions of the paper, sharing one LPIPS-AlexNet model."""

    def __init__(self, device):
        self.lpips = lpips.LPIPS(net="alex").to(device)
        for p in self.lpips.parameters():      # LPIPS is pre-trained and frozen
            p.requires_grad = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _perceptual(self, a, b):
        """LPIPS between two image tensors, 1-channel maps are repeated to 3."""
        if a.size(1) == 1:                     # depth / transmission maps
            a = a.repeat(1, 3, 1, 1)
        if b.size(1) == 1:
            b = b.repeat(1, 3, 1, 1)
        return self.lpips(a, b).mean()

    # ------------------------------------------------------------------
    # the four losses of the paper
    # ------------------------------------------------------------------
    def sup(self, pred_depth, gt_depth, pred_j, gt_j, pred_b, gt_b):
        """Eq. (6): synthetic supervision loss (needs synthetic ground truth)."""
        return (self._perceptual(pred_depth, gt_depth)
                + self._perceptual(pred_j, gt_j)
                + self._perceptual(pred_b, gt_b))

    def rec(self, image, reconstruction):
        """Eq. (8): LPIPS between an input image and its reconstruction."""
        return self._perceptual(reconstruction, image)

    def dec(self, mixed_components, original_components):
        """Eq. (9): LPIPS between 3 re-decomposed components of one mixed image
        and their original values. Components are ordered (A, J, depth)."""
        loss = 0.0
        for new, old in zip(mixed_components, original_components):
            loss = loss + self._perceptual(new, old)
        return loss

    @staticmethod
    def cc(J):
        """Eq. (10): gray-world color constancy.
        Pushes the mean R, G, B values of the scene radiance J towards each
        other, which reduces the color cast."""
        mean_rgb = J.mean(dim=(2, 3))                 # (B, 3): mean of each channel
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return (r - g).abs().mean() + (r - b).abs().mean() + (g - b).abs().mean()
