"""
Networks of LUIE (paper Sec. 3.2 "Network structure" and Fig. 2c).

The paper uses four tiny CNNs, all with the SAME structure:
    B-Net : underwater image -> background light A(x)        (3 channels out)
    J-Net : underwater image -> scene radiance J(x)          (3 channels out)
            J is the enhanced image and the ONLY network used at inference.
    D-Net : underwater image -> depth map d(x)               (1 channel out)
    T-Net : depth map d(x)   -> transmission t(x)            (1 channel out)

Each network is simply:
    5 convolutional layers (64 channels, 3x3 kernels)
    + instance normalization + ReLU after each of the first 4 layers
    + a sigmoid at the end so the output stays in [0, 1].

This tiny size is why the whole model has only ~0.11M parameters
(Table 4 in the paper).
"""

import torch
import torch.nn as nn


class FiveLayerCNN(nn.Module):
    """The small 5-conv-layer network shared by B-Net, J-Net, D-Net and T-Net."""

    def __init__(self, in_channels, out_channels, width=64):
        super().__init__()
        self.layers = nn.Sequential(
            # conv 1-4: conv -> instance norm -> ReLU
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(width),
            nn.ReLU(inplace=True),

            # conv 5: final conv -> sigmoid (output range [0, 1])
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class LUIE(nn.Module):
    """The whole LUIE model = B-Net + J-Net + D-Net + T-Net (paper Fig. 2)."""

    def __init__(self):
        super().__init__()
        self.b_net = FiveLayerCNN(3, 3)   # background light A(x)
        self.j_net = FiveLayerCNN(3, 3)   # scene radiance J(x) -> enhanced image
        self.d_net = FiveLayerCNN(3, 1)   # depth map d(x)
        self.t_net = FiveLayerCNN(1, 1)   # transmission t(x), learned from depth

    def decompose(self, image):
        """Run the four sub-networks on an image (paper Eq. 3).

        Returns:
            A     : background light  (B, 3, H, W)
            J     : scene radiance    (B, 3, H, W)
            depth : depth map         (B, 1, H, W)
            t     : transmission      (B, 1, H, W)
        """
        A = self.b_net(image)
        J = self.j_net(image)
        depth = self.d_net(image)
        t = self.t_net(depth)          # Eq. (3): t = f_t(depth), the "learnable" part
        return A, J, depth, t

    @staticmethod
    def reconstruct(J, t, A):
        """Underwater image formation model, Eq. (1) / Eq. (4):

               I = J * t + A * (1 - t)
        """
        return J * t + A * (1.0 - t)

    def forward(self, image):
        """Decompose an image and rebuild it with Eq. (4).
        At inference you normally only need `model.j_net(image)`."""
        A, J, depth, t = self.decompose(image)
        return J, self.reconstruct(J, t, A)
