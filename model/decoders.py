"""
CNN Decoders for Ptychography Reconstruction

This module contains decoder architectures used in ptychography reconstruction models.
"""

import torch
import torch.nn as nn


class TransposeConvBlock(nn.Module):
    """Transposed convolutional block with ConvTranspose2d, BatchNorm, activation, and optional dropout."""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation='relu', use_batchnorm=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation is not None:
            raise ValueError(f"Unsupported activation: {activation}")

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomActivation(nn.Module):
    """Custom activation function that can be used in decoders.

    LeCun, et al. Efficient BackProp 1998
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Example: You can modify this to any custom activation you want
        return 1.7159 * torch.tanh((2/3) * x)


class Decoder256(nn.Module):
    """CNN Decoder for 256x256 images with configurable depth.

    Mirrors encoder structure: Encoder has Conv2d, Conv2d, MaxPool2d per stage,
    Decoder has ConvTranspose2d, ConvTranspose2d, Upsample(bilinear) per stage.

    Args:
        latent_dim: Input channels from bottleneck (default: 512)
        base_channels: Base number of channels for final layer (default: 64)
        out_channels: Number of output channels (default: 1)
        use_batchnorm: Whether to use batch normalization (default: True)
        output_activation: Output activation function (default: None)
            Can be 'sigmoid', 'tanh', 'custom', or None
        dropout: Dropout rate (default: 0.0)
        num_stages: Number of upsampling stages, should match encoder (default: 4)
            - 4 stages: 16 -> 256 (spatial size)
            - 5 stages: 8 -> 256
            - 6 stages: 4 -> 256
    """
    def __init__(self, latent_dim=512, base_channels=64, out_channels=1, use_batchnorm=True, output_activation=None, dropout=0.0, num_stages=4):
        super().__init__()

        self.num_stages = num_stages
        self.stages = nn.ModuleList()

        # Calculate channel progression (reverse of encoder)
        # Build channel list that mirrors encoder
        encoder_channels = []
        for i in range(num_stages):
            if i < num_stages - 1:
                ch = min(base_channels * (2 ** i), latent_dim)
            else:
                ch = latent_dim
            encoder_channels.append(ch)

        # Reverse for decoder: start from latent_dim, go back to base_channels
        in_ch = latent_dim
        for i in range(num_stages):
            # Determine output channels for this stage
            if i < num_stages - 1:
                # Not the last stage - use the channel from encoder in reverse
                out_ch = encoder_channels[num_stages - 2 - i]
            else:
                # Last stage outputs base_channels
                out_ch = base_channels

            # Create stage with two ConvTranspose2d (mirroring two Conv2d) and bilinear upsample (undoing MaxPool2d)
            stage = nn.Sequential(
                TransposeConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
                TransposeConvBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            self.stages.append(stage)
            in_ch = out_ch

        # Final output layer
        # self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.output = nn.Linear(base_channels, out_channels)

        # Optional output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'custom':
            self.output_activation = CustomActivation()
        elif output_activation is None:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        # x = self.output(x)
        x = self.output(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.output_activation(x)
        return x
