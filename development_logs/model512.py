"""
512x512 Ptychography Reconstruction Models

This module contains legacy 512x512 image size models that are no longer actively used.
Kept for reference and compatibility with older experiments.
"""

import torch
import torch.nn as nn
import math


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


class Decoder(nn.Module):
    """CNN Decoder for 512x512 images with configurable depth.

    Mirrors encoder structure: Encoder has Conv2d, Conv2d, MaxPool2d per stage,
    Decoder has ConvTranspose2d, ConvTranspose2d, Upsample(bilinear) per stage.

    Args:
        latent_dim: Input channels from bottleneck (default: 512)
        base_channels: Base number of channels for final layer (default: 64)
        out_channels: Number of output channels (default: 1)
        use_batchnorm: Whether to use batch normalization (default: True)
        output_activation: Output activation function (default: None)
        dropout: Dropout rate (default: 0.0)
        num_stages: Number of upsampling stages, should match encoder (default: 5)
            - 5 stages: 16 -> 512 (spatial size)
            - 6 stages: 8 -> 512
            - 7 stages: 4 -> 512
    """
    def __init__(self, latent_dim=512, base_channels=64, out_channels=1, use_batchnorm=True, output_activation=None, dropout=0.0, num_stages=5):
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
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Optional output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation is None:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.output(x)
        x = self.output_activation(x)
        return x


class PtychoViT(nn.Module):
    """Vision Transformer-based ptychography reconstruction model for 512x512 images.

    Uses ViT encoder with CNN decoders (same as PtychoCNN) for parallel construction.
    """
    def __init__(self, config=None):
        super().__init__()
        # Import CustomViT here to avoid circular imports
        from model.vit import CustomViT

        # Use config if provided, otherwise use default values
        if config is None:
            config = {
                'encoder': {
                    'img_size': 512,
                    'patch_size': 16,
                    'in_channels': 1,
                    'embed_dim': 192,
                    'depth': 12,
                    'num_heads': 3,
                    'mlp_ratio': 4.0,
                    'use_cls_token': False,
                    'dropout': 0.1,
                    'attn_dropout': 0.0
                },
                'decoder': {
                    'base_channels': 64,
                    'latent_dim': 192,
                    'use_batchnorm': True,
                    'dropout': 0.0,
                    'num_stages': 5
                }
            }

        # Vision Transformer Encoder
        self.encoder = CustomViT(
            img_size=config['encoder']['img_size'],
            patch_size=config['encoder']['patch_size'],
            in_channels=config['encoder']['in_channels'],
            embed_dim=config['encoder']['embed_dim'],
            depth=config['encoder']['depth'],
            num_heads=config['encoder']['num_heads'],
            mlp_ratio=config['encoder']['mlp_ratio'],
            dropout=config['encoder'].get('dropout', 0.1),
            attn_dropout=config['encoder'].get('attn_dropout', 0.0),
            use_cls_token=config['encoder']['use_cls_token']
        )

        # Calculate the spatial size after ViT encoding
        # ViT outputs (B, num_patches, embed_dim) where num_patches = (img_size/patch_size)^2
        img_size = config['encoder']['img_size']
        patch_size = config['encoder']['patch_size']
        num_patches_per_side = img_size // patch_size

        # We need to reshape ViT output to spatial format for CNN decoder
        # Will reshape from (B, num_patches, embed_dim) to (B, embed_dim, H, W)
        self.num_patches_per_side = num_patches_per_side

        # Amplitude Decoder (CNN-based, from latent spatial representation to 512x512)
        self.amp_decoder = Decoder(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='sigmoid',
            dropout=config['decoder'].get('dropout', 0.0),
            num_stages=config['decoder'].get('num_stages', 5)
        )

        # Phase Decoder (CNN-based, from latent spatial representation to 512x512)
        self.ph_decoder = Decoder(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='tanh',
            dropout=config['decoder'].get('dropout', 0.0),
            num_stages=config['decoder'].get('num_stages', 5)
        )

    def forward(self, x, probe, normalization, scale):
        # FFT probe
        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])
        probe_intensity = torch.fft.fftshift(torch.fft.fft2(probe), dim=(-2, -1))
        probe_intensity = (probe_intensity.abs()**2).sum(2)[:, 0]

        # Normalization
        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)
        probe_intensity = (probe_intensity / normalization) * scale

        # Subtract probe contribution to total intensity
        x = x - torch.sqrt(probe_intensity.float().unsqueeze(1))

        # ViT Encoder: (B, 1, 512, 512) -> (B, num_patches, embed_dim)
        x = self.encoder(x)

        # Reshape from (B, num_patches, embed_dim) to (B, embed_dim, H, W)
        # where H = W = num_patches_per_side
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, -1, self.num_patches_per_side, self.num_patches_per_side)

        # Decode to amplitude and phase
        amp = self.amp_decoder(x).squeeze(1)
        ph = self.ph_decoder(x).squeeze(1) * math.pi

        # Complex object and diffraction
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)
