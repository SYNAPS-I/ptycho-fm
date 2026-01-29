"""
Ptychography Reconstruction Models for 256x256 Images

This module contains Vision Transformer-based models for ptychography reconstruction.
All models are fixed at 256x256 image size.
"""

import torch
import torch.nn as nn
import math
from model.vit import CustomViT
from model.vit_pretrained import VisionTransformer
from model.decoders import Decoder256
# from utils.math import create_logpolar_grid, apply_logpolar_transform


class PtychoViT(nn.Module):
    """Vision Transformer-based ptychography reconstruction model.

    Supports both CustomViT (train from scratch) and VisionTransformer (pretrained) encoders.
    Uses CNN decoders for amplitude and phase reconstruction.
    """
    def __init__(self, config=None):
        super().__init__()
        # Use config if provided, otherwise use default values
        if config is None:
            config = {
                'encoder_type': 'custom',  # 'custom' or 'pretrained'
                'encoder': {
                    'img_size': 256,
                    'patch_size': 16,
                    'in_channels': 1,
                    'embed_dim': 512,
                    'depth': 12,
                    'num_heads': 8,
                    'mlp_ratio': 4.0,
                    'use_cls_token': False,
                    'dropout': 0.1,
                    'attn_dropout': 0.0
                },
                'decoder': {
                    'base_channels': 64,
                    'latent_dim': None,  # None means use encoder.embed_dim
                    'use_batchnorm': True,
                    'dropout': 0.1,
                    'num_stages': 4
                }
            }

        # Determine encoder type
        encoder_type = config.get('encoder_type', 'custom')
        encoder_config = config.get('encoder', {})

        # Vision Transformer Encoder
        if encoder_type == 'pretrained':
            # Handle checkpoint_kwargs if present (for backward compatibility)
            # VisionTransformer expects strict/strict_load in timm_kwargs or as direct parameter
            checkpoint_kwargs = encoder_config.get('checkpoint_kwargs', {})
            strict_load = encoder_config.get('strict_load', checkpoint_kwargs.get('strict'))

            self.encoder = VisionTransformer(
                img_size=encoder_config.get('img_size', 256),
                patch_size=encoder_config.get('patch_size', 32),
                in_channels=encoder_config.get('in_channels', 1),
                embed_dim=encoder_config.get('embed_dim', 1024),
                depth=encoder_config.get('depth', 24),
                num_heads=encoder_config.get('num_heads', 16),
                mlp_ratio=encoder_config.get('mlp_ratio', 4.0),
                dropout=encoder_config.get('dropout', 0.1),
                attn_dropout=encoder_config.get('attn_dropout', 0.0),
                use_cls_token=encoder_config.get('use_cls_token', False),
                timm_model_name=encoder_config.get('timm_model_name', 'vit_large_patch32_224'),
                timm_kwargs=encoder_config.get('timm_kwargs'),
                checkpoint_path=encoder_config.get('checkpoint_path'),
                strict_load=strict_load
            )
        else:  # 'custom' or default
            self.encoder = CustomViT(
                img_size=encoder_config.get('img_size', 256),
                patch_size=encoder_config.get('patch_size', 16),
                in_channels=encoder_config.get('in_channels', 1),
                embed_dim=encoder_config.get('embed_dim', 512),
                depth=encoder_config.get('depth', 12),
                num_heads=encoder_config.get('num_heads', 8),
                mlp_ratio=encoder_config.get('mlp_ratio', 4.0),
                dropout=encoder_config.get('dropout', 0.1),
                attn_dropout=encoder_config.get('attn_dropout', 0.0),
                use_cls_token=encoder_config.get('use_cls_token', False)
            )

        # Decoder configuration
        decoder_config = config.get('decoder', {})
        # Use encoder's embed_dim if latent_dim is None/null
        latent_dim = decoder_config.get('latent_dim')
        if latent_dim is None:
            latent_dim = self.encoder.embed_dim

        # Amplitude Decoder (CNN-based, from latent spatial representation to 256x256)
        self.amp_decoder = Decoder256(
            latent_dim=latent_dim,
            base_channels=decoder_config.get('base_channels', 64),
            out_channels=1,
            use_batchnorm=decoder_config.get('use_batchnorm', True),
            output_activation='custom',
            dropout=decoder_config.get('dropout', 0.1),
            num_stages=decoder_config.get('num_stages', 4)
        )

        # Phase Decoder (CNN-based, from latent spatial representation to 256x256)
        self.ph_decoder = Decoder256(
            latent_dim=latent_dim,
            base_channels=decoder_config.get('base_channels', 64),
            out_channels=1,
            use_batchnorm=decoder_config.get('use_batchnorm', True),
            output_activation='custom',
            dropout=decoder_config.get('dropout', 0.1),
            num_stages=decoder_config.get('num_stages', 4)
        )

        # cache for log-polar grid
        self._logpolar_grid = None
        self._logpolar_hw = None

        # Scaling factors for the outputs
        self.log_scale_amp = nn.Parameter(torch.tensor(math.log(config.get("amp_scale", 0.2)), dtype=torch.float32), requires_grad=False)
        self.log_scale_ph = nn.Parameter(torch.tensor(math.log(config.get("ph_scale", math.pi)), dtype=torch.float32), requires_grad=False)

        self.amp_offset = config.get("amp_offset", 1.0)
        
        self.subtract_probe_intensity = config.get("subtract_probe_intensity", False)

    def forward(self, x, probe, normalization, scale):
        x = 2 * torch.log10(x + 1e-1)

        # FFT probe
        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])
        if self.subtract_probe_intensity:
            probe_intensity = torch.fft.fftshift(torch.fft.fft2(probe), dim=(-2, -1))
            probe_intensity = (probe_intensity.abs()**2).sum(2)[:, 0]

        # Normalization
        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)
        if self.subtract_probe_intensity:
            probe_intensity = (probe_intensity / normalization) * scale

        # Subtract probe contribution to total intensity
        if self.subtract_probe_intensity:
            x = x - torch.sqrt(probe_intensity.float().unsqueeze(1))

        # Apply log-polar coordinate transform
        # H, W = x.shape[-2], x.shape[-1]
        # if self._logpolar_grid is None or self._logpolar_hw != (H, W) or self._logpolar_grid.device != x.device:
        #     self._logpolar_grid = create_logpolar_grid(H, W, x.device)
        #     self._logpolar_hw = (H, W)
        # x_prime = apply_logpolar_transform(x, self._logpolar_grid, mode='bicubic')

        # ViT Encoder: (B, 1, 256, 256) -> (B, embed_dim, H, W)
        x = self.encoder(x)

        # Decode to amplitude and phase
        constrained_amp = self.amp_decoder(x).squeeze(1)
        constrained_ph = self.ph_decoder(x).squeeze(1)

        # Scale the constrained outputs 
        amp = (constrained_amp * torch.exp(self.log_scale_amp)) + self.amp_offset
        ph = constrained_ph * torch.exp(self.log_scale_ph)

        # Complex object and diffraction
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)