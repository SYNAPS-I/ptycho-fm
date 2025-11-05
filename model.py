import torch
import torch.nn as nn
import math
from vit import VisionTransformer
from model_cnn import Decoder, Decoder256, create_logpolar_grid, apply_logpolar_transform

class PtychoViT(nn.Module):
    """Vision Transformer-based ptychography reconstruction model for 512x512 images.

    Uses ViT encoder with CNN decoders (same as PtychoCNN) for parallel construction.
    """
    def __init__(self, config=None):
        super().__init__()
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
        self.encoder = VisionTransformer(
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


class PtychoViT256(nn.Module):
    """Vision Transformer-based ptychography reconstruction model for 256x256 images.

    Uses ViT encoder with CNN decoders (similar to PtychoCNN256) and log-polar transform.
    """
    def __init__(self, config=None):
        super().__init__()
        # Use config if provided, otherwise use default values
        if config is None:
            config = {
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
                    'latent_dim': 512,
                    'use_batchnorm': True,
                    'dropout': 0.1,
                    'num_stages': 4
                }
            }

        # Vision Transformer Encoder
        self.encoder = VisionTransformer(
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

        # Amplitude Decoder (CNN-based, from latent spatial representation to 256x256)
        self.amp_decoder = Decoder256(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='sigmoid',
            dropout=config['decoder'].get('dropout', 0.1),
            num_stages=config['decoder'].get('num_stages', 4)
        )

        # Phase Decoder (CNN-based, from latent spatial representation to 256x256)
        self.ph_decoder = Decoder256(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='tanh',
            dropout=config['decoder'].get('dropout', 0.1),
            num_stages=config['decoder'].get('num_stages', 4)
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

        # Apply log-polar coordinate transform
        grid = create_logpolar_grid(x.shape[-2], x.shape[-1], x.device)
        x_prime = apply_logpolar_transform(x, grid, mode='bicubic')

        # ViT Encoder: (B, 1, 256, 256) -> (B, num_patches, embed_dim)
        x = self.encoder(x_prime)

        # Reshape from (B, num_patches, embed_dim) to (B, embed_dim, H, W)
        # where H = W = num_patches_per_side
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, -1, self.num_patches_per_side, self.num_patches_per_side)

        # Decode to amplitude and phase
        #amp = self.amp_decoder(x).squeeze(1)
        amp = (self.amp_decoder(x).squeeze(1) + 1) / 2
        ph = self.ph_decoder(x).squeeze(1) * math.pi

        # Complex object and diffraction
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)