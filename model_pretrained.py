import math
import torch
import torch.nn as nn

from model_cnn import Decoder256, create_logpolar_grid, apply_logpolar_transform
from vit_pretrained import VisionTransformer  # timm-backed ViT with optional checkpoint loading

class PretrainedViT256(nn.Module):
    """Pretrained ViT encoder + CNN decoders matching PtychoViT256 behavior.

    - Input image_size=256, patch_size=16 (grid 16x16)
    - Probe subtraction and log-polar transform before the encoder
    - Decoder256 for amplitude/phase, with the same output scaling as model.PtychoViT256
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                "encoder": {
                    "img_size": 256,
                    "patch_size": 32,
                    "in_channels": 1,
                    "embed_dim": 1024,
                    "depth": 24,
                    "num_heads": 16,
                    "mlp_ratio": 4.0,
                    "use_cls_token": False,
                    "timm_model_name": "vit_large_patch32_224",
                    # stay offline + load local weights if provided
                    "timm_kwargs": {"pretrained": False},
                    "checkpoint_path": None,
                    # "strict_load": True,
                },
                "decoder": {
                    "base_channels": 64,
                    "latent_dim": None,  # will default to encoder.embed_dim
                    "use_batchnorm": True,
                    "dropout": 0.1,
                    "num_stages": 4,
                },
            }
        enc = config["encoder"]
        self.encoder = VisionTransformer(
            img_size=enc.get("img_size", 256),
            patch_size=enc.get("patch_size", 32),
            in_channels=enc.get("in_channels", 1),
            embed_dim=enc.get("embed_dim", 1024),
            depth=enc.get("depth", 24),
            num_heads=enc.get("num_heads", 16),
            mlp_ratio=enc.get("mlp_ratio", 4.0),
            dropout=enc.get("dropout", 0.1),
            attn_dropout=enc.get("attn_dropout", 0.0),
            use_cls_token=enc.get("use_cls_token", False),
            timm_model_name=enc.get("timm_model_name", "vit_large_patch32_224"),
            timm_kwargs=enc.get("timm_kwargs"),
            checkpoint_path=enc.get("checkpoint_path"),
            strict_load=enc.get("strict_load"),
        )

        dec = config.get("decoder", {})
        latent_dim = dec.get("latent_dim", self.encoder.embed_dim)
        self.amp_decoder = Decoder256(
            latent_dim=latent_dim,
            base_channels=dec.get("base_channels", 64),
            out_channels=1,
            use_batchnorm=dec.get("use_batchnorm", True),
            output_activation='sigmoid',
            dropout=dec.get("dropout", 0.1),
            num_stages=dec.get("num_stages", 5),
        )
        self.ph_decoder = Decoder256(
            latent_dim=latent_dim,
            base_channels=dec.get("base_channels", 64),
            out_channels=1,
            use_batchnorm=dec.get("use_batchnorm", True),
            output_activation='tanh',
            dropout=dec.get("dropout", 0.1),
            num_stages=dec.get("num_stages", 5),
        )

        # cache for log-polar grid
        self._logpolar_grid = None
        self._logpolar_hw = None

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

        # Apply log-polar coordinate transform (cache grid per H,W,device)
        H, W = x.shape[-2], x.shape[-1]
        if self._logpolar_grid is None or self._logpolar_hw != (H, W) or self._logpolar_grid.device != x.device:
            self._logpolar_grid = create_logpolar_grid(H, W, x.device)
            self._logpolar_hw = (H, W)
        x_prime = apply_logpolar_transform(x, self._logpolar_grid, mode='bicubic')

        # ViT Encoder: (B, 1, 256, 256) -> spatial latent (B, C, 16, 16)
        x_latent = self.encoder(x_prime)

        # Decode to amplitude and phase
        amp = (self.amp_decoder(x_latent).squeeze(1) + 1) / 2
        ph = self.ph_decoder(x_latent).squeeze(1) * math.pi

        # Complex object and diffraction
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)
