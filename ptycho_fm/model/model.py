"""
Ptychography Reconstruction Models for 256x256 Images

This module contains Vision Transformer-based models for ptychography reconstruction.
All models are fixed at 256x256 image size.
"""

import math

import torch
from torch import nn

from ptycho_fm.model.decoders import CoupledDecoder256, Decoder256, ResNetDecoder256
from ptycho_fm.model.vit import CustomViT
from ptycho_fm.model.vit_pretrained import VisionTransformer


def _output_norm_config(config):
    if config is None:
        return {}
    return config.get('output_norm', config.get('scaling', {}))


def _log_scale_parameter(output_norm_config, key, default, legacy_key=None):
    if key in output_norm_config:
        scale = output_norm_config[key]
        if scale <= 0:
            raise ValueError(f"Output norm scale '{key}' must be positive, got {scale}")
        value = math.log(scale)
    elif legacy_key is not None and legacy_key in output_norm_config:
        value = output_norm_config[legacy_key]
    else:
        value = math.log(default)

    return nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=False)


def _offset_parameter(output_norm_config, key, default, legacy_key=None):
    if key in output_norm_config:
        value = output_norm_config[key]
    elif legacy_key is not None and legacy_key in output_norm_config:
        value = output_norm_config[legacy_key]
    else:
        value = default

    return nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=False)


class PtychoFM(nn.Module):
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
        init_config = config.get('init', {})
        init_enabled = bool(init_config.get('enabled', False))
        init_method = init_config.get('method', 'trunc_normal')
        init_mean = float(init_config.get('mean', 0.0))
        init_std = float(init_config.get('std', 0.02))
        kaiming_cfg = init_config.get('kaiming', {}) or {}
        kaiming_distribution = kaiming_cfg.get('distribution', 'uniform')
        kaiming_mode = kaiming_cfg.get('mode', 'fan_in')
        kaiming_nonlinearity = kaiming_cfg.get('nonlinearity', 'relu')
        kaiming_a = float(kaiming_cfg.get('a', 0.0))
        apply_to = init_config.get('apply_to', ['encoder', 'decoders'])
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        apply_to = set(apply_to)
        override_pretrained = bool(init_config.get('override_pretrained', False))
        if init_enabled and init_method not in ['trunc_normal', 'kaiming']:
            raise ValueError(f"Unknown init method: {init_method}. Use 'trunc_normal' or 'kaiming'.")
        if init_enabled and init_method == 'kaiming' and kaiming_distribution not in ['uniform', 'normal']:
            raise ValueError(
                f"Unknown kaiming distribution: {kaiming_distribution}. "
                "Use 'uniform' or 'normal'."
            )

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
                use_cls_token=encoder_config.get('use_cls_token', False),
                init_mean=init_mean if init_enabled and 'encoder' in apply_to else 0.0,
                init_std=init_std if init_enabled and 'encoder' in apply_to else 0.02,
                init_method=init_method if init_enabled and 'encoder' in apply_to else 'trunc_normal',
                kaiming_a=kaiming_a,
                kaiming_mode=kaiming_mode,
                kaiming_nonlinearity=kaiming_nonlinearity,
                kaiming_distribution=kaiming_distribution
            )

        # Decoder configuration
        decoder_config = config.get('decoder', {})
        # Use encoder's embed_dim if latent_dim is None/null
        latent_dim = decoder_config.get('latent_dim')
        if latent_dim is None:
            latent_dim = self.encoder.embed_dim

        # Determine decoder type
        decoder_type = config.get('decoder_type', 'standard')

        phase_activation = decoder_config.get('phase_activation', 'custom')

        # Select decoder class based on type
        if decoder_type == 'resnet':
            decoder_class = ResNetDecoder256
            amp_decoder_kwargs = {
                'latent_dim': latent_dim,
                'base_channels': decoder_config.get('base_channels', 64),
                'out_channels': 1,
                'use_batchnorm': decoder_config.get('use_batchnorm', True),
                'output_activation': 'tanh',
                'dropout': decoder_config.get('dropout', 0.1),
                'num_stages': decoder_config.get('num_stages', 4),
                'blocks_per_stage': decoder_config.get('blocks_per_stage', 2)
            }
            ph_decoder_kwargs = {
                **amp_decoder_kwargs,
                'output_activation': phase_activation,
            }
        else:  # 'standard' or default
            decoder_class = Decoder256
            amp_decoder_kwargs = {
                'latent_dim': latent_dim,
                'base_channels': decoder_config.get('base_channels', 64),
                'out_channels': 1,
                'use_batchnorm': decoder_config.get('use_batchnorm', True),
                'output_activation': 'tanh',
                'dropout': decoder_config.get('dropout', 0.1),
                'num_stages': decoder_config.get('num_stages', 4)
            }
            ph_decoder_kwargs = {
                **amp_decoder_kwargs,
                'output_activation': phase_activation,
            }

        # Amplitude Decoder (from latent spatial representation to 256x256)
        self.amp_decoder = decoder_class(**amp_decoder_kwargs)

        # Phase Decoder (from latent spatial representation to 256x256)
        self.ph_decoder = decoder_class(**ph_decoder_kwargs)

        if init_enabled:
            if 'decoders' in apply_to:
                if init_method == 'kaiming':
                    self._init_module_kaiming(
                        self.amp_decoder,
                        kaiming_a,
                        kaiming_mode,
                        kaiming_nonlinearity,
                        kaiming_distribution,
                    )
                    self._init_module_kaiming(
                        self.ph_decoder,
                        kaiming_a,
                        kaiming_mode,
                        kaiming_nonlinearity,
                        kaiming_distribution,
                    )
                else:
                    self._init_module_trunc_normal(self.amp_decoder, init_mean, init_std)
                    self._init_module_trunc_normal(self.ph_decoder, init_mean, init_std)
            if encoder_type == 'pretrained' and 'encoder' in apply_to and override_pretrained:
                if init_method == 'kaiming':
                    self._init_module_kaiming(
                        self.encoder,
                        kaiming_a,
                        kaiming_mode,
                        kaiming_nonlinearity,
                        kaiming_distribution,
                    )
                else:
                    self._init_module_trunc_normal(self.encoder, init_mean, init_std)

        # Output normalization parameters are stored in log-space for positive scales.
        output_norm_config = _output_norm_config(config)
        self.amp_scale = _log_scale_parameter(output_norm_config, 'amp_scale', 0.2, legacy_key='log_scale_amp')
        self.ph_scale = _log_scale_parameter(output_norm_config, 'ph_scale', math.pi, legacy_key='log_scale_ph')
        self.amp_offset = _offset_parameter(output_norm_config, 'amp_offset', 1.0, legacy_key='offset_amp')
        
        self.subtract_probe_intensity = config.get("subtract_probe_intensity", False)

    @staticmethod
    def _init_module_trunc_normal(module, mean, std):
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _init_module_kaiming(module, a, mode, nonlinearity, distribution):
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if distribution == "normal":
                    nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, probe, normalization, scale):
        eps = 1e-6
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

        # ViT Encoder: (B, 1, 256, 256) -> (B, embed_dim, H, W)
        x = self.encoder(x)

        # Decode to amplitude and phase
        constrained_amp = self.amp_decoder(x).squeeze(1)
        constrained_ph = self.ph_decoder(x).squeeze(1)

        # Scale the constrained outputs
        amp = (constrained_amp * torch.exp(self.amp_scale)) + self.amp_offset
        ph = constrained_ph * torch.exp(self.ph_scale)

        # Complex object and diffraction
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale
        pred_diff_amp = torch.sqrt(intensity + eps)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)


class PtychoFMInference(PtychoFM):
    """Inference-only variant of PtychoFM without probe or physics model.

    Takes input diffraction patterns and returns predicted amplitude and phase directly.
    """
    def forward(self, x):
        x = 2 * torch.log10(x + 1e-1)
        x = self.encoder(x)
        constrained_amp = self.amp_decoder(x).squeeze(1)
        constrained_ph = self.ph_decoder(x).squeeze(1)
        amp = (constrained_amp * torch.exp(self.amp_scale)) + self.amp_offset
        ph = constrained_ph * torch.exp(self.ph_scale)
        return amp.unsqueeze(1), ph.unsqueeze(1)


class PtychoFMReIm(PtychoFM):
    """Vision Transformer-based ptychography reconstruction with coupled real/imaginary decoders.

    This model predicts real and imaginary parts of the complex object separately but with
    shared feature representations, enforcing physical coupling between amplitude and phase
    through the Kramers-Kronig relationship.

    Inherits encoder setup from PtychoFM but uses CoupledDecoder256 for real/imaginary prediction.
    """
    def __init__(self, config=None):
        # Initialize parent class to set up encoder
        super().__init__(config)

        # Get decoder config
        if config is None:
            config = {}
        decoder_config = config.get('decoder', {})
        decoder_type = config.get('decoder_type', 'standard')

        # Use encoder's embed_dim if latent_dim is None/null
        latent_dim = decoder_config.get('latent_dim')
        if latent_dim is None:
            latent_dim = self.encoder.embed_dim

        # Replace separate amp/ph decoders with coupled real/imag decoder
        del self.amp_decoder
        del self.ph_decoder

        self.coupled_decoder = CoupledDecoder256(
            latent_dim=latent_dim,
            base_channels=decoder_config.get('base_channels', 64),
            use_batchnorm=decoder_config.get('use_batchnorm', True),
            dropout=decoder_config.get('dropout', 0.1),
            num_stages=decoder_config.get('num_stages', 4),
            use_resnet_blocks=(decoder_type == 'resnet'),
            blocks_per_stage=decoder_config.get('blocks_per_stage', 2)
        )

        del self.amp_scale
        del self.ph_scale
        del self.amp_offset

        output_norm_config = _output_norm_config(config)
        self.real_scale = _log_scale_parameter(output_norm_config, 'real_scale', 0.1, legacy_key='log_scale_real')
        self.imag_scale = _log_scale_parameter(output_norm_config, 'imag_scale', 0.1, legacy_key='log_scale_imag')
        self.real_offset = _offset_parameter(output_norm_config, 'real_offset', 1.0, legacy_key='offset_real')
        self.imag_offset = _offset_parameter(output_norm_config, 'imag_offset', 0.0, legacy_key='offset_imag')

    def forward(self, x, probe, normalization, scale):
        """
        Forward pass predicting real/imaginary components.

        Args:
            x: Input diffraction pattern
            probe: Complex probe (B, modes, OPR, H, W, 2) where last dim is [real, imag]
            normalization: Normalization factor
            scale: Scaling factor

        Returns:
            Tuple of (pred_diff_amp, amp, ph) for compatibility with training loop
        """
        # Preprocess input (log transform)
        x = 2 * torch.log10(x + 1e-1)

        # Convert probe to complex tensor
        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])

        # Normalization preparation
        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)

        # ViT Encoder: (B, 1, 256, 256) -> (B, embed_dim, H, W)
        x = self.encoder(x)

        # Decode to real and imaginary parts in separate channels
        # Output shape: (B, 2, H, W) where channel 0=real, channel 1=imaginary
        complex_output = self.coupled_decoder(x)
        real_constrained = complex_output[:, 0]  # (B, H, W)
        imag_constrained = complex_output[:, 1]  # (B, H, W)

        # Apply output normalization scale and offset.
        # tanh output is in [-1, 1], scale to appropriate range
        real = real_constrained * torch.exp(self.real_scale) + self.real_offset
        imag = imag_constrained * torch.exp(self.imag_scale) + self.imag_offset

        # Construct complex object directly from real and imaginary parts
        complex_object = torch.complex(real, imag)

        # Compute amplitude and phase for logging/visualization and loss computation
        amp = torch.abs(complex_object)
        ph = torch.angle(complex_object)

        # Forward physics model: compute diffraction pattern
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1) #, real.unsqueeze(1), imag.unsqueeze(1)


class PtychoFMCoupledAmpPh(PtychoFM):
    """Vision Transformer-based ptychography reconstruction with coupled amplitude/phase decoders.

    This model predicts amplitude and phase of the complex object with shared feature
    representations, as an alternative to the real/imaginary parameterization in PtychoFMCoupled.

    Inherits encoder setup from PtychoFM but uses CoupledDecoder256 for amplitude/phase prediction.
    """
    def __init__(self, config=None):
        # Initialize parent class to set up encoder
        super().__init__(config)

        # Get decoder config
        if config is None:
            config = {}
        decoder_config = config.get('decoder', {})
        decoder_type = config.get('decoder_type', 'standard')

        # Use encoder's embed_dim if latent_dim is None/null
        latent_dim = decoder_config.get('latent_dim')
        if latent_dim is None:
            latent_dim = self.encoder.embed_dim

        # Replace separate amp/ph decoders with coupled amp/phase decoder
        del self.amp_decoder
        del self.ph_decoder

        circular_phase = decoder_config.get('circular_phase', False)
        self.coupled_decoder = CoupledDecoder256(
            latent_dim=latent_dim,
            base_channels=decoder_config.get('base_channels', 64),
            use_batchnorm=decoder_config.get('use_batchnorm', True),
            dropout=decoder_config.get('dropout', 0.1),
            num_stages=decoder_config.get('num_stages', 4),
            use_resnet_blocks=(decoder_type == 'resnet'),
            blocks_per_stage=decoder_config.get('blocks_per_stage', 2),
            circular_phase=circular_phase,
        )

        output_norm_config = _output_norm_config(config)
        self.amp_scale = _log_scale_parameter(output_norm_config, 'amp_scale', 0.1, legacy_key='log_scale_amp')
        self.amp_offset = _offset_parameter(output_norm_config, 'amp_offset', 1.0, legacy_key='offset_amp')
        ph_scale_default = 0.0 if circular_phase else math.log(math.pi)
        self.ph_scale = _log_scale_parameter(output_norm_config, 'ph_scale', math.exp(ph_scale_default), legacy_key='log_scale_ph')

    def forward(self, x, probe, normalization, scale):
        """
        Forward pass predicting amplitude/phase components.

        Args:
            x: Input diffraction pattern
            probe: Complex probe (B, modes, OPR, H, W, 2) where last dim is [real, imag]
            normalization: Normalization factor
            scale: Scaling factor

        Returns:
            Tuple of (pred_diff_amp, amp, ph) for compatibility with training loop
        """
        # Preprocess input (log transform)
        x = 2 * torch.log10(x + 1e-1)

        # Convert probe to complex tensor
        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])

        # Normalization preparation
        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)

        # ViT Encoder: (B, 1, 256, 256) -> (B, embed_dim, H, W)
        x = self.encoder(x)

        # Decode to amplitude and phase in separate channels
        # Output shape: (B, 2, H, W) where channel 0=amplitude, channel 1=phase
        complex_output = self.coupled_decoder(x)
        amp_constrained = complex_output[:, 0]  # (B, H, W)
        ph_constrained = complex_output[:, 1]   # (B, H, W)

        # Apply scaling and offset
        # Amplitude: tanh in [-1, 1] scaled around 1.0
        amp = amp_constrained * torch.exp(self.amp_scale) + self.amp_offset
        # Phase: tanh in [-1, 1] scaled to [-pi, pi]
        ph = ph_constrained * torch.exp(self.ph_scale)

        # Construct complex object from amplitude and phase
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))

        # Forward physics model: compute diffraction pattern
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0]  # sum over incoherent modes, select first OPR mode

        # Normalization
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)