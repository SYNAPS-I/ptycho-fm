import torch
import torch.nn as nn
import math
import numpy as np
from scipy.ndimage import map_coordinates


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, activation, and optional dropout."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu', use_batchnorm=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
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


class Encoder(nn.Module):
    """CNN Encoder for 512x512 images with configurable depth.

    Args:
        in_channels: Number of input channels (default: 1)
        base_channels: Base number of channels, doubles at each stage (default: 64)
        latent_dim: Output channels at bottleneck (default: 512)
        use_batchnorm: Whether to use batch normalization (default: True)
        dropout: Dropout rate (default: 0.0)
        num_stages: Number of downsampling stages (default: 5)
            - 5 stages: 512 -> 16 (spatial size)
            - 6 stages: 512 -> 8
            - 7 stages: 512 -> 4
    """
    def __init__(self, in_channels=1, base_channels=64, latent_dim=512, use_batchnorm=True, dropout=0.0, num_stages=5):
        super().__init__()

        self.num_stages = num_stages
        self.stages = nn.ModuleList()

        # Calculate channel progression
        in_ch = in_channels
        for i in range(num_stages):
            # For all stages except the last, double channels (with a cap)
            if i < num_stages - 1:
                out_ch = min(base_channels * (2 ** i), latent_dim)
            else:
                # Last stage outputs latent_dim
                out_ch = latent_dim

            # Create stage with two conv blocks and pooling
            stage = nn.Sequential(
                ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
                ConvBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
                nn.MaxPool2d(2, 2)
            )
            self.stages.append(stage)
            in_ch = out_ch

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


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


class Encoder256(nn.Module):
    """CNN Encoder for 256x256 images with configurable depth.

    Args:
        in_channels: Number of input channels (default: 1)
        base_channels: Base number of channels, doubles at each stage (default: 64)
        latent_dim: Output channels at bottleneck (default: 512)
        use_batchnorm: Whether to use batch normalization (default: True)
        dropout: Dropout rate (default: 0.0)
        num_stages: Number of downsampling stages (default: 4)
            - 4 stages: 256 -> 16 (spatial size)
            - 5 stages: 256 -> 8
            - 6 stages: 256 -> 4
    """
    def __init__(self, in_channels=1, base_channels=64, latent_dim=512, use_batchnorm=True, dropout=0.0, num_stages=4):
        super().__init__()

        self.num_stages = num_stages
        self.stages = nn.ModuleList()

        # Calculate channel progression
        in_ch = in_channels
        for i in range(num_stages):
            # For all stages except the last, double channels (with a cap)
            if i < num_stages - 1:
                out_ch = min(base_channels * (2 ** i), latent_dim)
            else:
                # Last stage outputs latent_dim
                out_ch = latent_dim

            # Create stage with two conv blocks and pooling
            stage = nn.Sequential(
                ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
                ConvBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
                nn.MaxPool2d(2, 2)
            )
            self.stages.append(stage)
            in_ch = out_ch

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


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
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Optional output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Tanh()
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


class PtychoCNN256(nn.Module):
    """CNN-based ptychography reconstruction model for 256x256 images.

    This is a 4-stage version of PtychoCNN optimized for 256x256 inputs.
    Architecture: 256 -> 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128 -> 256
    """
    def __init__(self, config=None):
        super().__init__()
        # Use config if provided, otherwise use default values
        if config is None:
            config = {
                'encoder': {
                    'img_size': 256,
                    'in_channels': 1,
                    'base_channels': 64,
                    'latent_dim': 512,
                    'use_batchnorm': True,
                    'dropout': 0.0,
                    'num_stages': 4
                },
                'decoder': {
                    'base_channels': 64,
                    'latent_dim': 512,
                    'use_batchnorm': True,
                    'dropout': 0.0,
                    'num_stages': 4
                }
            }

        # CNN Encoder (256x256 -> configurable bottleneck)
        self.encoder = Encoder256(
            in_channels=config['encoder']['in_channels'],
            base_channels=config['encoder']['base_channels'],
            latent_dim=config['encoder']['latent_dim'],
            use_batchnorm=config['encoder']['use_batchnorm'],
            dropout=config['encoder'].get('dropout', 0.0),
            num_stages=config['encoder'].get('num_stages', 4)
        )

        # Amplitude Decoder (configurable bottleneck -> 256x256)
        self.amp_decoder = Decoder256(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='sigmoid',
            dropout=config['decoder'].get('dropout', 0.0),
            num_stages=config['decoder'].get('num_stages', 4)
        )

        # Phase Decoder (configurable bottleneck -> 256x256)
        self.ph_decoder = Decoder256(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='tanh',
            dropout=config['decoder'].get('dropout', 0.0),
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

        # Log10 and subtract probe contribution to total intensity
        #x = torch.log10(x + 1.0e-6) 
        x = x - torch.sqrt(probe_intensity.float().unsqueeze(1))

        # Apply log-polar coordinate transform
        grid = create_logpolar_grid(x.shape[-2], x.shape[-1], x.device)
        x_prime = apply_logpolar_transform(x, grid, mode='bicubic')

        # CNN Encoder
        x = self.encoder(x_prime)

        # Decode to amplitude and phase
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


def create_logpolar_grid(height, width, device='cpu'):
    """
    Create a log-polar sampling grid for torch.nn.functional.grid_sample.
    Follows numpy 'ij' indexing convention to match scipy behavior.

    Args:
        height, width: Output dimensions
        device: torch device

    Returns:
        grid: Tensor of shape (1, height, width, 2) with normalized coordinates
    """
    # Match scipy's np.mgrid convention with 'ij' indexing
    # np.mgrid[-h//2:h//2, -w//2:w//2] creates grids where:
    # - first dimension (y) varies along rows (axis 0)
    # - second dimension (x) varies along columns (axis 1)
    half_h, half_w = height // 2, width // 2
    y, x = torch.meshgrid(
        torch.linspace(-half_h, half_h, height, device=device),
        torch.linspace(-half_w, half_w, width, device=device),
        indexing='ij'
    )

    # Compute log-polar coordinates
    rho = torch.log(torch.sqrt(x**2 + y**2) + 1e-10)
    theta = torch.atan2(y, x)

    # Define output log-polar grid (evenly sampled in log-polar space)
    valid_rho = rho[torch.isfinite(rho)]
    rho_out = torch.linspace(valid_rho.min(), rho.max(), height, device=device)
    theta_out = torch.linspace(-torch.pi, torch.pi, width, device=device)

    # np.meshgrid default is 'xy', but we need to match scipy exactly
    # In the scipy code: rho_grid, theta_grid = np.meshgrid(rho_out, theta_out)
    # This uses 'xy' indexing, so theta varies along axis 0, rho along axis 1
    theta_grid, rho_grid = torch.meshgrid(theta_out, rho_out, indexing='ij')

    # Convert log-polar grid back to Cartesian coordinates
    x_sample = torch.exp(rho_grid) * torch.cos(theta_grid)
    y_sample = torch.exp(rho_grid) * torch.sin(theta_grid)

    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects (x, y) in the last dimension
    x_norm = x_sample / half_w
    y_norm = y_sample / half_h

    # Stack to (1, H, W, 2) - last dim is (x, y)
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    return grid


def apply_logpolar_transform(images, grid, mode='bilinear'):
    """
    Apply log-polar transform to batched images.

    Args:
        images: Tensor of shape (B, C, H, W)
        grid: Pre-computed grid from create_logpolar_grid
        mode: Interpolation mode 

    Returns:
        Transformed images of shape (B, C, H, W)
    """
    import torch.nn.functional as F

    batch_size = images.shape[0]

    # Expand grid to batch size
    grid_expanded = grid.expand(batch_size, -1, -1, -1)

    # Apply transformation
    transformed = F.grid_sample(
        images,
        grid_expanded,
        mode=mode,
        padding_mode='zeros',
        align_corners=False
    )

    return transformed


class PtychoCNN(nn.Module):
    """CNN-based ptychography reconstruction model.

    This is a CNN alternative to PtychoViT, using convolutional encoder/decoder
    instead of Vision Transformer encoder.
    """
    def __init__(self, config=None):
        super().__init__()
        # Use config if provided, otherwise use default values
        if config is None:
            config = {
                'encoder': {
                    'img_size': 512,
                    'in_channels': 1,
                    'base_channels': 64,
                    'latent_dim': 512,
                    'use_batchnorm': True,
                    'dropout': 0.0,
                    'num_stages': 5
                },
                'decoder': {
                    'base_channels': 64,
                    'latent_dim': 512,
                    'use_batchnorm': True,
                    'dropout': 0.0,
                    'num_stages': 5
                }
            }

        # CNN Encoder
        self.encoder = Encoder(
            in_channels=config['encoder']['in_channels'],
            base_channels=config['encoder']['base_channels'],
            latent_dim=config['encoder']['latent_dim'],
            use_batchnorm=config['encoder']['use_batchnorm'],
            dropout=config['encoder'].get('dropout', 0.0),
            num_stages=config['encoder'].get('num_stages', 5)
        )

        # Amplitude Decoder
        self.amp_decoder = Decoder(
            latent_dim=config['decoder']['latent_dim'],
            base_channels=config['decoder']['base_channels'],
            out_channels=1,
            use_batchnorm=config['decoder']['use_batchnorm'],
            output_activation='sigmoid',
            dropout=config['decoder'].get('dropout', 0.0),
            num_stages=config['decoder'].get('num_stages', 5)
        )

        # Phase Decoder
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

        # CNN Encoder
        x = self.encoder(x)

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


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PtychoCNN Models")
    print("=" * 60)

    # Test PtychoCNN (512x512)
    print("\n1. Testing PtychoCNN (512x512)...")
    config_512 = {
        'encoder': {
            'img_size': 512,
            'in_channels': 1,
            'base_channels': 64,
            'latent_dim': 512,
            'use_batchnorm': True,
            'dropout': 0.1
        },
        'amp_decoder': {
            'base_channels': 64,
            'latent_dim': 512,
            'out_channels': 1,
            'use_batchnorm': True,
            'output_activation': 'sigmoid',
            'dropout': 0.1
        },
        'ph_decoder': {
            'base_channels': 64,
            'latent_dim': 512,
            'out_channels': 1,
            'use_batchnorm': True,
            'output_activation': 'tanh',
            'dropout': 0.1
        }
    }

    model_512 = PtychoCNN(config=config_512)

    # Create dummy inputs for 512x512
    batch_size = 2
    x_512 = torch.randn(batch_size, 1, 512, 512)
    probe_512 = torch.randn(batch_size, 1, 8, 512, 512, 2)
    normalization = torch.randn(batch_size, 1)
    scale = torch.randn(batch_size, 1)

    # Forward pass
    pred_diff_amp, amp, ph = model_512(x_512, probe_512, normalization, scale)

    print(f"   Input shape: {x_512.shape}")
    print(f"   Predicted diffraction amplitude shape: {pred_diff_amp.shape}")
    print(f"   Amplitude shape: {amp.shape}")
    print(f"   Phase shape: {ph.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model_512.parameters())
    trainable_params = sum(p.numel() for p in model_512.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test PtychoCNN256 (256x256)
    print("\n2. Testing PtychoCNN256 (256x256)...")
    config_256 = {
        'encoder': {
            'img_size': 256,
            'in_channels': 1,
            'base_channels': 64,
            'latent_dim': 512,
            'use_batchnorm': True,
            'dropout': 0.1
        },
        'amp_decoder': {
            'base_channels': 64,
            'latent_dim': 512,
            'out_channels': 1,
            'use_batchnorm': True,
            'output_activation': 'sigmoid',
            'dropout': 0.1
        },
        'ph_decoder': {
            'base_channels': 64,
            'latent_dim': 512,
            'out_channels': 1,
            'use_batchnorm': True,
            'output_activation': 'tanh',
            'dropout': 0.1
        }
    }

    model_256 = PtychoCNN256(config=config_256)

    # Create dummy inputs for 256x256
    x_256 = torch.randn(batch_size, 1, 256, 256)
    probe_256 = torch.randn(batch_size, 1, 8, 256, 256, 2)

    # Forward pass
    pred_diff_amp, amp, ph = model_256(x_256, probe_256, normalization, scale)

    print(f"   Input shape: {x_256.shape}")
    print(f"   Predicted diffraction amplitude shape: {pred_diff_amp.shape}")
    print(f"   Amplitude shape: {amp.shape}")
    print(f"   Phase shape: {ph.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model_256.parameters())
    trainable_params = sum(p.numel() for p in model_256.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
