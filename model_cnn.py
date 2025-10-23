import torch
import torch.nn as nn
import math


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
    """Symmetric CNN Encoder for 512x512 images."""
    def __init__(self, in_channels=1, base_channels=64, latent_dim=512, use_batchnorm=True, dropout=0.0):
        super().__init__()

        # Encoder: 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.MaxPool2d(2, 2)  # 512 -> 256
        )

        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels*2, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            ConvBlock(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )

        self.enc3 = nn.Sequential(
            ConvBlock(base_channels*2, base_channels*4, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            ConvBlock(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )

        self.enc4 = nn.Sequential(
            ConvBlock(base_channels*4, base_channels*8, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            ConvBlock(base_channels*8, base_channels*8, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

        self.enc5 = nn.Sequential(
            ConvBlock(base_channels*8, latent_dim, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            ConvBlock(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )

    def forward(self, x):
        x1 = self.enc1(x)    # 256
        x2 = self.enc2(x1)   # 128
        x3 = self.enc3(x2)   # 64
        x4 = self.enc4(x3)   # 32
        x5 = self.enc5(x4)   # 16
        return x5


class Decoder(nn.Module):
    """Symmetric CNN Decoder for 512x512 images."""
    def __init__(self, latent_dim=512, base_channels=64, out_channels=1, use_batchnorm=True, output_activation=None, dropout=0.0):
        super().__init__()

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256 -> 512
        self.dec1 = nn.Sequential(
            TransposeConvBlock(latent_dim, base_channels*8, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),  # 16 -> 32
            ConvBlock(base_channels*8, base_channels*8, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout)
        )

        self.dec2 = nn.Sequential(
            TransposeConvBlock(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),  # 32 -> 64
            ConvBlock(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout)
        )

        self.dec3 = nn.Sequential(
            TransposeConvBlock(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),  # 64 -> 128
            ConvBlock(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout)
        )

        self.dec4 = nn.Sequential(
            TransposeConvBlock(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),  # 128 -> 256
            ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout)
        )

        self.dec5 = nn.Sequential(
            TransposeConvBlock(base_channels, base_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=use_batchnorm, dropout=dropout),  # 256 -> 512
            ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=use_batchnorm, dropout=dropout)
        )

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
        x = self.dec1(x)    # 32
        x = self.dec2(x)    # 64
        x = self.dec3(x)    # 128
        x = self.dec4(x)    # 256
        x = self.dec5(x)    # 512
        x = self.output(x)
        x = self.output_activation(x)
        return x

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
                    'dropout': 0.0
                },
                'amp_decoder': {
                    'base_channels': 64,
                    'latent_dim': 512,
                    'out_channels': 1,
                    'use_batchnorm': True,
                    'output_activation': 'sigmoid',
                    'dropout': 0.0
                },
                'ph_decoder': {
                    'base_channels': 64,
                    'latent_dim': 512,
                    'out_channels': 1,
                    'use_batchnorm': True,
                    'output_activation': 'tanh',
                    'dropout': 0.0
                }
            }

        # CNN Encoder
        self.encoder = Encoder(
            in_channels=config['encoder']['in_channels'],
            base_channels=config['encoder']['base_channels'],
            latent_dim=config['encoder']['latent_dim'],
            use_batchnorm=config['encoder']['use_batchnorm'],
            dropout=config['encoder'].get('dropout', 0.0)
        )

        # Amplitude Decoder
        self.amp_decoder = Decoder(
            latent_dim=config['amp_decoder']['latent_dim'],
            base_channels=config['amp_decoder']['base_channels'],
            out_channels=config['amp_decoder']['out_channels'],
            use_batchnorm=config['amp_decoder']['use_batchnorm'],
            output_activation=config['amp_decoder']['output_activation'],
            dropout=config['amp_decoder'].get('dropout', 0.0)
        )

        # Phase Decoder
        self.ph_decoder = Decoder(
            latent_dim=config['ph_decoder']['latent_dim'],
            base_channels=config['ph_decoder']['base_channels'],
            out_channels=config['ph_decoder']['out_channels'],
            use_batchnorm=config['ph_decoder']['use_batchnorm'],
            output_activation=config['ph_decoder']['output_activation'],
            dropout=config['ph_decoder'].get('dropout', 0.0)
        )

    def forward(self, x, probe, normalization, scale):
        # FFT probe
        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])
        #probe_intensity = torch.fft.fftshift(torch.fft.fft2(probe), dim=(-2, -1))
        #probe_intensity = (probe_intensity.abs()**2).sum(2)[:, 0]

        # Normalization
        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)
        #probe_intensity = (probe_intensity / normalization) * scale

        # Subtract probe contribution to total intensity
        #x = x - probe_intensity.float().unsqueeze(1)

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
    # Test the PtychoCNN model
    print("Testing PtychoCNN model...")

    config = {
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

    model = PtychoCNN(config=config)

    # Create dummy inputs matching the expected format
    batch_size = 2
    img_size = 512

    x = torch.randn(batch_size, 1, img_size, img_size)
    probe = torch.randn(batch_size, 1, 8, img_size, img_size, 2)  # Complex probe
    normalization = torch.randn(batch_size, 1)
    scale = torch.randn(batch_size, 1)

    # Forward pass
    pred_diff_amp, amp, ph = model(x, probe, normalization, scale)

    print(f"Input shape: {x.shape}")
    print(f"Predicted diffraction amplitude shape: {pred_diff_amp.shape}")
    print(f"Amplitude shape: {amp.shape}")
    print(f"Phase shape: {ph.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
