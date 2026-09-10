"""
CNN Decoders for Ptychography Reconstruction

This module contains decoder architectures used in ptychography reconstruction models.
"""

import torch
from torch import nn


class CustomActivation(nn.Module):
    """Modified tanh activation function to mitigate diminishing gradients at boundaries.

    LeCun, et al. Efficient BackProp 1998
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.7159 * torch.tanh((2/3) * x)


class _SafeAtan2(torch.autograd.Function):
    """atan2 with gradient stabilized near the origin.

    Standard atan2 has gradient proportional to 1/(x²+y²), which diverges at (0,0).
    This replaces that denominator with max(x²+y², eps²) in the backward pass only,
    leaving forward values identical to torch.atan2.
    """
    @staticmethod
    def forward(ctx, y, x, eps):
        ctx.save_for_backward(y, x)
        ctx.eps = eps
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = (x ** 2 + y ** 2).clamp(min=ctx.eps ** 2)
        return grad_output * x / denom, grad_output * (-y) / denom, None


class Atan2Activation(nn.Module):
    """Circular activation for phase outputs using atan2.

    Expects a 2-channel input: channel 0 is the 'y' (sin) component and
    channel 1 is the 'x' (cos) component. Returns atan2(y, x) in [-pi, pi].
    Gradients flow freely across the ±pi boundary because the two components
    are unconstrained — the network is never stuck at a saturation extreme.

    Uses a stabilized backward pass that clamps the gradient denominator to
    eps², preventing gradient blow-up when both components are near zero.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return _SafeAtan2.apply(x[:, 0:1], x[:, 1:2], self.eps)


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


class ResidualBlock(nn.Module):
    """Residual block with skip connection for decoder.

    Uses two conv layers with a skip connection. If input and output channels differ,
    uses a 1x1 conv to match dimensions in the skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True, dropout=0.0):
        super().__init__()

        # Main path
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.main = nn.Sequential(*layers)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            # Need to adjust dimensions
            skip_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)]
            if use_batchnorm:
                skip_layers.append(nn.BatchNorm2d(out_channels))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.main(x)
        out = out + identity
        out = self.relu(out)
        return out


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
        x = self.output(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.output_activation(x)
        return x


class CoupledDecoder256(nn.Module):
    """CNN decoder with shared features for joint real/imaginary or amplitude/phase prediction.

    Outputs both components in a single tensor of shape (B, 2, H, W). With
    circular_phase=False, both channels use CustomActivation; the calling model
    interprets them as [real, imaginary] (PtychoViTCoupled) or [amplitude, phase]
    (PtychoViTCoupledAmpPhase) and applies output scaling and offsets. With
    circular_phase=True, the decoder returns [amplitude, phase], computing phase
    from two unconstrained sin/cos components via atan2.

    Args:
        latent_dim: Input channels from bottleneck (default: 512)
        base_channels: Base number of channels for final layer (default: 64)
        use_batchnorm: Whether to use batch normalization (default: True)
        dropout: Dropout rate (default: 0.0)
        num_stages: Number of upsampling stages, should match encoder (default: 4)
        use_resnet_blocks: Use ResidualBlock instead of TransposeConvBlock (default: False)
        blocks_per_stage: Number of residual blocks per stage when use_resnet_blocks=True (default: 2)
        circular_phase: If True, project to 3 raw channels: amplitude and unconstrained
            sin/cos components. Apply CustomActivation to amplitude and collapse sin/cos
            to phase via atan2, returning (B, 2, H, W): [amplitude, phase]. If False,
            return 2 channels with CustomActivation for either output representation.
            (default: False)
        atan2_eps: Gradient stabilization clamp for the atan2 denominator when circular_phase=True.
            Prevents blow-up when both sin/cos outputs are near zero. (default: 1e-7)
    """
    def __init__(self, latent_dim=512, base_channels=64, use_batchnorm=True, dropout=0.0,
                 num_stages=4, use_resnet_blocks=False, blocks_per_stage=2, circular_phase=False,
                 atan2_eps=1e-7):
        super().__init__()

        self.num_stages = num_stages
        self.use_resnet_blocks = use_resnet_blocks
        self.circular_phase = circular_phase
        self.atan2_eps = atan2_eps
        self.stages = nn.ModuleList()

        # Calculate channel progression (same as Decoder256)
        encoder_channels = []
        for i in range(num_stages):
            if i < num_stages - 1:
                ch = min(base_channels * (2 ** i), latent_dim)
            else:
                ch = latent_dim
            encoder_channels.append(ch)

        # Build shared decoder backbone
        in_ch = latent_dim
        for i in range(num_stages):
            # Determine output channels for this stage
            if i < num_stages - 1:
                out_ch = encoder_channels[num_stages - 2 - i]
            else:
                out_ch = base_channels

            if use_resnet_blocks:
                # Create stage with residual blocks and bilinear upsample
                stage_blocks = []
                stage_blocks.append(
                    ResidualBlock(in_ch, out_ch, stride=1,
                                  use_batchnorm=use_batchnorm, dropout=dropout)
                )
                for _ in range(blocks_per_stage - 1):
                    stage_blocks.append(
                        ResidualBlock(out_ch, out_ch, stride=1,
                                      use_batchnorm=use_batchnorm, dropout=dropout)
                    )
                stage_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                stage = nn.Sequential(*stage_blocks)
            else:
                # Create stage with two ConvTranspose2d and bilinear upsample
                stage = nn.Sequential(
                    TransposeConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                                     use_batchnorm=use_batchnorm, dropout=dropout),
                    TransposeConvBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1,
                                     use_batchnorm=use_batchnorm, dropout=dropout),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            self.stages.append(stage)
            in_ch = out_ch

        # Shared projection before the output split
        self.pre_output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels // 2) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        if circular_phase:
            # 3 raw channels: 1 for amplitude + 2 (sin, cos) for phase via atan2
            self.output = nn.Conv2d(base_channels // 2, 3, kernel_size=1, stride=1, padding=0)
            self.amp_activation = CustomActivation()
        else:
            # Standard 2-channel output with shared activation
            self.output = nn.Sequential(
                nn.Conv2d(base_channels // 2, 2, kernel_size=1, stride=1, padding=0),
                CustomActivation()
            )

    def forward(self, x):
        """
        Forward pass through shared decoder.

        Args:
            x: Latent representation from encoder

        Returns:
            Tensor of shape (B, 2, H, W).
            Default: channel 0 is real, channel 1 is imaginary (both via CustomActivation).
            circular_phase=True: channel 0 is amplitude (CustomActivation),
                channel 1 is phase in [-pi, pi] via atan2(sin, cos).
        """
        for stage in self.stages:
            x = stage(x)

        x = self.pre_output(x)
        x = self.output(x)

        if self.circular_phase:
            amp = self.amp_activation(x[:, 0:1])
            ph = _SafeAtan2.apply(x[:, 1:2], x[:, 2:3], self.atan2_eps)
            x = torch.cat([amp, ph], dim=1)

        return x


class ResNetDecoder256(nn.Module):
    """ResNet-style decoder for 256x256 images with skip connections.

    This decoder uses residual blocks with skip connections for better gradient flow
    and improved feature learning. It can be used as a drop-in replacement for Decoder256.

    Key differences from Decoder256:
    - Uses residual blocks instead of simple conv blocks
    - Better gradient flow through skip connections
    - More stable training with deeper networks

    Args:
        latent_dim: Input channels from bottleneck (default: 512)
        base_channels: Base number of channels for final layer (default: 64)
        out_channels: Number of output channels (default: 1)
        use_batchnorm: Whether to use batch normalization (default: True)
        output_activation: Output activation function (default: None)
            Can be 'sigmoid', 'tanh', 'custom', or None
        dropout: Dropout rate (default: 0.0)
        num_stages: Number of upsampling stages, should match encoder (default: 4)
        blocks_per_stage: Number of residual blocks per stage (default: 2)
    """
    def __init__(self, latent_dim=512, base_channels=64, out_channels=1, use_batchnorm=True,
                 output_activation=None, dropout=0.0, num_stages=4, blocks_per_stage=2):
        super().__init__()

        self.num_stages = num_stages
        self.blocks_per_stage = blocks_per_stage
        self.stages = nn.ModuleList()

        # Calculate channel progression (reverse of encoder)
        encoder_channels = []
        for i in range(num_stages):
            if i < num_stages - 1:
                ch = min(base_channels * (2 ** i), latent_dim)
            else:
                ch = latent_dim
            encoder_channels.append(ch)

        # Build ResNet decoder stages
        in_ch = latent_dim
        for i in range(num_stages):
            # Determine output channels for this stage
            if i < num_stages - 1:
                out_ch = encoder_channels[num_stages - 2 - i]
            else:
                out_ch = base_channels

            # Create stage with residual blocks
            stage_blocks = []

            # First block in stage (may change channels)
            stage_blocks.append(
                ResidualBlock(in_ch, out_ch, stride=1,
                            use_batchnorm=use_batchnorm, dropout=dropout)
            )

            # Additional residual blocks in this stage (same channels)
            for _ in range(blocks_per_stage - 1):
                stage_blocks.append(
                    ResidualBlock(out_ch, out_ch, stride=1,
                                use_batchnorm=use_batchnorm, dropout=dropout)
                )

            # Upsample at end of stage (2x spatial size)
            stage_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            self.stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch

        # Final output layer
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Optional output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'custom':
            self.output_activation = CustomActivation()
        elif output_activation == 'atan2':
            # Two output channels (sin, cos components); atan2 collapses to 1-channel phase in [-pi, pi]
            self.output = nn.Conv2d(base_channels, 2, kernel_size=1, stride=1, padding=0)
            self.output_activation = Atan2Activation()
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