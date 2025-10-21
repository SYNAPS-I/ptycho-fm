import torch
import torch.nn as nn

class Decoder(nn.Module):
    """CNN decoder that upsamples ViT encoder output to a predicted amplitude or phase of the original image size."""

    def __init__(
        self,
        embed_dim=192,
        out_channels=1,
        hidden_dims=[128, 64, 32, 16], 
        activation: str = "sigmoid"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_dim = embed_dim

        for hidden_dim in hidden_dims:
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_dim,
                        hidden_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )
            in_dim = hidden_dim

        # Final output layer
        self.final_conv = nn.Conv2d(in_dim, out_channels, kernel_size=3, stride=1, padding=1)
        # Activation (determines whether amplitude or phase is predicted)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Un-physical activation function: {activation}")

    def forward(self, x):
        """
        Args:
            x: Encoder output with shape (batch, embed_dim, H, W)

        Returns:
            Decoded output with shape (batch, out_channels, 512, 512)
        """
        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.final_conv(x)
        x = self.activation(x)
        return x