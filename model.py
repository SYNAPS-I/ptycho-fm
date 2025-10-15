import torch
import torch.nn as nn
from vit import VisionTransformer
from decoders import Decoder

class PtychoViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=512,
            patch_size=16,
            in_channels=1,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            use_cls_token=False
        )
        self.amp_decoder = Decoder(
            embed_dim=192,
            out_channels=1,
            hidden_dims=[128, 64, 32, 16], 
            activation="sigmoid"
        )
        self.ph_decoder = Decoder(
            embed_dim=192,
            out_channels=1,
            hidden_dims=[128, 64, 32, 16], 
            activation="tanh"
        )
    
    def forward(self, x, probe, normalization, scale):
        x = self.encoder(x)
        amp = self.amp_decoder(x).squeeze(1)
        ph = self.ph_decoder(x).squeeze(1)
        #print('Probe info before fix: ', probe.shape, probe.dtype)
        probe = torch.complex(probe[:, :, :, :, :, 0], probe[:, :, :, :, :, 1])
        #print('Probe info after fix: ', probe.shape, probe.dtype)

        # Complex object and diffraction
        complex_object = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))
        #print('Object info: ', complex_object.shape, complex_object.dtype)
        Psi = torch.fft.fftshift(torch.fft.fft2(complex_object[:, None, None, :] * probe), dim=(-2, -1))
        intensity = (Psi.abs()**2).sum(2)[:, 0] # sum over incoherent modes, select first OPR mode, final shape (B, H, W)

        # Normalization
        normalization = normalization.view(normalization.shape[0], 1, 1)
        scale = scale.view(scale.shape[0], 1, 1)
        intensity = (intensity.float() / normalization) * scale

        pred_diff_amp = torch.sqrt(intensity)

        return pred_diff_amp.unsqueeze(1), amp.unsqueeze(1), ph.unsqueeze(1)