import copy
from dataclasses import dataclass

import torch
import yaml
from model.model import PtychoViT

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None  # type: ignore[misc, assignment]

FVCORE_AVAILABLE = FlopCountAnalysis is not None

HEAD_DIM = 64
EMBED_BASE_RATIO = 8

# (embed_dim, depth): (512, 12) … (2048, 48) in equal steps; num_heads = embed_dim / 64
MODEL_SPECS = [(512, 12), (768, 18), (1024, 24), (1536, 32), (2048, 48)]

TFLOPS = 1e-12


def attention_flops(batch, seq, embed, heads):
    head_dim = embed // heads
    qkv = 2 * batch * seq * embed * embed * 3 + batch * seq * embed * 3
    logits = 2 * batch * heads * seq * head_dim * seq
    softmax = 5 * batch * heads * seq * seq
    attend = 2 * batch * heads * seq * seq * head_dim
    project = 2 * batch * seq * embed * embed + batch * seq * embed
    return (qkv + logits + softmax + attend + project) * TFLOPS


def mlp_flops(batchseq, embed, hidden):
    fc1 = 2 * batchseq * embed * hidden + batchseq * hidden
    fc2 = 2 * batchseq * hidden * embed + batchseq * embed
    gelu = 8 * batchseq * hidden
    return (fc1 + fc2 + gelu) * TFLOPS


def patch_embed_flops(batch, h, w, c, patch_size, embed):
    seq = h // patch_size * w // patch_size
    patchify = 2 * batch * h * w * c * embed + batch * seq * embed
    return patchify * TFLOPS


def layer_norm_flops(size):
    return 8 * size * TFLOPS


def pos_embed_add_flops(batch_tokens, embed):
    return batch_tokens * embed * TFLOPS


def conv_transpose2d_flops(batch, h, w, cin, cout, kernel, stride=1, padding=1, output_padding=0):
    """ConvTranspose2d FLOPs using 2 FLOPs per multiply-add."""
    if stride != 1:
        raise NotImplementedError("Decoder uses stride-1 transpose convs only.")
    h_out = (h - 1) * stride - 2 * padding + kernel + output_padding
    w_out = (w - 1) * stride - 2 * padding + kernel + output_padding
    macs = 2 * batch * h_out * w_out * cin * cout * kernel * kernel
    bias = batch * h_out * w_out * cout
    return (macs + bias) * TFLOPS


def batch_norm2d_flops(batch, h, w, channels):
    return 4 * batch * channels * h * w * TFLOPS


def relu_flops(batch, h, w, channels):
    return batch * channels * h * w * TFLOPS


def bilinear_upsample2x_flops(batch, h_in, w_in, channels):
    h_out, w_out = 2 * h_in, 2 * w_in
    return 6 * batch * channels * h_out * w_out * TFLOPS


def _decoder_stage_channel_map(num_stages, base_channels, latent_dim):
    encoder_channels = []
    for i in range(num_stages):
        if i < num_stages - 1:
            ch = min(base_channels * (2**i), latent_dim)
        else:
            ch = latent_dim
        encoder_channels.append(ch)
    return encoder_channels


def decoder256_flops(
    batch,
    grid0,
    num_stages,
    base_channels,
    latent_dim,
    out_channels=1,
    use_batchnorm=True,
    kernel=3,
    include_output_linear=True,
    include_activation=True,
):
    """FLOPs for one Decoder256 (matches model/decoders.py)."""
    enc_ch = _decoder_stage_channel_map(num_stages, base_channels, latent_dim)
    in_ch = latent_dim
    total = 0.0
    h = w = grid0
    for i in range(num_stages):
        if i < num_stages - 1:
            out_ch = enc_ch[num_stages - 2 - i]
        else:
            out_ch = base_channels

        total += conv_transpose2d_flops(batch, h, w, in_ch, out_ch, kernel)
        if use_batchnorm:
            total += batch_norm2d_flops(batch, h, w, out_ch)
        total += relu_flops(batch, h, w, out_ch)
        total += conv_transpose2d_flops(batch, h, w, out_ch, out_ch, kernel)
        if use_batchnorm:
            total += batch_norm2d_flops(batch, h, w, out_ch)
        total += relu_flops(batch, h, w, out_ch)

        total += bilinear_upsample2x_flops(batch, h, w, out_ch)
        in_ch = out_ch
        h, w = h * 2, w * 2

    if include_output_linear:
        total += (2 * batch * h * w * base_channels * out_channels + batch * h * w * out_channels) * TFLOPS
    if include_activation:
        total += 10 * batch * out_channels * h * w * TFLOPS
    return total


def _convt2d_weight_bias(in_ch: int, out_ch: int, kernel: int) -> int:
    return in_ch * out_ch * kernel * kernel + out_ch


def decoder256_param_count(
    latent_dim: int,
    base_channels: int,
    num_stages: int,
    out_channels: int = 1,
    use_batchnorm: bool = True,
    kernel: int = 3,
) -> int:
    """Trainable parameters in one ``Decoder256`` (ConvTranspose + optional BN + final Linear)."""
    enc_ch = _decoder_stage_channel_map(num_stages, base_channels, latent_dim)
    in_ch = latent_dim
    total = 0
    for i in range(num_stages):
        if i < num_stages - 1:
            out_ch = enc_ch[num_stages - 2 - i]
        else:
            out_ch = base_channels
        total += _convt2d_weight_bias(in_ch, out_ch, kernel)
        if use_batchnorm:
            total += 2 * out_ch
        total += _convt2d_weight_bias(out_ch, out_ch, kernel)
        if use_batchnorm:
            total += 2 * out_ch
        in_ch = out_ch
    total += base_channels * out_channels + out_channels
    return total


def custom_vit_encoder_param_count(
    img_h: int,
    img_w: int,
    patch_size: int,
    in_channels: int,
    embed: int,
    depth: int,
    mlp_ratio: float,
    use_cls_token: bool,
) -> int:
    """Trainable parameters in ``CustomViT`` (patch + pos + blocks + final LN), cf. neural-scaling style accounting."""
    h_patch = img_h // patch_size
    w_patch = img_w // patch_size
    n_patches = h_patch * w_patch
    hidden = int(embed * mlp_ratio)

    patch = in_channels * embed * patch_size * patch_size + embed
    n_pos = n_patches + (1 if use_cls_token else 0)
    pos = n_pos * embed
    cls_params = embed if use_cls_token else 0

    # One LayerNorm: gamma + beta
    ln = 2 * embed
    qkv = embed * embed * 3 + 3 * embed
    proj = embed * embed + embed
    mlp_w = embed * hidden + hidden + hidden * embed + embed
    block = 2 * ln + qkv + proj + mlp_w

    final_norm = ln
    return patch + pos + cls_params + block * depth + final_norm


def custom_vit_encoder_flops(
    batch,
    img_h,
    img_w,
    patch_size,
    in_channels,
    embed,
    depth,
    heads,
    mlp_ratio,
    use_cls_token,
    verbose=False,
):
    h_patch = img_h // patch_size
    w_patch = img_w // patch_size
    n_patches = h_patch * w_patch
    n_tokens = n_patches + (1 if use_cls_token else 0)
    hidden = int(embed * mlp_ratio)

    patch = patch_embed_flops(batch, img_h, img_w, in_channels, patch_size, embed)
    pos = pos_embed_add_flops(batch * n_tokens, embed)
    attn = attention_flops(batch, n_tokens, embed, heads)
    mlp = mlp_flops(batch * n_tokens, embed, hidden)
    norm_tokens = batch * n_tokens * embed
    norm_per_block = layer_norm_flops(norm_tokens)
    block_flops = attn + mlp + 2 * norm_per_block
    final_norm = layer_norm_flops(norm_tokens)

    if verbose:
        print(f"  patch embedding: {patch} TFLOPs")
        print(f"  pos add: {pos} TFLOPs")
        print(f"  per-block (attn+mlp+2LN): {block_flops} TFLOPs")
        print(f"  final LN: {final_norm} TFLOPs")

    return patch + pos + block_flops * depth + final_norm


def training_inputs(device, *, batch_size=1, spatial=256, probe_modes=10, norm=72033.211, scale=10000.0):
    x = torch.randn(batch_size, 1, spatial, spatial, device=device)
    probe = torch.randn(batch_size, 1, probe_modes, spatial, spatial, 2, device=device)
    norm_t = torch.full((batch_size,), norm, device=device)
    scale_t = torch.full((batch_size,), scale, device=device)
    return x, probe, norm_t, scale_t


def forward_flops(model, x, probe, norm_t, scale_t):
    if not FVCORE_AVAILABLE:
        raise RuntimeError("fvcore is not installed; pip install fvcore to use forward_flops.")
    model.eval()
    with torch.inference_mode():
        return FlopCountAnalysis(model, (x, probe, norm_t, scale_t)).total()


def model_config_from_base(base_model: dict, embed_dim: int, depth: int) -> dict:
    m = copy.deepcopy(base_model)
    m["encoder"]["embed_dim"] = embed_dim
    m["encoder"]["depth"] = depth
    m["encoder"]["num_heads"] = embed_dim // HEAD_DIM
    m["decoder"]["latent_dim"] = embed_dim
    m["decoder"]["base_channels"] = embed_dim // EMBED_BASE_RATIO
    return m


class PtychoViTFlopsCalculator:
    """Analytical FLOPs for ViT encoder + twin decoders.

    Optional: ``forward_flops_fvcore`` if the ``fvcore`` package is installed.
    """

    def __init__(
        self,
        model_cfg: dict,
        *,
        batch_size: int,
        spatial: int | None = None,
        probe_modes: int = 10,
        model: PtychoViT | None = None,
    ):
        self.model_cfg = model_cfg
        self.batch_size = batch_size
        enc = model_cfg.get("encoder", {})
        self.img_h = int(enc.get("img_size", 256 if spatial is None else spatial))
        self.img_w = self.img_h
        self.patch_size = int(enc.get("patch_size", 16))
        self.in_channels = int(enc.get("in_channels", 1))
        self.embed = int(enc.get("embed_dim", 512))
        self.depth = int(enc.get("depth", 12))
        self.mlp_ratio = float(enc.get("mlp_ratio", 4.0))
        self.use_cls_token = bool(enc.get("use_cls_token", False))
        self.encoder_type = model_cfg.get("encoder_type", "custom")
        self.heads = int(enc.get("num_heads", self.embed // HEAD_DIM))

        dec = model_cfg.get("decoder", {})
        self.base_channels = int(dec.get("base_channels", self.embed // EMBED_BASE_RATIO))
        self.num_stages = int(dec.get("num_stages", 4))
        latent = dec.get("latent_dim")
        self.latent_dim = int(self.embed if latent is None else latent)
        self.use_batchnorm = bool(dec.get("use_batchnorm", True))

        self.probe_modes = probe_modes
        self.model = model

    def grid_after_vit(self):
        return self.img_h // self.patch_size

    def flops_encoder_decoder(self, verbose=False):
        if self.encoder_type != "custom":
            raise NotImplementedError(
                "Analytical encoder FLOPs are implemented for CustomViT only (encoder_type='custom')."
            )
        enc = custom_vit_encoder_flops(
            self.batch_size,
            self.img_h,
            self.img_w,
            self.patch_size,
            self.in_channels,
            self.embed,
            self.depth,
            self.heads,
            self.mlp_ratio,
            self.use_cls_token,
            verbose=verbose,
        )
        g0 = self.grid_after_vit()
        dec = decoder256_flops(
            self.batch_size,
            g0,
            self.num_stages,
            self.base_channels,
            self.latent_dim,
            use_batchnorm=self.use_batchnorm,
        )
        if verbose:
            print(f"  one Decoder256: {dec} TFLOPs")
            print(f"  twin decoders: {2 * dec} TFLOPs")
        return enc + 2 * dec

    def flops_analytical(self, verbose=False):
        return self.flops_encoder_decoder(verbose=verbose)

    def param_count(self) -> float:
        """Estimated parameter count (millions): CustomViT + 2× Decoder256 + two scalar output scales."""
        if self.encoder_type != "custom":
            raise NotImplementedError("param_count is implemented for encoder_type='custom' only.")
        enc_p = custom_vit_encoder_param_count(
            self.img_h,
            self.img_w,
            self.patch_size,
            self.in_channels,
            self.embed,
            self.depth,
            self.mlp_ratio,
            self.use_cls_token,
        )
        dec_p = decoder256_param_count(
            self.latent_dim,
            self.base_channels,
            self.num_stages,
            out_channels=1,
            use_batchnorm=self.use_batchnorm,
        )
        head_extra = 2
        return (enc_p + 2 * dec_p + head_extra) * 1e-6

    def forward_flops_fvcore(self, device, norm=72033.211, scale=10000.0):
        """fvcore FlopCountAnalysis total (full forward), scaled to TFLOPs for display.

        Returns ``None`` if fvcore is not installed.
        """
        if not FVCORE_AVAILABLE:
            return None
        if self.model is None:
            raise ValueError("forward_flops_fvcore requires a PtychoViT instance; pass model=... to __init__.")
        inputs = training_inputs(
            device,
            batch_size=self.batch_size,
            spatial=self.img_h,
            probe_modes=self.probe_modes,
            norm=norm,
            scale=scale,
        )
        return forward_flops(self.model, *inputs) * 1e-12


@dataclass(frozen=True)
class AnalyticalFlopsForTrainer:
    """``PtychoViTFlopsCalculator`` at batch 1, plus factors for cumulative TFLOPs on W&B."""

    calculator: PtychoViTFlopsCalculator
    train_batch_size: int
    world_size: int


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = training_inputs(device)
    global_batch_size, num_batches = 32768, 1099

    for embed_dim, depth in MODEL_SPECS:
        model_cfg = model_config_from_base(cfg["model"], embed_dim, depth)
        model = PtychoViT(config=model_cfg).to(device)
        calc = PtychoViTFlopsCalculator(
            model_cfg,
            batch_size=inputs[0].shape[0],
            spatial=model_cfg["encoder"]["img_size"],
            probe_modes=inputs[1].shape[2],
            model=model,
        )
        analytical = calc.flops_analytical(verbose=False)
        fv_tflops = calc.forward_flops_fvcore(device)
        if fv_tflops is not None:
            fv_tflops *= 2
            epoch_flops = fv_tflops * 3 * global_batch_size * num_batches
            print(
                f"embed_dim={embed_dim} depth={depth} num_heads={embed_dim // HEAD_DIM} | "
                f"analytical={analytical:.3e} TFLOPs  fvcore={fv_tflops:.3e} TFLOPs  "
                f"epoch={epoch_flops:.3e} TFLOPs"
            )
        else:
            print(
                f"embed_dim={embed_dim} depth={depth} num_heads={embed_dim // HEAD_DIM} | "
                f"analytical={analytical:.3e} TFLOPs  (fvcore not installed)"
            )
