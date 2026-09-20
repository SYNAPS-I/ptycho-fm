"""Analytical compute for the current custom ViT and standard twin decoders.

All public FLOP methods return TFLOPs (1e12 operations). A multiply-add is
counted as two operations. Nonlinear costs are estimates: GELU 8, tanh 10,
custom activation 12, sigmoid 4, and softmax 5 per element. Attention scaling
and residual additions are included. BatchNorm includes training statistics.

The boundary is encoder input through decoder activations. Input transforms,
external output normalization, physics propagation/loss, optimizer updates,
validation, dropout/RNG and memory movement are excluded. Training FLOPs are
estimated as three forward passes. This is an analytical workload estimate,
not a measurement of kernel instructions or hardware throughput.
"""

import copy
from dataclasses import dataclass

ACCOUNTING_VERSION = "ptycho_fm_v1"
FORWARD_BACKWARD_FACTOR = 3.0

HEAD_DIM = 64
EMBED_BASE_RATIO = 8

# (embed_dim, depth): (512, 12) … (2048, 48) in equal steps; num_heads = embed_dim / 64
MODEL_SPECS = [(512, 12), (768, 18), (1024, 24), (1536, 32), (2048, 48)]

TFLOPS = 1e-12


def attention_flops(batch, seq, embed, heads):
    head_dim = embed // heads
    qkv = 2 * batch * seq * embed * embed * 3 + batch * seq * embed * 3
    logits = 2 * batch * heads * seq * head_dim * seq
    scaling = batch * heads * seq * seq
    softmax = 5 * batch * heads * seq * seq
    attend = 2 * batch * heads * seq * seq * head_dim
    project = 2 * batch * seq * embed * embed + batch * seq * embed
    return (qkv + logits + scaling + softmax + attend + project) * TFLOPS


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


def batch_norm2d_flops(batch, h, w, channels, training=True):
    """Analytical BN cost including training statistics and running updates.

    For n = batch*h*w values per channel: mean costs n operations, biased
    variance 3n, normalization/affine 4n. Ten per-channel operations cover
    epsilon/sqrt (2), unbiased running-variance conversion (2), and the two
    exponential moving averages (6). Arithmetic and sqrt each count as one.
    RNG, memory traffic, and integer bookkeeping are excluded throughout.
    """
    n = batch * h * w
    if training:
        if n <= 1:
            raise ValueError("Training BatchNorm needs more than one value per channel")
        return channels * (8 * n + 10) * TFLOPS
    return channels * (4 * n + 2) * TFLOPS


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
    output_activation='custom',
    training=True,
):
    """FLOPs for one Decoder256 (matches ptycho_fm/model/decoders.py)."""
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
            total += batch_norm2d_flops(batch, h, w, out_ch, training=training)
        total += relu_flops(batch, h, w, out_ch)
        total += conv_transpose2d_flops(batch, h, w, out_ch, out_ch, kernel)
        if use_batchnorm:
            total += batch_norm2d_flops(batch, h, w, out_ch, training=training)
        total += relu_flops(batch, h, w, out_ch)

        total += bilinear_upsample2x_flops(batch, h, w, out_ch)
        in_ch = out_ch
        h, w = h * 2, w * 2

    if include_output_linear:
        total += (2 * batch * h * w * base_channels * out_channels + batch * h * w * out_channels) * TFLOPS
    if include_activation:
        costs = {None: 0, 'tanh': 10, 'custom': 12, 'sigmoid': 4}
        if output_activation not in costs:
            raise ValueError(f"Unsupported decoder activation: {output_activation}")
        total += costs[output_activation] * batch * out_channels * h * w * TFLOPS
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
    residual = 2 * norm_tokens * TFLOPS
    block_flops = attn + mlp + 2 * norm_per_block + residual
    final_norm = layer_norm_flops(norm_tokens)

    if verbose:
        print(f"  patch embedding: {patch} TFLOPs")
        print(f"  pos add: {pos} TFLOPs")
        print(f"  per-block (attn+mlp+2LN): {block_flops} TFLOPs")
        print(f"  final LN: {final_norm} TFLOPs")

    return patch + pos + block_flops * depth + final_norm



def model_config_from_base(base_model: dict, embed_dim: int, depth: int) -> dict:
    model = copy.deepcopy(base_model)
    model.setdefault('encoder', {}).update(
        embed_dim=embed_dim, depth=depth, num_heads=embed_dim // HEAD_DIM,
    )
    model.setdefault('decoder', {}).update(
        latent_dim=embed_dim, base_channels=embed_dim // EMBED_BASE_RATIO,
    )
    return model


class PtychoFMFlopsCalculator:
    """Current PtychoFM custom encoder + standard independent decoders.

    Parameter counts include the three registered output-normalization
    parameters, whether frozen or trainable; their external arithmetic is
    outside the encoder/decoder FLOP boundary.
    """

    def __init__(self, model_cfg, *, batch_size, spatial=None, probe_modes=10, model=None):
        self.model_cfg = copy.deepcopy(model_cfg)
        if model_cfg.get('encoder_type', 'custom') != 'custom':
            raise NotImplementedError("FLOP accounting requires encoder_type='custom'")
        if model_cfg.get('decoder_type', 'standard') != 'standard':
            raise NotImplementedError("FLOP accounting requires decoder_type='standard'")
        if model_cfg.get('coupled_decoder_mode') is not None:
            raise NotImplementedError("FLOP accounting requires independent amplitude/phase decoders")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        self.batch_size = batch_size
        enc = model_cfg.get('encoder', {})
        self.img_h = int(enc.get('img_size', 256 if spatial is None else spatial))
        self.img_w = self.img_h
        self.patch_size = int(enc.get('patch_size', 16))
        self.in_channels = int(enc.get('in_channels', 1))
        self.embed = int(enc.get('embed_dim', 512))
        self.depth = int(enc.get('depth', 12))
        self.heads = int(enc.get('num_heads', 8))
        self.mlp_ratio = float(enc.get('mlp_ratio', 4.0))
        self.use_cls_token = bool(enc.get('use_cls_token', False))
        dec = model_cfg.get('decoder', {})
        self.base_channels = int(dec.get('base_channels', 64))
        self.num_stages = int(dec.get('num_stages', 4))
        latent = dec.get('latent_dim')
        self.latent_dim = self.embed if latent is None else int(latent)
        self.use_batchnorm = bool(dec.get('use_batchnorm', True))
        self.phase_activation = dec.get('phase_activation', 'custom')
        self.probe_modes = probe_modes
        self.model = model
        dimensions = (self.img_h, self.patch_size, self.in_channels, self.embed,
                      self.depth, self.heads, self.base_channels, self.num_stages)
        if any(value <= 0 for value in dimensions) or self.mlp_ratio <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.img_h % self.patch_size or self.embed % self.heads:
            raise ValueError('Image size and embedding must be divisible by patch size and heads')
        if self.latent_dim != self.embed:
            raise ValueError('Decoder latent_dim must equal encoder embed_dim')
        if 2 ** self.num_stages != self.patch_size:
            raise ValueError('Decoder upsampling must restore the input image size')
        if self.phase_activation not in (None, 'tanh', 'custom', 'sigmoid'):
            raise ValueError(f'Unsupported phase activation: {self.phase_activation}')

    def grid_after_vit(self):
        return self.img_h // self.patch_size

    def flops_encoder_decoder(self, verbose=False, *, batch_size=None, training=True):
        batch = self.batch_size if batch_size is None else batch_size
        if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
            raise ValueError('batch_size must be a positive integer')
        encoder = custom_vit_encoder_flops(
            batch, self.img_h, self.img_w, self.patch_size, self.in_channels,
            self.embed, self.depth, self.heads, self.mlp_ratio, self.use_cls_token,
            verbose=verbose,
        )
        decoders = sum(
            decoder256_flops(
                batch, self.grid_after_vit(), self.num_stages, self.base_channels,
                self.latent_dim, use_batchnorm=self.use_batchnorm,
                output_activation=activation, training=training,
            ) for activation in ('tanh', self.phase_activation)
        )
        return encoder + decoders

    def flops_analytical(self, verbose=False, *, batch_size=None, training=True):
        return self.flops_encoder_decoder(verbose, batch_size=batch_size, training=training)

    def training_tflops(self, batch_size):
        """Compute one rank's actual batch before summing across ranks.

        BN's per-channel statistics cost means batch-1 FLOPs multiplied by
        the sample count is not exactly the cost of a larger batch.
        """
        return FORWARD_BACKWARD_FACTOR * self.flops_analytical(batch_size=batch_size)

    def profile_fvcore(self, device='cpu'):
        """Optional encoder/decoder trace in fvcore's native one-FMA units.

        Returns None when fvcore is absent. Unsupported operators (including
        fused attention, if unhandled by the installed fvcore) are reported;
        this diagnostic partial total must never drive training budgets.
        """
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            return None
        import torch
        from torch import nn

        from ptycho_fm.model.model import PtychoFM

        class EncoderDecoders(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                features = self.model.encoder(x)
                return self.model.amp_decoder(features), self.model.ph_decoder(features)

        # Clone supplied weights so profiling cannot alter live training state.
        # Restore RNG, including any CUDA initialization/trace consumption.
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            model = copy.deepcopy(self.model) if self.model is not None else PtychoFM(config=self.model_cfg)
            if hasattr(model, 'module'):
                model = model.module
            wrapper = EncoderDecoders(model).to(device).eval()
            inputs = torch.zeros(self.batch_size, self.in_channels, self.img_h, self.img_w, device=device)
            with torch.no_grad():
                analysis = FlopCountAnalysis(wrapper, (inputs,))
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                total = analysis.total()
            return {'total_tflops_fvcore': total * TFLOPS,
                    'units': 'fvcore native; one fused multiply-add counts as one',
                    'scope': 'encoder and twin decoders, evaluation mode',
                    'unsupported_ops': dict(analysis.unsupported_ops()),
                    'by_operator': dict(analysis.by_operator())}

    def forward_flops_fvcore(self, device='cpu', norm=None, scale=None):
        """Compatibility scalar; use profile_fvcore for unsupported-op details.

        norm/scale are accepted for old callers; the profiling boundary now
        excludes preprocessing and physics, matching the analytical boundary.
        """
        import warnings

        result = self.profile_fvcore(device)
        if result is None:
            return None
        warnings.warn(f"Diagnostic fvcore total uses one-FMA units; unsupported ops: {result['unsupported_ops']}. "
                      'Use analytical FLOPs for budgets.', stacklevel=2)
        return result['total_tflops_fvcore']

    def param_count(self):
        """Total registered parameters, in millions, matching current PtychoFM."""
        encoder = custom_vit_encoder_param_count(
            self.img_h, self.img_w, self.patch_size, self.in_channels, self.embed,
            self.depth, self.mlp_ratio, self.use_cls_token,
        )
        decoder = decoder256_param_count(
            self.latent_dim, self.base_channels, self.num_stages,
            use_batchnorm=self.use_batchnorm,
        )
        return (encoder + 2 * decoder + 3) * 1e-6


# Read-only analysis callers may still import the old calculator name.
PtychoViTFlopsCalculator = PtychoFMFlopsCalculator


@dataclass(frozen=True)
class AnalyticalFlopsForTrainer:
    calculator: PtychoFMFlopsCalculator
    train_batch_size: int
    world_size: int
