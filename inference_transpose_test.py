"""Quick inference script to compare transpose=False vs True stitching.
Runs on cuda:1. Results saved to checkpoints/hxn_finetune/run003/.
"""
import os, numpy as np, torch, yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import PtychographyDataset
from model.model import PtychoViT
from utils.ptychi_utils import stitch_patches

DEVICE      = torch.device('cuda:0')  # cuda:0 because CUDA_VISIBLE_DEVICES=1 remaps it
CONFIG_PATH = 'checkpoints/hxn_finetune/run003/config.yaml'
MODEL_PATH  = 'checkpoints/hxn_finetune/run003/model_epoch_021.pth'
DATA_PATH   = 'train_data/405667_dp.hdf5'
OUT_DIR     = 'checkpoints/hxn_finetune/run003'
TRANSPOSE   = True    # ← the variable under test
BATCH_SIZE  = 128

# ── load model ──────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

model = PtychoViT(config=config['model'])
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE).eval()
print(f"Model loaded. amp_offset={model.amp_offset:.3f}  "
      f"amp_scale={model.log_scale_amp.exp().item():.3f}  "
      f"ph_scale={model.log_scale_ph.exp().item():.3f}")

# ── dataset ──────────────────────────────────────────────────────────────────
norm_path = config['data'].get('test_normalization') or config['data'].get('normalization_dict_path')
print("Initialising dataset…", flush=True)
dataset = PtychographyDataset(
    file_path=DATA_PATH,
    scale=config['data']['scale'],
    normalization_dict_path=norm_path,
    apply_noise=False,
    cache_object=False,   # not needed for inference — avoids slow GT patch caching
    max_probe_modes=config['data'].get('max_probe_modes', 1),
    transpose_object_patches=TRANSPOSE,
)
print(f"Dataset ready: {len(dataset)} patterns  transpose={TRANSPOSE}", flush=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── inference ────────────────────────────────────────────────────────────────
all_ph, all_pos = [], []
target_size  = config['model']['encoder']['img_size']
central_crop = config['training'].get('test_plot_central_crop', 0)

# Probe-intensity weight: |probe|² tells us where the beam was illuminating.
# Computed once from the first batch (probe is position-independent).
probe_weight = None  # (H_p, W_p) float, at cropped-patch resolution

with torch.no_grad():
    for i, batch in enumerate(loader):
        diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale = batch

        inp = diff_amp.to(DEVICE)
        h, w = inp.shape[-2], inp.shape[-1]
        if h > target_size or w > target_size:
            inp = torch.fft.fftshift(inp, dim=(-2, -1))
            sh = (h - target_size) // 2
            sw = (w - target_size) // 2
            inp = inp[..., sh:sh+target_size, sw:sw+target_size]

        # Compute probe weight before converting to real stack
        if probe_weight is None and probe.is_complex():
            # Sum modes → (B, H, W), take first batch element
            pw_map = probe.abs().pow(2).sum(dim=1)[0].cpu()  # (H_probe, W_probe)
            # Apply the same spatial padding used for the model input
            ph0, pw0 = pw_map.shape[-2], pw_map.shape[-1]
            if ph0 < target_size or pw0 < target_size:
                pt = (target_size - ph0) // 2
                pb = target_size - ph0 - pt
                pl = (target_size - pw0) // 2
                pr = target_size - pw0 - pl
                pw_map = torch.nn.functional.pad(pw_map, (pl, pr, pt, pb))
            # Apply central crop to match stitched patch size
            if central_crop > 0:
                pw_map = pw_map[central_crop:-central_crop, central_crop:-central_crop]
            pw_max = pw_map.max()
            if pw_max > 0:
                pw_map = pw_map / pw_max
            probe_weight = pw_map.squeeze()  # ensure 2-D (H, W)
            print(f"Probe weight computed: shape={probe_weight.shape}, "
                  f"nonzero fraction={float((probe_weight > 0.01).float().mean()):.2%}")

        if probe.is_complex():
            probe = torch.stack([probe.real, probe.imag], dim=-1)
        ph, pw = probe.shape[-3], probe.shape[-2]
        if ph < target_size or pw < target_size:
            pt, pb = (target_size-ph)//2, target_size-ph-(target_size-ph)//2
            pl, pr = (target_size-pw)//2, target_size-pw-(target_size-pw)//2
            probe = torch.nn.functional.pad(probe, (0,0,pl,pr,pt,pb))

        _, _, out_ph = model(inp, probe.to(DEVICE), norm.to(DEVICE), scale.to(DEVICE))

        ph_batch = out_ph.squeeze(1).cpu()
        if TRANSPOSE:   # un-transpose before stitching (same logic as run_inference.py)
            ph_batch = ph_batch.transpose(-2, -1).contiguous()
        all_ph.append(ph_batch.numpy())
        all_pos.append(probe_pos.cpu().numpy())

        if (i+1) % 10 == 0:
            print(f"  batch {i+1}/{len(loader)}  ({(i+1)*BATCH_SIZE}/{len(dataset)} patterns)", flush=True)

pred_ph  = np.concatenate(all_ph,  axis=0)
positions = np.concatenate(all_pos, axis=0)
print(f"Inference done. pred_ph shape: {pred_ph.shape}")

# ── stitch ───────────────────────────────────────────────────────────────────
patch_size = pred_ph.shape[-1]
print("Stitching phase…")
stitched_ph, _ = stitch_patches(
    pred_ph, positions, patch_size,
    crop=central_crop, canvas_pad=64,
    mode="phase",
    patch_weights=probe_weight,  # |probe|² weighting — zero outside beam support
)
print(f"Stitched phase shape: {stitched_ph.shape}")

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(f'run003 best_model — transpose={TRANSPOSE}', fontsize=14)

# centre crop for detail — compute its range first, use for both panels
H, W = stitched_ph.shape
ch, cw = H//4, W//4
crop = stitched_ph[ch:H-ch, cw:W-cw]
vc_mean, vc_std = crop.mean(), crop.std()
vc1, vc2 = vc_mean - 2*vc_std, vc_mean + 2*vc_std

im = axes[0].imshow(stitched_ph, cmap='magma', vmin=vc1, vmax=vc2)
axes[0].set_title('Predicted Phase (full)')
axes[0].axis('off')
plt.colorbar(im, ax=axes[0], fraction=0.03)

im2 = axes[1].imshow(crop, cmap='magma', vmin=vc1, vmax=vc2)
axes[1].set_title('Predicted Phase (centre crop)')
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1], fraction=0.03)

out_path = os.path.join(OUT_DIR, f'infer_{"with" if TRANSPOSE else "no"}_transpose.png')
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
