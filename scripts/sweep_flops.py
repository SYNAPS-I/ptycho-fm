import math
import yaml

from utils.flops_utils import HEAD_DIM, model_config_from_base, PtychoViTFlopsCalculator

CONFIGS = [
    (192, 6),
    (256, 8),
    (384, 8),
    (512, 8),
    (512, 12),
    (640, 12),
    (768, 12),
    (768, 16),
    (1024, 12),
    (1024, 16),
    (1024, 24),
    (1536, 16),
    (1536, 32),
    (2048, 48),
]

PATCH_SIZES = [16]
GLOBAL_BATCH_SIZE = 512
FORWARD_BACKWARD_FACTOR = 3

BUDGETS = [6e17, 1e18, 3e18, 6e18, 1e19, 3e19, 6e19]
MIN_ITERS = 1000
MAX_ITERS = 120_000


def decoder_num_stages(img_size: int, patch_size: int) -> int:
    grid = img_size // patch_size
    ratio = img_size / grid
    k = math.log2(ratio)
    if abs(k - round(k)) > 1e-6:
        raise ValueError(f"No integer num_stages for img_size={img_size}, patch_size={patch_size}")
    return int(round(k))


if __name__ == "__main__":
    with open("config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    base_model = full_cfg["model"]
    probe_modes = int(full_cfg.get("data", {}).get("max_probe_modes", 10))
    img_size = int(base_model.get("encoder", {}).get("img_size", 256))

    results = []

    for budget in BUDGETS:
        for embed, depth in CONFIGS:
            if embed % HEAD_DIM != 0:
                continue
            for patch_size in PATCH_SIZES:
                if img_size % patch_size != 0:
                    continue
                model_cfg = model_config_from_base(base_model, embed, depth)
                model_cfg["encoder_type"] = "custom"
                model_cfg["encoder"]["patch_size"] = patch_size
                model_cfg["encoder"]["img_size"] = img_size
                model_cfg["decoder"]["num_stages"] = decoder_num_stages(img_size, patch_size)

                calc = PtychoViTFlopsCalculator(
                    model_cfg,
                    batch_size=1,
                    spatial=img_size,
                    probe_modes=probe_modes,
                )
                per_forward_tflops = calc.flops_analytical()
                per_step_flops = per_forward_tflops * 1e12 * FORWARD_BACKWARD_FACTOR * GLOBAL_BATCH_SIZE
                params = calc.param_count()
                num_iters = budget / per_step_flops
                if num_iters < MIN_ITERS or num_iters > MAX_ITERS:
                    continue
                used_flops = per_step_flops * num_iters
                results.append((embed, depth, patch_size, per_step_flops, params, num_iters, used_flops))

    print(
        f"{'embed':>6} {'depth':>6} {'patch':>5} "
        f"{'per_step_flop':>14} {'param(m)':>10} {'num_iters':>10} {'used_flop':>14}"
    )
    print("-" * 100)

    results.sort(key=lambda x: x[-1])
    for embed, depth, patch_size, per_step_flops, params, num_iters, used_flops in results:
        print(
            f"{embed:6} {depth:6} {patch_size:5} "
            f"{per_step_flops:14.2e} {params:10.1f} {num_iters:10.0f} {used_flops:14.2e}"
        )

    print("-" * 100)
    print("-" * 100)

    results.sort(key=lambda x: x[4])
    print(
        f"{'embed':>6} {'depth':>6} {'patch':>5} "
        f"{'per_step_flop':>14} {'param(m)':>10} {'num_iters':>10} {'used_flop':>14}"
    )
    print("-" * 100)
    for embed, depth, patch_size, per_step_flops, params, num_iters, used_flops in results:
        print(
            f"{embed:6} {depth:6} {patch_size:5} "
            f"{per_step_flops:14.2e} {params:10.1f} {num_iters:10.1f} {used_flops:14.2e}"
        )
