import argparse
import math

import torch


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint, None

    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key

    return checkpoint, None


def rename_keys(state_dict, key_map):
    renamed = []
    for old_key, new_key in key_map.items():
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
            renamed.append((old_key, new_key))
    return renamed


def convert_decoder_output_to_linear(state_dict):
    """Convert standard decoder pointwise convolution weights in place.

    Conv2d [out, in, 1, 1] weights become
    Linear [out, in] weights. Biases and already-linear weights are unchanged.
    Common wrapper prefixes (such as module.) are preserved. This conversion
    targets the separate amplitude/phase decoders, not the coupled decoder.
    """
    converted = []
    for key, weight in state_dict.items():
        if not any(
            key == name or key.endswith("." + name)
            for name in ("amp_decoder.output.weight", "ph_decoder.output.weight")
        ):
            continue
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected a tensor for {key}, got {type(weight).__name__}")
        if weight.ndim == 2:
            continue
        if weight.ndim != 4 or any(size != 1 for size in weight.shape[2:]):
            raise ValueError(
                f"Cannot convert {key} with shape {tuple(weight.shape)} to Linear: "
                "expected a pointwise Conv2d weight with singleton kernel dimensions."
            )
        old_shape = tuple(weight.shape)
        state_dict[key] = weight.reshape(weight.shape[0], weight.shape[1])
        converted.append((key, old_shape, tuple(state_dict[key].shape)))
    return converted


def add_missing_current_keys(state_dict, amp_scale, ph_scale, amp_offset):
    defaults = {
        "amp_scale": math.log(amp_scale),
        "ph_scale": math.log(ph_scale),
        "amp_offset": amp_offset,
    }
    added = []
    for key, value in defaults.items():
        if key not in state_dict:
            state_dict[key] = torch.tensor(value, dtype=torch.float32)
            added.append(key)
    return added


def main():
    parser = argparse.ArgumentParser(
        description="Migrate output normalization keys and optional decoder weights in a PyTorch checkpoint."
    )
    parser.add_argument("src", help="Input .pth checkpoint path.")
    parser.add_argument("dst", help="Output .pth checkpoint path.")
    parser.add_argument(
        "--direction",
        choices=("current-to-legacy", "legacy-to-current"),
        default="current-to-legacy",
        help=(
            "current-to-legacy renames amp_scale/ph_scale/amp_offset to "
            "log_scale_amp/log_scale_ph/offset_amp. legacy-to-current does the reverse."
        ),
    )
    parser.add_argument(
        "--convert-decoder-output-to-linear",
        action="store_true",
        help=(
            "Convert standard amp/ph decoder output weights from pointwise Conv2d "
            "to Linear. Requires --direction legacy-to-current."
        ),
    )
    parser.add_argument(
        "--add-missing-current",
        action="store_true",
        help=(
            "When writing current-format keys, add missing amp_scale, ph_scale, "
            "and amp_offset defaults so older checkpoints load strictly."
        ),
    )
    parser.add_argument(
        "--amp-scale",
        type=float,
        default=0.1,
        help="Default linear amp_scale to add with --add-missing-current.",
    )
    parser.add_argument(
        "--ph-scale",
        type=float,
        default=math.pi,
        help="Default linear ph_scale to add with --add-missing-current.",
    )
    parser.add_argument(
        "--amp-offset",
        type=float,
        default=1.0,
        help="Default amp_offset to add with --add-missing-current.",
    )
    args = parser.parse_args()
    if args.convert_decoder_output_to_linear and args.direction != "legacy-to-current":
        parser.error("--convert-decoder-output-to-linear requires --direction legacy-to-current")

    checkpoint = torch.load(args.src, map_location="cpu")
    state_dict, state_key = get_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected checkpoint to contain a state_dict mapping, got {type(state_dict).__name__}."
        )

    current_to_legacy = {
        "amp_scale": "log_scale_amp",
        "ph_scale": "log_scale_ph",
        "amp_offset": "offset_amp",
    }

    if args.direction == "legacy-to-current":
        key_map = {value: key for key, value in current_to_legacy.items()}
    else:
        key_map = current_to_legacy

    renamed = rename_keys(state_dict, key_map)
    converted = []
    if args.convert_decoder_output_to_linear:
        converted = convert_decoder_output_to_linear(state_dict)
    added = []
    if args.add_missing_current:
        if args.direction != "legacy-to-current":
            raise ValueError("--add-missing-current can only be used with --direction legacy-to-current")
        added = add_missing_current_keys(
            state_dict,
            amp_scale=args.amp_scale,
            ph_scale=args.ph_scale,
            amp_offset=args.amp_offset,
        )

    if state_key is None:
        torch.save(state_dict, args.dst)
    else:
        checkpoint[state_key] = state_dict
        torch.save(checkpoint, args.dst)

    if renamed:
        for old_key, new_key in renamed:
            print(f"{old_key} -> {new_key}")
    else:
        print("No keys renamed.")

    if added:
        for key in added:
            print(f"Added missing key: {key}")

    if args.convert_decoder_output_to_linear:
        if converted:
            for key, old_shape, new_shape in converted:
                print(f"Converted {key}: {old_shape} -> {new_shape}")
        else:
            print("No decoder output weights needed conversion.")

    print(f"Saved: {args.dst}")


if __name__ == "__main__":
    main()
