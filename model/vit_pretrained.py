import math
import os
import warnings
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn


class VisionTransformer(nn.Module):
    """Vision Transformer encoder backed by timm, with optional local checkpoint loading."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 32,
        in_channels: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_cls_token: bool = False,
        *,
        timm_model_name: str = "vit_large_patch32_224",
        timm_kwargs: Optional[Dict[str, object]] = None,
        checkpoint_path: Optional[str] = None,
        strict_load: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # Keep the attributes for compatibility with previous manual implementation.
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_cls_token = use_cls_token
        self.img_size = img_size

        timm_kwargs = dict(timm_kwargs) if timm_kwargs else {}
        checkpoint_from_kwargs = timm_kwargs.pop("checkpoint_path", None)
        strict_from_kwargs = timm_kwargs.pop("strict", None)
        strict_load_from_kwargs = timm_kwargs.pop("strict_load", None)
        if strict_from_kwargs is None:
            strict_from_kwargs = strict_load_from_kwargs

        if checkpoint_path is None:
            checkpoint_path = checkpoint_from_kwargs
        if strict_load is None:
            if isinstance(strict_from_kwargs, str):
                strict_load = strict_from_kwargs.strip().lower() in {"1", "true", "yes", "on"}
            elif strict_from_kwargs is not None:
                strict_load = bool(strict_from_kwargs)
            else:
                strict_load = False

        # --- Build kwargs for timm.create_model ---
        create_kwargs: Dict[str, object] = {
            "pretrained": True,                   # default: download/cached weights
            "in_chans": in_channels,
            "img_size": (img_size, img_size),
            "num_classes": 0,                     # headless encoder
            "drop_rate": dropout,
            "attn_drop_rate": attn_dropout,
        }
        create_kwargs.update(timm_kwargs)

        # If a local checkpoint is provided, build the model first and load weights manually.
        if checkpoint_path is not None:
            # Avoid letting timm attempt to reconcile incompatible shapes automatically.
            create_kwargs["pretrained"] = bool(create_kwargs.get("pretrained", False))
            if create_kwargs["pretrained"]:
                warnings.warn(
                    "Ignoring pretrained=True because a checkpoint_path was provided; loading weights manually.",
                    RuntimeWarning,
                )
            create_kwargs["pretrained"] = False
            create_kwargs.pop("checkpoint_path", None)

        backbone = timm.create_model(timm_model_name, **create_kwargs)
        if checkpoint_path is not None:
            self._load_checkpoint(backbone, checkpoint_path, strict_load=strict_load)

        # Update attributes with actual backbone configuration.
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "embed_dim")
        blocks = getattr(backbone, "blocks", [])
        if blocks:
            self.depth = len(blocks)
            first_block = blocks[0]
            attn = getattr(first_block, "attn", None)
            if attn is not None and hasattr(attn, "num_heads"):
                self.num_heads = attn.num_heads
            mlp = getattr(first_block, "mlp", None)
            if mlp is not None and hasattr(mlp, "fc1") and hasattr(mlp.fc1, "weight"):
                self.mlp_ratio = mlp.fc1.weight.shape[0] / mlp.fc1.weight.shape[1]

        patch_embed = getattr(backbone, "patch_embed", None)
        if patch_embed is None:
            raise ValueError(f"Model '{timm_model_name}' does not expose a patch_embed module.")

        patch_hw = getattr(patch_embed, "patch_size", patch_size)
        if isinstance(patch_hw, tuple):
            if len(patch_hw) == 1:
                patch_hw = patch_hw[0]
            elif patch_hw[0] != patch_hw[1]:
                raise ValueError("Only square patches are supported.")
            else:
                patch_hw = patch_hw[0]
        self.patch_size = patch_hw

        grid_size = getattr(patch_embed, "grid_size", None)
        if grid_size is None:
            num_patches = getattr(patch_embed, "num_patches", None)
            if num_patches is None:
                raise ValueError("Unable to infer patch grid from the backbone.")
            side_len = int(math.sqrt(num_patches))
            if side_len * side_len != num_patches:
                raise ValueError("Patch tokens do not form a square grid.")
            grid_size = (side_len, side_len)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        # Keep these for API parity; timm handles actual dropout inside.
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        # Internal cache for the last class token (useful if requested downstream).
        self._last_cls_token: Optional[torch.Tensor] = None

    def _load_checkpoint(self, backbone: nn.Module, checkpoint_path: str, strict_load: bool) -> None:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = self._read_state_dict(checkpoint_path)
        target_state = backbone.state_dict()

        cleaned_state: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            stripped_key = self._strip_prefix(key)
            if stripped_key in target_state:
                cleaned_state[stripped_key] = tensor

        if not cleaned_state:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' does not share parameters with the backbone."
            )

        target_patch_key = "patch_embed.proj.weight"
        if target_patch_key in cleaned_state:
            cleaned_state[target_patch_key] = self._adapt_patch_embed(
                cleaned_state[target_patch_key],
                target_state[target_patch_key].shape,
            )

        target_pos_key = "pos_embed"
        if target_pos_key in cleaned_state:
            cleaned_state[target_pos_key] = self._resize_pos_embed(
                cleaned_state[target_pos_key],
                target_state[target_pos_key],
            )

        device = next(backbone.parameters()).device
        for key, value in list(cleaned_state.items()):
            cleaned_state[key] = value.to(device=device, dtype=target_state[key].dtype)

        load_result = backbone.load_state_dict(cleaned_state, strict=False)

        missing_keys, unexpected_keys = load_result
        if strict_load and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Strict checkpoint loading failed. Missing keys: {missing_keys}. "
                f"Unexpected keys: {unexpected_keys}."
            )
        if not strict_load:
            if missing_keys:
                warnings.warn(
                    f"Ignored missing keys when loading checkpoint: {missing_keys}",
                    RuntimeWarning,
                )
            if unexpected_keys:
                warnings.warn(
                    f"Ignored unexpected keys when loading checkpoint: {unexpected_keys}",
                    RuntimeWarning,
                )

    @staticmethod
    def _read_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency missing at runtime
                raise ImportError(
                    "safetensors is required to load .safetensors checkpoints."
                ) from exc
            state_dict = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = VisionTransformer._extract_state_dict(checkpoint)
        return state_dict

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                nested = checkpoint.get(key)
                if isinstance(nested, dict):
                    return nested
            return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        raise ValueError("Unable to extract a state_dict from the provided checkpoint.")

    @staticmethod
    def _strip_prefix(key: str) -> str:
        prefixes = (
            "module.",
            "backbone.",
            "encoder.",
            "encoder.backbone.",
            "model.",
            "model.encoder.",
            "net.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    @staticmethod
    def _resolve_pos_grid(num_tokens: int) -> Tuple[int, bool]:
        if num_tokens <= 0:
            raise ValueError("Number of tokens must be positive.")
        sqrt_tokens = int(round(num_tokens ** 0.5))
        if sqrt_tokens * sqrt_tokens == num_tokens:
            return sqrt_tokens, False
        sqrt_minus_cls = int(round((num_tokens - 1) ** 0.5)) if num_tokens > 1 else 0
        if sqrt_minus_cls * sqrt_minus_cls == num_tokens - 1:
            return sqrt_minus_cls, True
        raise ValueError(f"Cannot infer positional grid for token count {num_tokens}.")

    def _resize_pos_embed(
        self,
        checkpoint_pos_embed: torch.Tensor,
        target_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        if checkpoint_pos_embed.shape == target_pos_embed.shape:
            return checkpoint_pos_embed

        target_tokens = target_pos_embed.shape[1]
        target_size, target_has_cls = self._resolve_pos_grid(target_tokens)

        source_tokens = checkpoint_pos_embed.shape[1]
        source_size, source_has_cls = self._resolve_pos_grid(source_tokens)

        cls_token = None
        pos_tokens = checkpoint_pos_embed
        if source_has_cls:
            cls_token = pos_tokens[:, :1]
            pos_tokens = pos_tokens[:, 1:]

        pos_tokens = pos_tokens.reshape(1, source_size, source_size, -1)
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
            1, target_size * target_size, -1
        )

        if target_has_cls:
            if cls_token is None:
                cls_token = target_pos_embed[:, :1]
            pos_tokens = torch.cat([cls_token, pos_tokens], dim=1)

        return pos_tokens

    @staticmethod
    def _adapt_patch_embed(
        weight: torch.Tensor, target_shape: torch.Size
    ) -> torch.Tensor:
        if weight.shape == target_shape:
            return weight

        out_channels, in_channels_target, *_ = target_shape
        _, in_channels_source, *_ = weight.shape

        if in_channels_source == in_channels_target:
            return weight

        # Average across source channels and replicate to the requested count.
        reduced = weight.mean(dim=1, keepdim=True)
        repeats = in_channels_target
        adapted = reduced.repeat(1, repeats, 1, 1)
        return adapted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract token embeddings from the pretrained ViT backbone.
        feats = self.backbone.forward_features(x)
        if feats.ndim != 3:
            raise ValueError("Backbone forward_features is expected to return (B, N, C) token embeddings.")

        cls_token: Optional[torch.Tensor] = None
        if feats.shape[1] == self.n_patches + 1:
            cls_token, tokens = feats[:, :1], feats[:, 1:]
        elif feats.shape[1] == self.n_patches:
            tokens = feats
        else:
            raise ValueError(
                f"Unexpected token count {feats.shape[1]} for grid {self.grid_size} "
                f"(expected {self.n_patches} or {self.n_patches + 1})."
            )

        self._last_cls_token = cls_token

        batch_size = x.shape[0]
        x = tokens.transpose(1, 2)  # (B, C, N)
        x = x.reshape(batch_size, self.embed_dim, self.grid_size[0], self.grid_size[1]).contiguous()
        return x

    @property
    def cls_token(self) -> Optional[torch.Tensor]:
        """Return the last class token emitted by forward, if available."""
        if self._last_cls_token is None:
            return None
        return self._last_cls_token.squeeze(1)
