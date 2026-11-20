"""Probe-power-weighted placement of full object patches."""

import warnings

import torch
from ptychi.image_proc import place_patches_fourier_shift
from torch.nn.functional import pad as pad_tensor


def warn_ignored_central_crop(setting):
    warnings.warn(
        f"{setting} is deprecated and ignored; stitching now uses full patches "
        "weighted by probe power.",
        UserWarning,
        stacklevel=2,
    )


def probe_power(probe):
    """Return (B, H, W) power from complex (B, OPR, modes, H, W) probes.

    Match the forward model: use the first OPR component and sum intensities
    over incoherent probe modes, including any zero-padded modes.
    """
    return probe[:, 0].abs().square().sum(dim=1)


class ProbeWeightedStitcher:
    """Accumulate several object channels using the same illumination weights."""

    def __init__(self, object_shape, num_channels, pad=32):
        self.objects = [torch.zeros(object_shape, device="cpu") for _ in range(num_channels)]
        self.power = torch.zeros(object_shape, device="cpu")
        self.pad = pad

    def _place(self, image, positions, patches):
        # PtyChi's non-adjoint placement crops `pad` pixels after shifting.
        # Supply a zero border so it removes padding, never measured pixels.
        patches = pad_tensor(patches, (self.pad,) * 4)
        return place_patches_fourier_shift(
            image, positions, patches, op="add", adjoint_mode=False, pad=self.pad,
        )

    def add(self, positions, patches, power):
        positions = positions.detach().cpu()
        power = power.detach().to(device="cpu", dtype=self.power.dtype)
        for index, patch in enumerate(patches):
            weighted = patch.detach().to(device="cpu", dtype=self.power.dtype) * power
            self.objects[index] = self._place(self.objects[index], positions, weighted)
        self.power = self._place(self.power, positions, power)

    def result(self):
        # Reject nonpositive weights and Fourier roundoff in unilluminated areas.
        # A relative tolerance preserves invariance to the overall probe scale.
        threshold = self.power.amax() * torch.finfo(self.power.dtype).eps
        illuminated = self.power > threshold
        denominator = torch.where(illuminated, self.power, 1)
        return tuple(
            torch.where(illuminated, image / denominator, 0)
            for image in self.objects
        )
