import torch
from torch import nn


def _as_float(value, name):
    """Convert numeric config values to float with a clear error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


class WeightedLoss(nn.Module):
    def __init__(self, loss_type='mse', threshold=0.0, alpha=1.0):
        """
        Weighted loss that emphasizes low-intensity pixels.

        Args:
            loss_type: Type of loss - 'mse' or 'mae' (default: 'mse')
            threshold: Pixels below this value in BOTH input and target are ignored (default: 0.0)
            alpha: Weight exponent controlling emphasis on low intensities (default: 1.0)
                   alpha=1 → inverse weighting (1/intensity)
        """
        super().__init__()
        if loss_type not in ['mse', 'mae']:
            raise ValueError(f"loss_type must be 'mse' or 'mae', got '{loss_type}'")
        self.loss_type = loss_type
        self.threshold = _as_float(threshold, 'threshold')
        self.alpha = _as_float(alpha, 'alpha')

    def forward(self, input, target):
        """
        Args:
            input: Model output (predicted diffraction - no noise)
            target: Noisy diffraction pattern (with Poisson noise)
        """
        # Create mask: include pixels where EITHER input OR target is above threshold
        # This handles cases where Poisson noise zeros out low-intensity pixels
        #mask = ((target > self.threshold) | (input > self.threshold)).float()

        # Mask that includes pixels where ONLY input is above threshold
        mask = (input > self.threshold).float()

        if self.alpha > 0:
            weight = mask / (input + 1e-6) ** self.alpha
        else:
            weight = mask  # Uniform weighting

        # Normalize weights to keep loss magnitude reasonable
        num_valid = mask.sum()
        if num_valid > 0:
            weight = weight * (num_valid / (weight.sum() + 1e-10))

        # Compute weighted loss
        if self.loss_type == 'mse':
            error = (input - target) ** 2
        else:  # mae
            error = torch.abs(input - target)

        weighted_error = error * weight

        return weighted_error.mean()


class TotalFluxLoss(nn.Module):
    """Match the total measured and predicted diffraction intensity per sample.

    Inputs are diffraction amplitudes, so total flux is the spatial sum of the
    squared values. Comparing log flux makes the loss respond to relative flux
    errors without letting the brightest samples dominate the batch.
    """

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(f"input and target must have matching shapes, got {input.shape} and {target.shape}")
        if input.ndim < 3:
            raise ValueError(f"input must have at least 3 dimensions, got shape {input.shape}")

        eps = 1e-6
        input_flux = input.square().sum(dim=(-2, -1))
        target_flux = target.square().sum(dim=(-2, -1))
        log_input_flux = torch.log(torch.clamp(input_flux, min=eps))
        log_target_flux = torch.log(torch.clamp(target_flux, min=eps))
        return nn.functional.smooth_l1_loss(log_input_flux, log_target_flux)


class ProbeAwareLoss(nn.Module):
    def __init__(
        self,
        loss_type='mse',
        threshold=0.0,
        alpha=1.0,
        envelope_floor=0.05,
        eps=1e-6
    ):
        """
        Probe-aware diffraction loss that emphasizes sample-induced modulations
        relative to the probe far-field envelope.

        Args:
            loss_type: Type of loss - 'mse' or 'mae' (default: 'mse')
            threshold: Pixels below this value in BOTH input and target are ignored (default: 0.0)
            alpha: Power applied to the probe amplitude envelope in the denominator.
                   alpha=1.0 gives relative amplitude error; alpha=0.5 is softer.
            envelope_floor: Per-sample floor as a fraction of the mean probe envelope.
                            This prevents near-zero probe regions from dominating.
            eps: Small value for numerical stability.
        """
        super().__init__()
        if loss_type not in ['mse', 'mae']:
            raise ValueError(f"loss_type must be 'mse' or 'mae', got '{loss_type}'")
        alpha = _as_float(alpha, 'alpha')
        envelope_floor = _as_float(envelope_floor, 'envelope_floor')
        eps = _as_float(eps, 'eps')
        threshold = _as_float(threshold, 'threshold')

        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if envelope_floor < 0:
            raise ValueError(f"envelope_floor must be non-negative, got {envelope_floor}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.loss_type = loss_type
        self.threshold = threshold
        self.alpha = alpha
        self.envelope_floor = envelope_floor
        self.eps = eps
        self.uses_probe_envelope = True

    def _probe_envelope(self, probe, normalization=None, scale=None):
        """
        Compute the probe far-field amplitude envelope using the same convention
        as the forward model.

        Args:
            probe: Real-view complex probe with shape (B, OPR, modes, H, W, 2),
                   or complex probe with shape (B, OPR, modes, H, W)
            normalization: Optional per-sample intensity normalization
            scale: Optional per-sample intensity scale

        Returns:
            Probe amplitude envelope with shape (B, 1, H, W)
        """
        if torch.is_complex(probe):
            complex_probe = probe
        else:
            if probe.shape[-1] != 2:
                raise ValueError(
                    "probe must be complex or a real-view tensor with final dimension 2, "
                    f"got shape {probe.shape}"
                )
            complex_probe = torch.complex(probe[..., 0], probe[..., 1])

        probe_fft = torch.fft.fftshift(torch.fft.fft2(complex_probe), dim=(-2, -1))
        probe_intensity = (probe_fft.abs() ** 2).sum(2)[:, 0]

        if normalization is not None:
            normalization = normalization.view(normalization.shape[0], 1, 1)
            probe_intensity = probe_intensity.float() / torch.clamp(normalization, min=self.eps)
        if scale is not None:
            scale = scale.view(scale.shape[0], 1, 1)
            probe_intensity = probe_intensity * scale

        return torch.sqrt(torch.clamp(probe_intensity, min=self.eps)).unsqueeze(1)

    def forward(self, input, target, probe=None, normalization=None, scale=None):
        """
        Args:
            input: Model output diffraction amplitude
            target: Target diffraction amplitude
            probe: Optional probe used to generate the diffraction pattern
            normalization: Optional intensity normalization matching the forward model
            scale: Optional intensity scale matching the forward model
        """
        if input.shape != target.shape:
            raise ValueError(f"input and target must have matching shapes, got {input.shape} and {target.shape}")

        mask = ((input > self.threshold) | (target > self.threshold)).float()

        diff = input - target
        if probe is not None:
            envelope = self._probe_envelope(probe, normalization, scale).to(device=input.device, dtype=input.dtype)
            envelope_mean = envelope.mean(dim=(-2, -1), keepdim=True)
            envelope_floor = self.envelope_floor * envelope_mean
            denominator = torch.clamp(envelope, min=self.eps) + envelope_floor + self.eps
            diff = diff / denominator.pow(self.alpha)

        if self.loss_type == 'mse':
            error = diff ** 2
        else:  # mae
            error = torch.abs(diff)

        masked_error = error * mask
        valid = mask.sum()
        if valid > 0:
            return masked_error.sum() / torch.clamp(valid, min=1.0)
        return masked_error.mean()


class QDependentLoss(nn.Module):
    def __init__(
        self,
        loss_type='mse',
        alpha=1.0,
        q_beta=1.0,
        q_floor=0.05,
        threshold=0.0,
        log_error=True,
        normalize_q=True,
        eps=1e-6
    ):
        """
        Q-dependent loss that emphasizes high-Q detector pixels and hard-to-match errors.

        Args:
            loss_type: Type of loss - 'mse' or 'mae' (default: 'mse')
            alpha: Focal exponent controlling emphasis on poorly matched pixels (default: 1.0)
                   alpha=0 → no focal weighting, alpha>1 → stronger hard-pixel focus
            q_beta: Radial Q exponent controlling emphasis on high-Q pixels (default: 1.0)
                    q_beta=0 → no Q weighting, larger values emphasize detector edges/corners
            q_floor: Minimum normalized Q offset so low-Q pixels are not zero-weighted (default: 0.05)
            threshold: Pixels below this value in BOTH input and target are ignored (default: 0.0)
            log_error: Whether to compute the data error on log10 amplitudes (default: True)
            normalize_q: Whether to normalize the Q weight map to unit mean (default: True)
            eps: Small value for numerical stability (default: 1e-6)
        """
        super().__init__()
        if loss_type not in ['mse', 'mae']:
            raise ValueError(f"loss_type must be 'mse' or 'mae', got '{loss_type}'")

        alpha = _as_float(alpha, 'alpha')
        q_beta = _as_float(q_beta, 'q_beta')
        q_floor = _as_float(q_floor, 'q_floor')
        threshold = _as_float(threshold, 'threshold')
        eps = _as_float(eps, 'eps')

        if q_floor < 0:
            raise ValueError(f"q_floor must be non-negative, got {q_floor}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.loss_type = loss_type
        self.alpha = alpha
        self.q_beta = q_beta
        self.q_floor = q_floor
        self.threshold = threshold
        self.log_error = log_error
        self.normalize_q = normalize_q
        self.eps = eps
        self._q_weight = None
        self._q_weight_shape = None

    def _get_q_weight(self, input):
        """
        Create a radial Q weight map for fftshifted detector images.

        Args:
            input: Tensor with reciprocal-space image dimensions in the last two axes

        Returns:
            Q weight map broadcastable to input
        """
        h, w = input.shape[-2:]
        cache_key = (input.ndim, h, w, input.device, input.dtype)
        if self._q_weight is not None and self._q_weight_shape == cache_key:
            return self._q_weight

        y = torch.arange(h, device=input.device, dtype=input.dtype)
        x = torch.arange(w, device=input.device, dtype=input.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        q = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        q = q / torch.clamp(q.max(), min=self.eps)

        q_weight = (q + self.q_floor) ** self.q_beta
        if self.normalize_q:
            q_weight = q_weight / torch.clamp(q_weight.mean(), min=self.eps)

        self._q_weight = q_weight.view(*([1] * (input.ndim - 2)), h, w)
        self._q_weight_shape = cache_key
        return self._q_weight

    def forward(self, input, target):
        """
        Args:
            input: Model output diffraction amplitude with fftshifted low-Q center
            target: Target diffraction amplitude with matching detector coordinates

        Returns:
            Scalar Q-dependent loss value
        """
        if input.shape != target.shape:
            raise ValueError(f"input and target must have matching shapes, got {input.shape} and {target.shape}")
        if input.ndim < 3:
            raise ValueError(f"input must have at least 3 dimensions, got shape {input.shape}")

        mask = ((input > self.threshold) | (target > self.threshold)).float()

        if self.log_error:
            input_for_error = torch.log10(torch.clamp(input, min=self.eps))
            target_for_error = torch.log10(torch.clamp(target, min=self.eps))
        else:
            input_for_error = input
            target_for_error = target

        diff = input_for_error - target_for_error
        if self.loss_type == 'mse':
            error = diff ** 2
        else:  # mae
            error = torch.abs(diff)

        focal_weight = torch.abs(diff) ** self.alpha
        focal_max = torch.amax(focal_weight, dim=(-2, -1), keepdim=True)
        focal_weight = focal_weight / torch.clamp(focal_max, min=self.eps)
        focal_weight = torch.clamp(focal_weight, min=0.0, max=1.0).detach()

        q_weight = self._get_q_weight(input)
        weight = focal_weight * q_weight * mask

        # Normalize weights to keep loss magnitude reasonable
        num_valid = mask.sum()
        if num_valid > 0:
            weight_sum = weight.sum()
            if weight_sum > 0:
                weight = weight * (num_valid / (weight_sum + self.eps))

        weighted_error = error * weight

        return weighted_error.mean()


class KramersKronigLoss(nn.Module):
    """
    Kramers-Kronig consistency loss for complex object reconstruction.

    The Kramers-Kronig relations are dispersion relations that connect the real and imaginary
    parts of any complex function that is analytic in the upper half-plane. For X-ray scattering,
    these relations connect the absorption (imaginary part) and phase shift (real part) of the
    complex refractive index.

    For weak phase/amplitude objects common in X-ray ptychography:
    - Real part: n_real = 1 - δ (δ is phase shift, positive)
    - Imaginary part: n_imag = -β (β is absorption, positive)

    This loss encourages physical consistency between real and imaginary parts
    by enforcing correlation in their spatial gradients.
    """
    def __init__(self, weight=1.0, mode='gradient_correlation'):
        """
        Args:
            weight: Loss weight (default: 1.0)
            mode: Type of K-K constraint to enforce
                  'gradient_correlation': Enforce correlation between ∇real and ∇imag
                  'gradient_magnitude': Penalize difference in gradient magnitudes
                  'laplacian': Enforce correlation between ∇²real and ∇²imag
        """
        super().__init__()
        self.weight = weight
        self.mode = mode

    def forward(self, real_part, imag_part):
        """
        Compute Kramers-Kronig consistency loss.

        Args:
            real_part: Real part of complex object (B, H, W) or (B, 1, H, W)
            imag_part: Imaginary part of complex object (B, H, W) or (B, 1, H, W)

        Returns:
            Scalar loss value
        """
        # Handle both (B, H, W) and (B, 1, H, W) shapes
        if real_part.ndim == 4:
            real_part = real_part.squeeze(1)
            imag_part = imag_part.squeeze(1)

        if self.mode == 'gradient_correlation':
            return self.weight * self._gradient_correlation_loss(real_part, imag_part)
        elif self.mode == 'gradient_magnitude':
            return self.weight * self._gradient_magnitude_loss(real_part, imag_part)
        elif self.mode == 'laplacian':
            return self.weight * self._laplacian_loss(real_part, imag_part)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _gradient_correlation_loss(self, real_part, imag_part):
        """
        Enforce correlation between gradients of real and imaginary parts.

        For K-K relations, the spatial derivatives of real and imaginary parts
        should be correlated (not necessarily equal, but related).
        """
        # Compute gradients in x and y directions
        grad_real_x = real_part[:, :, 1:] - real_part[:, :, :-1]
        grad_real_y = real_part[:, 1:, :] - real_part[:, :-1, :]
        grad_imag_x = imag_part[:, :, 1:] - imag_part[:, :, :-1]
        grad_imag_y = imag_part[:, 1:, :] - imag_part[:, :-1, :]

        # K-K implies gradients should be correlated
        # Use MSE between normalized gradients as a proxy
        # Normalize to avoid scale issues
        grad_real_x_norm = grad_real_x / (torch.std(grad_real_x) + 1e-6)
        grad_real_y_norm = grad_real_y / (torch.std(grad_real_y) + 1e-6)
        grad_imag_x_norm = grad_imag_x / (torch.std(grad_imag_x) + 1e-6)
        grad_imag_y_norm = grad_imag_y / (torch.std(grad_imag_y) + 1e-6)

        # Correlation loss: penalize when gradients are uncorrelated
        # For weak objects, gradients should have similar patterns
        loss_x = torch.mean((grad_real_x_norm - grad_imag_x_norm) ** 2)
        loss_y = torch.mean((grad_real_y_norm - grad_imag_y_norm) ** 2)

        return (loss_x + loss_y) / 2

    def _gradient_magnitude_loss(self, real_part, imag_part):
        """
        Enforce similarity in gradient magnitudes.

        This is a weaker constraint than gradient correlation, enforcing that
        the magnitude of spatial variations should be similar.
        """
        # Compute gradient magnitudes
        grad_real_x = real_part[:, :, 1:] - real_part[:, :, :-1]
        grad_real_y = real_part[:, 1:, :] - real_part[:, :-1, :]
        grad_imag_x = imag_part[:, :, 1:] - imag_part[:, :, :-1]
        grad_imag_y = imag_part[:, 1:, :] - imag_part[:, :-1, :]

        # Compute magnitude of gradients
        grad_real_mag = torch.sqrt(grad_real_x[:, :-1, :] ** 2 + grad_real_y[:, :, :-1] ** 2 + 1e-8)
        grad_imag_mag = torch.sqrt(grad_imag_x[:, :-1, :] ** 2 + grad_imag_y[:, :, :-1] ** 2 + 1e-8)

        # Penalize difference in gradient magnitudes
        return torch.mean((grad_real_mag - grad_imag_mag) ** 2)

    def _laplacian_loss(self, real_part, imag_part):
        """
        Enforce correlation between Laplacians (second derivatives).

        The Laplacian captures curvature and can be more robust to constant offsets.
        """
        # Compute Laplacian (∇²) using finite differences
        # ∇²f ≈ f(x+1) + f(x-1) + f(y+1) + f(y-1) - 4f(x,y)

        # Pad to handle boundaries
        real_padded = torch.nn.functional.pad(real_part.unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
        imag_padded = torch.nn.functional.pad(imag_part.unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)

        # Compute Laplacian
        laplacian_real = (
            real_padded[:, 2:, 1:-1] +    # f(y+1)
            real_padded[:, :-2, 1:-1] +   # f(y-1)
            real_padded[:, 1:-1, 2:] +    # f(x+1)
            real_padded[:, 1:-1, :-2] -   # f(x-1)
            4 * real_padded[:, 1:-1, 1:-1]  # -4f(x,y)
        )

        laplacian_imag = (
            imag_padded[:, 2:, 1:-1] +
            imag_padded[:, :-2, 1:-1] +
            imag_padded[:, 1:-1, 2:] +
            imag_padded[:, 1:-1, :-2] -
            4 * imag_padded[:, 1:-1, 1:-1]
        )

        # Normalize Laplacians
        laplacian_real_norm = laplacian_real / (torch.std(laplacian_real) + 1e-6)
        laplacian_imag_norm = laplacian_imag / (torch.std(laplacian_imag) + 1e-6)

        # Correlation loss
        return torch.mean((laplacian_real_norm - laplacian_imag_norm) ** 2)


def kramers_kronig_loss_simple(real_part, imag_part, weight=1.0):
    """
    Simple standalone function for Kramers-Kronig consistency loss.

    This is a convenience function that can be used directly without instantiating
    the KramersKronigLoss class.

    Args:
        real_part: Real part of complex object (B, H, W) or (B, 1, H, W)
        imag_part: Imaginary part of complex object (B, H, W) or (B, 1, H, W)
        weight: Loss weight (default: 1.0)

    Returns:
        Scalar loss value
    """
    kk_loss = KramersKronigLoss(weight=weight, mode='gradient_correlation')
    return kk_loss(real_part, imag_part)


class CombinedLoss(nn.Module):
    """
    Combined loss function that computes a weighted sum of multiple loss functions.

    Args:
        loss_configs: List of dictionaries, each containing:
            - 'type': Loss function type ('l1', 'mse', 'weighted', 'probe_aware', 'q_dependent', 'smooth_l1', 'poisson_nll')
            - 'weight': Weight for this loss component (default: 1.0)
        loss_params: Optional dictionary containing individual loss configurations

    Example:
        # L1 + 5*WeightedMSE
        loss_configs = [
            {'type': 'l1', 'weight': 1.0},
            {'type': 'weighted', 'weight': 5.0}
        ]
        loss_params = {
            'weighted_loss': {'loss_type': 'mse', 'threshold': 0.0, 'alpha': 1.0}
        }
        criterion = CombinedLoss(loss_configs, loss_params)
    """
    def __init__(self, loss_configs, loss_params=None):
        super().__init__()
        if not loss_configs or len(loss_configs) == 0:
            raise ValueError("loss_configs must contain at least one loss configuration")

        self.loss_functions = nn.ModuleList()
        self.loss_weights = []
        self.loss_params = loss_params or {}
        self.uses_probe_envelope = False

        for config in loss_configs:
            loss_type = config.get('type')
            weight = _as_float(config.get('weight', 1.0), f'{loss_type}.weight')
            params = self.loss_params.get(f'{loss_type}_loss', {})

            # Create loss function based on type
            if loss_type == 'l1':
                loss_fn = nn.L1Loss()
            elif loss_type == 'mse':
                loss_fn = nn.MSELoss()
            elif loss_type == 'smooth_l1':
                loss_fn = nn.SmoothL1Loss()
            elif loss_type == 'poisson_nll':
                loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)
            elif loss_type == 'weighted':
                # WeightedLoss requires specific parameters
                loss_fn = WeightedLoss(
                    loss_type=params.get('loss_type', 'mse'),
                    threshold=params.get('threshold', 0.0),
                    alpha=params.get('alpha', 1.0)
                )
            elif loss_type == 'probe_aware':
                loss_fn = ProbeAwareLoss(
                    loss_type=params.get('loss_type', 'mse'),
                    threshold=params.get('threshold', 0.0),
                    alpha=params.get('alpha', 1.0),
                    envelope_floor=params.get('envelope_floor', 0.05),
                    eps=params.get('eps', 1e-6)
                )
            elif loss_type == 'q_dependent':
                loss_fn = QDependentLoss(
                    loss_type=params.get('loss_type', 'mse'),
                    alpha=params.get('alpha', 1.0),
                    q_beta=params.get('q_beta', 1.0),
                    q_floor=params.get('q_floor', 0.05),
                    threshold=params.get('threshold', 0.0),
                    log_error=params.get('log_error', True),
                    normalize_q=params.get('normalize_q', True),
                    eps=params.get('eps', 1e-6)
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.loss_functions.append(loss_fn)
            self.loss_weights.append(weight)
            self.uses_probe_envelope = self.uses_probe_envelope or getattr(loss_fn, 'uses_probe_envelope', False)

    def forward(self, input, target, probe=None, normalization=None, scale=None):
        """
        Compute combined loss as weighted sum of individual losses.

        Args:
            input: Model output
            target: Ground truth

        Returns:
            Scalar combined loss value
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
            if getattr(loss_fn, 'uses_probe_envelope', False):
                total_loss += weight * loss_fn(input, target, probe=probe, normalization=normalization, scale=scale)
            else:
                total_loss += weight * loss_fn(input, target)
        return total_loss
