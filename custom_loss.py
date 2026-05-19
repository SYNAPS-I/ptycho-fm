import torch
import torch.nn as nn

class WeightedLoss(nn.Module):
    def __init__(self, loss_type='mse', threshold=0.0, alpha=1.0):
        """
        Weighted loss that emphasizes low-intensity pixels.

        Args:
            loss_type: Type of loss - 'mse' or 'mae' (default: 'mse')
            threshold: Target pixels at or below this value are ignored (default: 0.0)
            alpha: Weight exponent controlling emphasis on low intensities (default: 1.0)
                   alpha=1 → inverse weighting (1/intensity)
        """
        super(WeightedLoss, self).__init__()
        if loss_type not in ['mse', 'mae']:
            raise ValueError(f"loss_type must be 'mse' or 'mae', got '{loss_type}'")
        self.loss_type = loss_type
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, input, target):
        """
        Args:
            input: Model output (predicted diffraction - no noise)
            target: Noisy diffraction pattern (with Poisson noise)
        """
        # [MODIFIED] Fix #2: mask and weights MUST come from `target` (the measurement),
        # not from `input` (the prediction). Using the prediction here creates a
        # positive-feedback loop: the model can minimize the loss by predicting ~0
        # wherever it struggles, which masks those pixels out and removes the gradient
        # signal. On HXN data with many true-zero detector pixels this was catastrophic.
        mask = (target > self.threshold).float()

        # Inverse-intensity weighting (Poisson-style): upweight dim *measured* pixels.
        # Again use `target` so weighting is determined by the data, not the prediction.
        if self.alpha > 0:
            weight = mask / (target + 1e-6) ** self.alpha
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
