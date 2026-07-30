import torch
import torch.nn as nn

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
        # Create mask: include pixels where EITHER input OR target is above threshold
        # This handles cases where Poisson noise zeros out low-intensity pixels
        #mask = ((target > self.threshold) | (input > self.threshold)).float()

        # Mask that includes pixels where ONLY input is above threshold
        mask = (input > self.threshold).float()

        # Compute weights based on input intensity where available
        # For pixels where target is zero but prediction is non-zero,
        # use the prediction intensity for weighting
        #intensity_for_weighting = torch.where(target > self.threshold, target, input)

        if self.alpha > 0:
            #weight = mask / (intensity_for_weighting + 1e-6) ** self.alpha
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
