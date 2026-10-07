import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ptycho_fm.model.decoders import ResNetDecoder256


@pytest.mark.parametrize("activation", [None, "sigmoid", "tanh", "custom", "atan2"])
def test_linear_output_matches_pointwise_convolution(activation):
    torch.manual_seed(42)
    decoder = ResNetDecoder256(
        latent_dim=4, base_channels=2, out_channels=3,
        num_stages=1, blocks_per_stage=1, use_batchnorm=False,
        output_activation=activation,
    ).double()
    assert isinstance(decoder.output, nn.Linear)
    assert decoder.output.out_features == (2 if activation == "atan2" else 3)
    inputs = torch.randn(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    features = decoder.stages[0](inputs)
    expected = decoder.output_activation(F.conv2d(
        features, decoder.output.weight[..., None, None], decoder.output.bias,
    ))
    actual = decoder(inputs)
    assert actual.shape == (2, 1 if activation == "atan2" else 3, 6, 10)
    torch.testing.assert_close(actual, expected)

    targets = (inputs, decoder.output.weight, decoder.output.bias)
    expected_grads = torch.autograd.grad(expected.square().sum(), targets)
    actual_grads = torch.autograd.grad(actual.square().sum(), targets)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
