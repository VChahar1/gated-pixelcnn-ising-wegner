"""CPU checks for masking and known Ising configurations."""

from itertools import product

import torch

from gate_pixel import AutoregressiveColorPixelRCNN, MaskedConv3d
from py_func import energy_ising3d


def test_exclusive_mask_excludes_current_and_future_sites():
    layer = MaskedConv3d(1, 1, kernel_size=3, padding=1, exclusive=True)
    assert layer.mask[1, 1, 1] == 0
    assert layer.mask[2].sum() == 0
    assert layer.mask[1, 2].sum() == 0
    assert layer.mask[1, 1, 2] == 0


def test_network_first_site_ignores_current_and_future_inputs():
    torch.manual_seed(0)
    model = AutoregressiveColorPixelRCNN(1, 1, 2).eval()
    with torch.no_grad():
        first = model(torch.zeros(1, 1, 3, 3, 3))
        second = model(torch.ones(1, 1, 3, 3, 3))
    assert torch.isfinite(first).all()
    assert ((first >= 0) & (first <= 1)).all()
    torch.testing.assert_close(first[:, :, 0, 0, 0], second[:, :, 0, 0, 0])


def test_periodic_ising_energy_matches_known_spin_configurations():
    length, beta = 3, 0.22
    spins = torch.ones(2, 1, length, length, length)
    spins[1, 0, 0, 0, 0] = -1
    observed = energy_ising3d(
        2, length, spins, list(product(range(length), repeat=3)), beta
    )
    expected = torch.tensor([-3 * length**3 * beta, (-3 * length**3 + 12) * beta])
    torch.testing.assert_close(observed, expected)
