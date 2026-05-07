import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.data import SignalDataset, NUM_FREQUENCIES

NOISE_MIN = 0.05
NOISE_MAX = 0.25


@pytest.fixture(scope="module")
def dataset():
    return SignalDataset(num_samples=50, noise_min=NOISE_MIN, noise_max=NOISE_MAX, seed=0)


def test_dataset_length(dataset):
    assert len(dataset) == 50


def test_input_shape(dataset):
    x, _ = dataset[0]
    assert x.shape == (15,)


def test_target_shape(dataset):
    _, y = dataset[0]
    assert y.shape == (10,)


def test_one_hot_valid(dataset):
    for i in range(len(dataset)):
        x, _ = dataset[i]
        one_hot = x[:NUM_FREQUENCIES]
        assert one_hot.sum().item() == pytest.approx(1.0)
        assert (one_hot == 1.0).sum().item() == 1


def test_sigma_in_range(dataset):
    for i in range(len(dataset)):
        x, _ = dataset[i]
        sigma = x[NUM_FREQUENCIES].item()
        assert NOISE_MIN <= sigma <= NOISE_MAX


def test_noisy_differs_from_clean(dataset):
    for i in range(len(dataset)):
        x, y = dataset[i]
        noisy = x[NUM_FREQUENCIES + 1:]
        assert not torch.equal(noisy, y)
