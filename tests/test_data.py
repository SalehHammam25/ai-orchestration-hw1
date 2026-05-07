import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.data import SignalDataset, NUM_FREQUENCIES


@pytest.fixture(scope="module")
def dataset():
    return SignalDataset(num_samples=50, seed=0)


def test_dataset_length(dataset):
    assert len(dataset) == 50


def test_input_shape(dataset):
    x, _ = dataset[0]
    assert x.shape == (14,)


def test_target_shape(dataset):
    _, y = dataset[0]
    assert y.shape == (10,)


def test_one_hot_valid(dataset):
    for i in range(len(dataset)):
        x, _ = dataset[i]
        one_hot = x[:NUM_FREQUENCIES]
        assert one_hot.sum().item() == pytest.approx(1.0)
        assert (one_hot == 1.0).sum().item() == 1


def test_noisy_window_in_valid_range(dataset):
    # sigma_noisy is the sum of 4 unit-amplitude signals; absolute values stay well below 5
    for i in range(len(dataset)):
        x, _ = dataset[i]
        noisy_window = x[NUM_FREQUENCIES:]
        assert noisy_window.abs().max().item() <= 5.0


def test_noisy_differs_from_clean(dataset):
    for i in range(len(dataset)):
        x, y = dataset[i]
        noisy = x[NUM_FREQUENCIES:]
        assert not torch.equal(noisy, y)
