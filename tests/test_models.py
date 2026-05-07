import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models import MLPModel, RNNModel, LSTMModel

BATCH_SIZE = 8
INPUT_SIZE = 15
OUTPUT_SIZE = 10


@pytest.fixture
def batch():
    return torch.randn(BATCH_SIZE, INPUT_SIZE)


def test_mlp_forward(batch):
    model = MLPModel()
    out = model(batch)
    assert out.shape == (BATCH_SIZE, OUTPUT_SIZE)


def test_rnn_forward(batch):
    model = RNNModel()
    out = model(batch)
    assert out.shape == (BATCH_SIZE, OUTPUT_SIZE)


def test_lstm_forward(batch):
    model = LSTMModel()
    out = model(batch)
    assert out.shape == (BATCH_SIZE, OUTPUT_SIZE)
