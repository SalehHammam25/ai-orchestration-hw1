import numpy as np
import torch
from torch.utils.data import Dataset


FREQUENCIES = [1.0, 2.0, 4.0, 8.0]
NUM_FREQUENCIES = len(FREQUENCIES)


class SignalDataset(Dataset):
    """Synthetic noisy sine signal dataset for denoising.

    Each input: [one_hot(4), sigma(1), noisy_signal(10)] → shape (15,)
    Each target: clean_signal(10)                         → shape (10,)
    """

    def __init__(
        self,
        num_samples=10000,
        window_size=10,
        sampling_rate=32,
        amplitude=1.0,
        noise_min=0.05,
        noise_max=0.25,
        seed=42,
    ):
        self.num_samples = num_samples
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.amplitude = amplitude
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.rng = np.random.default_rng(seed)

        self.inputs, self.targets = self._generate()

    def _one_hot(self, index):
        vec = np.zeros(NUM_FREQUENCIES, dtype=np.float32)
        vec[index] = 1.0
        return vec

    def _clean_signal(self, frequency, phase):
        t = np.arange(self.window_size) / self.sampling_rate
        return (self.amplitude * np.sin(2 * np.pi * frequency * t + phase)).astype(np.float32)

    def _generate(self):
        inputs, targets = [], []

        for _ in range(self.num_samples):
            freq_idx = self.rng.integers(0, NUM_FREQUENCIES)
            frequency = FREQUENCIES[freq_idx]
            sigma = float(self.rng.uniform(self.noise_min, self.noise_max))
            phase = self.rng.uniform(0, 2 * np.pi)

            clean = self._clean_signal(frequency, phase)
            noise = self.rng.normal(0, sigma, self.window_size).astype(np.float32)
            noisy = clean + noise

            model_input = np.concatenate([
                self._one_hot(freq_idx),
                np.array([sigma], dtype=np.float32),
                noisy,
            ])
            inputs.append(model_input)
            targets.append(clean)

        return (
            torch.tensor(np.array(inputs), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
