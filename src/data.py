import numpy as np
import torch
from torch.utils.data import Dataset


FREQUENCIES     = [1.0, 3.0, 5.0, 7.0]
NUM_FREQUENCIES = len(FREQUENCIES)
SAMPLING_RATE   = 1000   # Hz
SIGNAL_DURATION = 10     # seconds
SIGNAL_LENGTH   = SAMPLING_RATE * SIGNAL_DURATION  # 10,000 samples
CONTEXT_WINDOW  = 10     # samples per training window


class SignalDataset(Dataset):
    """Synthetic signal denoising dataset (lecture specification).

    Builds 4 clean sine signals and 4 amplitude/phase-perturbed noisy versions,
    sums them into sigma_clean and sigma_noisy, then draws context windows.

    Each input:  [one_hot(4) | sigma_noisy_window(10)] → shape (14,)
    Each target: selected_clean_signal_window(10)      → shape (10,)
    """

    def __init__(
        self,
        num_samples=10000,
        sampling_rate=SAMPLING_RATE,
        signal_duration=SIGNAL_DURATION,
        context_window=CONTEXT_WINDOW,
        amplitude=1.0,
        noise_amp_std=0.1,
        noise_phase_std=0.2,
        seed=42,
    ):
        self.num_samples     = num_samples
        self.sampling_rate   = sampling_rate
        self.signal_duration = signal_duration
        self.context_window  = context_window
        self.amplitude       = amplitude
        self.noise_amp_std   = noise_amp_std
        self.noise_phase_std = noise_phase_std
        self.rng             = np.random.default_rng(seed)

        self.inputs, self.targets = self._generate()

    def _generate(self):
        signal_length = self.sampling_rate * self.signal_duration
        t = np.arange(signal_length) / self.sampling_rate

        # 4 clean signals:  A * sin(2π * f * t)
        clean_signals = [
            self.amplitude * np.sin(2 * np.pi * f * t)
            for f in FREQUENCIES
        ]

        # 4 noisy signals:  (A + α) * sin(2π * f * t + β)
        #   α ~ N(0, noise_amp_std),  β ~ N(0, noise_phase_std)  — scalar per signal
        noisy_signals = []
        for f in FREQUENCIES:
            alpha = self.rng.normal(0.0, self.noise_amp_std)
            beta  = self.rng.normal(0.0, self.noise_phase_std)
            noisy_signals.append(
                (self.amplitude + alpha) * np.sin(2 * np.pi * f * t + beta)
            )

        # sigma_clean and sigma_noisy: element-wise sums of all 4 signals
        sigma_noisy = np.sum(noisy_signals, axis=0)   # shape (signal_length,)

        max_start = signal_length - self.context_window
        inputs, targets = [], []

        for _ in range(self.num_samples):
            freq_idx = self.rng.integers(0, NUM_FREQUENCIES)
            start    = self.rng.integers(0, max_start + 1)

            one_hot = np.zeros(NUM_FREQUENCIES, dtype=np.float32)
            one_hot[freq_idx] = 1.0

            window_noisy = sigma_noisy[start : start + self.context_window].astype(np.float32)
            window_clean = clean_signals[freq_idx][start : start + self.context_window].astype(np.float32)

            inputs.append(np.concatenate([one_hot, window_noisy]))
            targets.append(window_clean)

        return (
            torch.tensor(np.array(inputs),  dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
