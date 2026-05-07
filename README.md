# AI Orchestration HW1 — Signal Reconstruction with MLP, RNN, and LSTM

**GitHub:** https://github.com/SalehHammam25/ai-orchestration-hw1

---

## 1. Project Overview

This project is Homework 1 for the AI Agents Orchestration course.

The goal is to build a small deep learning experiment for time-series signal reconstruction. A synthetic dataset of noisy sine signals is generated, and three neural network architectures — MLP, RNN, and LSTM — are trained to reconstruct the clean signal from noisy input samples. The models are then compared by their test MSE.

---

## 2. AI-Assisted Development Workflow

This project was developed using an organized AI-agent-assisted workflow. The purpose was to practice structured development, not just to produce working code.

The workflow followed these steps in order:

1. Write a Product Requirements Document (`prd.md`) before writing any code.
2. Create an implementation plan (`plan.md`) describing the project structure and approach.
3. Create a task list (`todo.md`) to track progress step by step.
4. Implement each component incrementally: data → models → training → evaluation → main script.
5. Write unit tests for each component.
6. Run the code and fix any issues.
7. Document the work in this README.
8. Push to GitHub with meaningful commits.

Planning documents were committed before any code was written. Each phase was completed and tested before moving to the next.

---

## 3. Theoretical Background

### 3.1 Sequence Prediction

Sequence prediction is the task of using a model to predict future or corrected values based on a sequence of past observations. In this project, the task is not future prediction but signal denoising: given 10 noisy samples from a sine wave, the model must reconstruct the 10 corresponding clean samples.

### 3.2 RNN and Memory

A Recurrent Neural Network (RNN) processes sequences one step at a time. At each timestep, it takes the current input and a hidden state from the previous step, producing a new hidden state. This gives the network a form of short-term memory — it can carry information forward through the sequence. The hidden state is updated at every step:

```
h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
```

A known limitation of vanilla RNNs is the vanishing gradient problem: gradients shrink as they are backpropagated through many timesteps, making it hard to learn long-range dependencies.

### 3.3 LSTM and Gated Memory

Long Short-Term Memory (LSTM) networks address the vanishing gradient problem using a cell state and three gates: input, forget, and output. The forget gate decides what information to discard from the cell state, the input gate decides what new information to add, and the output gate controls what part of the cell state is exposed as the hidden state.

This gating mechanism allows LSTMs to retain or discard information over longer sequences more reliably than vanilla RNNs.

### 3.4 Sine Signal Basics

A sine signal is a periodic waveform defined by:

```
y(t) = A * sin(2π * f * t + φ)
```

- **Amplitude (A):** the peak value of the signal.
- **Frequency (f):** how many full cycles occur per second (Hz). Higher frequency means a faster oscillation.
- **Phase (φ):** a horizontal shift of the waveform. A phase of π/2 turns a sine into a cosine.
- **Sampling rate:** how many discrete samples are taken per second. This project uses 32 samples per second, so a 10-sample window covers 0.3125 seconds. A rate of 32 Hz is sufficient because the highest frequency in the dataset is 8 Hz; the Nyquist theorem requires the sampling rate to be at least twice the highest frequency (2 × 8 = 16 Hz), so 32 Hz provides a comfortable margin with no aliasing.

---

## 4. Dataset

The dataset is fully synthetic and generated in code (`src/data.py`).

### 4.1 Frequencies

Four frequencies were chosen:

| Index | Frequency (Hz) |
|-------|----------------|
| 0     | 1.0            |
| 1     | 2.0            |
| 2     | 4.0            |
| 3     | 8.0            |

These are powers of two, covering a range from slow to fast oscillations. Using a doubling pattern ensures the frequencies are well-separated and easy to distinguish.

### 4.2 Sample Structure

Each sample in the dataset is constructed as follows:

1. A frequency is chosen randomly from the four options.
2. A random phase φ ∈ [0, 2π) is drawn. This randomizes the starting position within the cycle and means the dataset naturally includes cosine-like signals as well (since cos(x) = sin(x + π/2)).
3. A noise level σ is drawn uniformly from [0.05, 0.25].
4. A clean signal window of 10 samples is generated using the sine formula.
5. Gaussian noise with standard deviation σ is added to produce the noisy signal.

### 4.3 Input and Target Format

Each sample is represented as a flat input tensor of size **15** and a target tensor of size **10**:

```
Input  (15): [ one_hot(4) | sigma(1) | noisy_signal(10) ]
Target (10): [ clean_signal(10) ]
```

- **One-hot vector (4):** encodes which frequency the signal belongs to. Exactly one value is 1.0 and the rest are 0.0.
- **Sigma (1):** the noise level that was applied, so the model knows how much noise to expect.
- **Noisy signal (10):** the 10 observed noisy samples.
- **Target (10):** the 10 corresponding clean samples the model must reconstruct.

The dataset is generated with 2000 samples (seed=42 for reproducibility).

---

## 5. Model Architectures

All three models accept input of shape `(batch, 15)` and produce output of shape `(batch, 10)`.

### 5.1 MLP

A fully connected feedforward network with two hidden layers:

```
Linear(15 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 10)
```

The MLP treats the entire input as a flat feature vector. It has no notion of sequence order but can freely mix all 15 input features.

### 5.2 RNN

The input is split into:
- **Condition** (`x[:5]`): one-hot + sigma, repeated at every timestep.
- **Sequence** (`x[5:]`): the 10 noisy samples, treated as a sequence of length 10.

Each timestep receives a 6-dimensional input `[noisy_sample, one_hot(4), sigma]`. The condition is repeated at every step so the network has access to frequency and noise information at each denoising decision.

```
RNN(input=6, hidden=32, batch_first=True) → Linear(32 → 1) per step → output (batch, 10)
```

### 5.3 LSTM

Identical structure to the RNN model, but uses an LSTM cell instead of a vanilla RNN cell:

```
LSTM(input=6, hidden=32, batch_first=True) → Linear(32 → 1) per step → output (batch, 10)
```

---

## 6. Training Setup

| Setting         | Value                    |
|-----------------|--------------------------|
| Loss function   | MSELoss                  |
| Optimizer       | Adam                     |
| Learning rate   | 0.001                    |
| Batch size      | 32                       |
| Epochs          | 20                       |
| Dataset size    | 2000 samples             |
| Train/test split| 80% train / 20% test     |
| Random seed     | 42                       |

Training and evaluation are implemented in `src/train.py`. The training loop records the average MSE loss per epoch. Evaluation computes the average MSE on the held-out test set using `torch.no_grad()`.

---

## 7. Results

| Model | Test MSE  |
|-------|-----------|
| MLP   | 0.003767  |
| LSTM  | 0.009794  |
| RNN   | 0.011066  |

Prediction plots comparing noisy input, clean target, and model prediction are saved to the `results/` folder when running `main.py`.

---

## 8. Discussion

The MLP achieved the lowest test MSE in this experiment, outperforming both the RNN and LSTM. This result is somewhat counterintuitive — sequence models are generally expected to have an advantage on time-series tasks — but there are a few reasons why MLP performed best here:

- **The task is local, not temporal.** Each input already contains all 10 noisy samples together, so there is no need to maintain memory across timesteps. The MLP can directly learn a mapping from all 15 features to the 10 clean outputs.
- **The sequences are short.** With only 10 timesteps, the vanishing gradient problem is less of a concern, and the advantage of LSTM gating is reduced.
- **The dataset is relatively small.** Recurrent models generally benefit more from larger datasets. With 1600 training samples and 20 epochs, the RNN and LSTM may not have fully converged.

The LSTM performed better than the RNN, which is consistent with its more expressive gating mechanism helping it learn faster.

These results should not be generalized too broadly. With more training data, more epochs, larger hidden sizes, or longer sequences, the RNN and LSTM might close the gap or outperform the MLP.

---

## 9. How to Run

### Install dependencies

```bash
py -m pip install -r requirements.txt
```

### Run the full experiment

```bash
py main.py
```

This will train all three models, print per-epoch losses and final test MSEs, print a comparison table, and save prediction plots to `results/`.

---

## 10. How to Run Tests

```bash
py -m pytest
```

Tests are located in `tests/test_data.py` and `tests/test_models.py`. They verify dataset output shapes, one-hot encoding validity, sigma range, noise correctness, and model forward pass shapes.

---

## 11. Project Structure

```
ai-orchestration-hw1/
│
├── README.md
├── prd.md
├── plan.md
├── todo.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── data.py        # Dataset generation
│   ├── models.py      # MLP, RNN, LSTM definitions
│   ├── train.py       # Training and evaluation functions
│   └── evaluate.py    # Results comparison and prediction plots
│
├── tests/
│   ├── test_data.py
│   └── test_models.py
│
└── results/           # Prediction plots saved here after running main.py
```
