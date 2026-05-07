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
- **Sampling rate:** how many discrete samples are taken per second. This project uses 1000 Hz, so each full signal is 10 seconds × 1000 Hz = 10,000 samples. A 10-sample context window covers exactly 10 ms.

---

## 4. Dataset

The dataset is fully synthetic and generated in code (`src/data.py`).

### 4.1 Frequencies

Four odd frequencies were chosen per the lecture specification:

| Index | Frequency (Hz) |
|-------|----------------|
| 0     | 1.0            |
| 1     | 3.0            |
| 2     | 5.0            |
| 3     | 7.0            |

### 4.2 Signal Construction

The dataset is built from 4 fixed full-length signals, each 10 seconds at 1000 Hz (10,000 samples per signal).

**Clean signals** — pure sine at each frequency with amplitude A = 1.0:

```
clean_i(t) = A * sin(2π * f_i * t)
```

**Noisy signals** — each clean signal is perturbed by scalar amplitude noise α and phase noise β:

```
noisy_i(t) = (A + α_i) * sin(2π * f_i * t + β_i)
```

where α_i ~ N(0, 0.1) and β_i ~ N(0, 0.2) are drawn once per signal (constant across all 10,000 samples).

**Summed signals:**

```
sigma_clean(t) = clean_0(t) + clean_1(t) + clean_2(t) + clean_3(t)
sigma_noisy(t) = noisy_0(t) + noisy_1(t) + noisy_2(t) + noisy_3(t)
```

### 4.3 Dataset Sampling

Each training sample is drawn as follows:

1. Randomly select one of the 4 frequencies (index k).
2. Encode it as a one-hot vector **C** of size 4.
3. Randomly choose a start index (0 to 9,990) in `sigma_noisy`.
4. Extract a 10-sample context window from `sigma_noisy` at that position.
5. Extract the same 10-sample window from `clean_k` as the target **Sc**.

### 4.4 Input and Target Format

Each sample is represented as a flat input tensor of size **14** and a target tensor of size **10**:

```
Input  (14): [ C (one_hot, 4) | sigma_noisy_window (10) ]
Target (10): [ Sc = clean_k_window (10) ]
```

- **C (4):** one-hot vector identifying which of the 4 frequencies the target belongs to.
- **sigma_noisy_window (10):** 10 consecutive samples from the sum of all 4 noisy signals.
- **Sc (10):** the 10 corresponding samples from the selected clean signal — the denoising target.

The dataset draws 10,000 such windows (seed=42 for reproducibility).

---

## 5. Model Architectures

All three models accept input of shape `(batch, 14)` and produce output of shape `(batch, 10)`.

### 5.1 MLP

A fully connected feedforward network with two hidden layers:

```
Linear(14 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 10)
```

The MLP treats the entire input as a flat feature vector. It has no notion of sequence order but can freely mix all 14 input features.

### 5.2 RNN

The input is split into:
- **Condition** (`x[:4]`): one-hot vector C, repeated at every timestep.
- **Sequence** (`x[4:]`): the 10 sigma_noisy samples, treated as a sequence of length 10.

Each timestep receives a 5-dimensional input `[sigma_noisy_sample, one_hot(4)]`. The condition is repeated at every step so the network always knows which clean signal to target.

```
RNN(input=5, hidden=32, batch_first=True) → Linear(32 → 1) per step → output (batch, 10)
```

### 5.3 LSTM

Identical structure to the RNN model, but uses an LSTM cell instead of a vanilla RNN cell:

```
LSTM(input=5, hidden=32, batch_first=True) → Linear(32 → 1) per step → output (batch, 10)
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
| Dataset size    | 10,000 windows           |
| Train/test split| 80% train / 20% test     |
| Random seed     | 42                       |

Training and evaluation are implemented in `src/train.py`. The training loop records the average MSE loss per epoch. Evaluation computes the average MSE on the held-out test set using `torch.no_grad()`.

---

## 7. Results

| Model | Test MSE  |
|-------|-----------|
| MLP   | 0.173525  |
| LSTM  | 0.294412  |
| RNN   | 0.309764  |

Prediction plots comparing noisy input, clean target, and model prediction are saved to the `results/` folder when running `main.py`.

---

## 8. Discussion

The MLP achieved the lowest test MSE in this experiment, outperforming both the RNN and LSTM. This result is somewhat counterintuitive — sequence models are generally expected to have an advantage on time-series tasks — but there are a few reasons why MLP performed best here:

- **The task is local, not temporal.** Each input already contains all 10 noisy samples together, so there is no need to maintain memory across timesteps. The MLP can directly learn a mapping from all 15 features to the 10 clean outputs.
- **The sequences are short.** With only 10 timesteps, the vanishing gradient problem is less of a concern, and the advantage of LSTM gating is reduced.
- **The dataset is relatively small.** Recurrent models generally benefit more from larger datasets. With 8,000 training windows and 20 epochs, the RNN and LSTM may not have fully converged.

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
