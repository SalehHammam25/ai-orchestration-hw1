import os
import torch
from torch.utils.data import DataLoader, random_split

from src.data import SignalDataset
from src.models import MLPModel, RNNModel, LSTMModel, CONDITION_SIZE
from src.train import train_model, evaluate_model
from src.evaluate import compare_results, save_prediction_plot

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SAMPLES   = 10000
TRAIN_RATIO   = 0.8
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-3
SEED          = 42
RESULTS_DIR   = "results"

# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = SignalDataset(num_samples=NUM_SAMPLES, seed=SEED)

train_size = int(NUM_SAMPLES * TRAIN_RATIO)
test_size  = NUM_SAMPLES - train_size
train_ds, test_ds = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── Train & Evaluate ──────────────────────────────────────────────────────────
models = {
    "MLP":  MLPModel(),
    "RNN":  RNNModel(),
    "LSTM": LSTMModel(),
}

results = {}
for name, model in models.items():
    print(f"=== {name} ===")
    train_model(model, train_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)
    mse = evaluate_model(model, test_loader, device=device)
    results[name] = mse
    print(f"{name} test MSE: {mse:.6f}\n")

compare_results(results)

# ── Save prediction plots ─────────────────────────────────────────────────────
sample_x, sample_y = test_ds[0]
noisy = sample_x[CONDITION_SIZE:]   # indices 5-14 are the 10 noisy samples

for name, model in models.items():
    model.eval()
    with torch.no_grad():
        pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()

    plot_path = os.path.join(RESULTS_DIR, f"prediction_{name.lower()}.png")
    save_prediction_plot(
        noisy_samples=noisy,
        clean_samples=sample_y,
        predicted_samples=pred,
        output_path=plot_path,
        title=f"{name} — Signal Reconstruction",
    )
    print(f"Saved: {plot_path}")
