import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, train_loader, epochs, learning_rate=1e-3, device="cpu"):
    """Train model and return per-epoch average MSE loss."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}  loss: {avg_loss:.6f}")

    return epoch_losses


def evaluate_model(model, data_loader, device="cpu"):
    """Return average MSE over all batches in data_loader."""
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * x.size(0)

    return total_loss / len(data_loader.dataset)
