import os
import torch
import matplotlib.pyplot as plt


def compare_results(results_dict):
    """Print a sorted comparison table of model names and their test MSE."""
    print(f"\n{'Model':<12}  {'Test MSE':>10}")
    print("-" * 26)
    for name, mse in sorted(results_dict.items(), key=lambda kv: kv[1]):
        print(f"{name:<12}  {mse:>10.6f}")
    print()


def save_prediction_plot(noisy_samples, clean_samples, predicted_samples, output_path, title="Signal Prediction"):
    """Plot noisy input, clean target, and model prediction, then save to output_path."""
    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t

    noisy = to_numpy(noisy_samples)
    clean = to_numpy(clean_samples)
    pred  = to_numpy(predicted_samples)
    steps = range(len(clean))

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, noisy, "o--", color="gray",  label="Noisy input",  alpha=0.7)
    ax.plot(steps, clean, "o-",  color="green", label="Clean target", linewidth=2)
    ax.plot(steps, pred,  "o-",  color="red",   label="Prediction",   linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
