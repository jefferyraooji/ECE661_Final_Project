import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr.astype('uint8'))
    im.save(path)

def save_history_to_json(loss_history, lr_history, path):
    """Saves training history to a JSON file."""
    data = {
        "loss": loss_history,
        "lr": lr_history
    }
    with open(path, 'w') as f:
        json.dump(data, f)

def load_history_from_json(path):
    """Loads training history from a JSON file."""
    if not os.path.exists(path):
        return [], []
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get("loss", []), data.get("lr", [])

def plot_training_history(loss_history, lr_history, save_path):
    """
    Plots the full training curve.
    """
    if not loss_history:
        return

    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss', color='tab:red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training Loss (Total Epochs: {len(loss_history)})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Plot 2: Learning Rate
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lr_history, label='Learning Rate', color='tab:blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()