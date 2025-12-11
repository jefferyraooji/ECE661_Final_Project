import torch
import torch.nn.functional as F
import os
import sys
import random

# --- PATH HACK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# -----------------

from config import cfg
from diffusion import Diffusion
from utils import save_images

# Import local model
from model_attention import ContextUnetWithAttention

CIFAR_CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def load_attention_model():
    # Load from 'attention_modification/results'
    results_dir = os.path.join(current_dir, "results")
    path = os.path.join(results_dir, f"ckpt_{cfg.EPOCHS-1}.pt")
    
    if not os.path.exists(path):
        if not os.path.exists(results_dir):
            raise FileNotFoundError("Local results folder not found. Run 'train_attention.py' first.")
        files = [f for f in os.listdir(results_dir) if f.endswith(".pt")]
        if not files: raise FileNotFoundError("No models found in local results.")
        path = os.path.join(results_dir, files[-1])
    
    print(f"Loading Attention model from {path}...")
    model = ContextUnetWithAttention(cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
    model.eval()
    return model

def run_visualization():
    diffusion = Diffusion(cfg)
    model = load_attention_model()
    
    # Save outputs to 'attention_modification/outputs'
    out_dir = os.path.join(current_dir, "outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    target_idx = random.randint(0, 9)
    target_label = torch.tensor([target_idx]).long()
    
    print(f"\n--- Generating With Attention for Class: {CIFAR_CLASSES[target_idx]} ---")
    
    n_snapshots = 10
    steps = torch.linspace(0, cfg.TIMESTEPS-1, n_snapshots).long().tolist()
    steps = [int(s) for s in steps]
    capture_list = steps + [cfg.TIMESTEPS]
    
    trajectory = diffusion.sample_trajectory(
        model, 
        n_sample=1, 
        labels=target_label, 
        guide_w=2.0, 
        capture_steps=capture_list
    )
    
    img_list = []
    unique_steps = sorted(list(set(steps + [cfg.TIMESTEPS])), reverse=True)

    for t in unique_steps:
        if t in trajectory:
            img_list.append(trajectory[t])

    GAP = 4 
    processed_list = []
    
    for img in img_list:
        rgb = (img + 1) / 2
        rgb = rgb.clamp(0, 1)
        padded = F.pad(rgb, (0, GAP, 0, 0), value=1.0)
        processed_list.append(padded)

    final_grid = torch.cat(processed_list, dim=3)

    save_path = os.path.join(out_dir, "inference_attention_process.png")
    save_images(final_grid, save_path, nrow=1)
    
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    run_visualization()