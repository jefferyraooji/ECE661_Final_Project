import torch
import torch.nn.functional as F
import torchvision
import os
import random
from config import cfg
from model import ContextUnet
from diffusion import Diffusion
from utils import save_images

CIFAR_CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def load_model():
    # Use config path
    path = os.path.join(cfg.RESULTS_PATH, f"ckpt_{cfg.EPOCHS-1}.pt")
    
    if not os.path.exists(path):
        if not os.path.exists(cfg.RESULTS_PATH):
            raise FileNotFoundError(f"Results folder not found at {cfg.RESULTS_PATH}")
            
        files = [f for f in os.listdir(cfg.RESULTS_PATH) if f.endswith(".pt")]
        if not files: raise FileNotFoundError("No models found.")
        path = os.path.join(cfg.RESULTS_PATH, files[-1])
    
    print(f"Loading model from {path}...")
    model = ContextUnet(cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(path, map_location=cfg.DEVICE))
    model.eval()
    return model

def run_denoising_visualization():
    diffusion = Diffusion(cfg)
    model = load_model()
    
    # Ensure outputs folder exists in FastDiff Code/outputs
    if not os.path.exists(cfg.OUTPUTS_PATH):
        os.makedirs(cfg.OUTPUTS_PATH)
        print(f"Created outputs directory: {cfg.OUTPUTS_PATH}")

    target_idx = random.randint(0, 9)
    target_label = torch.tensor([target_idx]).long()
    
    print(f"\n--- Generating CIFAR-10 Visualization for Class: {CIFAR_CLASSES[target_idx]} ---")
    
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

    # Process
    GAP = 4 
    processed_list = []
    
    for img in img_list:
        rgb = (img + 1) / 2
        rgb = rgb.clamp(0, 1)
        padded = F.pad(rgb, (0, GAP, 0, 0), value=1.0)
        processed_list.append(padded)

    final_grid = torch.cat(processed_list, dim=3)

    save_path = os.path.join(cfg.OUTPUTS_PATH, "inference_cifar_process.png")
    save_images(final_grid, save_path, nrow=1)
    
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    run_denoising_visualization()