import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import time
from datetime import datetime, timedelta

# --- PATH HACK: Import from parent directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# -----------------------------------------------

from config import cfg
from diffusion import Diffusion
from utils import save_images, plot_training_history, save_history_to_json, load_history_from_json

# Import local model
from model_attention import ContextUnetWithAttention

# --- Helper: Format Time ---
def format_time(seconds):
    """Converts seconds to HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))

def train():
    print("Loading CIFAR-10 Data (For Attention Model)...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Data is still downloaded to the main 'data' folder to avoid re-downloading
    dataset = datasets.CIFAR10(root=cfg.DATASET_PATH, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize New Model
    model = ContextUnetWithAttention(cfg).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    
    diffusion = Diffusion(cfg)
    loss_fn = nn.MSELoss()

    print(f"Starting Training with Attention on {cfg.DEVICE}...")
    
    # --- LOCAL RESULTS DIRECTORY ---
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created local results directory: {results_dir}")
    
    history_path = os.path.join(results_dir, "training_history.json")
    loss_history, lr_history = load_history_from_json(history_path)
    start_epoch = len(loss_history)
    print(f"Resuming from Epoch {start_epoch}")
    
    if start_epoch > 0:
        last_ckpt = os.path.join(results_dir, f"ckpt_{start_epoch-1}.pt")
        if os.path.exists(last_ckpt):
            print(f"Loading weights from {last_ckpt}...")
            model.load_state_dict(torch.load(last_ckpt, map_location=cfg.DEVICE))
        for _ in range(start_epoch): scheduler.step()

    # --- Start Timer ---
    session_start_time = time.time()
    print("-" * 95)
    # Print Table Header
    print(f"{'Timestamp':<20} | {'Epoch':<10} | {'Loss':<8} | {'LR':<10} | {'Duration':<10} | {'Elapsed':<10} | {'ETA':<10}")
    print("-" * 95)

    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start_time = time.time() # Record start of epoch
        model.train()
        total_loss = 0
        
        viz_real_image = None
        viz_real_label = None

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            
            if i == 0:
                viz_real_image = images[0].unsqueeze(0)
                viz_real_label = labels[0].unsqueeze(0)

            optimizer.zero_grad()
            loss = diffusion.train_step(model, images, labels, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        lr_history.append(current_lr)

        # --- Calculate Timing Metrics ---
        current_time = time.time()
        epoch_duration = current_time - epoch_start_time
        total_elapsed = current_time - session_start_time
        
        # Calculate ETA (Estimated Time of Arrival)
        epochs_remaining = cfg.EPOCHS - (epoch + 1)
        eta_seconds = epochs_remaining * epoch_duration
        
        # Format strings
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_str = format_time(epoch_duration)
        elapsed_str = format_time(total_elapsed)
        eta_str = format_time(eta_seconds)

        # Print Info Row
        print(f"{timestamp_str:<20} | {epoch}/{cfg.EPOCHS:<9} | {avg_loss:.4f}   | {current_lr:.6f}   | {duration_str:<10} | {elapsed_str:<10} | {eta_str:<10}")

        save_history_to_json(loss_history, lr_history, history_path)
        plot_training_history(loss_history, lr_history, os.path.join(results_dir, "training_curves.png"))

        # Visualization
        if epoch % 5 == 0 or epoch == cfg.EPOCHS - 1:
            # print(f"Saving visualization...") # Optional: Comment out to keep logs clean
            
            n_snapshots = 10
            steps = torch.linspace(0, cfg.TIMESTEPS-1, n_snapshots).long().tolist()
            steps = [int(s) for s in steps]
            
            # Row 1
            row1_raw_imgs = []
            for t in steps:
                if t == 0: row1_raw_imgs.append(viz_real_image)
                else:
                    t_tensor = torch.tensor([t]).to(cfg.DEVICE)
                    noisy, _ = diffusion.noise_images(viz_real_image, t_tensor)
                    row1_raw_imgs.append(noisy)

            # Row 2
            capture_list = steps + [cfg.TIMESTEPS]
            trajectory = diffusion.sample_trajectory(
                model, n_sample=1, labels=viz_real_label, guide_w=2.0, capture_steps=capture_list
            )
            
            row2_raw_imgs = []
            if cfg.TIMESTEPS in trajectory: row2_raw_imgs.append(trajectory[cfg.TIMESTEPS])
            for t in sorted(steps, reverse=True):
                if t in trajectory: row2_raw_imgs.append(trajectory[t])
            row2_raw_imgs = row2_raw_imgs[:len(steps)]

            # Process
            def process_and_pad(img_list, pad_pixels=2):
                processed = []
                for img in img_list:
                    rgb = (img + 1) / 2
                    rgb = rgb.clamp(0, 1)
                    padded = F.pad(rgb, (0, pad_pixels, 0, 0), value=1.0)
                    processed.append(padded)
                return processed

            GAP = 4
            row1_ready = process_and_pad(row1_raw_imgs, pad_pixels=GAP)
            row2_ready = process_and_pad(row2_raw_imgs, pad_pixels=GAP)
            
            row1_strip = torch.cat(row1_ready, dim=3)
            row2_strip = torch.cat(row2_ready, dim=3)
            row1_strip = F.pad(row1_strip, (0, 0, 0, GAP), value=1.0)
            final_img = torch.cat([row1_strip, row2_strip], dim=2)
            
            save_path = os.path.join(results_dir, f"epoch_{epoch}_process.png")
            save_images(final_img, save_path, nrow=1)
            
            torch.save(model.state_dict(), os.path.join(results_dir, f"ckpt_{epoch}.pt"))

if __name__ == "__main__":
    train()