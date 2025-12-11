import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from datetime import datetime, timedelta

# --- AMP Imports ---
from torch.cuda.amp import autocast, GradScaler

# --- Enable TF32 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config import cfg
from model import ContextUnet
from diffusion import Diffusion
from utils import save_images, plot_training_history, save_history_to_json, load_history_from_json

def format_time(seconds):
    """Converts seconds to HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))

def train():
    print(f"Loading CIFAR-10 Data from: {cfg.DATASET_PATH}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root=cfg.DATASET_PATH, train=True, download=True, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    model = ContextUnet(cfg).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    
    diffusion = Diffusion(cfg)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()

    print(f"Starting Training on {cfg.DEVICE} (AMP Enabled)...")
    
    if not os.path.exists(cfg.RESULTS_PATH):
        os.makedirs(cfg.RESULTS_PATH)
    
    # --- RESUME LOGIC ---
    history_path = os.path.join(cfg.RESULTS_PATH, "training_history.json")
    loss_history, lr_history = load_history_from_json(history_path)
    start_epoch = len(loss_history)
    print(f"Resuming from Epoch {start_epoch}")
    
    if start_epoch > 0:
        last_ckpt = os.path.join(cfg.RESULTS_PATH, f"ckpt_{start_epoch-1}.pt")
        if os.path.exists(last_ckpt):
            print(f"Loading weights from {last_ckpt}...")
            checkpoint = torch.load(last_ckpt, map_location=cfg.DEVICE)
            model.load_state_dict(checkpoint)
        for _ in range(start_epoch): scheduler.step()

    # --- TIMING VARIABLES ---
    session_start_time = time.time()
    print("-" * 80)
    print(f"{'Timestamp':<20} | {'Epoch':<10} | {'Loss':<8} | {'LR':<10} | {'Duration':<10} | {'Elapsed':<10} | {'ETA':<10}")
    print("-" * 80)

    # --- Training Loop ---
    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        
        viz_real_image = None
        viz_real_label = None

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(cfg.DEVICE, non_blocking=True)
            labels = labels.to(cfg.DEVICE, non_blocking=True)
            
            if i == 0:
                viz_real_image = images[0].unsqueeze(0)
                viz_real_label = labels[0].unsqueeze(0)

            optimizer.zero_grad()
            
            with autocast():
                loss = diffusion.train_step(model, images, labels, loss_fn)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        lr_history.append(current_lr)

        # --- CALCULATE TIMING METRICS ---
        current_time = time.time()
        epoch_duration = current_time - epoch_start_time
        total_elapsed = current_time - session_start_time
        
        # Calculate ETA
        epochs_remaining = cfg.EPOCHS - (epoch + 1)
        eta_seconds = epochs_remaining * epoch_duration
        
        # Get formatting strings
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_str = format_time(epoch_duration)
        elapsed_str = format_time(total_elapsed)
        eta_str = format_time(eta_seconds)

        # Print Info
        print(f"{timestamp_str:<20} | {epoch}/{cfg.EPOCHS:<9} | {avg_loss:.4f}   | {current_lr:.6f}   | {duration_str:<10} | {elapsed_str:<10} | {eta_str:<10}")

        save_history_to_json(loss_history, lr_history, history_path)
        plot_training_history(loss_history, lr_history, os.path.join(cfg.RESULTS_PATH, "training_curves.png"))

        # --- VISUALIZATION ---
        if epoch % 5 == 0 or epoch == cfg.EPOCHS - 1:
            
            n_snapshots = 10
            steps = torch.linspace(0, cfg.TIMESTEPS-1, n_snapshots).long().tolist()
            steps = [int(s) for s in steps]
            
            # Row 1: Forward
            row1_raw_imgs = []
            for t in steps:
                if t == 0:
                    row1_raw_imgs.append(viz_real_image)
                else:
                    t_tensor = torch.tensor([t]).to(cfg.DEVICE)
                    noisy, _ = diffusion.noise_images(viz_real_image, t_tensor)
                    row1_raw_imgs.append(noisy)

            # Row 2: Reverse
            capture_list = steps + [cfg.TIMESTEPS]
            trajectory = diffusion.sample_trajectory(
                model, n_sample=1, labels=viz_real_label, guide_w=2.0, capture_steps=capture_list
            )
            
            row2_raw_imgs = []
            if cfg.TIMESTEPS in trajectory: row2_raw_imgs.append(trajectory[cfg.TIMESTEPS])
            for t in sorted(steps, reverse=True):
                if t in trajectory: row2_raw_imgs.append(trajectory[t])
            row2_raw_imgs = row2_raw_imgs[:len(steps)]

            # Process & Pad
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
            
            save_path = os.path.join(cfg.RESULTS_PATH, f"epoch_{epoch}_process.png")
            save_images(final_img, save_path, nrow=1)
            
            torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, f"ckpt_{epoch}.pt"))

if __name__ == "__main__":
    train()