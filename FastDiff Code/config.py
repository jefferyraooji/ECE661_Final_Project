import torch
import os

class Config:
    PROJECT_NAME = "FastDiff_CIFAR10"
    
    # --- PATH SETTINGS (Ensures everything stays in 'FastDiff Code') ---
    # Get the absolute path of the folder containing this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Data will be stored in: FastDiff Code/data
    DATASET_PATH = os.path.join(BASE_DIR, "data")
    
    # Results (Checkpoints) will be in: FastDiff Code/results
    RESULTS_PATH = os.path.join(BASE_DIR, "results")
    
    # Inference outputs will be in: FastDiff Code/outputs
    OUTPUTS_PATH = os.path.join(BASE_DIR, "outputs")
    
    # --- CIFAR-10 Specifics ---
    IMG_SIZE = 32
    CHANNELS = 3
    
    # --- Model Architecture ---
    N_FEAT = 256        
    N_CLASSES = 10 
    
    # --- Diffusion Hyperparameters ---
    TIMESTEPS = 500
    BETA_START = 1e-4
    BETA_END = 0.02
    DROP_PROB = 0.1
    GUIDE_W = 2.0
    
    # --- Training ---
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 1e-4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()