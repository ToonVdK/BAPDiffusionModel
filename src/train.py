import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from split_utils import get_chronological_split

from dataset import SatelliteDataset
from model import get_satellite_unet

class EarlyStopper:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model, save_path="./data/model_output"):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            model.save_pretrained(save_path) # Only save when it gets BETTER
            print(f"    --> Validation loss decreased to {val_loss:.4f}. Model saved!")
        else:
            self.counter += 1
            print(f"    --> No improvement. EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def train():
    """
    Trains the model using a masked loss function
    """
    # --- CONFIGURATIONS ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    batch_size = 16  # Power of 2
    epochs = 300
    learning_rate = 1e-4
    weight_decay = 1e-4  # L2 Regularization

    # --- INITIALIZE DATA & SPLIT ---
    train_idx, _ = get_chronological_split(train_ratio=0.8)   # 80% for training, 20% for test (held out)
    full_dataset = SatelliteDataset(
        lst_path='./data/processed/aligned_lst.nc',
        sm_path='./data/processed/aligned_sm.nc',
        indices=train_idx  # Only the chronological training days
    )
    # Split the training days further into train/validation (random)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total days: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_satellite_unet().to(device)
    # Create the Shadow Model
    # power=3/4 is the standard optimization for diffusion EMA
    ema_model = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True, inv_gamma=1.0, power=3/4)
    ema_model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- VARIANCE PRESERVING SDE ---
    # The MissDiff paper adopts a Variance Preserving (VP) SDE.
    # In diffusers, DDPMScheduler is the discrete equivalent of this.
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize Early Stopper & History tracking
    early_stopper = EarlyStopper(patience=40, min_delta=0.0)  # Wait for 50 consecutive epochs of no improvement before killing the run
    train_loss_history = []
    val_loss_history = []

    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        model.train() # Make sure model is in training mode
        epoch_train_loss = 0.0
        # tqdm provides a nice progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # Unpack the dataset yields
            clean_images, masks = batch
            clean_images = clean_images.to(device)
            masks = masks.to(device)

            # Sample standard Gaussian noise to add to the images
            noise = torch.randn_like(clean_images).to(device)

            # Sample a random timestep for each image in the batch
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (= forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # --- NEURAL NETWORK PREDICTION ---
            # Ask the U-Net to predict the noise that was added
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # --- MASKING THE LOSS (from MissDiff paper) ---
            # According to the Denoising Score Matching objective on missing data:
            # We multiply both the predicted noise and the target noise by our binary mask.
            # This completely zeroes out the error in the cloudy/missing regions,
            # meaning the model is NOT penalized for what it guesses inside the clouds.
            masked_noise_pred = noise_pred * masks
            masked_noise_target = noise * masks

            # Calculate MSE loss ONLY on the valid pixels
            loss = F.mse_loss(masked_noise_pred, masked_noise_target)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update the shadow model with the newest active weights
            ema_model.step(model.parameters())

            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION LOOP---
        model.eval() # Turn off dropout/batchnorm for testing
        epoch_val_loss = 0.0
        
        with torch.no_grad(): # Don't calculate gradients (saves massive VRAM)
            for batch in val_dataloader:
                clean_images, masks = batch
                clean_images, masks = clean_images.to(device), masks.to(device)

                noise = torch.randn_like(clean_images).to(device)
                bsz = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                masked_noise_pred = noise_pred * masks
                masked_noise_target = noise * masks

                loss = F.mse_loss(masked_noise_pred, masked_noise_target)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        lr_scheduler.step()
        print(f"Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        early_stopper(avg_val_loss, model)
        # Save a snapshot of the smooth EMA weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Saving EMA Checkpoint at Epoch {epoch + 1} ---")
            
            # 1. Store the noisy active weights in a temporary buffer
            ema_model.store(model.parameters())
            
            # 2. Copy the smooth EMA weights into the active model framework
            ema_model.copy_to(model.parameters())
            
            # 3. Save the model (which currently holds the perfect EMA weights)
            model.save_pretrained(f"./data/model_output/unet_ema_epoch_{epoch + 1}")
            
            # 4. Restore the noisy active weights back into the model so training can continue
            ema_model.restore(model.parameters())
            print("Checkpoint saved successfully. Resuming training...\n")

    print("\nTraining finished!")
    print("\nCopying final EMA weights to active model...")
    ema_model.copy_to(model.parameters())
    model.save_pretrained("./data/model_output/unet_ema_final")

    # --- PLOT & SAVE LOSS CURVE ---
    print("Generating loss curve plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot both lines
    actual_epochs = len(train_loss_history)
    plt.plot(range(1, actual_epochs + 1), train_loss_history, label='Training Loss', marker='o', linestyle='-', color='b', markersize=4)
    plt.plot(range(1, actual_epochs + 1), val_loss_history, label='Validation Loss', marker='x', linestyle='--', color='r', markersize=4)
    
    plt.title("Training & Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f"./data/model_output/loss_curve.png", dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to ./data/model_output/loss_curve.png")

    plt.show()


if __name__ == "__main__":
    train()