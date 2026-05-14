import torch
import numpy as np
import os
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

def generate_bulk_samples(num_days=100, epoch_to_load=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating generation of {num_days} days | Device: {device}")

    model_path = f"./data/model_output/unet_ema_epoch_{epoch_to_load}"
    output_dir = "./data/generated"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- LOAD MODEL ---
    print(f"Loading model from {model_path}...")
    model = UNet2DModel.from_pretrained(model_path).to(device)
    # Take the same noise timetable as during training (1000 steps)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(1000)

    # --- GENERATION ---
    torch.manual_seed(42)  # Optional for testing: Set the seed
    sample = torch.randn(num_days, 2, 32, 32).to(device) # Create a gaussian noise tensor (n days, 2 channels, 32x32 pixels)

    for t in tqdm(scheduler.timesteps, desc="Generating"):
        with torch.no_grad():  # We don't need gradients during inference
            residual = model(sample, t).sample  # Let the model make a prediction of the noise
        sample = scheduler.step(residual, t, sample).prev_sample  # Subtract the predicted noise for the next sample

    # Convert the generated samples to NumPy arrays
    gen_lst_norm = sample[:, 0, :, :].cpu().numpy()
    gen_sm_norm = sample[:, 1, :, :].cpu().numpy()
    
    lst_save_path = os.path.join(output_dir, f'ai_generated_lst_{num_days}days_epoch_{epoch_to_load}.npy')
    sm_save_path = os.path.join(output_dir, f'ai_generated_sm_{num_days}days_epoch{epoch_to_load}.npy')
    
    np.save(lst_save_path, gen_lst_norm)
    np.save(sm_save_path, gen_sm_norm)
    
    print("=== SAVED SUCCESSFULLY ===")
    print(f"LST Data shape {gen_lst_norm.shape} saved to: {lst_save_path}")
    print(f"SM Data shape  {gen_sm_norm.shape} saved to: {sm_save_path}")

if __name__ == "__main__":
    generate_bulk_samples(num_days=3000, epoch_to_load=140)