import torch
import numpy as np
import os
import xarray as xr
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

def compute_ocean_mask():
    """
    Extracts the boolean ocean mask from the historical dataset.
    """
    print("Computing ocean mask from real LST data...")
    ds = xr.open_dataset("./data/processed/aligned_lst.nc")
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds["time"].dt.month.isin(valid_months))
    lst_data = summer["LST_PMW"].values[:, 0:32, 0:32]
    ds.close()

    # If a pixel is NaN for all 40 years, it is the ocean
    ocean_mask = np.all(np.isnan(lst_data), axis=0) 
    print(f"Ocean pixels identified: {ocean_mask.sum()}")
    return ocean_mask

def generate_bulk_samples(num_days=3000, epoch_to_load=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- INITIATING DIFFUSINO GENERATION ---")
    print(f"Generating {num_days} days | Device: {device}")

    model_path = f"./data/model_output/unet_ema_epoch_{epoch_to_load}"
    output_dir = "./data/generated"
    os.makedirs(output_dir, exist_ok=True)

    # ---LOAD MODEL & SCHEDULER ---
    print(f"Loading U-Net from {model_path}...")
    model = UNet2DModel.from_pretrained(model_path).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(1000)

    # --- LOAD & FORMAT OCEAN MASK ---
    ocean_mask_np = compute_ocean_mask()
    # Convert to boolean tensor
    ocean_mask = torch.from_numpy(ocean_mask_np).bool().to(device)

    # --- REVERSE DIFFUSION LOOP ---
    torch.manual_seed(42)
    sample = torch.randn(num_days, 2, 32, 32).to(device)

    for t in tqdm(scheduler.timesteps, desc="Denoising Steps"):
        with torch.no_grad():
            residual = model(sample, t).sample
        
        # Step previous sample
        sample = scheduler.step(residual, t, sample).prev_sample
        
        # This applies the ocean mask constraint across all batches and both channels (LST and SM)
        sample[:, :, ocean_mask] = 0.0

    # --- SAVE ---
    print("--- SAVING GENERATED ARRAYS ---")
    gen_lst_norm = sample[:, 0, :, :].cpu().numpy()
    gen_sm_norm = sample[:, 1, :, :].cpu().numpy()

    lst_save_path = os.path.join(output_dir, f"ai_generated_lst_{num_days}days_epoch_{epoch_to_load}_masked_ocean.npy")
    sm_save_path = os.path.join(output_dir, f"ai_generated_sm_{num_days}days_epoch_{epoch_to_load}_masked_ocean.npy")

    np.save(lst_save_path, gen_lst_norm)
    np.save(sm_save_path, gen_sm_norm)

    print("--- GENERATION COMPLETE ---")
    print(f"LST Data shape {gen_lst_norm.shape} saved to: {lst_save_path}")
    print(f"SM Data shape  {gen_sm_norm.shape} saved to: {sm_save_path}")

if __name__ == "__main__":
    generate_bulk_samples(num_days=3000, epoch_to_load=100)