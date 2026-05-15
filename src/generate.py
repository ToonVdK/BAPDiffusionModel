# generate.py  (ocean fill with nearest-land pixel values)

import torch
import numpy as np
import os
import xarray as xr
from scipy.spatial import cKDTree
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm


def compute_masks_and_mapping():
    """
    Returns:
        ocean_mask : (32,32) bool array, True for ocean
        land_mask  : (32,32) bool array, True for land
        ocean_to_land_idx : list of (land_i, land_j) indices for each ocean pixel,
                            in the order of flat ocean pixel index
    """
    print("Computing ocean/land masks and nearest-land mapping...")
    ds = xr.open_dataset("./data/processed/aligned_lst.nc")
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds["time"].dt.month.isin(valid_months))
    lst_data = summer["LST_PMW"].values[:, 0:32, 0:32]
    ds.close()

    ocean_mask = np.all(np.isnan(lst_data), axis=0)   # True for ocean
    land_mask = ~ocean_mask

    # Build list of land coordinates
    land_coords = np.argwhere(land_mask)               # (N_land, 2)
    tree = cKDTree(land_coords)

    # For each ocean pixel, find nearest land
    ocean_coords = np.argwhere(ocean_mask)             # (N_ocean, 2)
    _, nn_indices = tree.query(ocean_coords, k=1)      # indices into land_coords

    # Create a mapping: for each ocean pixel (in order of ocean_coords), store target (i, j)
    ocean_to_land_idx = land_coords[nn_indices]        # (N_ocean, 2)

    print(f"Ocean pixels: {ocean_mask.sum()}, Land pixels: {land_mask.sum()}")
    print(f"Nearest-land mapping computed.")

    return ocean_mask, land_mask, ocean_to_land_idx


def generate_bulk_samples(num_days=100, epoch_to_load=100,
                          ocean_mask=None, land_mask=None,
                          ocean_to_land_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating generation of {num_days} days | Device: {device}")

    model_path = f"./data/model_output/unet_ema_epoch_{epoch_to_load}"
    output_dir = "./data/generated"
    os.makedirs(output_dir, exist_ok=True)

    # --- LOAD MODEL ---
    print(f"Loading model from {model_path}...")
    model = UNet2DModel.from_pretrained(model_path).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(1000)

    # --- MASKS & MAPPING ---
    if ocean_mask is None or ocean_to_land_idx is None:
        ocean_mask, land_mask, ocean_to_land_idx = compute_masks_and_mapping()

    # Convert to torch tensors on device
    ocean_mask_torch = torch.from_numpy(ocean_mask).to(device)   # (H,W)
    # We'll need to index ocean pixels efficiently. Flatten the mapping.
    # ocean_to_land_idx is (N_ocean, 2) with (i, j) coordinates of target land pixel.
    ocean_target_i = torch.tensor(ocean_to_land_idx[:, 0], dtype=torch.long, device=device)
    ocean_target_j = torch.tensor(ocean_to_land_idx[:, 1], dtype=torch.long, device=device)

    # Precompute the indices of ocean pixels in the flattened 2D grid
    ocean_coords = torch.nonzero(ocean_mask_torch)   # (N_ocean, 2)
    ocean_i = ocean_coords[:, 0]
    ocean_j = ocean_coords[:, 1]

    # --- GENERATION ---
    torch.manual_seed(42)
    sample = torch.randn(num_days, 2, 32, 32).to(device)   # (B, C, H, W)

    for t in tqdm(scheduler.timesteps, desc="Generating"):
        with torch.no_grad():
            residual = model(sample, t).sample
        sample = scheduler.step(residual, t, sample).prev_sample

        # --- NEAREST-LAND IMPUTATION ---
        # For each ocean pixel, copy the value from its nearest land pixel (across batch and channels)
        # sample[:, :, ocean_i, ocean_j] = sample[:, :, ocean_target_i, ocean_target_j]
        sample[:, :, ocean_i, ocean_j] = sample[:, :, ocean_target_i, ocean_target_j]

    # Convert to numpy
    gen_lst_norm = sample[:, 0, :, :].cpu().numpy()
    gen_sm_norm = sample[:, 1, :, :].cpu().numpy()

    lst_save_path = os.path.join(output_dir,
                                 f"ai_generated_lst_{num_days}days_epoch_{epoch_to_load}_nn_ocean.npy")
    sm_save_path = os.path.join(output_dir,
                                f"ai_generated_sm_{num_days}days_epoch_{epoch_to_load}_nn_ocean.npy")

    np.save(lst_save_path, gen_lst_norm)
    np.save(sm_save_path, gen_sm_norm)

    print("=== SAVED SUCCESSFULLY ===")
    print(f"LST Data shape {gen_lst_norm.shape} saved to: {lst_save_path}")
    print(f"SM Data shape  {gen_sm_norm.shape} saved to: {sm_save_path}")


if __name__ == "__main__":
    generate_bulk_samples(num_days=3000, epoch_to_load=100)