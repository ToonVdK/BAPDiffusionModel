import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel
from scipy.spatial.distance import cdist
import xarray as xr
# from model import get_satellite_unet # Import your model architecture

def calculate_spatial_energy_score(real_vectors, gen_vectors):
    """
    Calculates the Spatial Energy Score.
    real_vectors and gen_vectors should be 2D arrays: (n_samples, valid_pixels)
    """
    if len(real_vectors) == 0 or len(gen_vectors) == 0:
        return np.nan
        
    # 1. Distance between Real and Generated
    dist_RG = cdist(real_vectors, gen_vectors, metric='euclidean')
    term1 = np.mean(dist_RG)
    
    # 2. Internal spread of Generated (Penalty for being too chaotic/wide)
    dist_GG = cdist(gen_vectors, gen_vectors, metric='euclidean')
    term2 = np.mean(dist_GG)
    
    return term1 - (0.5 * term2)

def plot_generative_u_curve():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # --- 1. PREPARE REAL BASELINE DATA ---
    print("Loading real baseline data...")
    lst_ds = xr.open_dataset('./data/processed/aligned_lst.nc')
    valid_months = [5, 6, 7, 8, 9]
    lst_summer = lst_ds.sel(time=lst_ds['time'].dt.month.isin(valid_months))['LST_PMW'].values
    
    # Extract 32x32 grid to match AI output
    real_grid = lst_summer[:, 0:32, 0:32]
    
    # Normalize it roughly to [-1, 1] just like the AI output
    real_min, real_max = np.nanmin(real_grid), np.nanmax(real_grid)
    real_grid_norm = 2 * ((real_grid - real_min) / (real_max - real_min)) - 1
    
    # Create land mask and extract 1D vectors for the Energy Score
    real_pixel_mean = np.nanmean(real_grid_norm, axis=0)
    land_mask = ~np.isnan(real_pixel_mean)
    
    # Impute the sparse NaNs in the real data so cdist doesn't crash
    real_imputed = np.where(np.isnan(real_grid_norm), np.broadcast_to(real_pixel_mean, real_grid_norm.shape), real_grid_norm)
    real_vectors = real_imputed[:, land_mask]
    
    # Optional: Subsample real data to match AI batch size for faster math
    np.random.seed(42)
    idx_real = np.random.choice(real_vectors.shape[0], 50, replace=False)
    real_vectors_sample = real_vectors[idx_real]

    # --- 2. EVALUATE EPOCHS ---
    epochs_to_test = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    ema_energy_scores = []
    
    for epoch in epochs_to_test:
        print(f"\n--- Evaluating Epoch {epoch} ---")
        model_path = f"./data/model_output/unet_ema_epoch_{epoch}"
        
        try:
            model = UNet2DModel.from_pretrained(model_path).to(device)
            model.eval()
        except Exception as e:
            print(f"Could not load epoch {epoch}, skipping. ({e})")
            continue
            
        batch_size = 300
        # NOTE: Ensure this matches your actual model input shape (using 32x32 based on previous context)
        noise = torch.randn(batch_size, 2, 32, 32).to(device)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                residual = model(noise, t, return_dict=False)[0]
                noise = scheduler.step(residual, t, noise).prev_sample
        
        # Extract LST channel (channel 0), ensure it matches 32x32
        generated_lst = noise[:, 0, 0:32, 0:32].cpu().numpy()
        
        # Extract valid land pixels
        ai_vectors = generated_lst[:, land_mask]
        
        # Calculate Spatial Energy Score
        es = calculate_spatial_energy_score(real_vectors_sample, ai_vectors)
        print(f"[EMA] Spatial Energy Score: {es:.4f}")
        
        ema_energy_scores.append((epoch, es))

    # --- 3. THE MONEY PLOT ---
    if not ema_energy_scores:
        print("No scores calculated, exiting.")
        return

    epochs_plotted, scores_plotted = zip(*ema_energy_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_plotted, scores_plotted, label='EMA Model (Spatial Focus)', color='purple', marker='o', linewidth=2)
    
    # Automatically highlight the absolute best epoch
    best_idx = np.argmin(scores_plotted)
    best_epoch = epochs_plotted[best_idx]
    best_score = scores_plotted[best_idx]
    
    plt.scatter(best_epoch, best_score, color='gold', s=200, zorder=5, edgecolors='black', label=f'Best Epoch: {best_epoch}')
    
    plt.title("Spatial Convergence: Energy Score vs Training Epochs", fontsize=14)
    plt.xlabel("Training Epoch")
    plt.ylabel("Spatial Energy Score (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('generative_u_curve_spatial.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\nSaved 'generative_u_curve_spatial.png'!")

if __name__ == "__main__":
    plot_generative_u_curve()