import torch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import wasserstein_distance, ks_2samp

def calculate_metrics():
    """
    Compares the AI generated data to the real data in a histogram, calculates the Wasserstein distance and does
    a Kolmogorov-Smirnov test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on: {device}")

    # --- LOAD GENERATED DATA ---
    gen_lst_norm = np.load('./data/generated/ai_generated_lst_3000days.npy')
    gen_sm_norm = np.load('./data/generated/ai_generated_sm_3000days.npy')

    # --- LOAD REAL DATA ---
    print("Loading Real Data...")
    lst_ds_full = xr.open_dataset('./data/processed/aligned_lst.nc')
    sm_ds_full = xr.open_dataset('./data/processed/aligned_sm.nc')

    # Filter for summer months (May=5 to Sept=9)
    valid_months = [5, 6, 7, 8, 9]
    lst_ds = lst_ds_full.sel(time=lst_ds_full['time'].dt.month.isin(valid_months))
    sm_ds = sm_ds_full.sel(time=sm_ds_full['time'].dt.month.isin(valid_months))
    
    real_lst = lst_ds['LST_PMW'].values
    real_sm = sm_ds['sm'].values

    # Find the global min and max used during training
    lst_min, lst_max = np.nanmin(real_lst), np.nanmax(real_lst)
    sm_min, sm_max = np.nanmin(real_sm), np.nanmax(real_sm)

    # --- REVERSE NORMALIZATION ---
    print("Reversing AI normalization to physical units...")
    # Formula: X_real = ((X_norm + 1) / 2) * (max - min) + min
    gen_lst = ((gen_lst_norm + 1) / 2) * (lst_max - lst_min) + lst_min
    gen_sm = ((gen_sm_norm + 1) / 2) * (sm_max - sm_min) + sm_min

    # --- EXTRACT VALID PIXELS ---
    # We only calculate metrics on valid land pixels
    valid_real_mask = ~np.isnan(real_lst) & ~np.isnan(real_sm)
    real_lst_flat = real_lst[valid_real_mask]
    real_sm_flat = real_sm[valid_real_mask]

    # Create a 2D spatial land mask (True if a pixel is EVER valid across the entire span of the dataset)
    spatial_land_mask = valid_real_mask.any(axis=0)
    # Broadcast to match the AI batch size
    batch_mask = np.broadcast_to(spatial_land_mask, gen_lst.shape)
    
    # Extract only the valid pixels from the dataset by applying the mask
    gen_lst_flat = gen_lst[batch_mask]
    gen_sm_flat = gen_sm[batch_mask]

    # --- CALCULATE METRICS ---
    # Wasserstein distance calculates how lazy you can be when moving stuff around while still getting it done
    wd_lst = wasserstein_distance(real_lst_flat, gen_lst_flat)
    wd_sm = wasserstein_distance(real_sm_flat, gen_sm_flat)

    # KS Test calculates the statistic (distance) and the p-value
    ks_lst_stat, ks_lst_pval = ks_2samp(real_lst_flat, gen_lst_flat)
    ks_sm_stat, ks_sm_pval = ks_2samp(real_sm_flat, gen_sm_flat)

    print("\n=== MODEL METRICS ===")
    print(f"LST | Wasserstein: {wd_lst:.2f} K | KS Stat: {ks_lst_stat:.3f} (p-value: {ks_lst_pval:.2e})")
    print(f"SM  | Wasserstein: {wd_sm:.4f} m³/m³ | KS Stat: {ks_sm_stat:.3f} (p-value: {ks_sm_pval:.2e})")

    # --- VISUALIZE DISTRIBUTIONS ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create a shared set of 75 bins based on the real data's range
    lst_bins = np.linspace(real_lst_flat.min(), real_lst_flat.max(), 75)
    sm_bins = np.linspace(real_sm_flat.min(), real_sm_flat.max(), 75)

    # LST Histogram (Force both to use lst_bins)
    axes[0].hist(real_lst_flat, bins=lst_bins, density=True, alpha=0.5, color='teal', label='Real LST')
    axes[0].hist(gen_lst_flat, bins=lst_bins, density=True, alpha=0.5, color='purple', label='AI Generated LST')
    axes[0].set_title(f"LST Match (WD: {wd_lst:.2f} K, KS: {ks_lst_stat:.2e})")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # SM Histogram (Force both to use sm_bins)
    axes[1].hist(real_sm_flat, bins=sm_bins, density=True, alpha=0.5, color='teal', label='Real SM')
    axes[1].hist(gen_sm_flat, bins=sm_bins, density=True, alpha=0.5, color='purple', label='AI Generated SM')
    axes[1].set_title(f"SM Match (WD: {wd_sm:.3f} m³/m³, KS: {ks_sm_stat:.2e})")
    axes[1].set_xlabel("Volumetric Soil Moisture (m³/m³)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=150)
    print("Saved 'histogram_analysis.png'")

if __name__ == "__main__":
    calculate_metrics()