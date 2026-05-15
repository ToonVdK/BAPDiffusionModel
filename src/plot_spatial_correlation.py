# plot_spatial_correlation.py  (chronological split, semivariogram only)
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def calculate_semivariogram(grid, max_dist=15):
    """
    Empirical semivariogram.
    """
    variances = []
    distances = list(range(1, max_dist + 1))
    for d in distances:
        diff_x = grid[:, :, :-d] - grid[:, :, d:]
        diff_y = grid[:, :-d, :] - grid[:, d:, :]
        val = 0.5 * (np.nanmean(diff_x**2) + np.nanmean(diff_y**2))
        variances.append(val)
    return distances, variances

def evaluate_spatial_metrics(target_var):
    print(f"--- 1. LOADING {target_var.upper()} DATASETS (Chronological split) ---")
    
    if target_var == "lst":
        nc_path   = './data/processed/aligned_lst.nc'
        nc_var    = 'LST_PMW'
        ai_path   = './data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy'
        copula_path = './data/generated/copula_gaussian_lst_chrono.npy'
        real_test_path = './data/generated/real_lst_chrono.npy'
        unit = "K²"
        title_var = "LST"
    elif target_var == "sm":
        nc_path   = './data/processed/aligned_sm.nc'
        nc_var    = 'sm'
        ai_path   = './data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy'
        copula_path = './data/generated/copula_gaussian_sm_chrono.npy'
        real_test_path = './data/generated/real_sm_chrono.npy'
        unit = "(m³/m³)²"
        title_var = "Soil Moisture"
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    # Load full dataset to get training min/max for un‑normalising AI
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    ds_summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = ds_summer[nc_var].values[:, 0:32, 0:32]
    ds.close()

    # For AI only: need training min/max. We'll use the same chronological train indices.
    # But note: this expects that you have retrained the AI model on the same split.
    # If not, the AI normalization will be wrong. We keep as is for demonstration.
    from split_utils import get_train_indices
    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    val_min, val_max = np.nanmin(train_data), np.nanmax(train_data)

    # Real TEST data
    real_test_grid = np.load(real_test_path)

    # AI generated (un‑normalize with training stats)
    gen_norm = np.load(ai_path)
    ai_grid = ((gen_norm + 1) / 2) * (val_max - val_min) + val_min

    # Copula test data
    copula_grid = np.load(copula_path)

    print(f"  Training min/max: {val_min:.2f} / {val_max:.2f}")
    print(f"  Real test days: {real_test_grid.shape[0]}")
    print(f"  AI generated samples: {ai_grid.shape[0]}")
    print(f"  Copula test samples: {copula_grid.shape[0]}")

    # --- Masks ---
    real_test_mean = np.nanmean(real_test_grid, axis=0)
    land_mask = ~np.isnan(real_test_mean)

    ai_masked = np.where(land_mask, ai_grid, np.nan)
    copula_masked = np.where(land_mask, copula_grid, np.nan)

    # --- Semivariograms ---
    print("--- 2. CALCULATING SEMIVARIOGRAMS ---")
    dist, var_real   = calculate_semivariogram(real_test_grid)
    _,    var_ai     = calculate_semivariogram(ai_masked)
    _,    var_copula = calculate_semivariogram(copula_masked)

    # --- RMSE ---
    rmse_ai = np.sqrt(np.mean((np.array(var_real) - np.array(var_ai))**2))
    rmse_copula = np.sqrt(np.mean((np.array(var_real) - np.array(var_copula))**2))

    # --- Plotting (single panel) ---
    print("--- 3. PLOTTING SEMIVARIOGRAM ---")
    plt.figure(figsize=(8, 6))
    plt.plot(dist, var_real, marker='o', color='teal', linewidth=3, label='Real Test Observations')
    plt.plot(dist, var_ai, marker='s', color='purple', linewidth=2, linestyle='--', label='U‑Net Generated')
    plt.plot(dist, var_copula, marker='^', color='darkorange', linewidth=2, linestyle=':', label='Copula Generated')
    plt.title(f"Spatial Correlation: Empirical Semivariogram ({title_var})\nU-Net RMSE = {rmse_ai:.2e} | Copula RMSE = {rmse_copula:.2e}", fontsize=14)
    plt.xlabel("Distance Between Pixels")
    plt.ylabel(f"Semivariance {unit}")
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    save_filename = f'spatial_semivariogram_chrono_{target_var}.png'
    plt.savefig(save_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_filename}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_spatial_correlation.py [lst|sm]")
        sys.exit(1)
    var = sys.argv[1].lower()
    evaluate_spatial_metrics(var)