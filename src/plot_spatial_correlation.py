import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from split_utils import get_train_indices

def calculate_semivariogram(grid, max_dist=15):
    """
    Empirical semivariogram calculating isotropic spatial variance.
    """
    variances = []
    distances = list(range(1, max_dist + 1))
    for d in distances:
        diff_x = grid[:, :, :-d] - grid[:, :, d:]
        diff_y = grid[:, :-d, :] - grid[:, d:, :]
        val = 0.5 * (np.nanmean(diff_x**2) + np.nanmean(diff_y**2))
        variances.append(val)
    return distances, variances

def process_variable(target_var):
    """
    Loads, masks, and calculates the semivariogram for a given variable.
    """
    print(f"--- PROCESSING {target_var.upper()} ---")
    if target_var == "lst":
        nc_path   = './data/processed/aligned_lst.nc'
        nc_var    = 'LST_PMW'
        ai_path   = './data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy'
        copula_path = './data/generated/copula_gaussian_lst_chrono.npy'
        real_test_path = './data/generated/real_lst_chrono.npy'
        unit = "(K²)"
        title_var = "Land Surface Temperature"
    elif target_var == "sm":
        nc_path   = './data/processed/aligned_sm.nc'
        nc_var    = 'sm'
        ai_path   = './data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy'
        copula_path = './data/generated/copula_gaussian_sm_chrono.npy'
        real_test_path = './data/generated/real_sm_chrono.npy'
        unit = "(m³/m³)²"
        title_var = "Soil Moisture"

    # Get training min/max for un-normalising AI
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    ds_summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = ds_summer[nc_var].values[:, 0:32, 0:32]
    ds.close()

    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    val_min, val_max = np.nanmin(train_data), np.nanmax(train_data)

    # Load test/generated arrays
    real_test_grid = np.load(real_test_path)
    gen_norm = np.load(ai_path)
    ai_grid = ((gen_norm + 1) / 2) * (val_max - val_min) + val_min
    copula_grid = np.load(copula_path)

    # Apply masks
    real_test_mean = np.nanmean(real_test_grid, axis=0)
    land_mask = ~np.isnan(real_test_mean)

    ai_masked = np.where(land_mask, ai_grid, np.nan)
    copula_masked = np.where(land_mask, copula_grid, np.nan)

    # Calculate Variograms
    dist, var_real   = calculate_semivariogram(real_test_grid)
    _,    var_ai     = calculate_semivariogram(ai_masked)
    _,    var_copula = calculate_semivariogram(copula_masked)

    # Calculate RMSE
    rmse_ai = np.sqrt(np.mean((np.array(var_real) - np.array(var_ai))**2))
    rmse_copula = np.sqrt(np.mean((np.array(var_real) - np.array(var_copula))**2))
    
    return dist, var_real, var_ai, var_copula, rmse_ai, rmse_copula, title_var, unit

def main():
    print("--- GENERATING MASTER SEMIVARIOGRAM PLOT ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, var in enumerate(['lst', 'sm']):
        dist, var_real, var_ai, var_copula, rmse_ai, rmse_copula, title_var, unit = process_variable(var)
        
        ax = axes[idx]
        
        # Plot lines
        ax.plot(dist, var_real, marker='o', color='teal', linewidth=3, label='Real Observations')
        ax.plot(dist, var_ai, marker='s', color='purple', linewidth=2, linestyle='--', label='Diffusion Generated')
        ax.plot(dist, var_copula, marker='^', color='darkorange', linewidth=2, linestyle=':', label='Copula Generated')
        
        # Formatting
        ax.set_title(f"{title_var}\nDiffusion RMSE: {rmse_ai:.2e} | Copula RMSE: {rmse_copula:.2e}", fontsize=13)
        ax.set_xlabel("Distance Between Pixels", fontsize=12)
        ax.set_ylabel(f"Semivariance {unit}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)

    # Global Title
    plt.suptitle("Spatial Correlation: Empirical Semivariograms", fontsize=18, y=1.05)
    plt.tight_layout()
    
    save_filename = 'spatial_semivariograms.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved highly condensed master figure to '{save_filename}'")

if __name__ == "__main__":
    main()