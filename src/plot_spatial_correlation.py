# plot_spatial_correlation.py  (fair out‑of‑sample version)
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ----------------------------------------------------------------------
# Systematic split config (must match copula & training)
TEST_STEP = 5

def get_train_indices():
    ds = xr.open_dataset('./data/processed/aligned_lst.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    n_total = len(summer['time'])
    ds.close()
    test_idx = np.arange(0, n_total, TEST_STEP)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    return train_idx

# ----------------------------------------------------------------------
def calculate_energy_score(real_vectors, gen_vectors, n_samples=500):
    """
    Calculates the Formal 2-Sample Energy Distance in 1024D space.
    Formula: ED = 2 * E[||R - G||] - E[||R - R||] - E[||G - G||]
    """
    np.random.seed(42)
    idx_real = np.random.choice(real_vectors.shape[0], min(n_samples, real_vectors.shape[0]), replace=False)
    idx_gen  = np.random.choice(gen_vectors.shape[0],  min(n_samples, gen_vectors.shape[0]),  replace=False)
    
    R = real_vectors[idx_real]
    G = gen_vectors[idx_gen]
    
    dist_RG = np.mean(cdist(R, G, metric='euclidean'))
    dist_RR = np.mean(cdist(R, R, metric='euclidean'))
    dist_GG = np.mean(cdist(G, G, metric='euclidean'))
    
    return max(0.0, 2 * dist_RG - dist_RR - dist_GG)

# ----------------------------------------------------------------------
def calculate_semivariogram(grid, max_dist=15):
    """
    Empirical semivariogram (unchanged).
    """
    variances = []
    distances = list(range(1, max_dist + 1))
    for d in distances:
        diff_x = grid[:, :, :-d] - grid[:, :, d:]
        diff_y = grid[:, :-d, :] - grid[:, d:, :]
        val = 0.5 * (np.nanmean(diff_x**2) + np.nanmean(diff_y**2))
        variances.append(val)
    return distances, variances

# ----------------------------------------------------------------------
def evaluate_spatial_metrics(target_var, test_step=TEST_STEP):
    print(f"--- 1. LOADING {target_var.upper()} DATASETS (OOS step={test_step}) ---")
    
    if target_var == "lst":
        nc_path   = './data/processed/aligned_lst.nc'
        nc_var    = 'LST_PMW'
        ai_path   = './data/generated/ai_generated_lst_3000days_epoch_300.npy'
        copula_path = f'./data/generated/copula_gaussian_lst_oos_test_step{test_step}.npy'
        real_test_path = f'./data/generated/real_lst_oos_test_step{test_step}.npy'
        unit = "K²"
    elif target_var == "sm":
        nc_path   = './data/processed/aligned_sm.nc'
        nc_var    = 'sm'
        ai_path   = './data/generated/ai_generated_sm_3000days_epoch_300.npy'
        copula_path = f'./data/generated/copula_gaussian_sm_oos_test_step{test_step}.npy'
        real_test_path = f'./data/generated/real_sm_oos_test_step{test_step}.npy'
        unit = "(m³/m³)²"
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    # Load full dataset for coordinates and training min/max
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    ds_summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = ds_summer[nc_var].values[:, 0:32, 0:32]

    # Training subset for normalization
    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    val_min, val_max = np.nanmin(train_data), np.nanmax(train_data)

    # Real TEST data (already saved as .npy)
    real_test_grid = np.load(real_test_path)   # shape (n_test, 32, 32)

    # AI generated (un‑normalize with training stats)
    gen_norm = np.load(ai_path)
    ai_grid = ((gen_norm + 1) / 2) * (val_max - val_min) + val_min

    # Copula test data
    copula_grid = np.load(copula_path)   # shape (n_test, 32, 32)

    print(f"  Training min/max: {val_min:.2f} / {val_max:.2f}")
    print(f"  Real test days: {real_test_grid.shape[0]}")
    print(f"  AI generated samples: {ai_grid.shape[0]}")
    print(f"  Copula test samples: {copula_grid.shape[0]}")

    # ------------------------------------------------------------------
    print("--- 2. APPLYING MASKS ---")
    # Land mask based on real test mean (where there is any valid data)
    real_test_mean = np.nanmean(real_test_grid, axis=0)
    land_mask = ~np.isnan(real_test_mean)

    # Mask synthetic grids for variogram (hide ocean)
    ai_masked = np.where(land_mask, ai_grid, np.nan)
    copula_masked = np.where(land_mask, copula_grid, np.nan)

    # Impute NaNs in real test data with its own pixel‑wise mean
    real_test_imputed = np.where(
        np.isnan(real_test_grid),
        np.broadcast_to(real_test_mean, real_test_grid.shape),
        real_test_grid
    )

    # Extract valid land vectors for energy score
    real_vectors = real_test_imputed[:, land_mask]
    ai_vectors   = ai_grid[:, land_mask]
    copula_vectors = copula_grid[:, land_mask]

    # ------------------------------------------------------------------
    print("--- 3. CALCULATING SPATIAL ENERGY SCORES (out‑of‑sample) ---")
    es_ai = calculate_energy_score(real_vectors, ai_vectors)
    es_copula = calculate_energy_score(real_vectors, copula_vectors)
    print(f"U‑Net Energy Score:  {es_ai:.2f}")
    print(f"Copula Energy Score: {es_copula:.2f}")

    # ------------------------------------------------------------------
    print("--- 4. CALCULATING EMPIRICAL SEMIVARIOGRAMS ---")
    dist, var_real   = calculate_semivariogram(real_test_grid)   # real test, NaNs handled internally
    _,    var_ai     = calculate_semivariogram(ai_masked)
    _,    var_copula = calculate_semivariogram(copula_masked)

    # ------------------------------------------------------------------
    print("--- 5. PLOTTING RESULTS ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PLOT 1: Energy Score bar chart
    bars = axes[0].bar(['U‑Net A.I.', 'Gaussian Copula'], [es_ai, es_copula],
                       color=['purple', 'darkorange'])
    axes[0].set_title("Spatial Energy Score (Multivariate Distance)\n(Out‑of‑Sample)", fontsize=13)
    axes[0].set_ylabel("Energy Score")
    axes[0].grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + max(0.02*yval, 0.1),
                     f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    # PLOT 2: Semivariogram
    axes[1].plot(dist, var_real, marker='o', color='teal', linewidth=3, label='Real Test Observations')
    axes[1].plot(dist, var_ai, marker='s', color='purple', linewidth=2, linestyle='--', label='U‑Net Generated')
    axes[1].plot(dist, var_copula, marker='^', color='darkorange', linewidth=2, linestyle=':', label='Copula Generated')
    axes[1].set_title("Spatial Correlation: Empirical Semivariogram\n(Out‑of‑Sample)", fontsize=13)
    axes[1].set_xlabel("Distance Between Pixels")
    axes[1].set_ylabel(f"Semivariance {unit}")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=11)

    plt.suptitle(f"Spatial Evaluation: {target_var.upper()} (Systematic Step={test_step})", fontsize=16, y=1.02)
    plt.tight_layout()
    save_filename = f'spatial_correlation_oos_step{test_step}_{target_var}.png'
    plt.savefig(save_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_filename}'")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [lst|sm] [test_step]")
        sys.exit(1)
    var = sys.argv[1].lower()
    step = int(sys.argv[2]) if len(sys.argv) > 2 else TEST_STEP
    evaluate_spatial_metrics(var, test_step=step)