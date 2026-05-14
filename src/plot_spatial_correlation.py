import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def calculate_energy_score(real_vectors, gen_vectors, n_samples=500):
    """
    Calculates the Formal 2-Sample Energy Distance in 1024D space.
    Formula: ED = 2 * E[||R - G||] - E[||R - R||] - E[||G - G||]
    """
    np.random.seed(42)
    idx_real = np.random.choice(real_vectors.shape[0], min(n_samples, real_vectors.shape[0]), replace=False)
    idx_gen = np.random.choice(gen_vectors.shape[0], min(n_samples, gen_vectors.shape[0]), replace=False)
    
    R = real_vectors[idx_real]
    G = gen_vectors[idx_gen]
    
    dist_RG = np.mean(cdist(R, G, metric='euclidean'))
    dist_RR = np.mean(cdist(R, R, metric='euclidean'))
    dist_GG = np.mean(cdist(G, G, metric='euclidean'))
    
    energy_distance = (2 * dist_RG) - dist_RR - dist_GG
    
    return max(0.0, energy_distance)

def calculate_semivariogram(grid, max_dist=15):
    """
    Calculates the empirical semivariogram.
    This is the mathematical proof of spatial correlation.
    """
    variances = []
    distances = list(range(1, max_dist + 1))
    
    for d in distances:
        # Shift the grid horizontally and vertically to compare pixels at distance 'd' instead of using a for loop (slow)
        # Then subtract the shifted map from the original map to calculate pixel_A - pixel_B
        diff_x = grid[:, :, :-d] - grid[:, :, d:]
        diff_y = grid[:, :-d, :] - grid[:, d:, :]
        
        # Calculate the mean squared difference, ignoring NaNs --> Derived from Matharon's Equation for Empirical Semivariance
        val = 0.5 * (np.nanmean(diff_x**2) + np.nanmean(diff_y**2))
        variances.append(val)
        
    return distances, variances

def evaluate_spatial_metrics(target_var):
    print(f"--- 1. LOADING {target_var.upper()} DATASETS ---")
    
    if target_var == "lst":
        real_path = './data/processed/aligned_lst.nc'
        nc_var = 'LST_PMW'
        copula_path = './data/generated/copula_gaussian_lst_3000days.npy'
        ai_path = './data/generated/ai_generated_lst_3000days.npy'
    elif target_var == "sm":
        real_path = './data/processed/aligned_sm.nc'
        nc_var = 'sm'
        copula_path = './data/generated/copula_gaussian_sm_3000days.npy'
        ai_path = './data/generated/ai_generated_sm_3000days.npy'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    # Real Data
    ds_full = xr.open_dataset(real_path)
    valid_months = [5, 6, 7, 8, 9]
    ds_summer = ds_full.sel(time=ds_full['time'].dt.month.isin(valid_months))
    real_grid = ds_summer[nc_var].values[:, 0:32, 0:32]
    val_min, val_max = np.nanmin(real_grid), np.nanmax(real_grid)

    # AI Data
    gen_norm = np.load(ai_path)
    ai_grid = ((gen_norm + 1) / 2) * (val_max - val_min) + val_min

    # Copula Data
    try:
        copula_flat = np.load(copula_path)
        copula_grid = copula_flat.reshape(3000, 32, 32)
    except FileNotFoundError:
        print(f"Error: Could not find {copula_path}. Ensure it was generated!")
        sys.exit(1)

    print("--- 2. APPLYING MASKS ---")
    # Get the land mask based on historical validity
    real_pixel_mean = np.nanmean(real_grid, axis=0)
    land_mask = ~np.isnan(real_pixel_mean)
    
    # Mask the grids for the Variogram (hides AI ocean hallucinations)
    ai_masked = np.where(land_mask, ai_grid, np.nan)
    copula_masked = np.where(land_mask, copula_grid, np.nan)
    
    # Prepare data for Energy Score: We need full arrays (no NaNs). 
    # Impute clouds in real data with that pixel's summer mean.
    real_imputed = np.where(np.isnan(real_grid), np.broadcast_to(real_pixel_mean, real_grid.shape), real_grid)
    
    # Extract ONLY the valid land pixels into 1D vectors for Energy Score
    real_vectors = real_imputed[:, land_mask]
    ai_vectors = ai_grid[:, land_mask]
    copula_vectors = copula_grid[:, land_mask]

    print("--- 3. CALCULATING SPATIAL ENERGY SCORES ---")
    es_ai = calculate_energy_score(real_vectors, ai_vectors)
    es_copula = calculate_energy_score(real_vectors, copula_vectors)
    
    print(f"U-Net Energy Score:  {es_ai:.2f}")
    print(f"Copula Energy Score: {es_copula:.2f}")

    print("--- 4. CALCULATING EMPIRICAL SEMIVARIOGRAMS ---")
    dist, var_real = calculate_semivariogram(real_grid)
    _, var_ai = calculate_semivariogram(ai_masked)
    _, var_copula = calculate_semivariogram(copula_masked)

    print("--- 5. PLOTTING RESULTS ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PLOT 1: The Energy Score Bar Chart
    bars = axes[0].bar(['U-Net A.I.', 'Gaussian Copula'], [es_ai, es_copula], color=['purple', 'darkorange'])
    axes[0].set_title("Spatial Energy Score (Multivariate Distance)\n", fontsize=13)
    axes[0].set_ylabel("Energy Score")
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    # PLOT 2: The Empirical Semivariogram (The Spatial Correlation Proof)
    axes[1].plot(dist, var_real, marker='o', color='teal', linewidth=3, label='Real Observations')
    axes[1].plot(dist, var_ai, marker='s', color='purple', linewidth=2, linestyle='--', label='U-Net Generated')
    axes[1].plot(dist, var_copula, marker='^', color='darkorange', linewidth=2, linestyle=':', label='Copula Generated')
    
    axes[1].set_title("Spatial Correlation: Empirical Semivariogram", fontsize=13)
    axes[1].set_xlabel("Distance Between Pixels")
    
    # Adjust Y-label dynamically based on variable
    unit = "K²" if target_var == "lst" else "(m³/m³)²"
    axes[1].set_ylabel(f"Semivariance {unit}")
    
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=11)

    plt.suptitle(f"Spatial Evaluation: ({target_var.upper()})", fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_filename = f'spatial_correlation_metrics_{target_var}.png'
    plt.savefig(save_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_filename}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [lst|sm]")
        sys.exit(1)
        
    target_variable = sys.argv[1].lower()
    evaluate_spatial_metrics(target_variable)