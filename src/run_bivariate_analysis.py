import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon

def calculate_2d_jsd(real_x, real_y, ai_x, ai_y, bins=50):
    """
    Calculates the Jensen-Shannon Divergence between two 2D distributions.
    """

    # Find the global min/max boundaries so both grids match perfectly
    x_min = min(np.min(real_x), np.min(ai_x))
    x_max = max(np.max(real_x), np.max(ai_x))
    y_min = min(np.min(real_y), np.min(ai_y))
    y_max = max(np.max(real_y), np.max(ai_y))
    
    bounds = [[x_min, x_max], [y_min, y_max]]
    
    # Create 2D histograms (Probability Density)
    real_hist, _, _ = np.histogram2d(real_x, real_y, bins=bins, range=bounds, density=True)
    ai_hist, _, _   = np.histogram2d(ai_x, ai_y, bins=bins, range=bounds, density=True)
    
    # Flatten the grids into 1D probability arrays
    real_prob = real_hist.flatten()
    ai_prob = ai_hist.flatten()
    
    # Add a tiny epsilon to avoid division by zero
    epsilon = 1e-10
    real_prob = real_prob + epsilon
    ai_prob = ai_prob + epsilon
    
    # Normalize so they sum to exactly 1.0
    real_prob /= np.sum(real_prob)
    ai_prob /= np.sum(ai_prob)
    
    # Calculate JSD (Distance)
    js_distance = jensenshannon(real_prob, ai_prob)
    
    return js_distance

def calculate_2d_wasserstein(real_x, real_y, ai_x, ai_y, max_samples=1000):
    """
    Calculates the 2D Wasserstein (Earth Mover's) Distance using random subsampling 
    to prevent computational freezing on large matrices.
    """
    # Stack the 1D arrays into arrays of 2D coordinates: shape (N, 2)
    real_coords = np.column_stack((real_x, real_y))
    ai_coords = np.column_stack((ai_x, ai_y))
    
    # --- Random Subsampling ---
    np.random.seed(42) # Optionally lock the seed for consistent tests
    
    if len(real_coords) > max_samples:
        idx_real = np.random.choice(len(real_coords), max_samples, replace=False)
        real_coords = real_coords[idx_real]
        
    if len(ai_coords) > max_samples:
        idx_ai = np.random.choice(len(ai_coords), max_samples, replace=False)
        ai_coords = ai_coords[idx_ai]
    
    print("Calculating Wasserstein Distance for {0} real coordinates and {1} A.I. coordinates".format(len(real_coords), len(ai_coords)))
    # Calculate the multi-dimensional Earth Mover's Distance on the manageable subset
    w_dist = wasserstein_distance_nd(real_coords, ai_coords)
    
    return w_dist

def plot_bivariate_dependencies():
    """
    Runs evaluation tests for the bivariant dependencies.
    """

    # --- LOAD REAL DATA ---
    print("Loading Real Data...")
    lst_full = xr.open_dataset('./data/processed/aligned_lst.nc')
    sm_full = xr.open_dataset('./data/processed/aligned_sm.nc')

    # Filter the real data to match the AI's training data (May - September)
    valid_months = [5, 6, 7, 8, 9]
    lst_summer = lst_full.sel(time=lst_full['time'].dt.month.isin(valid_months))
    sm_summer = sm_full.sel(time=sm_full['time'].dt.month.isin(valid_months))

    real_lst = lst_summer['LST_PMW'].values
    real_sm = sm_summer['sm'].values

    # Find the global min and max used during training (now correctly bounded by summer!)
    lst_min, lst_max = np.nanmin(real_lst), np.nanmax(real_lst)
    sm_min, sm_max = np.nanmin(real_sm), np.nanmax(real_sm)

    # --- LOAD GENERATED DATA ---
    print("Loading AI data...")
    gen_lst_norm = np.load('./data/generated/ai_generated_lst_3000days.npy')
    gen_sm_norm = np.load('./data/generated/ai_generated_sm_3000days.npy')

    # Un-normalize the output of the model
    # Formula: X_real = ((X_norm + 1) / 2) * (max - min) + min
    gen_lst = ((gen_lst_norm + 1) / 2) * (lst_max - lst_min) + lst_min
    gen_sm = ((gen_sm_norm + 1) / 2) * (sm_max - sm_min) + sm_min

    # --- CHOOSE PIXELS ---
    # row/col indices (indices start at 0)
    row_A, col_A = 11, 17  # Antwerp
    row_B, col_B = 12, 17  # Antwerp-adjacent

    # --- EXTRACT TIME SERIES ---
    # Real Data
    real_lst_A = real_lst[:, row_A, col_A]
    real_sm_A  = real_sm[:, row_A, col_A]
    real_lst_B = real_lst[:, row_B, col_B]

    # STRICT 9D SPATIAL MASK (To match Vine Copula comparison)
    # Extract the same 3x3 grid the Copula used
    lst_grid = real_lst[:, row_A-1:row_A+2, col_A-1:col_A+2]
    lst_flat = lst_grid.reshape(lst_grid.shape[0], 9)
    
    # A day is only valid if ALL 9 pixels are clear
    valid_spatial = ~np.isnan(lst_flat).any(axis=1)
    
    # Cross-variable can remain the same 2-variable mask
    valid_A = ~np.isnan(real_lst_A) & ~np.isnan(real_sm_A)

    real_lst_A_clean = real_lst_A[valid_A]
    real_sm_A_clean  = real_sm_A[valid_A]
    
    real_lst_A_spat = real_lst_A[valid_spatial]
    real_lst_B_spat = real_lst_B[valid_spatial]

    # AI Data (No NaNs)
    gen_lst_A = gen_lst[:, row_A, col_A]
    gen_sm_A  = gen_sm[:, row_A, col_A]
    gen_lst_B = gen_lst[:, row_B, col_B]

    # --- CALCULATE METRICS (Jensen-Shannon Divergence + Wasserstein Distance) ---
    # Spearman is better here because it doesn't assume a strict linear relationship
    real_cross_corr, _ = spearmanr(real_sm_A_clean, real_lst_A_clean)
    ai_cross_corr, _   = spearmanr(gen_sm_A, gen_lst_A)
    jsd_cross = calculate_2d_jsd(real_lst_A_clean, real_sm_A_clean, gen_lst_A, gen_sm_A)
    w_cross = calculate_2d_wasserstein(real_lst_A_clean, real_sm_A_clean, gen_lst_A, gen_sm_A)

    real_spat_corr, _ = spearmanr(real_lst_A_spat, real_lst_B_spat)
    ai_spat_corr, _   = spearmanr(gen_lst_A, gen_lst_B)
    jsd_spat = calculate_2d_jsd(real_lst_A_spat, real_lst_B_spat, gen_lst_A, gen_lst_B)
    w_spat = calculate_2d_wasserstein(real_lst_A_spat, real_lst_B_spat, gen_lst_A, gen_lst_B)

    print(f"Cross-Variable (Antwerp LST vs SM) | Real rho: {real_cross_corr:.3f} | AI rho: {ai_cross_corr:.3f} | JS-Divergence: {jsd_cross:.3f} | Wasserstein Distance: {w_cross:.3f}")
    print(f"Spatial (Antwerp vs Inland LST)    | Real rho: {real_spat_corr:.3f} | AI rho: {ai_spat_corr:.3f} | JS-Divergence: {jsd_spat:.3f} | Wasserstein Distance: {w_spat:.3f}")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PLOT 1: Cross-Variable Dependence (SM vs LST at Pixel A)
    axes[0].set_title(f"Cross-Variable Dependence Analysis (Pixel A)\nReal ρ={real_cross_corr:.2f} | AI ρ={ai_cross_corr:.2f} | JS-Divergence: {jsd_cross:.3f} | Wasserstein Distance: {w_cross:.3f}", fontsize=14)
    # Plot Real Data as filled contours (Teal)
    sns.kdeplot(x=real_sm_A_clean, y=real_lst_A_clean, ax=axes[0], 
                cmap="mako", fill=True, alpha=0.5, thresh=0.05, label='Real Data')
    # Plot AI Data as outline contours (Purple)
    sns.kdeplot(x=gen_sm_A, y=gen_lst_A, ax=axes[0], 
                color="purple", linewidths=2, levels=5, thresh=0.05, label='AI Data')
    axes[0].set_xlabel("Volumetric Soil Moisture (m³/m³)")
    axes[0].set_ylabel("LST (K)")
    axes[0].grid(alpha=0.3)

    # PLOT 2: Spatial Dependence (LST at Pixel A vs LST at Pixel B)
    axes[1].set_title(f"Spatial Dependence Analysis (LST Pixel A vs B)\nReal ρ={real_spat_corr:.2f} | AI ρ={ai_spat_corr:.2f} | JS-Divergence: {jsd_spat:.3f} | Wasserstein Distance: {w_spat:.3f}", fontsize=14)
    # Plot Real Data (Teal)
    sns.kdeplot(x=real_lst_A_spat, y=real_lst_B_spat, ax=axes[1], 
                cmap="mako", fill=True, alpha=0.5, thresh=0.05)
    # Plot AI Data (Purple)
    sns.kdeplot(x=gen_lst_A, y=gen_lst_B, ax=axes[1], 
                color="purple", linewidths=2, levels=5, thresh=0.05)
    axes[1].set_xlabel("LST at Pixel A (Antwerp) (K)")
    axes[1].set_ylabel("LST at Pixel B (Inland) (K)")
    axes[1].grid(alpha=0.3)

    # Add a custom legend
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    real_patch = mpatches.Patch(color='teal', alpha=0.5, label='Real Data (40 Years)')
    ai_line = mlines.Line2D([], [], color='purple', linewidth=2, label='AI Generated')
    fig.legend(handles=[real_patch, ai_line], loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig('bivariate_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved 'bivariate_analysis.png'")

if __name__ == "__main__":
    plot_bivariate_dependencies()