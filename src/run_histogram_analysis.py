import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import wasserstein_distance
from split_utils import get_train_indices  # chronological split indices

def load_real_test_and_masks(var):
    """
    Load real test data (chronological) and create a land mask of pixels ever valid.
    """
    real_path = f'./data/generated/real_{var}_chrono.npy'
    real_data = np.load(real_path)  # Shape (n_test, 32, 32)
    # Land mask: pixels that have at least one valid observation across all test days
    land_mask = ~np.isnan(np.nanmean(real_data, axis=0))
    # Flatten all valid pixels across all days
    real_flat = real_data[:, land_mask].flatten()
    real_flat = real_flat[~np.isnan(real_flat)]  # Remove any residual NaNs
    return real_flat, land_mask

def load_ai_data(var, land_mask):
    """
    Load AI generated data, un‑normalise using training stats (chronological split).
    """
    ds = xr.open_dataset(f'./data/processed/aligned_{var}.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = summer[f'{"LST_PMW" if var=="lst" else "sm"}'].values[:, 0:32, 0:32]
    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    vmin, vmax = np.nanmin(train_data), np.nanmax(train_data)
    ds.close()

    gen_norm = np.load(f'./data/generated/ai_generated_{var}_3000days_epoch_100_masked_ocean.npy')
    gen_phys = ((gen_norm + 1) / 2) * (vmax - vmin) + vmin
    # Apply the same land mask (real valid pixels) and flatten
    gen_flat = gen_phys[:, land_mask].flatten()
    return gen_flat

def load_copula_data(var, land_mask):
    """
    Load copula generated data (chronological split).
    """
    gen_data = np.load(f'./data/generated/copula_gaussian_{var}_chrono.npy')
    gen_flat = gen_data[:, land_mask].flatten()
    return gen_flat

def plot_histogram_comparison(ax, real_flat, gen_flat, title, var_name, unit):
    """
    Create a single subplot histogram directly on the provided axis.
    """
    # Determine bins based on combined range
    combined_min = min(real_flat.min(), gen_flat.min())
    combined_max = max(real_flat.max(), gen_flat.max())
    bins = np.linspace(combined_min, combined_max, 75)

    ax.hist(real_flat, bins=bins, density=True, alpha=0.6, color='teal', label='Real Observations')
    
    # Use purple for Diffusion to match your other plots, orange for Copula
    gen_color = 'purple' if 'Diffusion' in title else 'darkorange'
    ax.hist(gen_flat, bins=bins, density=True, alpha=0.6, color=gen_color, label='Generated')
    
    # Compute metrics
    wd = wasserstein_distance(real_flat, gen_flat)
    mean_diff = np.mean(gen_flat) - np.mean(real_flat)
    
    ax.set_title(f"{title}\n{var_name} | Wasserstein = {wd:.3f} {unit} | Mean diff = {mean_diff:+.3f} {unit}", fontsize=12)
    ax.set_xlabel(f"{var_name} ({unit})", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

def main():
    print("--- LOADING REAL TEST DATA AND MASKS ---")
    real_lst, mask_lst = load_real_test_and_masks('lst')
    real_sm,  mask_sm  = load_real_test_and_masks('sm')

    print("--- LOADING AI DATA ---")
    ai_lst = load_ai_data('lst', mask_lst)
    ai_sm  = load_ai_data('sm',  mask_sm)

    print("--- LOADING COPULA DATA ---")
    cop_lst = load_copula_data('lst', mask_lst)
    cop_sm  = load_copula_data('sm',  mask_sm)

    print("--- GENERATING MASTER PLOT ---")
    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Row 1: Diffusion vs Real
    plot_histogram_comparison(axes[0, 0], real_lst, ai_lst, "Diffusion vs Real", "LST", "K")
    plot_histogram_comparison(axes[0, 1], real_sm, ai_sm, "Diffusion vs Real", "Soil Moisture", "m³/m³")

    # Row 2: Copula vs Real
    plot_histogram_comparison(axes[1, 0], real_lst, cop_lst, "Copula vs Real", "LST", "K")
    plot_histogram_comparison(axes[1, 1], real_sm, cop_sm, "Copula vs Real", "Soil Moisture", "m³/m³")

    # Add a global title and format spacing
    plt.suptitle("1D Marginal Analysis: Global Distribution Comparison", fontsize=18, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout to make room for the suptitle
    
    save_name = 'global_histograms.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved highly condensed master figure to '{save_name}'")

if __name__ == "__main__":
    main()