# plot_global_histograms.py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import wasserstein_distance
from split_utils import get_train_indices  # chronological split indices

def load_real_test_and_masks(var):
    """Load real test data (chronological) and create a land mask of pixels ever valid."""
    real_path = f'./data/generated/real_{var}_chrono.npy'
    real_data = np.load(real_path)          # shape (n_test, 32, 32)
    # land mask: pixels that have at least one valid observation across all test days
    land_mask = ~np.isnan(np.nanmean(real_data, axis=0))
    # flatten all valid pixels across all days
    real_flat = real_data[:, land_mask].flatten()
    real_flat = real_flat[~np.isnan(real_flat)]  # remove any residual NaNs
    return real_flat, land_mask

def load_ai_data(var, land_mask):
    """Load AI generated data, un‑normalise using training stats (chronological split)."""
    ds = xr.open_dataset(f'./data/processed/aligned_{var}.nc')
    valid_months = [5,6,7,8,9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = summer[f'{"LST_PMW" if var=="lst" else "sm"}'].values[:, 0:32, 0:32]
    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    vmin, vmax = np.nanmin(train_data), np.nanmax(train_data)
    ds.close()

    gen_norm = np.load(f'./data/generated/ai_generated_{var}_3000days_epoch_100_masked_ocean.npy')
    gen_phys = ((gen_norm + 1) / 2) * (vmax - vmin) + vmin
    # apply the same land mask (real valid pixels) and flatten
    gen_flat = gen_phys[:, land_mask].flatten()
    return gen_flat

def load_copula_data(var, land_mask):
    """Load copula generated data (chronological split)."""
    gen_data = np.load(f'./data/generated/copula_gaussian_{var}_chrono.npy')
    gen_flat = gen_data[:, land_mask].flatten()
    return gen_flat

def plot_histogram_comparison(real_flat, gen_flat, title, var_name, unit, save_name):
    """Create a single subplot histogram."""
    # Determine bins based on combined range
    combined_min = min(real_flat.min(), gen_flat.min())
    combined_max = max(real_flat.max(), gen_flat.max())
    bins = np.linspace(combined_min, combined_max, 75)

    plt.hist(real_flat, bins=bins, density=True, alpha=0.6, color='teal', label='Real')
    plt.hist(gen_flat, bins=bins, density=True, alpha=0.6, color='darkorange', label='Generated')
    
    # Compute metrics
    wd = wasserstein_distance(real_flat, gen_flat)
    mean_diff = np.mean(gen_flat) - np.mean(real_flat)
    
    plt.title(f"{title}\n{var_name} | Wasserstein = {wd:.3f} {unit} | Mean diff = {mean_diff:+.3f} {unit}", fontsize=11)
    plt.xlabel(f"{var_name} ({unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

def main():
    print("Loading real test data and land masks...")
    real_lst, mask_lst = load_real_test_and_masks('lst')
    real_sm,  mask_sm  = load_real_test_and_masks('sm')

    print("Loading AI data...")
    ai_lst = load_ai_data('lst', mask_lst)
    ai_sm  = load_ai_data('sm',  mask_sm)

    print("Loading Copula data...")
    cop_lst = load_copula_data('lst', mask_lst)
    cop_sm  = load_copula_data('sm',  mask_sm)

    # Figure 1: AI vs Real
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.sca(axes[0])
    plot_histogram_comparison(real_lst, ai_lst, "AI vs Real", "LST", "K", save_name=None)
    plt.sca(axes[1])
    plot_histogram_comparison(real_sm, ai_sm, "AI vs Real", "Soil Moisture", "m³/m³", save_name=None)
    plt.suptitle("Global Distribution Comparison – AI Generated vs Real", fontsize=14)
    plt.tight_layout()
    plt.savefig('global_histogram_ai.png', dpi=150, bbox_inches='tight')
    print("Saved 'global_histogram_ai.png'")

    # Figure 2: Copula vs Real
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.sca(axes[0])
    plot_histogram_comparison(real_lst, cop_lst, "Copula vs Real", "LST", "K", save_name=None)
    plt.sca(axes[1])
    plot_histogram_comparison(real_sm, cop_sm, "Copula vs Real", "Soil Moisture", "m³/m³", save_name=None)
    plt.suptitle("Global Distribution Comparison – Copula Generated vs Real", fontsize=14)
    plt.tight_layout()
    plt.savefig('global_histogram_copula.png', dpi=150, bbox_inches='tight')
    print("Saved 'global_histogram_copula.png'")

if __name__ == "__main__":
    main()