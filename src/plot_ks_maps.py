# plot_ks_maps.py – side‑by‑side KS maps for AI and Copula (chronological split)
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp
from split_utils import get_train_indices

def compute_ks_map(real_data, gen_data, land_mask):
    """Pixel‑wise KS D‑statistic between real and generated time series."""
    ks_map = np.full((32, 32), np.nan)
    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                r_ts = real_data[:, i, j]
                g_ts = gen_data[:, i, j]
                # Remove NaNs (real may have NaNs; generated should have none)
                r_clean = r_ts[~np.isnan(r_ts)]
                g_clean = g_ts[~np.isnan(g_ts)]
                if len(r_clean) > 0 and len(g_clean) > 0:
                    ks_map[i, j] = ks_2samp(r_clean, g_clean).statistic
    return ks_map

def main():
    print("--- 1. LOADING REAL DATA & TRAINING STATS FOR AI ---")
    # For AI we need training min/max to un‑normalise
    ds_lst = xr.open_dataset('./data/processed/aligned_lst.nc')
    ds_sm  = xr.open_dataset('./data/processed/aligned_sm.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer_lst = ds_lst.sel(time=ds_lst['time'].dt.month.isin(valid_months))
    summer_sm  = ds_sm.sel(time=ds_sm['time'].dt.month.isin(valid_months))
    full_lst = summer_lst['LST_PMW'].values[:, 0:32, 0:32]
    full_sm  = summer_sm['sm'].values[:, 0:32, 0:32]
    train_idx = get_train_indices()
    train_lst = full_lst[train_idx]
    train_sm  = full_sm[train_idx]
    lst_min, lst_max = np.nanmin(train_lst), np.nanmax(train_lst)
    sm_min,  sm_max  = np.nanmin(train_sm),  np.nanmax(train_sm)

    # Real test sets (chronological)
    print("--- 2. LOADING REAL TEST SETS ---")
    real_lst = np.load('./data/generated/real_lst_chrono.npy')
    real_sm  = np.load('./data/generated/real_sm_chrono.npy')

    # AI generated data (un‑normalise using training stats)
    print("--- 3. LOADING AI DATA ---")
    ai_norm_lst = np.load('./data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy')
    ai_norm_sm  = np.load('./data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy')
    ai_lst = ((ai_norm_lst + 1) / 2) * (lst_max - lst_min) + lst_min
    ai_sm  = ((ai_norm_sm + 1) / 2) * (sm_max  - sm_min ) + sm_min

    # Copula generated data
    print("--- 4. LOADING COPULA DATA ---")
    copula_lst = np.load('./data/generated/copula_gaussian_lst_chrono.npy')
    copula_sm  = np.load('./data/generated/copula_gaussian_sm_chrono.npy')

    # Common land mask (pixels with at least some valid real data)
    land_mask = ~np.isnan(np.nanmean(real_lst, axis=0)) & ~np.isnan(np.nanmean(real_sm, axis=0))

    # Coordinates for plotting
    lat_name = 'lat' if 'lat' in ds_lst.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_lst.coords else 'longitude'
    lats = ds_lst[lat_name].values[0:32]
    lons = ds_lst[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds_lst.close()
    ds_sm.close()

    print("--- 5. COMPUTING KS MAPS ---")
    # For each variable we compute KS maps separately, but we'll plot them in two rows? 
    # The user wants "1 figure with 2 plots each portraying the KS statistics for A.I. and copula respectively"
    # That suggests we should pick one variable (e.g., LST) or combine? To keep it simple, we'll produce two separate figures:
    # one for LST and one for SM. Each figure will have two subplots (AI left, Copula right).
    # Alternatively, we can do 2x2 grid: (AI LST, Copula LST) and (AI SM, Copula SM). But the user said "2 plots" – likely one pair per variable.
    # We'll produce two figures: ks_maps_lst_chrono.png and ks_maps_sm_chrono.png.
    # Each figure contains two side-by-side maps: AI KS and Copula KS, with mean KS in titles.

    for var, real_data, ai_data, copula_data, title in [
        ("LST", real_lst, ai_lst, copula_lst, "Land Surface Temperature"),
        ("SM",  real_sm,  ai_sm,  copula_sm,  "Soil Moisture")
    ]:
        print(f"  Computing KS for {var}...")
        ks_ai = compute_ks_map(real_data, ai_data, land_mask)
        ks_copula = compute_ks_map(real_data, copula_data, land_mask)

        mean_ks_ai = np.nanmean(ks_ai)
        mean_ks_copula = np.nanmean(ks_copula)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # AI KS map
        im0 = axes[0].imshow(ks_ai, cmap='magma', vmin=0.0, vmax=max(np.nanmax(ks_ai), np.nanmax(ks_copula)),
                             origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
        axes[0].set_title(f"U‑Net {var} KS D‑Statistic\n(Mean = {mean_ks_ai:.4f})", fontsize=13)
        axes[0].add_feature(cfeature.COASTLINE, linewidth=1)
        axes[0].add_feature(cfeature.BORDERS, linewidth=1)
        axes[0].set_axis_off()

        # Copula KS map
        im1 = axes[1].imshow(ks_copula, cmap='magma', vmin=0.0, vmax=max(np.nanmax(ks_ai), np.nanmax(ks_copula)),
                             origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
        axes[1].set_title(f"2048D Gaussian Copula {var} KS D‑Statistic\n(Mean = {mean_ks_copula:.4f})", fontsize=13)
        axes[1].add_feature(cfeature.COASTLINE, linewidth=1)
        axes[1].add_feature(cfeature.BORDERS, linewidth=1)
        axes[1].set_axis_off()

        cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_label("KS D-Statistic", fontsize=12)

        plt.suptitle(f"1D Distribution Comparison: {title}", fontsize=16, y=1.02)
        save_name = f'ks_maps_chrono_{var.lower()}.png'
        plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved '{save_name}' (AI mean KS = {mean_ks_ai:.4f}, Copula mean KS = {mean_ks_copula:.4f})")

if __name__ == "__main__":
    main()