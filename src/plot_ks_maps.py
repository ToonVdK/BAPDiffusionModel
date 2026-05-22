import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp
from split_utils import get_train_indices

def compute_ks_map(real_data, gen_data, land_mask):
    """
    Pixel-wise KS D-statistic between real and generated time series.
    """
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
    print("--- LOADING REAL DATA & TRAINING STATS FOR AI ---")
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

    print("--- LOADING REAL TEST SETS ---")
    real_lst = np.load('./data/generated/real_lst_chrono.npy')
    real_sm  = np.load('./data/generated/real_sm_chrono.npy')

    print("--- LOADING AI DATA ---")
    ai_norm_lst = np.load('./data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy')
    ai_norm_sm  = np.load('./data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy')
    ai_lst = ((ai_norm_lst + 1) / 2) * (lst_max - lst_min) + lst_min
    ai_sm  = ((ai_norm_sm + 1) / 2) * (sm_max  - sm_min ) + sm_min

    print("--- LOADING COPULA DATA ---")
    copula_lst = np.load('./data/generated/copula_gaussian_lst_chrono.npy')
    copula_sm  = np.load('./data/generated/copula_gaussian_sm_chrono.npy')

    # LST mask for LST data, SM mask for SM data
    land_mask_lst = ~np.isnan(np.nanmean(real_lst, axis=0))
    land_mask_sm = ~np.isnan(np.nanmean(real_sm, axis=0))

    # Coordinates for plotting
    lat_name = 'lat' if 'lat' in ds_lst.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_lst.coords else 'longitude'
    lats = ds_lst[lat_name].values[0:32]
    lons = ds_lst[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds_lst.close()
    ds_sm.close()

    print("--- COMPUTING KS MAPS & GENERATING MASTER PLOT ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Pass the specific mask for each variable into the loop list
    variables = [
        ("LST", real_lst, ai_lst, copula_lst, 0, land_mask_lst),
        ("SM",  real_sm,  ai_sm,  copula_sm,  1, land_mask_sm)
    ]

    for var, real_data, ai_data, copula_data, row_idx, var_mask in variables:
        print(f"  Computing KS for {var}...")
        
        # Use the specific var_mask here instead of the combined one
        ks_ai = compute_ks_map(real_data, ai_data, var_mask)
        ks_copula = compute_ks_map(real_data, copula_data, var_mask)

        mean_ks_ai = np.nanmean(ks_ai)
        mean_ks_copula = np.nanmean(ks_copula)
        
        # Calculate shared color limits for this specific row
        vmax = max(np.nanmax(ks_ai), np.nanmax(ks_copula))

        # AI KS map (Left Column)
        im0 = axes[row_idx, 0].imshow(ks_ai, cmap='magma', vmin=0.0, vmax=vmax,
                                      origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
        axes[row_idx, 0].add_feature(cfeature.COASTLINE, linewidth=1)
        axes[row_idx, 0].add_feature(cfeature.BORDERS, linewidth=1)
        axes[row_idx, 0].set_axis_off()
        axes[row_idx, 0].set_title(f"Mean Error = {mean_ks_ai:.4f}", fontsize=12)

        # Copula KS map (Right Column)
        im1 = axes[row_idx, 1].imshow(ks_copula, cmap='magma', vmin=0.0, vmax=vmax,
                                      origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
        axes[row_idx, 1].add_feature(cfeature.COASTLINE, linewidth=1)
        axes[row_idx, 1].add_feature(cfeature.BORDERS, linewidth=1)
        axes[row_idx, 1].set_axis_off()
        axes[row_idx, 1].set_title(f"Mean Error = {mean_ks_copula:.4f}", fontsize=12)

        # Row Titles on the far left
        axes[row_idx, 0].text(-0.05, 0.5, f"{var} KS Statistic", va='center', ha='right', rotation=90, 
                              transform=axes[row_idx, 0].transAxes, fontsize=14, fontweight='bold')

        # One colorbar per row
        cbar = fig.colorbar(im0, ax=axes[row_idx, :].ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_label("KS D-Statistic", fontsize=11)

    # Column Titles at the very top
    axes[0, 0].text(0.5, 1.1, "Diffusion Generated", transform=axes[0, 0].transAxes, 
                    ha='center', va='center', fontsize=15)
    axes[0, 1].text(0.5, 1.1, "Gaussian Copula", transform=axes[0, 1].transAxes, 
                    ha='center', va='center', fontsize=15)

    plt.suptitle("1D Marginal Analysis: Pixel-wise KS D-Statistic", fontsize=18, y=0.96)
    
    save_name = 'ks_maps.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved highly condensed master figure to '{save_name}'")

if __name__ == "__main__":
    main()