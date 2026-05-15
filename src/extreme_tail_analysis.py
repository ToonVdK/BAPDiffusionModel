# extreme_tail_analysis.py  (pixel‑wise exceedance‑fraction maps)
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ------------------------------------------------------------
# SPLIT CONFIG (must match your training split)
TEST_STEP = 5   # change to 0 for chronological split, or import from split_utils

def get_train_indices():
    """Training indices for the split used in model training."""
    ds = xr.open_dataset('./data/processed/aligned_lst.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    n_total = len(summer['time'])
    ds.close()
    test_idx = np.arange(0, n_total, TEST_STEP)   # systematic
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    return train_idx

# ------------------------------------------------------------
def compute_pixel_percentiles(nc_path, nc_var, train_idx, lower_p=10, upper_p=90):
    """
    Returns per-pixel lower and upper percentile thresholds (using training data).
    Shape of each: (32, 32)
    """
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = summer[nc_var].values[:, 0:32, 0:32]
    ds.close()

    train_data = full_data[train_idx]
    # Compute percentiles ignoring NaN (ocean returns NaN)
    lower = np.nanpercentile(train_data, lower_p, axis=0)
    upper = np.nanpercentile(train_data, upper_p, axis=0)
    return lower, upper

def fraction_exceed(data, threshold, direction='upper'):
    """
    Fraction of days where value exceeds (upper) or falls below (lower) threshold.
    data: (T, 32, 32)
    threshold: (32, 32)
    """
    if direction == 'upper':
        exceed = data > threshold   # (T,32,32)
    else:
        exceed = data < threshold
    return np.nanmean(exceed, axis=0)   # fraction per pixel

# ------------------------------------------------------------
def tail_analysis(target_var, test_step=TEST_STEP):
    if target_var == "lst":
        nc_path   = './data/processed/aligned_lst.nc'
        nc_var    = 'LST_PMW'
        ai_path   = './data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy'
        copula_path = f'./data/generated/copula_gaussian_lst_oos_test_step{test_step}.npy'
        real_test_path = f'./data/generated/real_lst_oos_test_step{test_step}.npy'
        unit = "K"
        title_var = "LST"
    elif target_var == "sm":
        nc_path   = './data/processed/aligned_sm.nc'
        nc_var    = 'sm'
        ai_path   = './data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy'
        copula_path = f'./data/generated/copula_gaussian_sm_oos_test_step{test_step}.npy'
        real_test_path = f'./data/generated/real_sm_oos_test_step{test_step}.npy'
        unit = "m³/m³"
        title_var = "Soil Moisture"
    else:
        print("Error: target must be 'lst' or 'sm'")
        sys.exit(1)

    print(f"--- 1. LOADING & COMPUTING THRESHOLDS ({title_var}) ---")
    train_idx = get_train_indices()
    lower_thresh, upper_thresh = compute_pixel_percentiles(nc_path, nc_var, train_idx,
                                                           lower_p=10, upper_p=90)

    # Coordinates for maps
    ds = xr.open_dataset(nc_path)
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds.close()

    # --- Load real test, AI, copula ---
    real_test = np.load(real_test_path)      # (T_test, 32, 32)
    ai_norm = np.load(ai_path)
    # Un‑normalise AI
    train_ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    summer_ds = train_ds.sel(time=train_ds['time'].dt.month.isin(valid_months))
    full_data = summer_ds[nc_var].values[:, 0:32, 0:32]
    train_ds.close()
    train_data = full_data[train_idx]
    val_min, val_max = np.nanmin(train_data), np.nanmax(train_data)
    ai_grid = ((ai_norm + 1) / 2) * (val_max - val_min) + val_min

    copula_grid = np.load(copula_path)       # (T_copula, 32, 32)

    print(f"Real test days: {real_test.shape[0]}, AI days: {ai_grid.shape[0]}, Copula days: {copula_grid.shape[0]}")

    # --- Land mask (from real test mean) ---
    real_mean = np.nanmean(real_test, axis=0)
    land_mask = ~np.isnan(real_mean)

    # --- Exceedance fractions for upper tail ---
    real_upper_frac = fraction_exceed(real_test, upper_thresh, 'upper')
    ai_upper_frac   = fraction_exceed(ai_grid, upper_thresh, 'upper')
    copula_upper_frac = fraction_exceed(copula_grid, upper_thresh, 'upper')

    # --- Exceedance fractions for lower tail ---
    real_lower_frac = fraction_exceed(real_test, lower_thresh, 'lower')
    ai_lower_frac   = fraction_exceed(ai_grid, lower_thresh, 'lower')
    copula_lower_frac = fraction_exceed(copula_grid, lower_thresh, 'lower')

    # --- Ratio maps (generated / real, clipped for visualisation) ---
    # Use np.where to avoid division by zero; real fraction may be 0 if no extremes occurred.
    eps = 1e-6
    ratio_ai_upper = np.where(land_mask, ai_upper_frac / (real_upper_frac + eps), np.nan)
    ratio_cop_upper = np.where(land_mask, copula_upper_frac / (real_upper_frac + eps), np.nan)
    ratio_ai_lower = np.where(land_mask, ai_lower_frac / (real_lower_frac + eps), np.nan)
    ratio_cop_lower = np.where(land_mask, copula_lower_frac / (real_lower_frac + eps), np.nan)

    # Set reasonable colour limits: ratio = 1 perfect, 0 means never generated, >2 very over‑represented.
    vmin, vmax = 0.0, 2.0

    # --- Metrics (MAE of fractions) ---
    mae_ai_upper = np.nanmean(np.abs(ai_upper_frac - real_upper_frac))
    mae_cop_upper = np.nanmean(np.abs(copula_upper_frac - real_upper_frac))
    mae_ai_lower = np.nanmean(np.abs(ai_lower_frac - real_lower_frac))
    mae_cop_lower = np.nanmean(np.abs(copula_lower_frac - real_lower_frac))
    print(f"Upper tail MAE: AI={mae_ai_upper:.4f}, Copula={mae_cop_upper:.4f}")
    print(f"Lower tail MAE: AI={mae_ai_lower:.4f}, Copula={mae_cop_lower:.4f}")

    # --- Plotting ---
    for tail, ratio_ai, ratio_cop, mae_ai, mae_cop in [
        ('upper', ratio_ai_upper, ratio_cop_upper, mae_ai_upper, mae_cop_upper),
        ('lower', ratio_ai_lower, ratio_cop_lower, mae_ai_lower, mae_cop_lower)
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        # AI map
        im0 = axes[0].imshow(ratio_ai, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                             origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
        axes[0].set_title(f"U‑Net {title_var} {tail} tail\n(MAE={mae_ai:.3f})", fontsize=14)
        axes[0].add_feature(cfeature.COASTLINE, linewidth=1)
        axes[0].add_feature(cfeature.BORDERS, linewidth=1)
        axes[0].set_axis_off()

        # Copula map
        im1 = axes[1].imshow(ratio_cop, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                             origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
        axes[1].set_title(f"Copula {title_var} {tail} tail\n(MAE={mae_cop:.3f})", fontsize=14)
        axes[1].add_feature(cfeature.COASTLINE, linewidth=1)
        axes[1].add_feature(cfeature.BORDERS, linewidth=1)
        axes[1].set_axis_off()

        fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04,
                     label='Generated / Real fraction (1 = perfect)')
        plt.suptitle(f"Extreme {tail} tail representation – {title_var}", fontsize=16, y=1.02)
        plt.savefig(f'extreme_{tail}_tail_{target_var}.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        print(f"Saved 'extreme_{tail}_tail_{target_var}.png'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extreme_tail_analysis.py [lst|sm]")
        sys.exit(1)
    var = sys.argv[1].lower()
    tail_analysis(var, test_step=TEST_STEP)