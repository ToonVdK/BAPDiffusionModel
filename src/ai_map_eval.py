import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp

# ----------------------------------------------------------------------
# Helper: reproduce the exact systematic split used for copula & training
TEST_STEP = 5

def get_train_indices():
    """Returns the training indices (complement of every TEST_STEP‑th day)."""
    ds = xr.open_dataset('./data/processed/aligned_lst.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    n_total = len(summer['time'])
    ds.close()
    test_idx = np.arange(0, n_total, TEST_STEP)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    return train_idx

# ----------------------------------------------------------------------
def evaluate_ai_climatology(target_var, test_step=TEST_STEP):
    # --- 0. CONFIGURATION ---
    if target_var == "lst":
        nc_path   = './data/processed/aligned_lst.nc'
        nc_var    = 'LST_PMW'
        ai_path   = './data/generated/ai_generated_lst_3000days_epoch_300.npy'   # generated with retrained model
        real_test_npy = f'./data/generated/real_lst_oos_test_step{test_step}.npy'
        unit      = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
    elif target_var == "sm":
        nc_path   = './data/processed/aligned_sm.nc'
        nc_var    = 'sm'
        ai_path   = './data/generated/ai_generated_sm_3000days_epoch_300.npy'
        real_test_npy = f'./data/generated/real_sm_oos_test_step{test_step}.npy'
        unit      = "m³/m³"
        title_name = "Soil Moisture"
        cmap_mean = 'YlGnBu'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    print(f"--- 1. LOADING REAL DATA ({title_name}) & COMPUTING TRAINING STATS ---")
    # Open the full NetCDF only to get coordinates and training min/max
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = summer[nc_var].values[:, 0:32, 0:32]

    # Extract training portion (the same days the U‑Net saw)
    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    lst_min, lst_max = np.nanmin(train_data), np.nanmax(train_data)

    # Coordinates for plotting
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds.close()

    print(f"Training {title_name} range: {lst_min:.2f} – {lst_max:.2f}")

    print(f"--- 2. LOADING AI GENERATED DATA ({title_name}) ---")
    gen_norm = np.load(ai_path)
    # Un‑normalize using the TRAINING min/max
    ai_grid = ((gen_norm + 1) / 2) * (lst_max - lst_min) + lst_min

    print(f"--- 3. LOADING REAL TEST SET (Systematic Step {test_step}) ---")
    real_test = np.load(real_test_npy)   # shape (n_test, 32, 32)

    print("--- 4. CALCULATING CLIMATOLOGY & KS STATISTIC ---")
    real_mean = np.nanmean(real_test, axis=0)
    ai_mean   = np.mean(ai_grid, axis=0)

    real_std = np.nanstd(real_test, axis=0)
    ai_std   = np.std(ai_grid, axis=0)

    land_mask = ~np.isnan(real_mean)

    # Pixel‑wise KS D‑Statistic (test set vs. AI)
    print("Calculating pixel‑wise KS D‑Statistic...")
    ks_map = np.full((32, 32), np.nan)
    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                r_ts = real_test[:, i, j]
                a_ts = ai_grid[:, i, j]
                r_clean = r_ts[~np.isnan(r_ts)]
                a_clean = a_ts[~np.isnan(a_ts)]
                if len(r_clean) > 0 and len(a_clean) > 0:
                    ks_map[i, j] = ks_2samp(r_clean, a_clean).statistic

    print("--- 5. PLOTTING ---")
    ai_mean_masked = np.where(land_mask, ai_mean, np.nan)
    ai_std_masked  = np.where(land_mask, ai_std, np.nan)

    mean_vmin = min(np.nanmin(real_mean), np.nanmin(ai_mean_masked))
    mean_vmax = max(np.nanmax(real_mean), np.nanmax(ai_mean_masked))
    std_vmin  = min(np.nanmin(real_std),  np.nanmin(ai_std_masked))
    std_vmax  = max(np.nanmax(real_std),  np.nanmax(ai_std_masked))

    fig, axes = plt.subplots(3, 2, figsize=(12, 15),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Row 1: Mean
    axes[0,0].imshow(real_mean, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,0].set_title(f"Real Average {title_name} (Test Set, step={test_step})")
    axes[0,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,0].set_axis_off()

    axes[0,1].imshow(ai_mean_masked, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,1].set_title(f"U‑Net Average {title_name} (Out‑of‑Sample)")
    axes[0,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,1].set_axis_off()
    fig.colorbar(axes[0,0].images[0], ax=axes[0,:], label=f"{title_name} ({unit})", shrink=0.8)

    # Row 2: Standard Deviation
    axes[1,0].imshow(real_std, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,0].set_title(f"Real {title_name} Standard Deviation")
    axes[1,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,0].set_axis_off()

    axes[1,1].imshow(ai_std_masked, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,1].set_title(f"U‑Net {title_name} Standard Deviation")
    axes[1,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,1].set_axis_off()
    fig.colorbar(axes[1,0].images[0], ax=axes[1,:], label=f"Std Dev ({unit})", shrink=0.8)

    # Row 3: KS D‑Statistic
    axes[2,0].imshow(ks_map, cmap='magma', vmin=0.0, vmax=np.nanmax(ks_map),
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[2,0].set_title(f"KS D‑Statistic Error\n(Mean Error: {np.nanmean(ks_map):.4f})")
    axes[2,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[2,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[2,0].set_axis_off()
    axes[2,1].set_visible(False)
    fig.colorbar(axes[2,0].images[0], ax=axes[2,0], label="KS Statistic", shrink=0.8, pad=0.04)

    plt.suptitle(
        f"U‑Net Out‑Of‑Sample Evaluation (Systematic Step={test_step}) – {title_name}",
        fontsize=16, y=0.92
    )
    save_name = f'ai_oos_systematic_step{test_step}_{target_var}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_name}'")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_map_eval.py [lst|sm] [test_step]")
        sys.exit(1)
    var = sys.argv[1].lower()
    step = int(sys.argv[2]) if len(sys.argv) > 2 else TEST_STEP
    evaluate_ai_climatology(var, test_step=step)