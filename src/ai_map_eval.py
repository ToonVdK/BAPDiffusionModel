# ai_map_eval.py – chronological split, only mean and standard deviation
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from split_utils import get_train_indices

def evaluate_ai_climatology(target_var):
    # --- Configuration ---
    if target_var == "lst":
        nc_path = './data/processed/aligned_lst.nc'
        nc_var = 'LST_PMW'
        ai_path = './data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy'
        real_test_path = './data/generated/real_lst_chrono.npy'
        unit = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
    elif target_var == "sm":
        nc_path = './data/processed/aligned_sm.nc'
        nc_var = 'sm'
        ai_path = './data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy'
        real_test_path = './data/generated/real_sm_chrono.npy'
        unit = "m³/m³"
        title_name = "Soil Moisture"
        cmap_mean = 'YlGnBu'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    print(f"--- 1. LOADING REAL DATA ({title_name}) & TRAINING STATS ---")
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = summer[nc_var].values[:, 0:32, 0:32]
    train_idx = get_train_indices()   # chronological training indices
    train_data = full_data[train_idx]
    lst_min, lst_max = np.nanmin(train_data), np.nanmax(train_data)
    ds.close()

    print(f"Training {title_name} range: {lst_min:.2f} – {lst_max:.2f}")

    print("--- 2. LOADING AI GENERATED DATA ---")
    gen_norm = np.load(ai_path)
    ai_grid = ((gen_norm + 1) / 2) * (lst_max - lst_min) + lst_min

    print("--- 3. LOADING REAL TEST SET (Chronological) ---")
    real_test = np.load(real_test_path)

    print("--- 4. CALCULATING CLIMATOLOGY ---")
    real_mean = np.nanmean(real_test, axis=0)
    ai_mean = np.mean(ai_grid, axis=0)

    real_std = np.nanstd(real_test, axis=0)
    ai_std = np.std(ai_grid, axis=0)

    land_mask = ~np.isnan(real_mean)

    print("--- 5. PLOTTING MEAN AND STD ---")
    ai_mean_masked = np.where(land_mask, ai_mean, np.nan)
    ai_std_masked = np.where(land_mask, ai_std, np.nan)

    # Coordinates for plotting
    ds_coords = xr.open_dataset(nc_path)
    lat_name = 'lat' if 'lat' in ds_coords.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_coords.coords else 'longitude'
    lats = ds_coords[lat_name].values[0:32]
    lons = ds_coords[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds_coords.close()

    mean_vmin = min(np.nanmin(real_mean), np.nanmin(ai_mean_masked))
    mean_vmax = max(np.nanmax(real_mean), np.nanmax(ai_mean_masked))
    std_vmin = min(np.nanmin(real_std), np.nanmin(ai_std_masked))
    std_vmax = max(np.nanmax(real_std), np.nanmax(ai_std_masked))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Row 1: Mean
    im0 = axes[0,0].imshow(real_mean, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,0].set_title(f"Real Average {title_name}")
    axes[0,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,0].set_axis_off()

    im1 = axes[0,1].imshow(ai_mean_masked, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,1].set_title(f"U‑Net Average {title_name}")
    axes[0,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,1].set_axis_off()
    fig.colorbar(im0, ax=axes[0,:].ravel().tolist(), label=f"{title_name} ({unit})", shrink=0.8)

    # Row 2: Standard Deviation
    im2 = axes[1,0].imshow(real_std, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,0].set_title(f"Real {title_name} Standard Deviation")
    axes[1,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,0].set_axis_off()

    im3 = axes[1,1].imshow(ai_std_masked, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,1].set_title(f"U‑Net {title_name} Standard Deviation")
    axes[1,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,1].set_axis_off()
    fig.colorbar(im2, ax=axes[1,:].ravel().tolist(), label=f"Std Dev ({unit})", shrink=0.8)

    plt.suptitle(f"U‑Net 1D Evaluation – {title_name}",
                 fontsize=16, y=0.95)
    save_name = f'ai_mean_std_chrono_{target_var}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_name}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_map_eval.py [lst|sm]")
        sys.exit(1)
    var = sys.argv[1].lower()
    evaluate_ai_climatology(var)