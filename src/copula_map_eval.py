# copula_map_eval.py – chronological split, only mean and standard deviation
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def evaluate_copula_climatology(target_var):
    # --- Configuration ---
    if target_var == "lst":
        nc_path = './data/processed/aligned_lst.nc'
        real_npy_path = './data/generated/real_lst_chrono.npy'
        copula_path = './data/generated/copula_gaussian_lst_chrono.npy'
        unit = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
    elif target_var == "sm":
        nc_path = './data/processed/aligned_sm.nc'
        real_npy_path = './data/generated/real_sm_chrono.npy'
        copula_path = './data/generated/copula_gaussian_sm_chrono.npy'
        unit = "m³/m³"
        title_name = "Soil Moisture"
        cmap_mean = 'YlGnBu'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    print(f"--- 1. LOADING REAL TEST DATA ({title_name}) ---")
    real_data = np.load(real_npy_path)

    # Coordinates
    ds = xr.open_dataset(nc_path)
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds.close()

    print("--- 2. LOADING COPULA DATA ---")
    copula_grid = np.load(copula_path)

    print("--- 3. CALCULATING CLIMATOLOGY ---")
    real_mean = np.nanmean(real_data, axis=0)
    copula_mean = np.mean(copula_grid, axis=0)

    real_std = np.nanstd(real_data, axis=0)
    copula_std = np.std(copula_grid, axis=0)

    land_mask = ~np.isnan(real_mean)

    print("--- 4. PLOTTING MEAN AND STD ---")
    copula_mean_masked = np.where(land_mask, copula_mean, np.nan)
    copula_std_masked = np.where(land_mask, copula_std, np.nan)

    mean_vmin = min(np.nanmin(real_mean), np.nanmin(copula_mean_masked))
    mean_vmax = max(np.nanmax(real_mean), np.nanmax(copula_mean_masked))
    std_vmin = min(np.nanmin(real_std), np.nanmin(copula_std_masked))
    std_vmax = max(np.nanmax(real_std), np.nanmax(copula_std_masked))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Row 1: Mean
    im0 = axes[0,0].imshow(real_mean, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,0].set_title(f"Real Average {title_name}")
    axes[0,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,0].set_axis_off()

    im1 = axes[0,1].imshow(copula_mean_masked, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,1].set_title(f"Copula Average {title_name}")
    axes[0,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,1].set_axis_off()
    fig.colorbar(im0, ax=axes[0,:].ravel().tolist(), label=f"{title_name} ({unit})", shrink=0.8)

    # Row 2: Std
    im2 = axes[1,0].imshow(real_std, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,0].set_title(f"Real {title_name} Standard Deviation")
    axes[1,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,0].set_axis_off()

    im3 = axes[1,1].imshow(copula_std_masked, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                           origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,1].set_title(f"Copula {title_name} Standard Deviation")
    axes[1,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,1].set_axis_off()
    fig.colorbar(im2, ax=axes[1,:].ravel().tolist(), label=f"Std Dev ({unit})", shrink=0.8)

    plt.suptitle(f"Copula Evaluation – {title_name}",
                 fontsize=16, y=0.95)
    save_name = f'copula_mean_std_chrono_{target_var}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_name}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python copula_map_eval.py [lst|sm]")
        sys.exit(1)
    var = sys.argv[1].lower()
    evaluate_copula_climatology(var)