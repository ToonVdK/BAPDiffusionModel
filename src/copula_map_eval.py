import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp

def evaluate_copula_climatology(target_var):

    # --- 0A. TARGET VARIABLE CONFIGURATION ---
    if target_var == "lst":
        real_path = './data/processed/aligned_lst.nc'
        nc_var = 'LST_PMW'
        unit = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
    elif target_var == "sm":
        real_path = './data/processed/aligned_sm.nc'
        nc_var = 'sm'
        unit = "m³/m³"
        title_name = "SM"
        cmap_mean = 'YlGnBu'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    copula_path = f'./data/generated/copula_gaussian_{target_var}_3000days.npy'

    print(f"--- 1. LOADING REAL DATA ({title_name}) ---")
    ds = xr.open_dataset(real_path)

    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    real_data = summer[nc_var].values[:, 0:32, 0:32]

    print("--- 2. LOADING COPULA DATA ---")
    try:
        copula_flat = np.load(copula_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {copula_path}")
        sys.exit(1)

    copula_grid = copula_flat.reshape(-1, 32, 32)

    print("--- 3. CALCULATING CLIMATOLOGY & KS STATISTIC ---")
    real_mean = np.nanmean(real_data, axis=0)
    copula_mean = np.mean(copula_grid, axis=0)

    real_std = np.nanstd(real_data, axis=0)
    copula_std = np.std(copula_grid, axis=0)

    land_mask = ~np.isnan(real_mean)

    # Calculate KS D-Statistic for each pixel
    print("Calculating pixel-wise KS D-Statistic...")
    ks_map = np.full((32, 32), np.nan)
    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                r_ts = real_data[:, i, j]
                c_ts = copula_grid[:, i, j]
                
                # Drop NaNs for the test
                r_clean = r_ts[~np.isnan(r_ts)]
                c_clean = c_ts[~np.isnan(c_ts)]
                
                if len(r_clean) > 0 and len(c_clean) > 0:
                    ks_map[i, j] = ks_2samp(r_clean, c_clean).statistic

    print("--- 4. APPLYING MASKS AND PLOTTING ---")
    copula_mean_masked = np.where(land_mask, copula_mean, np.nan)
    copula_std_masked = np.where(land_mask, copula_std, np.nan)

    # Determine map extent from dataset coordinates
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'

    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    # --- CREATE CARTOPY FIGURE (3 rows, 2 columns) ---
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 15),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    mean_vmin = min(np.nanmin(real_mean), np.nanmin(copula_mean_masked))
    mean_vmax = max(np.nanmax(real_mean), np.nanmax(copula_mean_masked))

    std_vmin = min(np.nanmin(real_std), np.nanmin(copula_std_masked))
    std_vmax = max(np.nanmax(real_std), np.nanmax(copula_std_masked))

    # --- ROW 1: MEAN ---
    im0 = axes[0, 0].imshow(
        real_mean, cmap=cmap_mean,
        vmin=mean_vmin, vmax=mean_vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )
    axes[0, 0].set_title(f"Real Average {title_name} (40 years)")
    axes[0, 0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0, 0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0, 0].set_axis_off()

    im1 = axes[0, 1].imshow(
        copula_mean_masked, cmap=cmap_mean,
        vmin=mean_vmin, vmax=mean_vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )
    axes[0, 1].set_title(f"Gaussian Copula Generated Average {title_name}")
    axes[0, 1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0, 1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0, 1].set_axis_off()

    fig.colorbar(im1, ax=axes[0, :].ravel().tolist(),
                 label=f"{title_name} ({unit})", shrink=0.8)

    # --- ROW 2: STD ---
    im2 = axes[1, 0].imshow(
        real_std, cmap='viridis',
        vmin=std_vmin, vmax=std_vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )
    axes[1, 0].set_title(f"Real {title_name} Standard Deviation")
    axes[1, 0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1, 0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1, 0].set_axis_off()

    im3 = axes[1, 1].imshow(
        copula_std_masked, cmap='viridis',
        vmin=std_vmin, vmax=std_vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )
    axes[1, 1].set_title(f"Gaussian Copula Generated {title_name} Standard Deviation")
    axes[1, 1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1, 1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1, 1].set_axis_off()

    fig.colorbar(im3, ax=axes[1, :].ravel().tolist(),
                 label=f"Std Dev ({unit})", shrink=0.8)

    # --- ROW 3: KS D-STATISTIC ---
    im4 = axes[2, 0].imshow(
        ks_map, cmap='magma', vmin=0.0, vmax=np.nanmax(ks_map),
        origin='lower', extent=map_extent, transform=ccrs.PlateCarree()
    )
    axes[2, 0].set_title(f"KS D-Statistic Error\n(Mean Error: {np.nanmean(ks_map):.4f})")
    axes[2, 0].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[2, 0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[2, 0].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[2, 0].axis('off')

    # Turn off the bottom right plot to keep it clean
    axes[2, 1].axis('off')
    axes[2, 1].set_visible(False)

    # Add colorbar just for the KS plot
    fig.colorbar(im4, ax=axes[2, 0], label="KS Statistic (Lower is Better)", shrink=0.8, pad=0.04)

    plt.suptitle(
        f"Copula Evaluation: Spatial Mean, Variability, and KS Error ({title_name})",
        fontsize=16, y=0.92
    )

    save_filename = f'copula_map_climatology_{target_var}.png'
    plt.savefig(save_filename, dpi=150,
                bbox_inches='tight', facecolor='white')

    print(f"Saved '{save_filename}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py [lst|sm]")
        sys.exit(1)

    evaluate_copula_climatology(sys.argv[1].lower())