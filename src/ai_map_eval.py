import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp

def evaluate_ai_climatology(target_var):

    # --- 0. CONFIGURATION BASED ON PARAMETER ---
    if target_var == "lst":
        real_path = './data/processed/aligned_lst.nc'
        nc_var = 'LST_PMW'
        ai_path = './data/generated/ai_generated_lst_3000days.npy'
        unit = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
    elif target_var == "sm":
        real_path = './data/processed/aligned_sm.nc'
        nc_var = 'sm'
        ai_path = './data/generated/ai_generated_sm_3000days.npy'
        unit = "m³/m³"
        title_name = "Soil Moisture"
        cmap_mean = 'YlGnBu'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    print(f"--- 1. LOADING REAL DATA ({title_name}) ---")
    ds = xr.open_dataset(real_path)

    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    real_data = summer[nc_var].values[:, 0:32, 0:32]

    lst_min, lst_max = np.nanmin(real_data), np.nanmax(real_data)

    print(f"--- 2. LOADING AI DATA ({title_name}) ---")
    gen_norm = np.load(ai_path)

    ai_grid = ((gen_norm + 1) / 2) * (lst_max - lst_min) + lst_min

    print("--- 3. CALCULATING CLIMATOLOGY & KS STATISTIC ---")

    real_mean = np.nanmean(real_data, axis=0)
    ai_mean = np.mean(ai_grid, axis=0)

    real_std = np.nanstd(real_data, axis=0)
    ai_std = np.std(ai_grid, axis=0)

    land_mask = ~np.isnan(real_mean)

    # Calculate KS D-Statistic for each pixel
    print("Calculating pixel-wise KS D-Statistic (this takes about 15 seconds)...")
    ks_map = np.full((32, 32), np.nan)
    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                r_ts = real_data[:, i, j]
                a_ts = ai_grid[:, i, j]
                
                # Drop NaNs for the test
                r_clean = r_ts[~np.isnan(r_ts)]
                a_clean = a_ts[~np.isnan(a_ts)]
                
                if len(r_clean) > 0 and len(a_clean) > 0:
                    ks_map[i, j] = ks_2samp(r_clean, a_clean).statistic

    print("--- 4. APPLYING MASKS AND PLOTTING ---")

    ai_mean_masked = np.where(land_mask, ai_mean, np.nan)
    ai_std_masked = np.where(land_mask, ai_std, np.nan)

    # --- GET COORDINATES FOR EXTENT ---
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'

    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]

    map_extent = [
        np.min(lons),
        np.max(lons),
        np.min(lats),
        np.max(lats)
    ]

    # --- CREATE CARTOPY FIGURE (3 rows, 2 columns) ---
    fig, axes = plt.subplots(
        3, 2,
        figsize=(12, 15),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    mean_vmin = min(np.nanmin(real_mean), np.nanmin(ai_mean_masked))
    mean_vmax = max(np.nanmax(real_mean), np.nanmax(ai_mean_masked))

    std_vmin = min(np.nanmin(real_std), np.nanmin(ai_std_masked))
    std_vmax = max(np.nanmax(real_std), np.nanmax(ai_std_masked))

    # --- ROW 1: MEAN ---
    im0 = axes[0, 0].imshow(
        real_mean, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
        origin='lower', extent=map_extent, transform=ccrs.PlateCarree()
    )
    axes[0, 0].set_title(f"Real Average {title_name} (40 Years)")
    axes[0, 0].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[0, 0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[0, 0].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(
        ai_mean_masked, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
        origin='lower', extent=map_extent, transform=ccrs.PlateCarree()
    )
    axes[0, 1].set_title(f"AI Generated Average {title_name}")
    axes[0, 1].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[0, 1].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[0, 1].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[0, 1].axis('off')

    fig.colorbar(im1, ax=axes[0, :].ravel().tolist(), label=f"{title_name} ({unit})", shrink=0.8)

    # --- ROW 2: STD ---
    im2 = axes[1, 0].imshow(
        real_std, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
        origin='lower', extent=map_extent, transform=ccrs.PlateCarree()
    )
    axes[1, 0].set_title(f"Real {title_name} Standard Deviation")
    axes[1, 0].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[1, 0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[1, 0].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[1, 0].axis('off')

    im3 = axes[1, 1].imshow(
        ai_std_masked, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
        origin='lower', extent=map_extent, transform=ccrs.PlateCarree()
    )
    axes[1, 1].set_title(f"AI Generated {title_name} Standard Deviation")
    axes[1, 1].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[1, 1].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[1, 1].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[1, 1].axis('off')

    fig.colorbar(im3, ax=axes[1, :].ravel().tolist(), label=f"Std Dev ({unit})", shrink=0.8)

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
        f"Generative Model Evaluation: Spatial Mean, Variability, and KS Error ({title_name})",
        fontsize=16, y=0.92
    )

    save_filename = f'ai_map_climatology_with_ks_{target_var}.png'
    plt.savefig(save_filename, dpi=150, bbox_inches='tight', facecolor='white')

    print(f"Saved '{save_filename}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [lst|sm]")
        sys.exit(1)

    evaluate_ai_climatology(sys.argv[1].lower())