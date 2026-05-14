import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp

def evaluate_copula_climatology(target_var, test_step=5):
    # --- 0. CONFIGURATION ---
    if target_var == "lst":
        nc_path = './data/processed/aligned_lst.nc'   # for coordinates only
        real_npy_path = f'./data/generated/real_lst_oos_test_step{test_step}.npy'
        copula_path = f'./data/generated/copula_gaussian_lst_oos_test_step{test_step}.npy'
        unit = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
    elif target_var == "sm":
        nc_path = './data/processed/aligned_sm.nc'
        real_npy_path = f'./data/generated/real_sm_oos_test_step{test_step}.npy'
        copula_path = f'./data/generated/copula_gaussian_sm_oos_test_step{test_step}.npy'
        unit = "m³/m³"
        title_name = "SM"
        cmap_mean = 'YlGnBu'
    else:
        print("Error: Parameter must be 'lst' or 'sm'")
        sys.exit(1)

    print(f"--- 1. LOADING REAL DATA ({title_name}) ---")
    # Load the pre‑saved test array (already the correct split)
    real_data = np.load(real_npy_path)   # shape (n_test, 32, 32)

    # Load the full NetCDF only to grab the coordinates
    ds = xr.open_dataset(nc_path)
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]
    ds.close()  # we don't need the whole dataset anymore

    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    print("--- 2. LOADING COPULA DATA ---")
    copula_grid = np.load(copula_path)   # shape (n_test, 32, 32)

    print("--- 3. CALCULATING CLIMATOLOGY & KS STATISTIC ---")
    real_mean = np.nanmean(real_data, axis=0)
    copula_mean = np.mean(copula_grid, axis=0)

    real_std = np.nanstd(real_data, axis=0)
    copula_std = np.std(copula_grid, axis=0)

    land_mask = ~np.isnan(real_mean)

    print("Calculating pixel-wise KS D-Statistic...")
    ks_map = np.full((32, 32), np.nan)
    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                r_ts = real_data[:, i, j]
                c_ts = copula_grid[:, i, j]
                r_clean = r_ts[~np.isnan(r_ts)]
                c_clean = c_ts[~np.isnan(c_ts)]
                if len(r_clean) > 0 and len(c_clean) > 0:
                    ks_map[i, j] = ks_2samp(r_clean, c_clean).statistic

    print("--- 4. PLOTTING ---")
    copula_mean_masked = np.where(land_mask, copula_mean, np.nan)
    copula_std_masked = np.where(land_mask, copula_std, np.nan)

    mean_vmin = min(np.nanmin(real_mean), np.nanmin(copula_mean_masked))
    mean_vmax = max(np.nanmax(real_mean), np.nanmax(copula_mean_masked))
    std_vmin = min(np.nanmin(real_std), np.nanmin(copula_std_masked))
    std_vmax = max(np.nanmax(real_std), np.nanmax(copula_std_masked))

    fig, axes = plt.subplots(3, 2, figsize=(12, 15),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Row 1: Mean
    axes[0,0].imshow(real_mean, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,0].set_title(f"Real Average {title_name} (Test Set, step={test_step})")
    axes[0,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,0].set_axis_off()

    axes[0,1].imshow(copula_mean_masked, cmap=cmap_mean, vmin=mean_vmin, vmax=mean_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[0,1].set_title(f"Copula Average {title_name} (Test Set)")
    axes[0,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0,1].set_axis_off()
    fig.colorbar(axes[0,0].images[0], ax=axes[0,:], label=f"{title_name} ({unit})", shrink=0.8)

    # Row 2: Std
    axes[1,0].imshow(real_std, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,0].set_title(f"Real {title_name} Standard Deviation")
    axes[1,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,0].set_axis_off()

    axes[1,1].imshow(copula_std_masked, cmap='viridis', vmin=std_vmin, vmax=std_vmax,
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[1,1].set_title(f"Copula {title_name} Standard Deviation")
    axes[1,1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1,1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1,1].set_axis_off()
    fig.colorbar(axes[1,0].images[0], ax=axes[1,:], label=f"Std Dev ({unit})", shrink=0.8)

    # Row 3: KS
    axes[2,0].imshow(ks_map, cmap='magma', vmin=0.0, vmax=np.nanmax(ks_map),
                     origin='lower', extent=map_extent, transform=ccrs.PlateCarree())
    axes[2,0].set_title(f"KS D-Statistic\n(Mean Error: {np.nanmean(ks_map):.4f})")
    axes[2,0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[2,0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[2,0].set_axis_off()
    axes[2,1].set_visible(False)
    fig.colorbar(axes[2,0].images[0], ax=axes[2,0], label="KS Statistic", shrink=0.8, pad=0.04)

    plt.suptitle(f"Copula Out‑Of‑Sample (Systematic Step={test_step}) – {title_name}",
                 fontsize=16, y=0.92)
    save_name = f'copula_oos_systematic_step{test_step}_{target_var}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved '{save_name}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python copula_map_eval.py [lst|sm] [test_step]")
        sys.exit(1)
    var = sys.argv[1].lower()
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    evaluate_copula_climatology(var, test_step=step)