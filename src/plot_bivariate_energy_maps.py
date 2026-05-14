import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial.distance import cdist

def calculate_bivariate_energy_distance(real_pts, gen_pts, subsample=500):
    """
    Calculates the 2-Sample Energy Distance between two bivariate distributions.
    Formula: ED = 2 * E[||R - G||] - E[||R - R||] - E[||G - G||]
    """
    real_clean = real_pts[~np.isnan(real_pts).any(axis=1)]
    gen_clean = gen_pts[~np.isnan(gen_pts).any(axis=1)]

    if len(real_clean) == 0 or len(gen_clean) == 0:
        return np.nan

    # Grab 500 real days and 500 generated days for this pixel randomly
    np.random.seed(42)
    idx_r = np.random.choice(len(real_clean), min(subsample, len(real_clean)), replace=False)
    idx_g = np.random.choice(len(gen_clean), min(subsample, len(gen_clean)), replace=False)

    R = real_clean[idx_r]
    G = gen_clean[idx_g]

    # Standardize values into Z-score
    r_mean = np.mean(R, axis=0)
    r_std = np.std(R, axis=0) + 1e-8

    R_scaled = (R - r_mean) / r_std
    G_scaled = (G - r_mean) / r_std

    # Calculate each distance using cdist
    dist_RG = np.mean(cdist(R_scaled, G_scaled, metric='euclidean'))
    dist_RR = np.mean(cdist(R_scaled, R_scaled, metric='euclidean'))
    dist_GG = np.mean(cdist(G_scaled, G_scaled, metric='euclidean'))

    # Use the formula to calculate the energy distance
    energy_distance = (2 * dist_RG) - dist_RR - dist_GG
    return max(0.0, energy_distance)


def generate_bivariate_heatmaps():
    print("--- 1. LOADING REAL DATASETS ---")
    valid_months = [5, 6, 7, 8, 9]

    ds_lst = xr.open_dataset('./data/processed/aligned_lst.nc')
    summer_lst = ds_lst.sel(time=ds_lst['time'].dt.month.isin(valid_months))
    real_lst = summer_lst['LST_PMW'].values[:, 0:32, 0:32]

    ds_sm = xr.open_dataset('./data/processed/aligned_sm.nc')
    summer_sm = ds_sm.sel(time=ds_sm['time'].dt.month.isin(valid_months))
    real_sm = summer_sm['sm'].values[:, 0:32, 0:32]

    print("--- 2. LOADING GENERATED DATASETS ---")
    ai_norm_lst = np.load('./data/generated/ai_generated_lst_3000days.npy')
    ai_norm_sm = np.load('./data/generated/ai_generated_sm_3000days.npy')

    ai_lst = ((ai_norm_lst + 1) / 2) * (np.nanmax(real_lst) - np.nanmin(real_lst)) + np.nanmin(real_lst)
    ai_sm = ((ai_norm_sm + 1) / 2) * (np.nanmax(real_sm) - np.nanmin(real_sm)) + np.nanmin(real_sm)

    copula_lst_flat = np.load('./data/generated/copula_gaussian_lst_3000days.npy')
    copula_sm_flat = np.load('./data/generated/copula_gaussian_sm_3000days.npy')

    copula_lst = copula_lst_flat.reshape(3000, 32, 32)
    copula_sm = copula_sm_flat.reshape(3000, 32, 32)

    print("--- 3. CALCULATING BIVARIATE ENERGY DISTANCE ---")
    land_mask = ~np.isnan(np.nanmean(real_lst, axis=0)) & ~np.isnan(np.nanmean(real_sm, axis=0))

    ed_map_ai = np.full((32, 32), np.nan)
    ed_map_copula = np.full((32, 32), np.nan)

    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                real_pts = np.column_stack((real_lst[:, i, j], real_sm[:, i, j]))
                ai_pts = np.column_stack((ai_lst[:, i, j], ai_sm[:, i, j]))
                copula_pts = np.column_stack((copula_lst[:, i, j], copula_sm[:, i, j]))

                ed_map_ai[i, j] = calculate_bivariate_energy_distance(real_pts, ai_pts)
                ed_map_copula[i, j] = calculate_bivariate_energy_distance(real_pts, copula_pts)
    
    print(f"Copula energy at pixel 11, 17 equals: {ed_map_copula[11,18]}")
    print(f"AI energy at pixel 11, 17 equals {ed_map_ai[11,18]}")

    print("--- 4. PLOTTING HEATMAPS ---")

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    vmax = max(np.nanmax(ed_map_ai), np.nanmax(ed_map_copula))
    vmin = 0.0

    lat_name = 'lat' if 'lat' in ds_lst.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_lst.coords else 'longitude'

    lats = ds_lst[lat_name].values[0:32]
    lons = ds_lst[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    # --- AI MAP ---
    im0 = axes[0].imshow(
        ed_map_ai, cmap='magma',
        vmin=vmin, vmax=vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )
    axes[0].set_title("U-Net Cross-Variable Error", fontsize=14)
    axes[0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0].set_axis_off()

    # --- COPULA MAP ---
    im1 = axes[1].imshow(
        ed_map_copula, cmap='magma',
        vmin=vmin, vmax=vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )
    axes[1].set_title("2048D Gaussian Copula Cross-Variable Error", fontsize=14)
    axes[1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1].set_axis_off()

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("2-Sample Energy Distance (Standardized)", fontsize=12)

    plt.suptitle(
        "Joint LST & Soil Moisture Distribution Error",
        fontsize=16, y=1.02
    )

    plt.savefig('bivariate_energy_heatmaps.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("Saved 'bivariate_energy_heatmaps.png'")


if __name__ == "__main__":
    generate_bivariate_heatmaps()