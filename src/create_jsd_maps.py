import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial.distance import jensenshannon

def calculate_2d_jsd(real_x, real_y, gen_x, gen_y, bins=15):
    """
    Calculates the 2D JSD between two bivariate distributions.
    Bins are reduced to 15 to prevent sparsity in the 2D grid.
    """
    valid_real = ~np.isnan(real_x) & ~np.isnan(real_y)
    valid_gen = ~np.isnan(gen_x) & ~np.isnan(gen_y)

    rx, ry = real_x[valid_real], real_y[valid_real]
    gx, gy = gen_x[valid_gen], gen_y[valid_gen]

    if len(rx) == 0 or len(gx) == 0:
        return np.nan

    x_min = min(np.min(rx), np.min(gx))
    x_max = max(np.max(rx), np.max(gx))
    y_min = min(np.min(ry), np.min(gy))
    y_max = max(np.max(ry), np.max(gy))

    bounds = [[x_min, x_max], [y_min, y_max]]

    p_hist, _, _ = np.histogram2d(rx, ry, bins=bins, range=bounds, density=True)
    q_hist, _, _ = np.histogram2d(gx, gy, bins=bins, range=bounds, density=True)

    p_prob = p_hist.flatten()
    q_prob = q_hist.flatten()

    epsilon = 1e-10
    p_prob = p_prob + epsilon
    q_prob = q_prob + epsilon
    p_prob /= np.sum(p_prob)
    q_prob /= np.sum(q_prob)

    return jensenshannon(p_prob, q_prob)

def generate_2d_jsd_heatmaps():

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

    lst_min, lst_max = np.nanmin(real_lst), np.nanmax(real_lst)
    sm_min, sm_max = np.nanmin(real_sm), np.nanmax(real_sm)

    ai_lst = ((ai_norm_lst + 1) / 2) * (lst_max - lst_min) + lst_min
    ai_sm = ((ai_norm_sm + 1) / 2) * (sm_max - sm_min) + sm_min

    copula_lst_flat = np.load('./data/generated/copula_gaussian_lst_3000days.npy')
    copula_sm_flat = np.load('./data/generated/copula_gaussian_sm_3000days.npy')

    copula_lst = copula_lst_flat.reshape(3000, 32, 32)
    copula_sm = copula_sm_flat.reshape(3000, 32, 32)

    print("--- 3. CALCULATING PIXEL-WISE 2D JSD ---")

    land_mask = ~np.isnan(np.nanmean(real_lst, axis=0)) & \
                ~np.isnan(np.nanmean(real_sm, axis=0))

    jsd_map_ai = np.full((32, 32), np.nan)
    jsd_map_copula = np.full((32, 32), np.nan)

    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:

                r_lst, r_sm = real_lst[:, i, j], real_sm[:, i, j]
                a_lst, a_sm = ai_lst[:, i, j], ai_sm[:, i, j]
                c_lst, c_sm = copula_lst[:, i, j], copula_sm[:, i, j]

                jsd_map_ai[i, j] = calculate_2d_jsd(r_sm, r_lst, a_sm, a_lst)
                jsd_map_copula[i, j] = calculate_2d_jsd(r_sm, r_lst, c_sm, c_lst)

    print("--- 4. PLOTTING HEATMAPS WITH BORDERS ---")

    # Get spatial coordinates
    ds_lst = xr.open_dataset('./data/processed/aligned_lst.nc')
    lat_name = 'lat' if 'lat' in ds_lst.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_lst.coords else 'longitude'

    lats = ds_lst[lat_name].values[0:32]
    lons = ds_lst[lon_name].values[0:32]

    map_extent = [
        np.min(lons),
        np.max(lons),
        np.min(lats),
        np.max(lats)
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    vmax = max(np.nanmax(jsd_map_ai), np.nanmax(jsd_map_copula))
    vmin = 0.0

    # AI Heatmap
    im0 = axes[0].imshow(
        jsd_map_ai,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )

    axes[0].set_title("U-Net Cross-Variable JSD Error", fontsize=14)
    axes[0].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[0].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[0].axis('off')

    # Copula Heatmap
    im1 = axes[1].imshow(
        jsd_map_copula,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree()
    )

    axes[1].set_title("2048D Gaussian Copula Cross-Variable JSD Error", fontsize=14)
    axes[1].add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    axes[1].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    axes[1].set_extent(map_extent, crs=ccrs.PlateCarree())
    axes[1].axis('off')

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("2D Jensen-Shannon Divergence\n(Lower is better)", fontsize=12)

    plt.suptitle(
        "Bivariate Spatial Evaluation: LST vs. Soil Moisture Dependence Error",
        fontsize=16,
        y=1.02
    )

    plt.savefig('jsd_error_heatmaps.png', dpi=150,
                bbox_inches='tight', facecolor='white')

    print("Saved 'jsd_error_heatmaps.png'")

if __name__ == "__main__":
    generate_2d_jsd_heatmaps()