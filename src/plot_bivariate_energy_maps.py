# plot_bivariate_energy_maps.py  (out‑of‑sample, systematic split)
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial.distance import cdist

# ----------------------------------------------------------------------
# Systematic split config (must match copula & training)
TEST_STEP = 5

def get_train_indices():
    ds = xr.open_dataset('./data/processed/aligned_lst.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    n_total = len(summer['time'])
    ds.close()
    test_idx = np.arange(0, n_total, TEST_STEP)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    return train_idx

# ----------------------------------------------------------------------
def calculate_bivariate_energy_distance(real_pts, gen_pts, subsample=500):
    """
    Calculates the 2-Sample Energy Distance between two bivariate distributions.
    Formula: ED = 2 * E[||R - G||] - E[||R - R||] - E[||G - G||]
    (unchanged from original)
    """
    real_clean = real_pts[~np.isnan(real_pts).any(axis=1)]
    gen_clean = gen_pts[~np.isnan(gen_pts).any(axis=1)]

    if len(real_clean) == 0 or len(gen_clean) == 0:
        return np.nan

    np.random.seed(42)
    idx_r = np.random.choice(len(real_clean), min(subsample, len(real_clean)), replace=False)
    idx_g = np.random.choice(len(gen_clean), min(subsample, len(gen_clean)), replace=False)

    R = real_clean[idx_r]
    G = gen_clean[idx_g]

    r_mean = np.mean(R, axis=0)
    r_std = np.std(R, axis=0) + 1e-8
    R_scaled = (R - r_mean) / r_std
    G_scaled = (G - r_mean) / r_std

    dist_RG = np.mean(cdist(R_scaled, G_scaled, metric='euclidean'))
    dist_RR = np.mean(cdist(R_scaled, R_scaled, metric='euclidean'))
    dist_GG = np.mean(cdist(G_scaled, G_scaled, metric='euclidean'))

    energy_distance = (2 * dist_RG) - dist_RR - dist_GG
    return max(0.0, energy_distance)

# ----------------------------------------------------------------------
def generate_bivariate_heatmaps_oos(test_step=TEST_STEP):
    print("--- 1. LOADING REAL & TRAINING DATA FOR NORMALIZATION ---")
    # We need the training min/max for un-normalising the AI data
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

    # Coordinates for maps
    lat_name = 'lat' if 'lat' in ds_lst.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds_lst.coords else 'longitude'
    lats = ds_lst[lat_name].values[0:32]
    lons = ds_lst[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    print("--- 2. LOADING REAL TEST SET ---")
    real_lst_test = np.load(f'./data/generated/real_lst_oos_test_step{test_step}.npy')
    real_sm_test  = np.load(f'./data/generated/real_sm_oos_test_step{test_step}.npy')
    n_test = real_lst_test.shape[0]

    print("--- 3. LOADING AI GENERATED DATA ---")
    ai_norm_lst = np.load('./data/generated/ai_generated_lst_3000days_epoch_300.npy')
    ai_norm_sm  = np.load('./data/generated/ai_generated_sm_3000days_epoch_300.npy')
    ai_lst = ((ai_norm_lst + 1) / 2) * (lst_max - lst_min) + lst_min
    ai_sm  = ((ai_norm_sm + 1) / 2) * (sm_max  - sm_min ) + sm_min

    print("--- 4. LOADING COPULA TEST DATA ---")
    copula_lst = np.load(f'./data/generated/copula_gaussian_lst_oos_test_step{test_step}.npy')
    copula_sm  = np.load(f'./data/generated/copula_gaussian_sm_oos_test_step{test_step}.npy')

    print("--- 5. CALCULATING PIXEL‑WISE BIVARIATE ENERGY DISTANCE ---")
    land_mask = ~np.isnan(np.nanmean(real_lst_test, axis=0)) & \
                ~np.isnan(np.nanmean(real_sm_test, axis=0))

    ed_map_ai = np.full((32, 32), np.nan)
    ed_map_copula = np.full((32, 32), np.nan)

    for i in range(32):
        for j in range(32):
            if land_mask[i, j]:
                # Real test points for this pixel
                real_pts = np.column_stack((real_lst_test[:, i, j],
                                            real_sm_test[:, i, j]))
                # AI points
                ai_pts = np.column_stack((ai_lst[:, i, j],
                                          ai_sm[:, i, j]))
                # Copula points
                copula_pts = np.column_stack((copula_lst[:, i, j],
                                              copula_sm[:, i, j]))

                ed_map_ai[i, j] = calculate_bivariate_energy_distance(real_pts, ai_pts)
                ed_map_copula[i, j] = calculate_bivariate_energy_distance(real_pts, copula_pts)

    print("--- 6. PLOTTING HEATMAPS ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    vmax = max(np.nanmax(ed_map_ai), np.nanmax(ed_map_copula))
    vmin = 0.0

    # AI map
    im0 = axes[0].imshow(ed_map_ai, cmap='magma',
                         vmin=vmin, vmax=vmax,
                         origin='lower', extent=map_extent,
                         transform=ccrs.PlateCarree())
    axes[0].set_title("U‑Net Cross‑Variable Error\n(Out‑of‑Sample)", fontsize=14)
    axes[0].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[0].add_feature(cfeature.BORDERS, linewidth=1)
    axes[0].set_axis_off()

    # Copula map
    im1 = axes[1].imshow(ed_map_copula, cmap='magma',
                         vmin=vmin, vmax=vmax,
                         origin='lower', extent=map_extent,
                         transform=ccrs.PlateCarree())
    axes[1].set_title("2048D Gaussian Copula Cross‑Variable Error\n(Out‑of‑Sample)", fontsize=14)
    axes[1].add_feature(cfeature.COASTLINE, linewidth=1)
    axes[1].add_feature(cfeature.BORDERS, linewidth=1)
    axes[1].set_axis_off()

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("2‑Sample Energy Distance (Standardized)", fontsize=12)

    plt.suptitle("Bivariate Energy Distance: LST vs. Soil Moisture\n(Out‑Of‑Sample, Systematic Step={})".format(test_step),
                 fontsize=16, y=1.02)
    plt.savefig(f'bivariate_energy_oos_step{test_step}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"Saved 'bivariate_energy_oos_step{test_step}.png'")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    step = int(sys.argv[1]) if len(sys.argv) > 1 else TEST_STEP
    generate_bivariate_heatmaps_oos(test_step=step)