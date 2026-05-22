import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from split_utils import get_train_indices

def load_and_process_data(target_var):
    """
    Loads Real, AI, and Copula data, applies masks, and calculates statistics.
    """
    # Config based on variable
    if target_var == "lst":
        nc_path = './data/processed/aligned_lst.nc'
        nc_var = 'LST_PMW'
        ai_path = './data/generated/ai_generated_lst_3000days_epoch_100_masked_ocean.npy'
        real_test_path = './data/generated/real_lst_chrono.npy'
        copula_path = './data/generated/copula_gaussian_lst_chrono.npy'
        unit = "K"
        title_name = "LST"
        cmap_mean = 'inferno'
        cmap_std = 'viridis'
    elif target_var == "sm":
        nc_path = './data/processed/aligned_sm.nc'
        nc_var = 'sm'
        ai_path = './data/generated/ai_generated_sm_3000days_epoch_100_masked_ocean.npy'
        real_test_path = './data/generated/real_sm_chrono.npy'
        copula_path = './data/generated/copula_gaussian_sm_chrono.npy'
        unit = "m³/m³"
        title_name = "Soil Moisture"
        cmap_mean = 'YlGnBu'
        cmap_std = 'plasma'  # Different variance cmap for visual distinction

    # Get training min/max for un-normalizing AI
    ds = xr.open_dataset(nc_path)
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    full_data = summer[nc_var].values[:, 0:32, 0:32]
    
    train_idx = get_train_indices()
    train_data = full_data[train_idx]
    val_min, val_max = np.nanmin(train_data), np.nanmax(train_data)
    
    # Coordinates for mapping
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values[0:32]
    lons = ds[lon_name].values[0:32]
    map_extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    ds.close()

    # Load Datasets
    real_test = np.load(real_test_path)
    copula_grid = np.load(copula_path)
    
    gen_norm = np.load(ai_path)
    ai_grid = ((gen_norm + 1) / 2) * (val_max - val_min) + val_min

    # Calculate Stats
    real_mean = np.nanmean(real_test, axis=0)
    ai_mean = np.mean(ai_grid, axis=0)
    copula_mean = np.mean(copula_grid, axis=0)

    real_std = np.nanstd(real_test, axis=0)
    ai_std = np.std(ai_grid, axis=0)
    copula_std = np.std(copula_grid, axis=0)

    # Apply Land Mask strictly
    land_mask = ~np.isnan(real_mean)

    ai_mean_masked = np.where(land_mask, ai_mean, np.nan)
    copula_mean_masked = np.where(land_mask, copula_mean, np.nan)
    
    ai_std_masked = np.where(land_mask, ai_std, np.nan)
    copula_std_masked = np.where(land_mask, copula_std, np.nan)

    # Return a dictionary containing everything needed for plotting
    return {
        'title': title_name, 'unit': unit, 'extent': map_extent,
        'cmap_mean': cmap_mean, 'cmap_std': cmap_std,
        'means': [real_mean, ai_mean_masked, copula_mean_masked],
        'stds': [real_std, ai_std_masked, copula_std_masked]
    }

def plot_combined_climatology():
    print("--- PROCESSING LST DATA ---")
    lst_data = load_and_process_data('lst')
    
    print("--- PROCESSING SOIL MOISTURE DATA ---")
    sm_data = load_and_process_data('sm')

    print("--- GENERATING MASTER PLOT ---")
    # 4 rows, 3 columns layout
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    
    col_titles = ["Real Observations", "Diffusion Generated", "Gaussian Copula"]
    
    # Package data into a list of dictionaries corresponding to each row
    rows_info = [
        {'data': lst_data['means'], 'cmap': lst_data['cmap_mean'], 'name': f"LST Mean ({lst_data['unit']})"},
        {'data': lst_data['stds'],  'cmap': lst_data['cmap_std'],  'name': f"LST Std. Dev. ({lst_data['unit']})"},
        {'data': sm_data['means'],  'cmap': sm_data['cmap_mean'],  'name': f"SM Mean ({sm_data['unit']})"},
        {'data': sm_data['stds'],   'cmap': sm_data['cmap_std'],   'name': f"SM Std. Dev. ({sm_data['unit']})"}
    ]

    for row_idx, row in enumerate(rows_info):
        # Calculate shared color limits for this specific row so comparisons are mathematically fair
        vmin = min(np.nanmin(row['data'][0]), np.nanmin(row['data'][1]), np.nanmin(row['data'][2]))
        vmax = max(np.nanmax(row['data'][0]), np.nanmax(row['data'][1]), np.nanmax(row['data'][2]))

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(row['data'][col_idx], cmap=row['cmap'], vmin=vmin, vmax=vmax,
                           origin='lower', extent=lst_data['extent'], transform=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linewidth=1)
            ax.set_axis_off()

            # Set Column Titles only on the very top row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=15, pad=10)
                
            # Set Subtitles for clarity
            if col_idx == 0:
                ax.text(-0.05, 0.5, row['name'], va='center', ha='right', rotation=90, 
                        transform=ax.transAxes, fontsize=14, fontweight='bold')

        # Add one shared colorbar per row to the right of the axes
        cbar = fig.colorbar(im, ax=axes[row_idx, :].ravel().tolist(), fraction=0.015, pad=0.04)
        cbar.set_label(row['name'], fontsize=12)

    plt.suptitle("1D Evaluation: Mean and Variance Representivity", fontsize=20, y=0.92)
    save_name = 'master_climatology_grid.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved highly condensed master figure to '{save_name}'")

if __name__ == "__main__":
    plot_combined_climatology()