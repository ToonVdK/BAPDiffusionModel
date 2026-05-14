import os
import glob
import numpy as np
import xarray as xr

# --- GLOBAL SETTINGS & PATHS ---
LST_RAW_DIR = '/mnt/mvbc-diff/LST'
SM_RAW_DIR = '/mnt/mvbc-diff/SM'

INTERIM_DIR = './data/interim'
PROCESSED_DIR = './data/processed'

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 8x8 Degree bounding box centered on Benelux/Western Europe
# 0.25 degree resolution = 32x32 pixels
lat_slice = slice(46.0, 54.0)
lon_slice = slice(0.0, 8.0)

# Threshold for keeping a SM sample (a value of 0.50 means 50% of the box must have valid SM data)
COVERAGE_THRESHOLD = 0.50


# ----------------------------------------------------┐
# PROCESSING LST DATA (Crop -> Coarsen -> Daily Mean) |
# ----------------------------------------------------┘
lst_checkpoint_path = f'{INTERIM_DIR}/lst_coarsened_daily_ckpt.nc'

if not os.path.exists(lst_checkpoint_path):
    print("LST Checkpoint not found. Scanning files...")
    lst_files = sorted(glob.glob(f'{LST_RAW_DIR}/*.nc'))  # Get all LST files as a list
    
    # To avoid running into segmentation faults, we must process the dataset in batches
    CHUNK_SIZE = 1000
    total_chunks = (len(lst_files) // CHUNK_SIZE) + 1
    processed_batches = []

    for i in range(0, len(lst_files), CHUNK_SIZE):
        batch_files = lst_files[i:i + CHUNK_SIZE]  # Take just one batch from the entire dataset
        current_batch = (i // CHUNK_SIZE) + 1
        print(f"  -> Processing LST batch {current_batch} of {total_chunks} ({len(batch_files)} files)...")

        # Open just this batch
        ds_batch = xr.open_mfdataset(batch_files, combine='nested', concat_dim='time', 
                                     engine='netcdf4', data_vars='minimal', 
                                     coords='minimal', compat='override')
        
        # --- Crop the data to the bounding box ---
        cropped = ds_batch.sel(lat=lat_slice, lon=lon_slice)

        # --- Coarsen the data to match the resolution from the SM data ---
        coarse = cropped.coarsen(lat=5, lon=5, boundary='trim').mean()  # Averages 5x5 blocks of pixels into 1 pixel

        # --- Average hourly data to daily ---
        daily = coarse.resample(time='1D').mean()

        # Compute the result into RAM so we can close the original files
        daily.load()
        ds_batch.close()

        # Save the processed block
        processed_batches.append(daily)

    # Stitch together all of the processed batches
    print("\nStitching all processed batches together...")
    lst_daily = xr.concat(processed_batches, dim='time')
    
    print(f"Saving final LST checkpoint to {lst_checkpoint_path}...")
    lst_daily.to_netcdf(lst_checkpoint_path)
    lst_daily.close()  # Free up memory
else:
    print("LST Checkpoint found! Loading from checkpoint...")

# Load the daily LST back into memory
lst_daily = xr.open_dataset(lst_checkpoint_path)

# -----------------------------------------┐
# PROCESSING SOIL MOISTURE DATA (Cropping) |
# -----------------------------------------┘
sm_checkpoint_path = f'{INTERIM_DIR}/sm_cropped_ckpt.nc'

if not os.path.exists(sm_checkpoint_path):
    print("SM Checkpoint not found. Scanning files...")
    # Grab only the COMBINED Surface Soil Moisture files (not Root Zone SM)
    sm_files = sorted(glob.glob(f'{SM_RAW_DIR}/C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-*.nc'))
    
    # To avoid running into segmentation faults, we must process the dataset in batches
    SM_CHUNK_SIZE = 500
    sm_total_chunks = (len(sm_files) // SM_CHUNK_SIZE) + 1
    processed_sm_batches = []

    for i in range(0, len(sm_files), SM_CHUNK_SIZE):
        batch_files = sm_files[i:i + SM_CHUNK_SIZE]  # Take just one batch from the entire dataset
        current_batch = (i // SM_CHUNK_SIZE) + 1
        print(f"  -> Processing SM batch {current_batch} of {sm_total_chunks} ({len(batch_files)} files)...")

        # Open just this batch
        ds_batch = xr.open_mfdataset(batch_files, combine='nested', concat_dim='time', 
                                     engine='netcdf4', data_vars='minimal', 
                                     coords='minimal', compat='override')

        # Invert latitude and longitude to match the LST data
        ds_batch = ds_batch.sortby(['lat', 'lon'])

        # --- Crop the data to the bounding box ---
        cropped = ds_batch.sel(lat=lat_slice, lon=lon_slice)

        # Compute the result into RAM so we can close the original files
        cropped.load()
        ds_batch.close()
        
        # Save the processed block
        processed_sm_batches.append(cropped)

    # Stitch together all of the processed batches
    print("\nStitching all SM batches together...")
    sm_cropped = xr.concat(processed_sm_batches, dim='time')

    print(f"Saving SM checkpoint to {sm_checkpoint_path}...")
    sm_cropped.to_netcdf(sm_checkpoint_path)
    sm_cropped.close()  # Free up memory
else:
    print("\nSM Checkpoint found! Loading from checkpoint...")

# Load the SM dataset back into memory
sm_cropped = xr.open_dataset(sm_checkpoint_path)
sm_var = 'sm'


# ----------------------┐
# ALIGNMENT & FILTERING |
# ----------------------┘
# Clean up duplicate times caused by batch boundaries
lst_daily = lst_daily.groupby('time').mean()  # Find duplicate days and average them together for LST data
sm_cropped = sm_cropped.drop_duplicates(dim='time')  # Find duplicate days and delete the extras for SM data

# --- Align the datasets in time ---
print("\nAligning timestamps...")
common_times = lst_daily.indexes['time'].intersection(sm_cropped.indexes['time'])
final_lst = lst_daily.sel(time=common_times)
final_sm = sm_cropped.sel(time=common_times)

# --- Filter out SM samples with poor coverage ---
print("\nFiltering out days with poor SM satellite coverage...")
valid_counts = final_sm[sm_var].count(dim=['lat', 'lon'])  # Count how many valid (non-NaN) pixels exist per day
total_pixels = final_sm[sm_var].isel(time=0).size  # Calculate the total number of pixels in one 32x32 frame
# Create a boolean mask of days that meet the threshold (e.g. 50%)
coverage_ratios = valid_counts / total_pixels
good_coverage_mask = coverage_ratios >= COVERAGE_THRESHOLD
# Apply the mask to both datasets to throw away the bad days
filtered_lst = final_lst.where(good_coverage_mask, drop=True)
filtered_sm = final_sm.where(good_coverage_mask, drop=True)

days_kept = filtered_sm.time.size
days_total = common_times.size
print(f"Kept {days_kept} out of {days_total} overlapping days ({(days_kept/days_total)*100:.1f}%).")


# ------------------------------------┐
# COMPUTE AND SAVE PROCESSED DATASETS |
# ------------------------------------┘
print("\nExecuting computation and saving final arrays to disk...")
filtered_lst = filtered_lst.assign_coords(lat=filtered_sm.lat, lon=filtered_sm.lon)  # Ensure the coordinates match up perfectly
filtered_lst.to_netcdf(f'{PROCESSED_DIR}/aligned_lst.nc')
filtered_sm.to_netcdf(f'{PROCESSED_DIR}/aligned_sm.nc')

print("Preprocessing fully complete!")