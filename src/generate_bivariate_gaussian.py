import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import os
from scipy.stats import rankdata
from copulas.multivariate import GaussianMultivariate

def generate_2048d_gaussian():
    print(f"\n{'='*60}")
    print("INITIATING 2048D SPATIO-BIVARIATE GAUSSIAN COPULA")
    print(f"{'='*60}")

    # --- 1. DATA PREP (LST + SM) ---
    print("\n[1/4] Loading and Imputing LST and Soil Moisture...")
    t0 = time.time()
    
    valid_months = [5, 6, 7, 8, 9]
    flat_data_list = []
    
    # Paths and Variables (Hardcoded as requested)
    datasets = [
        {'path': './data/processed/aligned_lst.nc', 'var': 'LST_PMW', 'name': 'LST'},
        {'path': './data/processed/aligned_sm.nc', 'var': 'sm', 'name': 'Soil Moisture'}
    ]

    for ds_info in datasets:
        print(f" -> Processing {ds_info['name']}...")
        ds_full = xr.open_dataset(ds_info['path'])
        ds_summer = ds_full.sel(time=ds_full['time'].dt.month.isin(valid_months))
        real_data = ds_summer[ds_info['var']].values
        
        # Extract 32x32 grid
        grid = real_data[:, 0:32, 0:32]
        total_days = grid.shape[0]
        flat_data = grid.reshape(total_days, 1024)

        # Impute NaNs safely
        col_means = np.nanmean(flat_data, axis=0)
        global_mean = np.nanmean(flat_data)
        col_means[np.isnan(col_means)] = global_mean
        
        inds = np.where(np.isnan(flat_data))
        flat_data[inds] = np.take(col_means, inds[1])
        
        # Inject micro-noise to prevent crashing
        flat_data += np.random.normal(0, 1e-4, size=flat_data.shape)
        flat_data_list.append(flat_data)

    # Stack LST and SM horizontally -> Shape: (Days, 2048)
    combined_flat_data = np.hstack(flat_data_list)
    total_dims = combined_flat_data.shape[1]
    print(f"\nDatasets stacked. Total dimensions for Copula: {total_dims} (1024 LST + 1024 SM)")
    
    print(" -> Converting to Uniform Percentiles...")
    epsilon = 1e-10
    u_data = np.zeros_like(combined_flat_data, dtype=float)
    for i in range(total_dims):
        ranks = rankdata(combined_flat_data[:, i])
        u_data[:, i] = np.clip((ranks - 0.5) / len(combined_flat_data), epsilon, 1 - epsilon)

    df_uniform = pd.DataFrame(u_data, columns=[f'var_{i}' for i in range(total_dims)])
    print(f"Data Prep Complete. Time: {time.time() - t0:.2f} seconds")

    # --- 2. FITTING THE MODEL ---
    print("\n[2/4] Fitting 2048-Dimensional Gaussian Copula...")
    t1 = time.time()
    
    model = GaussianMultivariate()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(df_uniform)
        
    fit_time = time.time() - t1
    print(f">>> FITTING COMPLETE | Time elapsed: {fit_time/60:.2f} minutes")

    # --- 3. GENERATING SAMPLES ---
    print("\n[3/4] Generating 3000 Synthetic Spatio-Bivariate Maps...")
    t2 = time.time()
    
    synthetic_uniforms = model.sample(3000).to_numpy()
    
    gen_time = time.time() - t2
    print(f">>> GENERATION COMPLETE | Time elapsed: {gen_time/60:.2f} minutes")
    
    # --- 4. REVERSE TRANSFORM & SAVE ---
    print("\n[4/4] Reversing to Physical Units and Splitting Data...")
    
    # Force strict boundaries to prevent floating-point crashes
    synthetic_uniforms = np.clip(synthetic_uniforms, epsilon, 1.0 - epsilon)
    
    syn_physical = np.zeros_like(synthetic_uniforms)
    for i in range(total_dims):
        syn_physical[:, i] = np.quantile(combined_flat_data[:, i], synthetic_uniforms[:, i], method='linear')

    output_dir = "./data/generated"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the 2048 columns back into 1024 LST and 1024 SM
    syn_lst = syn_physical[:, :1024]
    syn_sm = syn_physical[:, 1024:]
    
    # Save using the exact naming convention expected by your plotting scripts!
    lst_path = os.path.join(output_dir, 'copula_gaussian_lst_3000days.npy')
    sm_path = os.path.join(output_dir, 'copula_gaussian_sm_3000days.npy')
    
    np.save(lst_path, syn_lst)
    np.save(sm_path, syn_sm)
    
    print(f"\nSUCCESS! Data safely split and locked away at:")
    print(f" -> LST: {lst_path}")
    print(f" -> SM:  {sm_path}")
    print(f"\nTotal Pipeline Time: {(time.time() - t0)/60:.2f} minutes")

if __name__ == "__main__":
    generate_2048d_gaussian()