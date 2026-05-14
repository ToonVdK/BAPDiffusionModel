import time, warnings, sys, os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from copulas.multivariate import GaussianMultivariate

def generate_2048d_systematic(test_step=5):
    """
    Fit a 2048‑D Gaussian copula on a systematic train/test split.
    test_step : int, keep every `test_step`‑th sample for testing (train = all others).
    """
    print(f"\n{'='*60}")
    print(f"2048D GAUSSIAN COPULA (SYSTEMATIC TEST STEP = {test_step})")
    print(f"{'='*60}")

    valid_months = [5,6,7,8,9]
    ds_lst = xr.open_dataset('./data/processed/aligned_lst.nc')
    ds_sm  = xr.open_dataset('./data/processed/aligned_sm.nc')

    lst_summer = ds_lst.sel(time=ds_lst['time'].dt.month.isin(valid_months))['LST_PMW'].values[:, 0:32, 0:32]
    sm_summer  = ds_sm.sel( time=ds_sm['time'].dt.month.isin(valid_months))['sm'].values[:, 0:32, 0:32]

    n_total = lst_summer.shape[0]
    # Systematic indices: e.g., test_step=5 => test indices 0,5,10,...
    test_idx = np.arange(0, n_total, test_step)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)

    n_train, n_test = len(train_idx), len(test_idx)
    print(f"Total summer days: {n_total}")
    print(f"Training: {n_train} (every {test_step}‑th day held out)")
    print(f"Test:     {n_test}")

    train_lst = lst_summer[train_idx].reshape(n_train, 1024)
    test_lst  = lst_summer[test_idx].reshape(n_test, 1024)
    train_sm  = sm_summer[train_idx].reshape(n_train, 1024)
    test_sm   = sm_summer[test_idx].reshape(n_test, 1024)

    # Impute training NaNs
    def impute(data):
        col_means = np.nanmean(data, axis=0)
        global_mean = np.nanmean(data)
        col_means[np.isnan(col_means)] = global_mean
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_means, inds[1])
        data += np.random.normal(0, 1e-4, size=data.shape)
        return data

    train_lst = impute(train_lst.copy())
    train_sm  = impute(train_sm.copy())

    combined_train = np.hstack([train_lst, train_sm])
    total_dims = combined_train.shape[1]

    # Uniform transform on training data only
    epsilon = 1e-10
    u_train = np.zeros_like(combined_train)
    for i in range(total_dims):
        ranks = rankdata(combined_train[:, i])
        u_train[:, i] = np.clip((ranks - 0.5) / n_train, epsilon, 1 - epsilon)

    df_train = pd.DataFrame(u_train, columns=[f'var_{i}' for i in range(total_dims)])

    print("Fitting 2048‑D Gaussian Copula...")
    t0 = time.time()
    model = GaussianMultivariate()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(df_train)
    print(f"Fitted in {time.time()-t0:.1f} s")

    # Generate as many synthetic test maps as real test days
    print(f"Generating {n_test} synthetic maps...")
    syn_uniforms = model.sample(n_test).to_numpy()
    syn_uniforms = np.clip(syn_uniforms, epsilon, 1 - epsilon)

    # Reverse transform using training quantiles
    syn_physical = np.zeros_like(syn_uniforms)
    for i in range(total_dims):
        syn_physical[:, i] = np.quantile(combined_train[:, i], syn_uniforms[:, i], method='linear')

    syn_lst = syn_physical[:, :1024].reshape(n_test, 32, 32)
    syn_sm  = syn_physical[:, 1024:].reshape(n_test, 32, 32)

    # Flatten the training data (the same combined_train used for fitting)
    train_flat = combined_train  # shape (n_train, 2048)
    syn_flat = syn_physical        # shape (n_test, 2048)

    # For each synthetic sample, find the nearest training sample (Euclidean distance)
    tree = cKDTree(train_flat)
    distances, indices = tree.query(syn_flat, k=1)
    print("Nearest neighbour distances to training set:")
    print(f"  Mean distance: {np.mean(distances):.6f}")
    print(f"  Std distance:  {np.std(distances):.6f}")
    print(f"  Min distance:  {np.min(distances):.6f}")
    print(f"  Max distance:  {np.max(distances):.6f}")
    print(f"  Fraction of samples with distance < 1e-6: {np.mean(distances < 1e-6)*100:.1f}%")

    # Save synthetic and real test arrays
    out_dir = "./data/generated"
    os.makedirs(out_dir, exist_ok=True)
    lst_cop_path = os.path.join(out_dir, f'copula_gaussian_lst_oos_test_step{test_step}.npy')
    sm_cop_path  = os.path.join(out_dir, f'copula_gaussian_sm_oos_test_step{test_step}.npy')
    lst_real_path = os.path.join(out_dir, f'real_lst_oos_test_step{test_step}.npy')
    sm_real_path  = os.path.join(out_dir, f'real_sm_oos_test_step{test_step}.npy')

    np.save(lst_cop_path, syn_lst)
    np.save(sm_cop_path, syn_sm)
    np.save(lst_real_path, test_lst.reshape(n_test, 32, 32))
    np.save(sm_real_path, test_sm.reshape(n_test, 32, 32))

    print("Files saved with suffix 'step{test_step}'.")
    print("Done.")


if __name__ == "__main__":
    # default test_step=5 (20% test)
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    generate_2048d_systematic(test_step=step)