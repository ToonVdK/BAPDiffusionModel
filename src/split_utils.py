import numpy as np
import xarray as xr

def get_chronological_split(train_ratio=0.8):
    """
    Returns (train_indices, test_indices) for a chronological split.
    Assumes the time dimension is already sorted ascending.
    """
    ds = xr.open_dataset('./data/processed/aligned_lst.nc')
    valid_months = [5, 6, 7, 8, 9]
    summer = ds.sel(time=ds['time'].dt.month.isin(valid_months))
    n_total = len(summer['time'])
    ds.close()

    split_point = int(n_total * train_ratio)
    train_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n_total)
    return train_idx, test_idx

def get_train_indices():
    """
    Convenience function for training scripts (returns only train indices).
    """
    train_idx, _ = get_chronological_split()
    return train_idx

def get_test_indices():
    _, test_idx = get_chronological_split()
    return test_idx