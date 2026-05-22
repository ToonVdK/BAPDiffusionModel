import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import xarray as xr
import numpy as np

class SatelliteDataset(Dataset):
    def __init__(self, lst_path, sm_path, indices=None):
        """
        indices : array-like or None. If provided, only those time indices
                  are used. Otherwise the full dataset is kept.
        """
        print("Loading datasets into memory...")
        lst_full = xr.open_dataset(lst_path)
        sm_full = xr.open_dataset(sm_path)

        valid_months = [5, 6, 7, 8, 9]
        print(f"Filtering dataset for months: {valid_months}...")
        lst_summer = lst_full.sel(time=lst_full['time'].dt.month.isin(valid_months))
        sm_summer = sm_full.sel(time=sm_full['time'].dt.month.isin(valid_months))

        self.lst_data = lst_summer['LST_PMW'].values
        self.sm_data = sm_summer['sm'].values

        # Apply optional index selection
        if indices is not None:
            self.lst_data = self.lst_data[indices]
            self.sm_data = self.sm_data[indices]

        assert self.lst_data.shape[0] == self.sm_data.shape[0], "Mismatch in number of days!"
        self.num_days = self.lst_data.shape[0]
        print(f"Data successfully filtered. Remaining Summer Days: {self.num_days}")

        print("Calculating global normalization statistics...")
        # Calculate global mins and maxes, ignoring the NaNs
        self.lst_min = np.nanmin(self.lst_data)
        self.lst_max = np.nanmax(self.lst_data)

        self.sm_min = np.nanmin(self.sm_data)
        self.sm_max = np.nanmax(self.sm_data)

        print(f"LST Range: {self.lst_min:.2f}K to {self.lst_max:.2f}K")
        print(f"SM Range: {self.sm_min:.2f}mm to {self.sm_max:.2f}mm")

    def __len__(self):
        return self.num_days

    def normalize(self, data, min_val, max_val):
        """
        Scales array to [-1, 1] range. Normalization is also used in the MisDiff paper.
        It's needed to avoid exploding gradients during the backward pass.
        """
        return 2 * ((data - min_val) / (max_val - min_val)) - 1

    def __getitem__(self, idx):
        """
        Fetches a single day's data, handles the NaNs, and converts to Tensors.
        """
        # Extract the 2D arrays for the specific day
        lst = self.lst_data[idx]
        sm = self.sm_data[idx]

        # Create MissDiff Mask (1 = observed, 0 = missing/cloud)
        # ~np.isnan() returns True for observed data (not NaN)
        # AND operation for SM and LST returns a mask that is 1 only for observed data in both channels
        mask = (~np.isnan(lst) & ~np.isnan(sm)).astype(np.float32)

        # Normalize the data to [-1, 1]
        # (The NaNs will remain NaNs during this math operation, which is fine)
        lst_norm = self.normalize(lst, self.lst_min, self.lst_max)
        sm_norm = self.normalize(sm, self.sm_min, self.sm_max)

        # Handle NaNs by filling missing LST values with 0.0
        lst_norm = np.nan_to_num(lst_norm, nan=0.0)
        sm_norm = np.nan_to_num(sm_norm, nan=0.0)

        # Convert to PyTorch tensors and add a channel dimension
        # PyTorch expects images in format (Channels, Height, Width)
        # .unsqueeze(0) changes shape from (H, W) to (1, H, W)
        lst_tensor = torch.tensor(lst_norm, dtype=torch.float32).unsqueeze(0)
        sm_tensor = torch.tensor(sm_norm, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Stack LST and SM into a single 2-channel image tensor
        # Shape becomes (2, Height, Width)
        x_tensor = torch.cat([lst_tensor, sm_tensor], dim=0)

        # Optional: Pad the data --> If you want to use other shapes such as (32, 64)
        # Pad rows/columns to multiples of 8 (see model.py for the reasoning behind this)
        # Since we're using MissDiff there is no harm in padding, it will be ignored by the loss function.
        # Format: (left, right, top, bottom)
        pad_amounts = (0, 0, 0, 0)  # For now, no padding is needed since the shape is 32 x 32 already
        x_tensor_padded = F.pad(x_tensor, pad_amounts, mode='constant', value=0.0)  # Pad the data
        mask_tensor_padded = F.pad(mask_tensor, pad_amounts, mode='constant', value=0.0)  # Pad the mask

        # Return the combined data and the cloud mask (which is needed for the masked loss function)
        return x_tensor_padded, mask_tensor_padded
