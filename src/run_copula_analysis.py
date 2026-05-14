import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import jensenshannon

try:
    # Newer SciPy versions
    from scipy.stats import wasserstein_distance_nd
    HAS_WD_ND = True
except Exception:
    # Fallback if wasserstein_distance_nd is unavailable
    from scipy.stats import wasserstein_distance
    HAS_WD_ND = False

from copulas.multivariate import GaussianMultivariate, VineCopula
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def calculate_2d_jsd(real_x, real_y, ai_x, ai_y, bins=50):
    """
    Jensen-Shannon distance between two 2D empirical distributions.
    Returns the Jensen-Shannon distance (not squared divergence).
    """
    x_min = min(np.min(real_x), np.min(ai_x))
    x_max = max(np.max(real_x), np.max(ai_x))
    y_min = min(np.min(real_y), np.min(ai_y))
    y_max = max(np.max(real_y), np.max(ai_y))

    bounds = [[x_min, x_max], [y_min, y_max]]

    real_hist, _, _ = np.histogram2d(real_x, real_y, bins=bins, range=bounds, density=True)
    ai_hist, _, _ = np.histogram2d(ai_x, ai_y, bins=bins, range=bounds, density=True)

    real_prob = real_hist.flatten().astype(float)
    ai_prob = ai_hist.flatten().astype(float)

    epsilon = 1e-6
    real_prob += epsilon
    ai_prob += epsilon

    real_prob /= np.sum(real_prob)
    ai_prob /= np.sum(ai_prob)

    return jensenshannon(real_prob, ai_prob)


def calculate_2d_wasserstein(real_x, real_y, ai_x, ai_y, max_samples=1000, seed=42):
    """
    2D Wasserstein distance with optional subsampling for stability/performance.
    If scipy.stats.wasserstein_distance_nd is unavailable, falls back to the
    mean of 1D Wasserstein distances across the two dimensions.
    """
    real_coords = np.column_stack((real_x, real_y))
    ai_coords = np.column_stack((ai_x, ai_y))

    rng = np.random.default_rng(seed)

    if len(real_coords) > max_samples:
        idx_real = rng.choice(len(real_coords), max_samples, replace=False)
        real_coords = real_coords[idx_real]

    if len(ai_coords) > max_samples:
        idx_ai = rng.choice(len(ai_coords), max_samples, replace=False)
        ai_coords = ai_coords[idx_ai]

    print(
        f"Calculating Wasserstein Distance for {len(real_coords)} real coordinates "
        f"and {len(ai_coords)} AI coordinates"
    )

    if HAS_WD_ND:
        return wasserstein_distance_nd(real_coords, ai_coords)

    # Fallback: average of dimension-wise Wasserstein distances
    w0 = wasserstein_distance(real_coords[:, 0], ai_coords[:, 0])
    w1 = wasserstein_distance(real_coords[:, 1], ai_coords[:, 1])
    return 0.5 * (w0 + w1)


def run_combined_plot():
    print("--- PHASE 1: DATA PREPARATION ---")
    print("Loading datasets...")
    lst_ds_full = xr.open_dataset("./data/processed/aligned_lst.nc")
    sm_ds_full = xr.open_dataset("./data/processed/aligned_sm.nc")

    # Strict fairness: summer only
    valid_months = [5, 6, 7, 8, 9]
    lst_summer = lst_ds_full.sel(time=lst_ds_full["time"].dt.month.isin(valid_months))
    sm_summer = sm_ds_full.sel(time=sm_ds_full["time"].dt.month.isin(valid_months))

    real_lst = lst_summer["LST_PMW"].values
    real_sm = sm_summer["sm"].values

    # Antwerp pixel
    row_A, col_A = 11, 17
    lst_A = real_lst[:, row_A, col_A]
    sm_A = real_sm[:, row_A, col_A]

    valid_mask_cross = ~np.isnan(lst_A) & ~np.isnan(sm_A)
    lst_clean = lst_A[valid_mask_cross]
    sm_clean = sm_A[valid_mask_cross]

    real_data_array = np.column_stack((sm_clean, lst_clean))
    print(f"Prepared {len(real_data_array)} mathematically valid summer days for Antwerp.")

    # Convert to uniform percentiles
    u_sm = (rankdata(sm_clean) - 0.5) / len(sm_clean)
    v_lst = (rankdata(lst_clean) - 0.5) / len(lst_clean)
    uniform_data_array = np.column_stack((u_sm, v_lst))

    print(f"Prepared {len(uniform_data_array)} mathematically valid uniform days for Antwerp.")

    print("\n--- PHASE 2: GAUSSIAN COPULA FITTING ---")
    real_rho_cross, _ = spearmanr(sm_clean, lst_clean)
    print(f"Real Cross-Variable Spearman ρ: {real_rho_cross:.3f}")

    print("Fitting Gaussian Copula...")
    baseline_copula = GaussianMultivariate()
    df_uniform = pd.DataFrame(uniform_data_array, columns=["sm", "lst"]) # Convert to Pandas Dataframe which is compatible with the copulas library

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data does not appear to be uniform")
        baseline_copula.fit(df_uniform)

    print("SUCCESS! Proceeding with Gaussian baseline model.")

    print("Generating 3000 synthetic baseline days...")
    synthetic_uniforms_cross = baseline_copula.sample(3000).to_numpy()

    syn_sm = np.quantile(sm_clean, synthetic_uniforms_cross[:, 0])
    syn_lst = np.quantile(lst_clean, synthetic_uniforms_cross[:, 1])

    syn_rho_cross, _ = spearmanr(syn_sm, syn_lst)
    js_cross = calculate_2d_jsd(lst_clean, sm_clean, syn_lst, syn_sm)
    w_cross = calculate_2d_wasserstein(lst_clean, sm_clean, syn_lst, syn_sm)

    print(
        f"Cross-Variable | Real ρ: {real_rho_cross:.3f} | "
        f"JSD: {js_cross:.3f} | Wasserstein: {w_cross:.3f}"
    )

    print("\n--- PHASE 3: SPATIAL VINE COPULA FITTING ---")
    print("Loading datasets for spatial analysis...")
    lst_ds_full_spatial = xr.open_dataset("./data/processed/aligned_lst.nc")

    lst_summer_spatial = lst_ds_full_spatial.sel(
        time=lst_ds_full_spatial["time"].dt.month.isin(valid_months)
    )
    real_lst_spatial = lst_summer_spatial["LST_PMW"].values

    # 3x3 grid around Antwerp
    lst_grid = real_lst_spatial[:, row_A - 1 : row_A + 2, col_A - 1 : col_A + 2]
    total_days = lst_grid.shape[0]
    lst_flat = lst_grid.reshape(total_days, 9)

    valid_mask_spatial = ~np.isnan(lst_flat).any(axis=1)
    lst_clean_spatial = lst_flat[valid_mask_spatial]
    print(f"Prepared {len(lst_clean_spatial)} fully valid 3x3 spatial days.")

    epsilon = 1e-10
    u_lst = np.zeros_like(lst_clean_spatial, dtype=float)
    for i in range(9):
        ranks = rankdata(lst_clean_spatial[:, i])
        u_lst[:, i] = np.clip((ranks - 0.5) / len(lst_clean_spatial), epsilon, 1 - epsilon)

    print(f"Uniform data ranges: [{u_lst.min():.6f}, {u_lst.max():.6f}]")

    print("Fitting 9-Dimensional Center-Vine Copula...")
    vine_copula = VineCopula("center")
    columns = [f"pixel_{i}" for i in range(9)]
    df_uniform_spatial = pd.DataFrame(u_lst, columns=columns)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Data does not appear to be uniform")
        vine_copula.fit(df_uniform_spatial)

    print("SUCCESS! Vine Copula fitted.")

    print("Generating 3000 synthetic spatial fields...")
    synthetic_uniforms_spatial = vine_copula.sample(3000).to_numpy()
    synthetic_uniforms_spatial = np.clip(synthetic_uniforms_spatial, epsilon, 1 - epsilon)
    print(
        f"Synthetic uniform ranges: "
        f"[{synthetic_uniforms_spatial.min():.6f}, {synthetic_uniforms_spatial.max():.6f}]"
    )

    syn_lst_spatial = np.zeros_like(synthetic_uniforms_spatial, dtype=float)
    for i in range(9):
        syn_lst_spatial[:, i] = np.quantile(
            lst_clean_spatial[:, i], synthetic_uniforms_spatial[:, i]
        )

    assert not np.any(np.isnan(syn_lst_spatial)), "NaN values detected in synthetic spatial data"
    print("Reverse transform completed successfully.")

    center_idx = 4
    south_idx = 7

    real_rho_spatial, _ = spearmanr(
        lst_clean_spatial[:, center_idx], lst_clean_spatial[:, south_idx]
    )
    syn_rho_spatial, _ = spearmanr(
        syn_lst_spatial[:, center_idx], syn_lst_spatial[:, south_idx]
    )

    js_spatial = calculate_2d_jsd(
        lst_clean_spatial[:, center_idx],
        lst_clean_spatial[:, south_idx],
        syn_lst_spatial[:, center_idx],
        syn_lst_spatial[:, south_idx],
    )
    w_spatial = calculate_2d_wasserstein(
        lst_clean_spatial[:, center_idx],
        lst_clean_spatial[:, south_idx],
        syn_lst_spatial[:, center_idx],
        syn_lst_spatial[:, south_idx],
    )

    print(
        f"Spatial | Real ρ: {real_rho_spatial:.3f} | Synthetic ρ: {syn_rho_spatial:.3f} | "
        f"JSD: {js_spatial:.3f} | Wasserstein: {w_spatial:.3f}"
    )

    # --- ONE FIGURE WITH BOTH PANELS ---
    print("\n--- PHASE 4: PLOTTING ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Cross-variable panel
    axes[0].set_title(
        "Cross-Variable Dependence (Gaussian Copula)\n"
        f"Real ρ={real_rho_cross:.2f} | Copula ρ={syn_rho_cross:.3f} | JSD={js_cross:.3f} | W={w_cross:.3f}",
        fontsize=13,
    )
    sns.kdeplot(
        x=sm_clean,
        y=lst_clean,
        ax=axes[0],
        cmap="mako",
        fill=True,
        alpha=0.5,
        thresh=0.05,
    )
    sns.kdeplot(
        x=syn_sm,
        y=syn_lst,
        ax=axes[0],
        color="darkorange",
        linewidths=2,
        levels=5,
        thresh=0.05,
    )
    axes[0].set_xlabel("Volumetric Soil Moisture (m³/m³)")
    axes[0].set_ylabel("LST (K)")
    axes[0].grid(alpha=0.3)

    # Spatial panel
    axes[1].set_title(
        "Spatial Dependence (LST Pixel A vs B) Using 9D Vine Copula\n"
        f"Real ρ={real_rho_spatial:.2f} | Copula ρ={syn_rho_spatial:.2f} | JSD={js_spatial:.3f} | W={w_spatial:.3f}",
        fontsize=13,
    )
    sns.kdeplot(
        x=lst_clean_spatial[:, center_idx],
        y=lst_clean_spatial[:, south_idx],
        ax=axes[1],
        cmap="mako",
        fill=True,
        alpha=0.5,
        thresh=0.05,
    )
    sns.kdeplot(
        x=syn_lst_spatial[:, center_idx],
        y=syn_lst_spatial[:, south_idx],
        ax=axes[1],
        color="darkorange",
        linewidths=2,
        levels=5,
        thresh=0.05,
    )
    axes[1].set_xlabel("LST at Antwerp (Center Pixel) (K)")
    axes[1].set_ylabel("LST at South Neighboring Pixel (K)")
    axes[1].grid(alpha=0.3)

    # Shared legend
    real_patch = mpatches.Patch(color="teal", alpha=0.5, label="Real Summer Data")
    copula_line = mlines.Line2D(
        [], [], color="darkorange", linewidth=2, label="Copula Baseline"
    )
    fig.legend(
        handles=[real_patch, copula_line],
        loc="lower center",
        ncol=2,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    plt.savefig("copula_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved 'copula_analysis.png'")


if __name__ == "__main__":
    run_combined_plot()