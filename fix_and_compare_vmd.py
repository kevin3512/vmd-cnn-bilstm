import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD
import matplotlib.pyplot as plt

from main import vmd_decompose


def load_raw_values(file_path, nrows=0, date_col='date', value_col='value'):
    if nrows == 0:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_excel(file_path, nrows=nrows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    vals = df[value_col].astype(float).values
    return df, vals


def simple_fix(series):
    """Fix zero values and isolated large jumps by interpolation.
    - Replace exact zeros or NaNs by linear interpolation of neighbors.
    - For large jumps (diff z-score > 5), replace that single point by interpolation between neighbors.
    Returns fixed copy and list of fixed indices.
    """
    s = series.astype(float).copy()
    fixed_idx = set()

    # Replace NaN / exact zeros
    isnan_or_zero = np.isnan(s) | (s == 0)
    if isnan_or_zero.any():
        idxs = np.where(isnan_or_zero)[0]
        fixed_idx.update(idxs.tolist())
        # linear interp over indices
        good = ~isnan_or_zero
        s[isnan_or_zero] = np.interp(idxs, np.where(good)[0], s[good])

    # Detect isolated large jumps (zscore of diffs)
    diffs = np.diff(s)
    mu = np.mean(diffs)
    sigma = np.std(diffs) if np.std(diffs) > 0 else 1.0
    z = (diffs - mu) / sigma
    jump_positions = np.where(np.abs(z) > 5)[0]  # position i means jump between i and i+1
    for pos in jump_positions:
        # attempt to repair by averaging neighbors at pos and pos+1
        i = pos
        j = pos + 1
        # replace the later point j by interpolation of i-1 and j+1 when available
        left = i-1
        right = j+1
        if left >= 0 and right < len(s):
            new = (s[left] + s[right]) / 2.0
            s[j] = new
            fixed_idx.add(j)
        elif left >= 0:
            s[j] = s[left]
            fixed_idx.add(j)
        elif right < len(s):
            s[j] = s[right]
            fixed_idx.add(j)

    return s, sorted(list(fixed_idx))


def vmd_reconstruct_metrics(series_scaled, imfs):
    vmd_sum = np.sum(imfs, axis=0)
    residual = series_scaled - vmd_sum
    rmse = np.sqrt(np.mean(residual**2))
    mae = np.mean(np.abs(residual))
    max_abs = np.max(np.abs(residual))
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'max_abs': float(max_abs),
        'residual': residual,
        'vmd_sum': vmd_sum
    }


def plot_comparison(original, vmd_sum, fixed_original, fixed_vmd_sum, save_path='vmd_fix_compare.png'):
    plt.figure(figsize=(12,5))
    L = min(200, len(original))
    plt.plot(original[-L:], label='Original (test tail)')
    plt.plot(vmd_sum[-L:], '--', label='VMD Sum (orig)')
    plt.plot(fixed_original[-L:], label='Fixed (test tail)')
    plt.plot(fixed_vmd_sum[-L:], '--', label='VMD Sum (fixed)')
    plt.legend()
    plt.title('Original vs VMD Sum — before/after fix (tail)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    file_path = '安徽数据集.xlsx'
    df, vals = load_raw_values(file_path)
    N = len(vals)
    print(f'Loaded {N} raw values')

    inspect_start = 390
    inspect_end = 400
    print('\nRaw rows 390-400:')
    for i in range(inspect_start, inspect_end+1):
        if i < N:
            print(i, df.loc[i].to_dict())

    fixed_vals, fixed_indices = simple_fix(vals)
    print('\nFixed indices:', fixed_indices)

    # scale both series separately (to keep VMD behavior comparable to earlier pipeline)
    scaler_orig = MinMaxScaler()
    s_scaled = scaler_orig.fit_transform(vals.reshape(-1,1)).flatten()

    scaler_fixed = MinMaxScaler()
    s_fixed_scaled = scaler_fixed.fit_transform(fixed_vals.reshape(-1,1)).flatten()

    # VMD
    imfs_orig = vmd_decompose(s_scaled)
    imfs_fixed = vmd_decompose(s_fixed_scaled)

    # VMD implementation may return imfs with length slightly shorter than input
    # align by trimming to the minimum length
    min_len = min(s_scaled.shape[0], imfs_orig.shape[1], imfs_fixed.shape[1])
    s_scaled_trim = s_scaled[:min_len]
    s_fixed_scaled_trim = s_fixed_scaled[:min_len]
    imfs_orig = imfs_orig[:, :min_len]
    imfs_fixed = imfs_fixed[:, :min_len]

    m_orig = vmd_reconstruct_metrics(s_scaled_trim, imfs_orig)
    m_fixed = vmd_reconstruct_metrics(s_fixed_scaled_trim, imfs_fixed)

    print('\nBefore fix: RMSE={:.8f}, MAE={:.8f}, max_abs={:.8f}'.format(m_orig['rmse'], m_orig['mae'], m_orig['max_abs']))
    print('After  fix: RMSE={:.8f}, MAE={:.8f}, max_abs={:.8f}'.format(m_fixed['rmse'], m_fixed['mae'], m_fixed['max_abs']))

    # save residual plots
    plt.figure(figsize=(10,4))
    plt.plot(m_orig['residual'], label='resid_before')
    plt.plot(m_fixed['residual'], label='resid_after')
    plt.legend()
    plt.title('VMD Reconstruction Residual — before vs after')
    plt.tight_layout()
    plt.savefig('vmd_resid_before_after.png', dpi=300)
    print('\nSaved plot: vmd_resid_before_after.png')

    # comparison plot (tail)
    plot_comparison(s_scaled_trim, m_orig['vmd_sum'], s_fixed_scaled_trim, m_fixed['vmd_sum'], save_path='vmd_fix_compare.png')
    print('Saved plot: vmd_fix_compare.png')

    # show a few sample values around previously found problematic index 396
    idx = 396
    if idx < N:
        print('\nSample around index', idx)
        for i in range(max(0, idx-5), min(N, idx+6)):
            print(i, 'orig=', vals[i], 'fixed=', fixed_vals[i], 'scaled_orig={:.6f}'.format(s_scaled[i]), 'scaled_fixed={:.6f}'.format(s_fixed_scaled[i]))
