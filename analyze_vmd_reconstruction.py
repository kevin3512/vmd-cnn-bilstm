import numpy as np
from main import load_data, vmd_decompose
import matplotlib.pyplot as plt


def analyze():
    series, scaler = load_data("安徽数据集.xlsx")
    imfs = vmd_decompose(series)

    # ensure shapes
    K, N = imfs.shape
    series = np.array(series).flatten()

    vmd_sum = np.sum(imfs, axis=0)

    residual = series - vmd_sum

    rmse = np.sqrt(np.mean(residual**2))
    mae = np.mean(np.abs(residual))
    max_abs = np.max(np.abs(residual))
    idx_max = int(np.argmax(np.abs(residual)))

    print(f"VMD reconstruction RMSE: {rmse:.8f}")
    print(f"VMD reconstruction MAE: {mae:.8f}")
    print(f"VMD reconstruction max abs: {max_abs:.8f} at index {idx_max}")

    # show some residual samples around max
    start = max(0, idx_max-5)
    end = min(N, idx_max+6)
    print("Residuals around largest error:")
    for i in range(start, end):
        print(f"i={i:4d}, series={series[i]:.8f}, vmd_sum={vmd_sum[i]:.8f}, resid={residual[i]:.8f}")

    # save residual plot
    plt.figure(figsize=(10,4))
    plt.plot(residual, linewidth=1)
    plt.axhline(0, color='k', linewidth=0.6)
    plt.title('VMD Reconstruction Residual (series - sum(imfs))')
    plt.xlabel('Time Index')
    plt.ylabel('Residual')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('vmd_reconstruction_residual.png', dpi=300)
    print('Saved plot: vmd_reconstruction_residual.png')

if __name__ == '__main__':
    analyze()
