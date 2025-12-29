import numpy as np
from vmdpy import VMD
from config import Config

def vmd_decompose(
    series,
    K=Config.K,
    alpha=Config.alpha,
    tau=Config.tau,
    DC=Config.DC,
    init=Config.init,
    tol=Config.tol
    ,
    fs=1.0  # sampling frequency in samples per unit time (default: 1.0)
):
    """
    VMD decomposition with explicit residual IMF.
    Ensures length alignment between series and reconstructed signal.
    """
    series = np.asarray(series).astype(float)
    T = len(series)

    # VMD decomposition
    imfs, _, _ = VMD(
        series,
        alpha=alpha,
        tau=tau,
        K=K,
        DC=DC,
        init=init,
        tol=tol
    )  # shape: (K, T') where T' may differ from T

    # Reconstructed signal
    recon = np.sum(imfs, axis=0)

    # ---- Length alignment (critical) ----
    if len(recon) > T:
        recon = recon[:T]
    elif len(recon) < T:
        # pad with last value (very conservative)
        pad_len = T - len(recon)
        recon = np.pad(recon, (0, pad_len), mode="edge")

    # Explicit residual IMF
    residual = series - recon

    # Align IMFs as well
    aligned_imfs = []
    for imf in imfs:
        if len(imf) > T:
            aligned_imfs.append(imf[:T])
        elif len(imf) < T:
            aligned_imfs.append(
                np.pad(imf, (0, T - len(imf)), mode="edge")
            )
        else:
            aligned_imfs.append(imf)

    aligned_imfs = np.array(aligned_imfs)

    # Append residual as the last IMF
    if(Config.hasResidual):
        print("Adding residual IMF as the last component.")
        imfs_with_residual = np.vstack([aligned_imfs, residual])
    else:
        print("Not adding residual IMF.")
        imfs_with_residual = aligned_imfs

    # ---- Compute and print center frequency for each IMF ----
    # Use power spectral centroid: sum(f * P(f)) / sum(P(f)) where P is power
    center_freqs = []
    N = imfs_with_residual.shape[1]
    # frequency bins for rfft
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    for idx, imf in enumerate(imfs_with_residual):
        # compute one-sided FFT and power
        X = np.fft.rfft(imf)
        P = np.abs(X) ** 2
        total_power = P.sum()
        if total_power <= 0:
            cf = 0.0
        else:
            cf = (freqs * P).sum() / total_power
        center_freqs.append(cf)
        print(f"IMF {idx}: center frequency = {cf:.6f} (Hz)")

    return imfs_with_residual
