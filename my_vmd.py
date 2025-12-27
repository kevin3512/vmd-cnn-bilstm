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
    imfs_with_residual = np.vstack([aligned_imfs, residual])

    return imfs_with_residual
