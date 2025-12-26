import itertools
import re
import time
import numpy as np
from torch import le
from my_vmd import VMD
from main import load_data
from config import Config
import matplotlib.pyplot as plt


def compute_mape(a, b, eps=1e-8):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    # avoid extreme values when a is near zero by using a small floor eps
    return np.mean(np.abs((a - b) / (a + eps))) * 100


def tune_vmd_params(max_evals=200, early_stop_mape=0.1):
    """Grid-search a small VMD parameter space to minimize reconstruction MAPE.
    Returns best_params dict and best_mape.
    If found mape <= early_stop_mape, update `config.py` VMD params automatically.
    """
    series, scaler = load_data("安徽数据集.xlsx")
    series = np.array(series).flatten()

    # parameter grids (kept small to limit runtime)
    K_list = [3, 4, 5, 6, 7, 8]
    alpha_list = [500, 1000, 1500, 2000, 3000, 4000, 5000, 10000]
    tau_list = [0,1]
    DC_list = [0,1]
    init_list = [0,1]
    tol_list = [1e-5, 1e-6, 1e-7, 1e-8]
    N_list = [500, 1000, 2000, 3000, 4000]

    best_mape = float('inf')
    best_params = None
    best_s_orig = None
    best_vmd_sum_orig = None
    eval_count = 0
    start_time = time.time()

    for K, alpha, tau, DC, init, tol, N in itertools.product(K_list, alpha_list, tau_list, DC_list, init_list, tol_list, N_list):
        if eval_count >= max_evals:
            break
        eval_count += 1

        try:
            imfs, _, _ = VMD(series, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol, N=N)
        except Exception as e:
            print(f"VMD failed for params K={K},alpha={alpha},init={init},tol={tol}: {e}")
            continue

        # align lengths
        min_len = min(series.shape[0], imfs.shape[1])
        s = series[:min_len]
        imfs = imfs[:, :min_len]

        vmd_sum = np.sum(imfs, axis=0)

        # compute MAPE in original (inverse-transformed) units to avoid division-by-zero on scaled values
        try:
            s_orig = scaler.inverse_transform(s.reshape(-1,1)).flatten()
            vmd_sum_orig = scaler.inverse_transform(vmd_sum.reshape(-1,1)).flatten()
            mape = compute_mape(s_orig, vmd_sum_orig, eps=1e-8)
        except Exception:
            # fallback to scaled-domain mape
            mape = compute_mape(s, vmd_sum, eps=1e-6)

        print(f"Eval {eval_count}: K={K},alpha={alpha},DC={DC},init={init},tol={tol},N={N} -> MAPE={mape:.6f}%")

        if mape < best_mape:
            best_mape = mape
            best_params = dict(K=K, alpha=alpha, tau=tau, DC=DC, init=init, tol=tol, N=N)
            # store best reconstruction for plotting
            try:
                best_s_orig = s_orig.copy()
                best_vmd_sum_orig = vmd_sum_orig.copy()
            except Exception:
                best_s_orig = s.copy()
                best_vmd_sum_orig = vmd_sum.copy()

            # if below threshold, write back to config
            if best_mape <= early_stop_mape:
                print("Found params under target MAPE. Updating config.py...")
                success = update_config_vmd_params(best_params)
                elapsed = time.time() - start_time
                print(f"Updated config.py: {success}. Time elapsed: {elapsed:.1f}s")
                # save comparison plots before returning
                try:
                    _save_tuner_plots(best_s_orig, best_vmd_sum_orig, best_params, best_mape)
                except Exception as e:
                    print('Failed to save tuner plots:', e)
                return best_params, best_mape

    elapsed = time.time() - start_time
    print(f"Tune finished. Best MAPE={best_mape:.6f} with params={best_params}. Time elapsed: {elapsed:.1f}s")
    # update config with best found even if above threshold
    if best_params is not None:
        update_config_vmd_params(best_params)
        # save comparison plots for best found
        try:
            _save_tuner_plots(best_s_orig, best_vmd_sum_orig, best_params, best_mape)
        except Exception as e:
            print('Failed to save tuner plots:', e)
    return best_params, best_mape


def update_config_vmd_params(params):
    """Safely update VMD params in config.py by replacing assignment lines.
    Returns True on success.
    """
    cfg_path = 'd:/vs_workspace/vmd-cnn-bilstm/config.py'
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        name_set = set(params.keys())
        new_lines = []
        replaced = set()
        for line in lines:
            stripped = line.lstrip()
            for name, val in params.items():
                if stripped.startswith(f"{name} =") or stripped.startswith(f"{name}="):
                    indent = line[:len(line) - len(stripped)]
                    new_lines.append(f"{indent}{name} = {val}\n")
                    replaced.add(name)
                    break
            else:
                new_lines.append(line)

        if not replaced:
            print('No replacements made in config.py; pattern mismatch')
            return False

        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return True
    except Exception as e:
        print('Failed to update config.py:', e)
        return False


def _save_tuner_plots(s_arr, vmd_sum_arr, params, mape, out_prefix='vmd_tuner_best'):
    """Save comparison and residual plots for best VMD reconstruction."""
    if s_arr is None or vmd_sum_arr is None:
        print('No data to plot')
        return

    s = np.array(s_arr).flatten()
    v = np.array(vmd_sum_arr).flatten()

    # main comparison
    plt.figure(figsize=(10,4))
    plt.plot(s, label='Original (best)')
    plt.plot(v, '--', label='VMD Sum (best)')
    plt.title(f"VMD Tuner Best Compare - MAPE={mape:.6f}%")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    comp_path = f"{out_prefix}_compare.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()

    # residual
    resid = s - v
    plt.figure(figsize=(10,3))
    plt.plot(resid, color='C3')
    plt.axhline(0, color='k', linewidth=0.6)
    plt.title('VMD Tuner Residual (orig - vmd_sum)')
    plt.xlabel('Index')
    plt.ylabel('Residual')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    resid_path = f"{out_prefix}_resid.png"
    plt.savefig(resid_path, dpi=300)
    plt.close()

    # write params summary
    try:
        with open(f"{out_prefix}_params.txt", 'w', encoding='utf-8') as f:
            f.write(f"best_mape={mape}\n")
            for k, val in params.items():
                f.write(f"{k}={val}\n")
    except Exception:
        pass

    print(f"Saved tuner plots: {comp_path}, {resid_path}")


if __name__ == '__main__':
    best_params, best_mape = tune_vmd_params()
    print('Result:', best_params, best_mape)
