"""ssa_vmd_tuner.py
Sparrow Search Algorithm (SSA) for tuning VMD parameters to minimize MAPE
between the sum of IMFs and the original series (in original units).

Usage:
    python ssa_vmd_tuner.py

The script uses `Config.file_name` via `main.load_data` for the dataset, and
saves best params and comparison plots into the working directory.
"""

import time
import math
import random
import numpy as np
from typing import Dict, Tuple

from config import Config
from main import load_data
from vmd_param_tuner import compute_mape, update_config_vmd_params, _save_tuner_plots

try:
    from vmdpy import VMD
except Exception:
    VMD = None


# Parameter bounds and types
PARAM_BOUNDS = {
    'K': (3, 12, int),               # number of modes
    'alpha': (100, 10000, float),    # bandwidth constraint
    'tau': (0.0, 1.0, float),        # time-step
    'DC': (0, 1, int),               # include DC (0 or 1)
    'init': (0, 1, int),             # init method (0/1)
    'tol': (1e-8, 1e-3, float),      # convergence tolerance
    'N': (100, 5000, int),           # (optional) max iterations
}


def _ensure_type(name, val):
    _, _, t = PARAM_BOUNDS[name]
    if t is int:
        return int(round(val))
    return float(val)


def _clip_param(name, val):
    lo, hi, _ = PARAM_BOUNDS[name]
    return max(lo, min(hi, val))


def vector_to_params(x: np.ndarray) -> Dict:
    """Map a continuous vector x in [0,1]^d to actual param dict."""
    names = list(PARAM_BOUNDS.keys())
    params = {}
    for i, name in enumerate(names):
        lo, hi, t = PARAM_BOUNDS[name]
        v = lo + x[i] * (hi - lo)
        v = _clip_param(name, v)
        params[name] = _ensure_type(name, v)
    return params


def evaluate_params(params: Dict, series: np.ndarray, scaler) -> float:
    """Return MAPE (lower is better). On failure return a large penalty."""
    try:
        # Try to call VMD; accept kwargs, fallback when not supported
        try:
            imfs, _, _ = VMD(series, alpha=params['alpha'], tau=params['tau'], K=params['K'], DC=params['DC'], init=params['init'], tol=params['tol'], N=params['N'])
        except TypeError:
            imfs, _, _ = VMD(series, alpha=params['alpha'], tau=params['tau'], K=params['K'], DC=params['DC'], init=params['init'], tol=params['tol'])

        # align lengths
        min_len = min(series.shape[0], imfs.shape[1])
        s = series[:min_len]
        imfs = imfs[:, :min_len]
        vmd_sum = np.sum(imfs, axis=0)

        # compute MAPE in original units
        try:
            s_orig = scaler.inverse_transform(s.reshape(-1,1)).flatten()
            vmd_sum_orig = scaler.inverse_transform(vmd_sum.reshape(-1,1)).flatten()
            mape = compute_mape(s_orig, vmd_sum_orig, eps=1e-8)
        except Exception:
            mape = compute_mape(s, vmd_sum, eps=1e-6)

        if not np.isfinite(mape):
            return 1e6
        return float(mape)
    except Exception as e:
        # VMD failed for these params; penalize
        # print short trace for diagnostics
        print(f"evaluate_params: VMD failed for {params}: {e}")
        return 1e6


class SSAOptimizer:
    def __init__(self, dim, pop_size=20, max_iter=50, pd=0.2, sd=0.1, stop_threshold=None):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        # safety checks
        if self.pop_size < 4:
            raise ValueError('pop_size must be >= 4')

    def optimize(self, fitness_fn):
        # initialize population in [0,1]^dim
        X = np.random.rand(self.pop_size, self.dim)
        fitness = np.full(self.pop_size, np.inf)

        # evaluate initial population
        for i in range(self.pop_size):
            fitness[i] = fitness_fn(X[i])

        best_idx = int(np.argmin(fitness))
        best = X[best_idx].copy()
        best_score = fitness[best_idx]

        print(f"SSA start: pop_size={self.pop_size}, max_iter={self.max_iter}, initial best={best_score:.6f}")

        for t in range(1, self.max_iter+1):
            # generate random helpers
            r2 = np.random.rand(self.pop_size)
            r3 = np.random.rand(self.pop_size)

            # identify producers (20%) and scroungers
            num_producers = max(1, int(0.2 * self.pop_size))
            indices = np.argsort(fitness)  # ascending
            producers = indices[:num_producers]
            scroungers = indices[num_producers:]

            # Producers update (exploit)
            for idx in producers:
                if np.random.rand() < 0.8:
                    X[idx] = X[idx] * np.exp(-idx / (t * np.random.rand() + 1e-6))
                else:
                    X[idx] = X[idx] + np.random.normal(0, 0.1, size=self.dim) * X[idx]
                X[idx] = np.clip(X[idx], 0.0, 1.0)

            # Scroungers update (follow good ones)
            for k, idx in enumerate(scroungers):
                best_vec = best
                r = np.random.rand()
                X[idx] = r * X[idx] + (1 - r) * best_vec
                X[idx] = np.clip(X[idx], 0.0, 1.0)

            # Awareness of danger (some will flee randomly)
            danger_prob = 0.15
            for idx in range(self.pop_size):
                if np.random.rand() < danger_prob:
                    X[idx] = np.random.rand(self.dim)

            # Evaluate
            for i in range(self.pop_size):
                fitness[i] = fitness_fn(X[i])

            # update best
            idx_min = int(np.argmin(fitness))
            if fitness[idx_min] < best_score:
                best_score = fitness[idx_min]
                best = X[idx_min].copy()

            print(f"Iter {t}/{self.max_iter}: best MAPE={best_score:.6f}")

            if self.stop_threshold is not None and best_score <= self.stop_threshold:
                print("Stop threshold reached")
                break

        return best, best_score


def run_ssa_tuner(pop_size=20, max_iter=50, early_stop_mape=None, save_best=True):
    # load series and scaler from main.load_data
    series, scaler = load_data(Config.file_name)
    series = np.array(series).flatten()

    dim = len(PARAM_BOUNDS)
    ssa = SSAOptimizer(dim=dim, pop_size=pop_size, max_iter=max_iter, stop_threshold=early_stop_mape)

    def fitness(x_vec):
        params = vector_to_params(x_vec)
        return evaluate_params(params, series, scaler)

    start = time.time()
    best_vec, best_score = ssa.optimize(fitness)
    elapsed = time.time() - start

    best_params = vector_to_params(best_vec)
    print(f"SSA finished in {elapsed:.1f}s: best MAPE={best_score:.6f}, params={best_params}")

    # save plots and optionally update config
    # recompute best arrays for plotting
    try:
        try:
            imfs, _, _ = VMD(series, alpha=best_params['alpha'], tau=best_params['tau'], K=best_params['K'], DC=best_params['DC'], init=best_params['init'], tol=best_params['tol'], N=best_params['N'])
        except TypeError:
            imfs, _, _ = VMD(series, alpha=best_params['alpha'], tau=best_params['tau'], K=best_params['K'], DC=best_params['DC'], init=best_params['init'], tol=best_params['tol'])

        min_len = min(series.shape[0], imfs.shape[1])
        s = series[:min_len]
        imfs = imfs[:, :min_len]
        vmd_sum = np.sum(imfs, axis=0)

        try:
            s_orig = scaler.inverse_transform(s.reshape(-1,1)).flatten()
            vmd_sum_orig = scaler.inverse_transform(vmd_sum.reshape(-1,1)).flatten()
        except Exception:
            s_orig = s.copy()
            vmd_sum_orig = vmd_sum.copy()

        _save_tuner_plots(s_orig, vmd_sum_orig, best_params, best_score, out_prefix='ssa_vmd_best')

    except Exception as e:
        print('Failed to save SSA best reconstruction plots:', e)

    if save_best:
        print('Updating config with best params...')
        success = update_config_vmd_params(best_params)
        print('update_config_vmd_params ->', success)

    return best_params, best_score


if __name__ == '__main__':
    # small default run
    best_params, best_score = run_ssa_tuner(pop_size=24, max_iter=100, early_stop_mape=None, save_best=True)
    print('Result:', best_params, best_score)
