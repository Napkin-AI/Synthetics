"""Dataset distances; no neural-network or model-download dependency."""
import numpy as np
from scipy.linalg import eigvalsh, svdvals
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def pair_arrays(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('Expected two feature matrices with the same feature dimension')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite features')
    return x, y


def insufficient(x, y, minimum):
    if min(len(x), len(y)) < minimum:
        return {'status': 'insufficient_samples', 'value': None,
                'real_count': len(x), 'synthetic_count': len(y), 'minimum_per_dataset': minimum}
    return None


def fid(x, y):
    x, y = pair_arrays(x, y)
    missing = insufficient(x, y, 2)
    if missing:
        return missing
    mean_term = np.sum((x.mean(axis=0) - y.mean(axis=0)) ** 2)
    a = (x - x.mean(axis=0)) / np.sqrt(len(x) - 1)
    b = (y - y.mean(axis=0)) / np.sqrt(len(y) - 1)
    # Exact low-rank identity avoids a 2048x2048 matrix square root for small sets.
    if max(len(x), len(y)) < x.shape[1]:
        cross_trace = svdvals(a @ b.T).sum()
    else:
        ca, cb = a.T @ a, b.T @ b
        eigenvalues, eigenvectors = np.linalg.eigh(ca)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
        sandwich = root @ cb @ root
        cross_trace = np.sqrt(np.maximum(eigvalsh((sandwich + sandwich.T) / 2), 0)).sum()
    value = mean_term + np.sum(a * a) + np.sum(b * b) - 2 * cross_trace
    return {'value': float(max(value, 0)), 'direction': 'lower', 'real_count': len(x),
            'synthetic_count': len(y), 'feature_dimension': x.shape[1]}


def polynomial_mmd(x, y):
    d = x.shape[1]
    xx, yy, xy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
    return ((xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1)) +
            (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1)) - 2 * xy.mean())


def kid(x, y, subset_size=1000, subsets=50, seed=0):
    x, y = pair_arrays(x, y)
    missing = insufficient(x, y, 2)
    if missing:
        return missing
    size = min(subset_size, len(x), len(y))
    rng = np.random.default_rng(seed)
    scores = [float(polynomial_mmd(x[rng.choice(len(x), size, replace=False)],
                                   y[rng.choice(len(y), size, replace=False)])) for _ in range(subsets)]
    return {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'direction': 'lower',
            'subset_size': size, 'subsets': subsets, 'estimator': 'unbiased_polynomial_mmd2',
            'note': 'Raw unscaled KID; negative estimates are valid. std is subset variability, not a confidence interval.'}


def rbf_mmd(x, y, sigma=10., scale=1., block=256):
    """Biased V-statistic, as in the released Google CMMD implementation."""
    x, y = pair_arrays(x, y)
    def kernel_mean(a, b):
        total = 0.
        for i in range(0, len(a), block):
            for j in range(0, len(b), block):
                squared = cdist(a[i:i+block], b[j:j+block], metric='sqeuclidean')
                total += np.exp(-squared / (2 * sigma ** 2)).sum()
        return total / (len(a) * len(b))
    missing = insufficient(x, y, 1)
    if missing:
        return missing
    value = scale * (kernel_mean(x, x) + kernel_mean(y, y) - 2 * kernel_mean(x, y))
    return {'value': float(max(value, 0)), 'direction': 'lower', 'sigma': sigma, 'scale': scale,
            'estimator': 'biased_rbf_mmd2', 'real_count': len(x), 'synthetic_count': len(y)}


def prdc(x, y, k=5, block=256):
    x, y = pair_arrays(x, y)
    missing = insufficient(x, y, k + 1)
    if missing:
        return missing
    def radii(a):
        result = np.empty(len(a))
        for i in range(0, len(a), block):
            distance = cdist(a[i:i+block], a)
            distance[np.arange(len(distance)), np.arange(i, i + len(distance))] = np.inf
            result[i:i+len(distance)] = np.partition(distance, k - 1, axis=1)[:, k - 1]
        return result
    rx, ry = radii(x), radii(y)
    precision_hits = np.zeros(len(y), dtype=bool)
    recall_hits = np.zeros(len(x), dtype=bool)
    density_hits = np.zeros(len(y), dtype=np.int64)
    coverage_hits = np.zeros(len(x), dtype=bool)
    for i in range(0, len(x), block):
        for j in range(0, len(y), block):
            distance = cdist(x[i:i+block], y[j:j+block])
            in_real = distance < rx[i:i+len(distance), None]
            in_fake = distance < ry[None, j:j+distance.shape[1]]
            precision_hits[j:j+distance.shape[1]] |= in_real.any(axis=0)
            recall_hits[i:i+len(distance)] |= in_fake.any(axis=1)
            density_hits[j:j+distance.shape[1]] += in_real.sum(axis=0)
            coverage_hits[i:i+len(distance)] |= in_real.any(axis=1)
    return {'precision': float(precision_hits.mean()), 'recall': float(recall_hits.mean()),
            'density': float(density_hits.mean() / k), 'coverage': float(coverage_hits.mean()),
            'nearest_k': k, 'direction': 'higher_fidelity_and_coverage',
            'zero_radius_real': int(np.count_nonzero(rx == 0)),
            'zero_radius_synthetic': int(np.count_nonzero(ry == 0)),
            'note': 'Strict radius comparison from PRDC; duplicates can produce zero radii. Density can exceed 1.'}


def vendi(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {'value': None, 'status': 'insufficient_samples'}
    norm = np.linalg.norm(x, axis=1)
    if np.any(norm == 0):
        return {'value': None, 'status': 'undefined_zero_norm_features'}
    normalized = x / norm[:, None]
    # Nonzero eigenvalues of XX^T and X^TX are identical.
    gram = normalized @ normalized.T if len(x) <= x.shape[1] else normalized.T @ normalized
    values = np.maximum(eigvalsh(gram / len(x)), 0)
    values = values[values > 0]
    values /= values.sum()
    return {'value': float(np.exp(-np.sum(values * np.log(values)))), 'count': len(x),
            'direction': 'diversity_only_not_quality', 'kernel': 'cosine'}


def sliced_wasserstein(x, y, projections=128, seed=0):
    x, y = pair_arrays(x, y)
    missing = insufficient(x, y, 1)
    if missing:
        return missing
    directions = np.random.default_rng(seed).normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0)
    xp, yp = x @ directions, y @ directions
    values = [wasserstein_distance(xp[:, i], yp[:, i]) for i in range(projections)]
    return {'value': float(np.mean(values)), 'direction': 'lower', 'projections': projections,
            'real_count': len(x), 'synthetic_count': len(y)}
