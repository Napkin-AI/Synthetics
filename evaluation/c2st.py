"""Group-held-out logistic domain classifier. No significance claim or pixel split."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


def group_folds(groups, folds, seed):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f'C2ST needs at least {folds} source groups per domain; got {len(unique)}')
    chunks = np.array_split(np.random.default_rng(seed).permutation(unique), folds)
    return [np.flatnonzero(np.isin(groups, chunk)) for chunk in chunks]


def auc_score(labels, scores):
    positive = labels == 1
    n = int(positive.sum())
    return float((rankdata(scores)[positive].sum() - n * (n + 1) / 2) / (n * (len(labels)-n)))


def c2st(x, y, real_groups, synthetic_groups, folds=3, seed=0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if min(len(set(real_groups)), len(set(synthetic_groups))) < folds:
        return {'status': 'insufficient_groups', 'auc': None, 'gap': None, 'folds': folds,
                'real_groups': len(set(real_groups)), 'synthetic_groups': len(set(synthetic_groups))}
    all_x = np.concatenate((x, y))
    labels = np.r_[np.zeros(len(x)), np.ones(len(y))]
    left = group_folds(real_groups, folds, seed)
    right = group_folds(synthetic_groups, folds, seed+1)
    results = []
    for a, b in zip(left, right):
        test = np.r_[a, b+len(x)]
        train = np.setdiff1d(np.arange(len(all_x)), test)
        mean, scale = all_x[train].mean(0), all_x[train].std(0)
        scale[scale < 1e-8] = 1.
        z = (all_x[train]-mean) / scale
        t = labels[train]
        weights = np.where(t == 1, .5 / (t == 1).sum(), .5 / (t == 0).sum())
        # Fixed L2 regularizer; no tuning on held-out domains.
        def objective(theta):
            score = z @ theta[:-1] + theta[-1]
            error = weights * (expit(score)-t)
            loss = np.sum(weights * (np.logaddexp(0, score)-t*score)) + .5 * np.dot(theta[:-1], theta[:-1])
            return loss, np.r_[z.T @ error + theta[:-1], error.sum()]
        fit = minimize(objective, np.zeros(z.shape[1]+1), jac=True, method='L-BFGS-B', options={'maxiter': 300})
        scores = (all_x[test]-mean) / scale @ fit.x[:-1] + fit.x[-1]
        prediction = scores >= 0
        truth = labels[test]
        accuracy = .5 * ((prediction[truth == 0] == 0).mean() + (prediction[truth == 1] == 1).mean())
        results.append({'auc': auc_score(truth, scores), 'balanced_accuracy': float(accuracy),
                        'converged': bool(fit.success), 'optimizer_message': str(fit.message),
                        'real_test_indices': a.tolist(), 'synthetic_test_indices': b.tolist(),
                        'real_test_groups': sorted(set(np.asarray(real_groups)[a].tolist())),
                        'synthetic_test_groups': sorted(set(np.asarray(synthetic_groups)[b].tolist()))})
    auc = float(np.mean([r['auc'] for r in results]))
    return {'auc': auc, 'gap': abs(auc-.5), 'balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in results])),
            'status': 'ok' if all(r['converged'] for r in results) else 'optimizer_not_converged',
            'folds': results, 'direction': 'auc_near_0.5_less_distinguishable',
            'variant': 'group_cv_balanced_logistic_l2_1_v1',
            'note': 'Scaler fitted on training folds only; mean fold AUC, no label inversion. '
                    'Group IDs are domain-local source units. Diagnostic, not a calibrated hypothesis test; '
                    'chance performance is not proof of equal distributions.'}
