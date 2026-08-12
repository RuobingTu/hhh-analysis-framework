#!/usr/bin/env python3
"""Which local GBDT accepts negative sample weights?  A signed-weight classifier
is the exact MUFFIN estimator (its per-z minimiser is A/(A+B) with A, B the NET
densities), so it is worth using whichever backend allows it.

Toy check: two Gaussians, class 1 = 'data' minus a 'sim' component injected with
negative weight.  The recovered exp(margin) must track the analytic net ratio.
"""
import numpy as np

rng = np.random.default_rng(0)
n = 20000
# data pass ~ N(1,1), data fail ~ N(0,1), sim contamination in pass ~ N(2,0.5)
x = np.concatenate([rng.normal(1, 1, n), rng.normal(0, 1, n), rng.normal(2, .5, n // 4)])
y = np.concatenate([np.ones(n), np.zeros(n), np.ones(n // 4)])
w = np.concatenate([np.ones(n), np.ones(n), -0.5 * np.ones(n // 4)])
X = x.reshape(-1, 1)

print('--- lightgbm ---')
try:
    import lightgbm as lgb
    print('version', lgb.__version__)
    d = lgb.Dataset(X, label=y, weight=w)
    b = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=8,
                       min_data_in_leaf=50, verbose=-1), d, num_boost_round=200)
    m = b.predict(X, raw_score=True)
    print('OK: exp(margin) range %.3f .. %.3f' % (np.exp(m).min(), np.exp(m).max()))
    for xv in (-1., 0., 1., 2.):
        i = np.argmin(np.abs(x - xv))
        A = n * np.exp(-(xv - 1) ** 2 / 2) - 0.5 * (n / 4) * np.exp(-(xv - 2) ** 2 / .5) / .5
        B = n * np.exp(-xv ** 2 / 2)
        print('   x=%5.1f  fitted %7.3f   analytic %7.3f' % (xv, np.exp(m[i]), A / B))
except Exception as e:
    print('FAILED:', type(e).__name__, e)

print('--- xgboost ---')
try:
    import xgboost as xgb
    print('version', xgb.__version__)
    xgb.train({'objective': 'binary:logistic'},
              xgb.DMatrix(X, label=y, weight=w), num_boost_round=10)
    print('OK')
except Exception as e:
    print('FAILED:', type(e).__name__, str(e)[:120])
