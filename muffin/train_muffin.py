#!/usr/bin/env python3
"""Train MUFFIN on the v29pre 1tau0l determination region (MR: jet4DeepFlavB >= 0.1).

Produces
  muffin_<tag>_nominal{,_cfg}.json          the fake factor itself
  muffin_<tag>_boot<NNN>_*                  Poisson-bootstrap replicas  (statistical)
  muffin_<tag>_var_<name>_*                 hyper-parameter variations  (modelling)
  muffin_<tag>_meta.json                    features, ranking, yields

and prints an out-of-fold DR self-closure -- the first thing to look at: it is
the only closure that does not also test the DR->AR extrapolation, so if it is
not flat the model itself is at fault.

Usage:
    bash muffin/run.sh train_muffin.py --features full --bootstrap 20 --variations
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc          # noqa: E402
from muffin_model import MuffinModel, VARIATIONS, NOMINAL  # noqa: E402


def closure_table(w_pred, y, wsig, keys, title='class'):
    """Net predicted pass yield (fake factor applied to the fail events) vs the
    net observed pass yield, split by `keys` = [(name, mask), ...]."""
    lines = ['  %-14s %10s %10s %8s' % (title, 'pred', 'obs', 'ratio')]
    f, p = y == 0, y == 1
    for name, m in list(keys) + [('TOTAL', np.ones_like(y, bool))]:
        pred = (w_pred[m & f] * wsig[m & f]).sum()
        obs = wsig[m & p].sum()
        lines.append('  %-14s %10.1f %10.1f %8.3f'
                     % (name, pred, obs, pred / max(obs, 1e-9)))
    return '\n'.join(lines)


def done(prefix):
    """Already-trained models are skipped, so a long run can be resumed."""
    return os.path.exists(prefix + '_cfg.json')


def class_masks(res, mask=None):
    nj6 = res['nsmalljets'] >= 6
    p3 = res['tau1decayMode'] >= 5
    if mask is not None:
        nj6, p3 = nj6[mask], p3[mask]
    return [('nj45 1p', ~nj6 & ~p3), ('nj45 3p', ~nj6 & p3),
            ('nj6 1p', nj6 & ~p3), ('nj6 3p', nj6 & p3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='poster', choices=sorted(mc.FEATURE_SETS))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--bootstrap', type=int, default=0)
    ap.add_argument('--variations', action='store_true')
    ap.add_argument('--bkgsub', type=float, default=0.0,
                    help='fractional variation of the subtracted simulation '
                         'normalisation, e.g. 0.10 -> two extra models at +-10%%')
    ap.add_argument('--kfold', type=int, default=4,
                    help='folds for the out-of-fold DR self-closure')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--resume', action='store_true',
                    help='skip the k-fold self-closure and any model already on disk')
    args = ap.parse_args()

    tag = args.tag or args.features
    mdir = os.path.join(mc.OUTDIR, 'models')
    names = mc.FEATURE_SETS[args.features]

    print('=== determination region (v29pre, jet4DeepFlavB >= %.2f) ===' % mc.J4CUT)
    dr = mc.load_region('DR', cache=not args.no_cache)
    mc.summarise(dr, 'DR')
    X = mc.select_features(dr, names)
    y, w = dr['y'], dr['w']
    print('features (%d): %s' % (len(names), ', '.join(names)))

    if args.resume:
        print('(--resume: skipping the self-closure, reusing existing models)')
    # ---- out-of-fold self-closure -------------------------------------------
    # every event is predicted by a model that never saw it, but the closure
    # still uses the full DR statistics -- a single hold-out split is too noisy
    # in the 3-prong classes to interpret.
    if not args.resume:
        print('\n=== DR self-closure (%d-fold, all predictions out-of-fold) ==='
              % args.kfold)
        rng = np.random.default_rng(1234)
        fold = rng.integers(0, args.kfold, size=y.size)
        w_cv = np.empty(y.size)
        for k in range(args.kfold):
            tr, te = fold != k, fold == k
            w_cv[te] = MuffinModel(names).fit(X[tr], y[tr], w[tr]).predict(X[te])
        print(closure_table(w_cv, y, w, class_masks(dr)))

    # ---- nominal model on the full DR --------------------------------------
    print('\n=== nominal model (full DR) ===')
    model = MuffinModel(names)
    model.fit(X, y, w)
    model.save(os.path.join(mdir, 'muffin_%s_nominal' % tag))
    wp = model.predict(X)
    print('  training sample: %s' % model.n_train)
    print('  w_MUFFIN: mean %.4f  median %.4f  1%%-99%% [%.4f, %.4f]  inclusive F_F %.4f'
          % (wp.mean(), np.median(wp), np.percentile(wp, 1), np.percentile(wp, 99),
             model.inclusive_ff))
    print('  in-sample DR closure (sanity, not a test):')
    print(closure_table(wp, y, w, class_masks(dr)))
    rank = model.ranking()
    print('  feature ranking (total gain):')
    for i, (f, g) in enumerate(rank):
        print('    %2d. %-20s %10.1f' % (i + 1, f, g))

    meta = dict(tag=tag, features=names, nominal=NOMINAL, base=mc.BASE,
                j4cut=mc.J4CUT, tau_pass=mc.TAU_PASS, n_dr=int(y.size),
                inclusive_ff=model.inclusive_ff,
                ranking=[f for f, _ in rank])

    # ---- bootstrap replicas (statistical uncertainty) -----------------------
    if args.bootstrap:
        print('\n=== %d Poisson-bootstrap replicas ===' % args.bootstrap)
        for b in range(args.bootstrap):
            pre = os.path.join(mdir, 'muffin_%s_boot%03d' % (tag, b))
            if done(pre):
                continue
            r = np.random.default_rng(9000 + b)
            MuffinModel(names).fit(X, y, w, seed=b,
                                   boot=r.poisson(1.0, size=w.size)).save(pre)
            gc.collect()
            if (b + 1) % 5 == 0:
                print('  ... %d/%d' % (b + 1, args.bootstrap))
        meta['bootstrap'] = args.bootstrap

    # ---- background-subtraction variation -----------------------------------
    # the poster's "subtracted simulation normalisation" component: scale the
    # genuine-tau simulation up and down and refit
    if args.bkgsub:
        print('\n=== background-subtraction variation (+-%.0f%% on the subtracted sim) ==='
              % (100 * args.bkgsub))
        for sign, nm in ((+1, 'bkgup'), (-1, 'bkgdn')):
            pre = os.path.join(mdir, 'muffin_%s_sys_%s' % (tag, nm))
            if done(pre):
                continue
            wv = np.where(w < 0, w * (1.0 + sign * args.bkgsub), w)
            MuffinModel(names).fit(X, y, wv).save(pre)
            gc.collect()
            print('  trained %s' % nm)
        meta['bkgsub'] = args.bkgsub

    # ---- modelling variations ----------------------------------------------
    if args.variations:
        print('\n=== modelling variations ===')
        for name, par in VARIATIONS.items():
            pre = os.path.join(mdir, 'muffin_%s_var_%s' % (tag, name))
            if done(pre):
                continue
            MuffinModel(names).fit(X, y, w, params=par).save(pre)
            gc.collect()
            print('  trained %-10s %s' % (name, par))
        meta['variations'] = sorted(VARIATIONS)

    with open(os.path.join(mdir, 'muffin_%s_meta.json' % tag), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('\nmodels -> %s' % mdir)


if __name__ == '__main__':
    main()
