#!/usr/bin/env python3
"""Does a njet-dependent correction, measured in the sideband, fix (tight,tight)?

The single-tight sideband closure carries a njet slope that MUFFIN and the
binned map share, even though one takes nsmalljets as an input and the other has
no njet axis at all.  Whatever it is, it is common to both -- and in 1tau0l the
njet dependence of the fake factor was measured to be negligible, which is why
the baseline map integrates that axis ("_nonj").

This derives the correction the 2tau0l data asks for,

    k(njet) = observed / predicted   in the single-tight sideband,

from 526 events, and then applies it to the (tight,tight) prediction, which is a
statistically independent set of 111 events.  If the same k fixes both, the njet
dependence is real and this is the repair.  If (tight,tight) does not move, the
two effects are unrelated and the 16% is something else (or nothing).

    bash muffin/run.sh njet_correction.py
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402
from closure_muffin import load_binned_ff  # noqa: E402
from sideband_2tau0l import load, ff_for_tau, TAU  # noqa: E402

NJ_EDGES = np.array([4, 5, 6, 7, 20])


def main():
    res = load()
    model = MuffinModel.load(os.path.join(mc.OUTDIR, 'models',
                                          'muffin_poster_nominal'))
    maps, yvar = load_binned_ff(mc.FR2D_REF)
    F = {t: ff_for_tau(res, t, model, model.feature_names, maps, yvar)
         for t in TAU}

    w, is_mc = res['w'], res['is_mc']
    g1, g2 = res['t1_gen'], res['t2_gen']
    TT_ = res['t1_pass'] & res['t2_pass']
    ONE = res['t1_pass'] & res['t2_anti']
    LL = res['t1_anti'] & res['t2_anti']
    nj = res['nsmalljets']

    for k_ff, tag in ((0, 'MUFFIN'), (1, 'binned map')):
        f1, f2 = F[1][k_ff], F[2][k_ff]
        print('\n=== %s ===' % tag)
        print('  njet    sideband obs   pred    k=obs/pred  +- ')
        kfac, lo, hi = [], [], []
        for i in range(len(NJ_EDGES) - 1):
            m = (nj >= NJ_EDGES[i]) & (nj < NJ_EDGES[i + 1])
            obs = w[m & ONE & ~is_mc].sum() - w[m & ONE & is_mc & g1].sum()
            pred = ((w * f1)[m & LL & ~is_mc].sum() + (w * f2)[m & LL & ~is_mc].sum()
                    - (w * f1)[m & LL & is_mc & g1].sum()
                    - (w * f2)[m & LL & is_mc & g2].sum())
            n_o = int((m & ONE & ~is_mc).sum())
            n_i = int((m & LL & ~is_mc).sum())
            e = np.sqrt(1.0 / max(n_o, 1) + 1.0 / max(n_i, 1))
            k = obs / pred if pred > 0 else 1.0
            kfac.append(k)
            lo.append(NJ_EDGES[i]); hi.append(NJ_EDGES[i + 1])
            print('  %2d-%-3d %10.1f %8.1f %10.3f  +- %.3f  (%d obs, %d input)'
                  % (NJ_EDGES[i], NJ_EDGES[i + 1] - 1, obs, pred, k, k * e, n_o, n_i))

        # per-event correction factor from the njet bin the event sits in
        kmap = np.ones(nj.size)
        for i, k in enumerate(kfac):
            kmap[(nj >= lo[i]) & (nj < hi[i])] = k

        # ---- apply it to (tight,tight), an independent 111 events ----------
        def tt_pred(c1, c2):
            return ((w * f2 * c2)[ONE & ~is_mc].sum()
                    - (w * f2 * c2)[ONE & is_mc & g2].sum()
                    - (w * f1 * c1 * f2 * c2)[LL & ~is_mc].sum()
                    + (w * f1 * c1 * f2 * c2)[LL & is_mc & (g1 | g2)].sum())

        one = np.ones(nj.size)
        prompt = w[TT_ & is_mc & g1 & g2].sum()
        n_obs = int((TT_ & ~is_mc).sum())
        print('\n  (tight,tight), %d observed events:' % n_obs)
        for lbl, c in (('uncorrected', one), ('with k(njet)', kmap)):
            p = tt_pred(c, c) + prompt
            sig = stats.norm.isf(1 - stats.poisson.cdf(n_obs - 1, p))
            print('    %-14s pred %7.1f   data/pred %6.3f   %.1f sigma'
                  % (lbl, p, n_obs / p, sig))


if __name__ == '__main__':
    main()
