#!/usr/bin/env python3
"""Is the 2tau0l (tight,tight) shortfall structural, or a 2-sigma fluctuation?

The single-tight sideband closes (0.998) while (tight,tight) falls ~16% short.
Both use the same fake factor and the same (anti,anti) input, so the difference
has to sit in what distinguishes them: the sideband is LINEAR in the fake
factor, the (tight,tight) prediction contains the PRODUCT F(tau1)F(tau2).

Candidate mechanism, which needs no correlation between the two jets: the taus
are ordered by raw DeepTau score, so in an (anti,anti) event tau1 is selected to
be the higher-score one and tau2 the lower.  Inside the anti window [2, 8) the
true pass probability rises steeply with that score, but the fake factor is a
function of (pT, eta, ...) only and cannot see it, so it returns nearly the same
value for both.  Writing the true values as F(1+d) and F(1-d):

    linear   F1 + F2  ->  2F          the bias cancels   (the sideband closes)
    product  F1 * F2  ->  F^2(1-d^2)  the map overestimates it

and since pred = A - C, overestimating C pushes the prediction DOWN.  A shortfall
of 16% needs d ~ 40%, which is the right size for that window.

ttbar MC has ~10^5 fake di-tau candidates, so both tests can be run there with
negligible statistical error.  If MC reproduces "sideband closes, (tight,tight)
falls short", the mechanism is structural.  If MC closes both, the data's 16% is
a fluctuation.

    bash muffin/run.sh order_bias_mc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402
from closure_muffin import load_binned_ff  # noqa: E402
from sideband_2tau0l import load, ff_for_tau, TAU  # noqa: E402


def main():
    res = load()
    mdir = os.path.join(mc.OUTDIR, 'models')
    model = MuffinModel.load(os.path.join(mdir, 'muffin_poster_nominal'))
    maps, yvar = load_binned_ff(mc.FR2D_REF)
    F = {t: ff_for_tau(res, t, model, model.feature_names, maps, yvar)
         for t in TAU}

    is_mc, w = res['is_mc'], res['w']
    t1p, t2p = res['t1_pass'], res['t2_pass']
    t1a, t2a = res['t1_anti'], res['t2_anti']
    # both taus FAKE: the population the template is meant to describe
    fake = is_mc & ~res['t1_gen'] & ~res['t2_gen']
    TT_ = fake & t1p & t2p
    ONE = fake & t1p & t2a
    LL = fake & t1a & t2a
    print('MC, both taus fake:  (tight,tight) %d   (tight,anti) %d   '
          '(anti,anti) %d entries'
          % (int(TT_.sum()), int(ONE.sum()), int(LL.sum())))
    print('  weighted yields:   %8.1f   %8.1f   %8.1f'
          % (w[TT_].sum(), w[ONE].sum(), w[LL].sum()))

    print('\n%-12s %10s %10s %8s   %10s %10s %8s'
          % ('fake factor', 'A pred', 'A obs', 'ratio', 'B pred', 'B obs', 'ratio'))
    print('-' * 78)
    for k, tag in ((0, 'MUFFIN'), (1, 'binned')):
        f1, f2 = F[1][k], F[2][k]
        a_pred = (w * f1)[LL].sum() + (w * f2)[LL].sum()
        a_obs = w[ONE].sum()
        b_pred = (w * f2)[ONE].sum() - (w * f1 * f2)[LL].sum()
        b_obs = w[TT_].sum()
        print('%-12s %10.1f %10.1f %8.3f   %10.1f %10.1f %8.3f'
              % (tag, a_pred, a_obs, a_pred / a_obs,
                 b_pred, b_obs, b_pred / b_obs))

    # how big is the ordering effect?  compare the map's product against the
    # product it would need to give to close B exactly
    f1, f2 = F[1][0], F[2][0]
    c_map = (w * f1 * f2)[LL].sum()
    c_needed = (w * f2)[ONE].sum() - w[TT_].sum()
    print('\n  C term (the F1*F2 sum over (anti,anti)):')
    print('    from the map        %10.1f' % c_map)
    print('    needed to close B   %10.1f' % c_needed)
    if c_map > 0:
        d2 = 1 - c_needed / c_map
        print('    => the map overestimates it by %5.1f%%, i.e. d = %.2f'
              % (100 * d2, np.sqrt(max(d2, 0))))

    # direct check of the mechanism: within the anti window, does the true pass
    # rate depend on where the tau sits relative to its partner?
    print('\n  direct check -- in (anti,anti) events the taus are score-ordered.')
    print('  Their fake factors from the map differ by only:')
    r = np.abs(f1[LL] - f2[LL]) / np.maximum(0.5 * (f1[LL] + f2[LL]), 1e-9)
    print('    median |F1-F2|/F = %.3f   (the map cannot see the ordering)' % np.median(r))


if __name__ == '__main__':
    main()
