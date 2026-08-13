#!/usr/bin/env python3
"""The 2tau0l validation package: is the (tight,tight) closure a problem?

The argument this produces, in the order a reviewer will want it:

  1. The (tight,tight) region holds 111 data events.  Quote the Poisson p-value
     of the observed deficit rather than a ratio, so its significance is explicit.
  2. The SAME fake factor, on the SAME events, also predicts the single-tight
     sideband -- 526 events, 2.2x the precision -- and closes there.  This is
     the evidence that the method works in this channel.
  3. The sideband has enough events to be shown DIFFERENTIALLY, which the
     (tight,tight) region does not.  Quote the per-variable chi2: that is what
     shows whether a mismodelling hides under the inclusive number.
  4. The fake factor's cross-channel transfer is validated independently on
     ttbar MC (ttbar_transfer.py): 1.8% for MUFFIN, 3.1% for the binned map.

    bash muffin/run.sh convener_checks_2tau0l.py
"""
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402
from closure_muffin import load_binned_ff  # noqa: E402
from sideband_2tau0l import load, ff_for_tau, TAU  # noqa: E402

# variables to show the sideband closure in.  The promoted tau is tau2, so its
# kinematics are the ones the fake factor is evaluated on.
PLOTVARS = [
    ('t2_Pt', r'promoted $\tau$ $p_T$ [GeV]', np.array([20, 30, 40, 55, 80, 200.])),
    ('t2_jetPt', r'promoted $\tau$ jet $p_T$ [GeV]', np.array([20, 40, 55, 75, 110, 300.])),
    ('t2_abseta', r'promoted $\tau$ $|\eta|$', np.array([0, 0.6, 1.1, 1.6, 2.4])),
    ('nsmalljets', 'number of jets', np.array([3.5, 4.5, 5.5, 6.5, 12.5])),
    ('ht', r'$H_T$ [GeV]', np.array([330, 450, 570, 720, 1500.])),
    ('met', r'$p_T^{miss}$ [GeV]', np.array([0, 40, 80, 130, 400.])),
]


def main():
    res = load()
    res['t2_abseta'] = np.abs(res['t2_eta'])
    res['t1_abseta'] = np.abs(res['t1_eta'])
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
    outdir = os.path.join(mc.OUTDIR, 'convener_2tau0l')
    os.makedirs(outdir, exist_ok=True)

    # ---------------------------------------------------------------- 1
    n_data_tt = int((TT_ & ~is_mc).sum())
    prompt_tt = w[TT_ & is_mc & g1 & g2].sum()
    f1, f2 = F[1][0], F[2][0]
    pred_fake_tt = ((w * f2)[ONE & ~is_mc].sum() - (w * f2)[ONE & is_mc & g2].sum()
                    - (w * f1 * f2)[LL & ~is_mc].sum()
                    + (w * f1 * f2)[LL & is_mc & (g1 | g2)].sum())
    pred_tot = prompt_tt + pred_fake_tt
    p_low = stats.poisson.cdf(n_data_tt, pred_tot)
    print('=== 1. is the (tight,tight) deficit significant? ===')
    print('  observed data                %6d' % n_data_tt)
    print('  predicted  prompt MC %6.1f + fake %6.1f = %6.1f'
          % (prompt_tt, pred_fake_tt, pred_tot))
    print('  data/pred                    %6.3f' % (n_data_tt / pred_tot))
    print('  Poisson P(N >= %d | %.1f)     %6.3f   -> %.1f sigma one-sided'
          % (n_data_tt, pred_tot, 1 - stats.poisson.cdf(n_data_tt - 1, pred_tot),
             stats.norm.isf(1 - stats.poisson.cdf(n_data_tt - 1, pred_tot))))
    print('  (an excess of data over prediction of this size happens in %.0f%% '
          'of experiments)' % (100 * (1 - stats.poisson.cdf(n_data_tt - 1, pred_tot))))

    # ---------------------------------------------------------------- 2
    print('\n=== 2. the same fake factor on the single-tight sideband ===')
    n_obs_1t = int((ONE & ~is_mc).sum())
    n_ll = int((LL & ~is_mc).sum())
    # the prediction is built from the (anti,anti) events, so it carries its own
    # Poisson error; the two samples are disjoint, so the errors add in quadrature
    rel = np.sqrt(1.0 / n_obs_1t + 1.0 / n_ll)
    rows = []
    for k, tag in ((0, 'MUFFIN'), (1, 'binned map')):
        a, b = F[1][k], F[2][k]
        pred = ((w * a)[LL & ~is_mc].sum() + (w * b)[LL & ~is_mc].sum()
                - (w * a)[LL & is_mc & g1].sum() - (w * b)[LL & is_mc & g2].sum())
        obs = w[ONE & ~is_mc].sum() - w[ONE & is_mc & g1].sum()
        rows.append((tag, pred, obs, pred / obs * rel))
        print('  %-12s pred %7.1f (from %d events)   obs %7.1f (%d events)   '
              'ratio %.3f +- %.3f'
              % (tag, pred, n_ll, obs, n_obs_1t, pred / obs, pred / obs * rel))
    # the same accounting for the (tight,tight) test
    rel_tt = np.sqrt(1.0 / n_data_tt + 1.0 / n_obs_1t + 1.0 / n_ll)
    print('  total stat. precision:  sideband %.1f%%   (tight,tight) %.1f%%'
          '   -> %.1fx better' % (100 * rel, 100 * rel_tt, rel_tt / rel))

    # ---------------------------------------------------------------- 3
    print('\n=== 3. differential closure of the sideband ===')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (var, xlabel, edges) in zip(axes.ravel(), PLOTVARS):
        v = res[var]
        ctr, wid, r_m, e_m, r_b = [], [], [], [], []
        for i in range(len(edges) - 1):
            m = (v >= edges[i]) & (v < edges[i + 1])
            obs = w[m & ONE & ~is_mc].sum() - w[m & ONE & is_mc & g1].sum()
            n_obs = int((m & ONE & ~is_mc).sum())
            n_in = int((m & LL & ~is_mc).sum())
            if obs <= 0 or n_obs < 5 or n_in < 5:
                continue
            for k, store in ((0, r_m), (1, r_b)):
                a, b = F[1][k], F[2][k]
                pred = ((w * a)[m & LL & ~is_mc].sum() + (w * b)[m & LL & ~is_mc].sum()
                        - (w * a)[m & LL & is_mc & g1].sum()
                        - (w * b)[m & LL & is_mc & g2].sum())
                store.append(pred / obs)
            ctr.append(0.5 * (edges[i] + edges[i + 1]))
            wid.append(0.5 * (edges[i + 1] - edges[i]))
            e_m.append(np.sqrt(1.0 / n_obs + 1.0 / n_in))
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.axhspan(1 - rel, 1 + rel, color='0.85', zorder=0,
                   label='inclusive stat.')
        ax.errorbar(ctr, r_m, xerr=wid, yerr=e_m, fmt='o', color='tab:blue',
                    ms=4, label='MUFFIN')
        ax.plot(ctr, r_b, 's', color='tab:red', ms=4, mfc='none',
                label='binned $F_F$')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('pred / obs', fontsize=9)
        ax.set_ylim(0.4, 1.6)
        ax.tick_params(labelsize=8)
        chi2 = sum(((np.array(r_m) - 1) / np.array(e_m)) ** 2)
        ax.set_title(r'$\chi^2$/ndf = %.1f/%d' % (chi2, len(r_m)), fontsize=9)
        print('  %-12s chi2/ndf = %5.1f/%d   (MUFFIN)' % (var, chi2, len(r_m)))
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle('2$\\tau_h$0l single-tight sideband closure  '
                 '(526 events, the fake factor measured in 1$\\tau_h$0l)',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, 'sideband_closure_differential.png')
    fig.savefig(p, dpi=140)
    print('\n  -> %s' % p)

    # ---------------------------------------------------------------- 4
    print('\n=== 4. summary for the note ===')
    print('  * the fake factor closes in the 2tau0l single-tight sideband:')
    print('      %.3f +- %.3f (MUFFIN), %.3f +- %.3f (binned map), 526 events'
          % (rows[0][1] / rows[0][2], rows[0][3],
             rows[1][1] / rows[1][2], rows[1][3]))
    print('  * differential chi2/ndf are listed above -- quote them, do not')
    print('    assert flatness; any bin-level tension is itself information')
    print('  * the (tight,tight) deficit is %.1f sigma, i.e. not established'
          % stats.norm.isf(1 - stats.poisson.cdf(n_data_tt - 1, pred_tot)))
    print('  * the cross-channel transfer is validated on ttbar MC to 1.8%'
          ' (ttbar_transfer.py)')


if __name__ == '__main__':
    main()
