#!/usr/bin/env python3
"""MUFFIN validation on v29pre 1tau0l: DR -> AR (MR -> VR) closure, benchmarked
against the current binned map, fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root
(x = tau1jetPt, y = |tau1Eta|, njet integrated).

Both predictions are built from exactly the same AR fail events with the same
signed weights, so the only difference is the fake factor itself:

    binned  F_F(jetPt, |eta| | prong) = FR/(1-FR)
    MUFFIN  w(z) = exp(margin) over the multi-dimensional feature vector z

Reported per (njet, prong) class and differentially, with the poster's figure-2
metric  sqrt( (1-r)^2 + sigma_r^2 )  where r = pred/obs -- i.e. the non-closure
and its statistical component added in quadrature.

Uncertainty bands: Poisson-bootstrap replicas (statistical) and hyper-parameter
variations (modelling), if those models were trained.

Usage:
    bash muffin/run.sh closure_muffin.py --features poster
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import uproot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402

# differential variables for the closure plots
PLOTVARS = [
    ('tau1jetPt', r'seeding jet $p_T$ [GeV]', np.array([20, 34, 41, 50, 70, 100, 150, 300.])),
    ('tau1Pt', r'$\tau_h$ $p_T$ [GeV]', np.array([20, 30, 40, 50, 65, 85, 120, 250.])),
    ('tau1jetDeepFlavB', 'seeding jet DeepFlavB', np.array([0, 0.0532, 0.304, 0.7476, 1.0])),
    ('nsmalljets', 'number of jets', np.array([3.5, 4.5, 5.5, 6.5, 7.5, 12.5])),
    ('abs_tau1Eta', r'$|\eta(\tau_h)|$', np.array([0, 0.5, 0.9, 1.3, 1.7, 2.4])),
    ('met', r'$p_T^{miss}$ [GeV]', np.array([0, 40, 80, 120, 180, 400.])),
]


def load_binned_ff(path):
    """FR/(1-FR) lookup from the baseline fr2d map.

    Returns (maps, yvar).  The y axis of the current 1tau0l map is |tau1Eta|
    (bins up to 2.4); the older flavour map used the mother-jet DeepFlavB (up to
    1.0), so the axis is identified from its range rather than assumed."""
    maps = {}
    with uproot.open(path) as f:
        for nj in ('nj45', 'nj6'):
            for pr in ('1p', '3p'):
                vals, xe, ye = f['fr2d_%s_%s' % (nj, pr)].to_numpy()
                maps[(nj, pr)] = (vals, xe, ye)
    ymax = max(m[2][-1] for m in maps.values())
    return maps, ('abs_tau1Eta' if ymax > 1.5 else 'tau1jetDeepFlavB')


def binned_ff(maps, jetpt, yval, nj6, p3):
    out = np.ones(jetpt.size)
    for nj, njm in (('nj45', ~nj6), ('nj6', nj6)):
        for pr, prm in (('1p', ~p3), ('3p', p3)):
            m = njm & prm
            if not m.any():
                continue
            vals, xe, ye = maps[(nj, pr)]
            i = np.clip(np.digitize(jetpt[m], xe) - 1, 0, vals.shape[0] - 1)
            j = np.clip(np.digitize(yval[m], ye) - 1, 0, vals.shape[1] - 1)
            fr = np.clip(vals[i, j], 1e-4, 0.95)
            out[m] = fr / (1.0 - fr)
    return out


def yields(w_ff, y, wsig, mask):
    """(predicted pass, observed pass, MC-stat-like error on each)."""
    f, p = (y == 0) & mask, (y == 1) & mask
    pred = (w_ff[f] * wsig[f]).sum()
    obs = wsig[p].sum()
    # statistical error: sum of squared weights (data + subtracted simulation)
    e_pred = np.sqrt(((w_ff[f] * wsig[f]) ** 2).sum())
    e_obs = np.sqrt((wsig[p] ** 2).sum())
    return pred, obs, e_pred, e_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='poster', choices=sorted(mc.FEATURE_SETS))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--region', default='AR', choices=['AR', 'DR'])
    args = ap.parse_args()
    tag = args.tag or args.features
    names = mc.FEATURE_SETS[args.features]
    mdir = os.path.join(mc.OUTDIR, 'models')
    pdir = os.path.join(mc.OUTDIR, 'plots_%s' % tag)
    os.makedirs(pdir, exist_ok=True)

    print('=== loading %s ===' % args.region)
    ar = mc.load_region(args.region)
    mc.summarise(ar, args.region)
    X = mc.select_features(ar, names)
    y, wsig = ar['y'], ar['w']
    nj6 = ar['nsmalljets'] >= 6
    p3 = ar['tau1decayMode'] >= 5

    model = MuffinModel.load(os.path.join(mdir, 'muffin_%s_nominal' % tag))
    assert model.feature_names == names, 'model/feature-set mismatch'
    w_muffin = model.predict(X)

    maps, yvar = load_binned_ff(mc.FR2D_REF)
    print('baseline map: %s  (y axis = %s)' % (os.path.basename(mc.FR2D_REF), yvar))
    w_binned = binned_ff(maps, ar['tau1jetPt'], ar[yvar], nj6, p3)

    # ---- replicas for the uncertainty bands ---------------------------------
    def _prefixes(pat):
        return sorted(f[:-len('_cfg.json')]
                      for f in glob.glob(os.path.join(mdir, pat + '_cfg.json')))
    w_boot = [MuffinModel.load(p).predict(X) for p in _prefixes('muffin_%s_boot*' % tag)]
    w_var = [MuffinModel.load(p).predict(X) for p in _prefixes('muffin_%s_var_*' % tag)]
    w_bkg = [MuffinModel.load(p).predict(X) for p in _prefixes('muffin_%s_sys_bkg*' % tag)]
    print('loaded %d bootstrap, %d modelling, %d background-subtraction replicas'
          % (len(w_boot), len(w_var), len(w_bkg)))

    # ---- inclusive / per-class closure --------------------------------------
    classes = [('nj45 1p', ~nj6 & ~p3), ('nj45 3p', ~nj6 & p3),
               ('nj6 1p', nj6 & ~p3), ('nj6 3p', nj6 & p3),
               ('TOTAL', np.ones_like(y, bool))]
    print('\n=== %s closure: predicted vs observed net pass yield ===' % args.region)
    print('  %-9s %10s | %10s %7s | %10s %7s' %
          ('class', 'observed', 'MUFFIN', 'ratio', 'binned FF', 'ratio'))
    summary = {}
    for name, m in classes:
        pm, obs, epm, eobs = yields(w_muffin, y, wsig, m)
        pb, _, _, _ = yields(w_binned, y, wsig, m)
        rm, rb = pm / max(obs, 1e-9), pb / max(obs, 1e-9)
        print('  %-9s %10.1f | %10.1f %7.3f | %10.1f %7.3f'
              % (name, obs, pm, rm, pb, rb))
        summary[name] = dict(obs=obs, muffin=pm, binned=pb, r_muffin=rm, r_binned=rb)

    # ---- differential closure + poster metric --------------------------------
    print('\n=== differential closure: sqrt((1-r)^2 + sigma_r^2), r = pred/obs ===')
    print('  (poster fig. 2 metric; smaller is better)')
    metric = {}
    for var, xlabel, edges in PLOTVARS:
        v = ar[var]
        rows = []
        for k in range(len(edges) - 1):
            m = (v >= edges[k]) & (v < edges[k + 1])
            if not m.any():
                continue
            pm, obs, epm, eobs = yields(w_muffin, y, wsig, m)
            pb, _, epb, _ = yields(w_binned, y, wsig, m)
            if obs <= 0:
                continue
            rm, rb = pm / obs, pb / obs
            sm = np.hypot(epm / obs, rm * eobs / obs)
            sb = np.hypot(epb / obs, rb * eobs / obs)
            rows.append((edges[k], edges[k + 1], obs, rm, sm, rb, sb,
                         np.hypot(1 - rm, sm), np.hypot(1 - rb, sb)))
        if not rows:
            continue
        a = np.array(rows)
        # yield-weighted average of the metric over the bins of this variable
        wgt = a[:, 2] / a[:, 2].sum()
        metric[var] = (float((a[:, 7] * wgt).sum()), float((a[:, 8] * wgt).sum()))
        print('  %-18s MUFFIN %6.1f%%   binned FF %6.1f%%   (%d bins)'
              % (var, 100 * metric[var][0], 100 * metric[var][1], len(rows)))

        # --- plot ---
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(5.2, 5.2), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.4, 1], hspace=0.06))
        ctr = 0.5 * (a[:, 0] + a[:, 1])
        wid = a[:, 1] - a[:, 0]
        ax.bar(ctr, a[:, 2] / wid, width=wid, color='0.85', edgecolor='0.4',
               label='observed (data $-$ sim, pass)')
        ax.plot(ctr, a[:, 2] * a[:, 3] / wid, 'o-', color='tab:blue', ms=4, label='MUFFIN')
        ax.plot(ctr, a[:, 2] * a[:, 5] / wid, 's--', color='tab:red', ms=4, label='binned $F_F$')
        ax.set_yscale('log')
        ax.set_ylabel('net pass yield / bin width')
        ax.legend(fontsize=8, frameon=False)
        ax.set_title('v29pre 1$\\tau_h$0l  %s closure' % args.region, fontsize=10, loc='left')
        rx.axhline(1.0, color='k', lw=0.8)
        rx.errorbar(ctr, a[:, 3], yerr=a[:, 4], fmt='o-', color='tab:blue', ms=4)
        rx.errorbar(ctr, a[:, 5], yerr=a[:, 6], fmt='s--', color='tab:red', ms=4)
        rx.set_ylim(0.6, 1.4)
        rx.set_ylabel('pred / obs', fontsize=9)
        rx.set_xlabel(xlabel)
        fig.savefig(os.path.join(pdir, 'closure_%s_%s.png' % (args.region, var)), dpi=140,
                    bbox_inches='tight')
        plt.close(fig)

    # ---- uncertainty decomposition -------------------------------------------
    unc = {}
    if w_boot:
        tot = np.array([(wb[y == 0] * wsig[y == 0]).sum() for wb in w_boot])
        nom = (w_muffin[y == 0] * wsig[y == 0]).sum()
        unc['statistical'] = float(tot.std() / abs(nom))
    nom = (w_muffin[y == 0] * wsig[y == 0]).sum()
    if w_var:
        tot = np.array([(wv[y == 0] * wsig[y == 0]).sum() for wv in w_var])
        unc['modelling'] = float(np.abs(tot - nom).max() / abs(nom))
    if w_bkg:
        tot = np.array([(wv[y == 0] * wsig[y == 0]).sum() for wv in w_bkg])
        unc['bkg_subtraction'] = float(np.abs(tot - nom).max() / abs(nom))
    # non-closure here also carries the DR->AR extrapolation: the AR differs from
    # the DR by the very cut (jet4DeepFlavB) that is not a feature, so the two
    # cannot be separated with this single validation region
    unc['non_closure_and_extrapolation'] = float(abs(1 - summary['TOTAL']['r_muffin']))
    if unc:
        print('\n=== MUFFIN uncertainty decomposition (inclusive, %s) ===' % args.region)
        for k, v in unc.items():
            print('  %-32s %6.2f%%' % (k, 100 * v))
        print('  %-32s %6.2f%%' % ('total (quadrature)',
                                   100 * np.sqrt(sum(v * v for v in unc.values()))))

    with open(os.path.join(pdir, 'closure_%s_summary.json' % args.region), 'w') as fh:
        json.dump(dict(summary=summary, metric=metric, uncertainty=unc,
                       features=names, n_boot=len(w_boot), n_var=len(w_var)), fh, indent=2)
    print('\nplots + summary -> %s' % pdir)


if __name__ == '__main__':
    main()
