#!/usr/bin/env python3
"""A higher-statistics closure test for the 2tau0l fake factor.

The channel's (tight,tight) region holds 111 data events, so the closure quoted
there carries a 9.5% Poisson error and cannot separate 1.00 from 1.17.  The
single-tight sideband holds 526, and the fake factor predicts it just as
directly.

The two taus are ordered by their raw DeepTau VSjet score (verified: tau1 > tau2
in 100% of the pool), so (anti1, tight2) is empty by construction and the pool
splits into three populations: both tight (111), exactly one tight (526, always
tight1 & anti2), neither tight (814).

Promoting either tau of a (anti,anti) event gives an exactly-one-tight event:

    pred N(1T) = sum over (anti,anti) of [ F(tau1) + F(tau2) ]

which is exact for independent per-object pass probabilities:
    N(1-p1)(1-p2) * [p1/(1-p1) + p2/(1-p2)] = N[p1(1-p2) + p2(1-p1)] = N(1T)

The (tight,tight) prediction is reported next to it, from the same events, as
the low-statistics reference:

    pred N(TT) = sum over (1T) F(tau2) - sum over (anti,anti) F(tau1)F(tau2)

Genuine taus are subtracted on both sides: the prediction is for events where
the tau that becomes tight is a FAKE, so MC events whose promoted tau is genuine
are removed from the predicted side, and MC events whose tight tau is genuine
from the observed side.

    bash muffin/run.sh sideband_2tau0l.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402
from muffin_model import MuffinModel  # noqa: E402
from closure_muffin import load_binned_ff, binned_ff  # noqa: E402

INCL_TT = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic')
TAU = [1, 2]

COLS = ['kind_category_FR', 'trigSF_pfHT', 'trigSF_caloHT', 'nsmalljets',
        'nbtags', 'met', 'ht']
for i in TAU:
    COLS += ['tau%didDeepTau2017v2p1VSjet' % i, 'tau%dgenPartFlav' % i,
             'tau%ddecayMode' % i, 'tau%dPt' % i, 'tau%dEta' % i,
             'tau%dPhi' % i, 'tau%djetPt' % i]
WCOLS = ['xsecWeight', 'genWeight', 'puWeight', 'l1PreFiringWeight',
         'btagWeight_shape', 'btagShapeR_weight', 'tauIDSF_weight',
         'triggerSF_perfilter_2nBtag_v24c', 'triggerLumiSF', 'genTtbarId']

FEATMAP = {'tau1decayMode': 'tau%ddecayMode', 'tau1Pt': 'tau%dPt',
           'tau1Eta': 'tau%dEta', 'tau1Phi': 'tau%dPhi',
           'tau1jetPt': 'tau%djetPt', 'nsmalljets': 'nsmalljets',
           'nbtags': 'nbtags'}


def load(cache=True):
    cpath = os.path.join(mc.OUTDIR, 'cache', 'sideband_2tau0l.npz')
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        print('  (cached) %d events' % z['is_mc'].size)
        return {k: z[k] for k in z.files}

    keep = ['is_mc', 'w', 'nsmalljets', 'nbtags', 'met', 'ht']
    for i in TAU:
        keep += ['t%d_pass' % i, 't%d_anti' % i, 't%d_gen' % i, 't%d_jetPt' % i,
                 't%d_eta' % i, 't%d_dm' % i, 't%d_Pt' % i, 't%d_Phi' % i]
    out = {k: [] for k in keep}

    def fill(fn, is_mc):
        cols = COLS + (WCOLS if is_mc else [])
        bnm = os.path.basename(fn)
        incl = any(p in bnm for p in INCL_TT) and 'TTbb_4f' not in bnm
        for a in uproot.iterate(fn + ':Events', cols, step_size='500 MB',
                                library='np'):
            m = ((a['kind_category_FR'] == 0)
                 & (a['trigSF_pfHT'] >= 300) & (a['trigSF_caloHT'] >= 160))
            if is_mc:
                w = mc.LUMI * (a['xsecWeight'] * a['genWeight'] * a['puWeight']
                               * a['l1PreFiringWeight'] * a['btagWeight_shape']
                               * a['btagShapeR_weight'] * a['tauIDSF_weight']
                               * a['triggerSF_perfilter_2nBtag_v24c']
                               * a['triggerLumiSF'])
                if incl:
                    m &= (a['genTtbarId'] % 100) < 51
            else:
                w = np.ones(m.size)
            if not m.any():
                continue
            i = np.where(m)[0]
            out['is_mc'].append(np.full(i.size, is_mc, bool))
            out['w'].append(w[i])
            for k in ('nsmalljets', 'nbtags', 'met', 'ht'):
                out[k].append(a[k][i])
            for t in TAU:
                v = a['tau%didDeepTau2017v2p1VSjet' % t][i]
                out['t%d_pass' % t].append(v >= mc.TAU_PASS)
                out['t%d_anti' % t].append((v >= 2) & (v < mc.TAU_PASS))
                out['t%d_gen' % t].append(a['tau%dgenPartFlav' % t][i] == 5)
                out['t%d_jetPt' % t].append(a['tau%djetPt' % t][i])
                out['t%d_eta' % t].append(a['tau%dEta' % t][i])
                out['t%d_dm' % t].append(a['tau%ddecayMode' % t][i])
                out['t%d_Pt' % t].append(a['tau%dPt' % t][i])
                out['t%d_Phi' % t].append(a['tau%dPhi' % t][i])

    fill(mc.BASE + '/data/parts/BTagCSV_tree.root', False)
    for fn in sorted(glob.glob(mc.BASE + '/mc/parts/*_tree.root')):
        if os.path.basename(fn).startswith('QCD'):
            continue
        fill(fn, True)
    res = {k: np.concatenate(v) for k, v in out.items()}
    if cache:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, **res)
    return res


def ff_for_tau(res, t, model, names, maps, yvar):
    """(MUFFIN, binned) fake factor evaluated on tau `t` of every event."""
    cols = []
    for f in names:
        src = FEATMAP.get(f)
        if f == 'ptratio':
            cols.append(res['t%d_jetPt' % t]
                        / np.maximum(res['t%d_Pt' % t], 1e-6))
        elif f == 'abs_tau1Eta':
            cols.append(np.abs(res['t%d_eta' % t]))
        elif src in ('nsmalljets', 'nbtags'):
            cols.append(res[src])
        elif src == 'tau%ddecayMode':
            cols.append(res['t%d_dm' % t])
        elif src == 'tau%dPt':
            cols.append(res['t%d_Pt' % t])
        elif src == 'tau%dEta':
            cols.append(res['t%d_eta' % t])          # signed, as the model wants
        elif src == 'tau%dPhi':
            cols.append(res['t%d_Phi' % t])
        elif src == 'tau%djetPt':
            cols.append(res['t%d_jetPt' % t])
        else:
            raise KeyError(f)
    X = np.column_stack(cols).astype(np.float32)
    w_muf = model.predict(X)
    nj6 = res['nsmalljets'] >= 6
    p3 = res['t%d_dm' % t] >= 5
    yv = (np.abs(res['t%d_eta' % t]) if yvar == 'abs_tau1Eta'
          else res['t%d_jetPt' % t])
    w_bin = binned_ff(maps, res['t%d_jetPt' % t], yv, nj6, p3)
    return w_muf, w_bin


def signed(res, mask):
    """Data minus MC on the given per-event mask."""
    w, is_mc = res['w'], res['is_mc']
    return w[mask & ~is_mc].sum() - w[mask & is_mc].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='poster')
    args = ap.parse_args()

    print('=== loading the 2tau0l pool ===')
    res = load()
    mdir = os.path.join(mc.OUTDIR, 'models')
    model = MuffinModel.load(os.path.join(mdir, 'muffin_%s_nominal' % args.tag))
    names = model.feature_names
    maps, yvar = load_binned_ff(mc.FR2D_REF)
    print('  MUFFIN features: %s' % ', '.join(names))
    print('  baseline map: %s (y = %s)' % (os.path.basename(mc.FR2D_REF), yvar))

    F = {}
    for t in TAU:
        F[t] = ff_for_tau(res, t, model, names, maps, yvar)

    t1p, t2p = res['t1_pass'], res['t2_pass']
    t1a, t2a = res['t1_anti'], res['t2_anti']
    g1, g2 = res['t1_gen'], res['t2_gen']
    is_mc = res['is_mc']
    w = res['w']

    TT_ = t1p & t2p
    ONE = t1p & t2a
    LL = t1a & t2a
    print('\n  populations (data): (tight,tight) %d   (tight,anti) %d   '
          '(anti,anti) %d'
          % (int((TT_ & ~is_mc).sum()), int((ONE & ~is_mc).sum()),
             int((LL & ~is_mc).sum())))

    print('\n=== test A: single-tight sideband, predicted from (anti,anti) ===')
    print('  %-12s %10s %10s %8s %8s' % ('fake factor', 'pred', 'obs', 'ratio',
                                         'obs stat'))
    n_obs = int((ONE & ~is_mc).sum())
    for k, tag in ((0, 'MUFFIN'), (1, 'binned')):
        f1, f2 = F[1][k], F[2][k]
        # predicted: promote either tau of an (anti,anti) event; drop the MC
        # contribution where the promoted tau is genuine
        pred = ((w * f1)[LL & ~is_mc].sum() + (w * f2)[LL & ~is_mc].sum()
                - (w * f1)[LL & is_mc & g1].sum()
                - (w * f2)[LL & is_mc & g2].sum())
        # observed: exactly-one-tight, minus MC where the tight tau is genuine
        obs = w[ONE & ~is_mc].sum() - w[ONE & is_mc & g1].sum()
        print('  %-12s %10.1f %10.1f %8.3f %7.1f%%'
              % (tag, pred, obs, pred / obs, 100 / np.sqrt(max(n_obs, 1))))

    print('\n=== test B: (tight,tight), the usual closure (reference) ===')
    n_obs_tt = int((TT_ & ~is_mc).sum())
    for k, tag in ((0, 'MUFFIN'), (1, 'binned')):
        f1, f2 = F[1][k], F[2][k]
        pred = ((w * f2)[ONE & ~is_mc].sum() - (w * f2)[ONE & is_mc & g2].sum()
                - (w * f1 * f2)[LL & ~is_mc].sum()
                + (w * f1 * f2)[LL & is_mc & (g1 | g2)].sum())
        obs = w[TT_ & ~is_mc].sum() - w[TT_ & is_mc & g1 & g2].sum()
        print('  %-12s %10.1f %10.1f %8.3f %7.1f%%'
              % (tag, pred, obs, pred / obs, 100 / np.sqrt(max(n_obs_tt, 1))))

    print('\n  the sideband test has %.1fx the precision of the (tight,tight) one'
          % (np.sqrt(n_obs / max(n_obs_tt, 1))))


if __name__ == '__main__':
    main()
