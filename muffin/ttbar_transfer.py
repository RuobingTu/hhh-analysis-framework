#!/usr/bin/env python3
"""Does a ttbar fake factor measured in 1tau1l carry over to 1tau0l?

This is the assumption behind a source-split fake factor: measure the ttbar
component where ttbar dominates the fakes (1tau1l, with a real lepton) and the
QCD component where QCD dominates, then combine.  The assumption is that the
ttbar jet->tau_h fake factor is the same object in both channels.

The test is pure MC truth, so it needs no data-driven subtraction at all: take
ttbar MC, keep the FAKE taus (tau1genPartFlav != 5), measure the fake factor in
one channel and predict the other.  ttbar MC gives ~800k fake taus in 1tau1l and
~350k in 1tau0l -- three orders of magnitude more than the 2tau0l data, so the
transfer can be tested to well under a percent.

Both fake factors are tested the same way:
  MUFFIN  w(z) = exp(margin), the poster feature set
  binned  FF(jetPt, |eta| | prong), the axes of the current 1tau0l map

and against two references: the same-channel out-of-fold self-closure (how well
each does when there is no transfer at all) and the inclusive fake factor.

    bash muffin/run.sh ttbar_transfer.py
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

TT = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic', 'TTbb_4f_TTTo')
INCL_TT = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic')

# the binned baseline: the axes of fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root
PT_E = np.array([0., 20., 27., 34., 41., 46., 50., 70., 100., 300.])
ETA_E = np.array([0., 0.8, 1.5, 2.4])

COLS = ['kind_category_FR', 'lep1passAnalysisWP', 'tau1genPartFlav',
        'tau1idDeepTau2017v2p1VSjet', 'tau1decayMode', 'tau1Pt', 'tau1Eta',
        'tau1Phi', 'tau1jetPt', 'nsmalljets', 'nbtags', 'genTtbarId', 'met',
        'tau1jetDeepFlavB', 'tau1jetQGL',   # _derive fills every FEATURES column
        'xsecWeight', 'genWeight', 'puWeight', 'l1PreFiringWeight',
        'btagWeight_shape', 'btagShapeR_weight', 'tauIDSF_weight',
        'triggerSF_perfilter_2nBtag_v24c', 'triggerLumiSF',
        'triggerSF_1tau1l_v1', 'Muon1IdSF', 'Ele1IdSF']


def load(channel, cache=True):
    """ttbar MC fake taus in '1tau0l' (kind_category_FR==2) or '1tau1l'
    (kind_category_FR==1 with a tight lepton)."""
    cpath = os.path.join(mc.OUTDIR, 'cache', 'ttbar_%s.npz' % channel)
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        res = {k: z[k] for k in z.files if k != 'feature_names'}
        res['feature_names'] = list(z['feature_names'])
        print('  (cached) %s: %d fake taus' % (channel, res['y'].size))
        return res

    names = list(mc.FEATURES)
    out = {k: [] for k in ('y', 'w', 'tau1jetPt', 'abs_tau1Eta', 'tau1Pt',
                           'nsmalljets', 'nbtags', 'tau1decayMode', 'met')}
    feats = {f: [] for f in names}
    for fn in sorted(glob.glob(mc.BASE + '/mc/parts/*_tree.root')):
        bnm = os.path.basename(fn)
        if not any(p in bnm for p in TT):
            continue
        incl = any(p in bnm for p in INCL_TT) and 'TTbb_4f' not in bnm
        for a in uproot.iterate(fn + ':Events', COLS, step_size='500 MB',
                                library='np'):
            if channel == '1tau0l':
                m = a['kind_category_FR'] == 2
                w = (mc.LUMI * a['xsecWeight'] * a['genWeight']
                     * a['l1PreFiringWeight'] * a['puWeight']
                     * a['btagWeight_shape'] * a['btagShapeR_weight']
                     * a['tauIDSF_weight'] * a['triggerSF_perfilter_2nBtag_v24c']
                     * a['triggerLumiSF'])
            else:
                m = (a['kind_category_FR'] == 1) & (a['lep1passAnalysisWP'] == 1)
                w = (mc.LUMI * a['xsecWeight'] * a['genWeight']
                     * a['l1PreFiringWeight'] * a['puWeight']
                     * a['btagWeight_shape'] * a['btagShapeR_weight']
                     * a['tauIDSF_weight'] * a['triggerSF_1tau1l_v1']
                     * a['Muon1IdSF'] * a['Ele1IdSF'])
            m &= a['tau1genPartFlav'] != 5          # FAKE taus only, MC truth
            if incl:
                m &= (a['genTtbarId'] % 100) < 51   # ttbar/ttbb overlap kill
            if not m.any():
                continue
            i = np.where(m)[0]
            d = mc._derive(a, i)
            for f in names:
                feats[f].append(d[f])
            out['y'].append((a['tau1idDeepTau2017v2p1VSjet'][i] >= mc.TAU_PASS).astype(np.int8))
            out['w'].append(w[i])
            for k in ('tau1jetPt', 'abs_tau1Eta', 'tau1Pt', 'nsmalljets',
                      'nbtags', 'tau1decayMode'):
                out[k].append(d[k])
            out['met'].append(a['met'][i])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res['X'] = np.column_stack([np.concatenate(feats[f]).astype(np.float32)
                                for f in names])
    res['feature_names'] = names
    if cache:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, **res)
    return res


# ---------------------------------------------------------------- binned FF
def measure_binned(res):
    """FF(jetPt, |eta| | prong) = sum w(pass) / sum w(fail), on the map's axes."""
    ff = np.zeros((2, len(PT_E) - 1, len(ETA_E) - 1))
    pr = (res['tau1decayMode'] >= 5).astype(int)
    ip = np.clip(np.digitize(res['tau1jetPt'], PT_E) - 1, 0, len(PT_E) - 2)
    ie = np.clip(np.digitize(res['abs_tau1Eta'], ETA_E) - 1, 0, len(ETA_E) - 2)
    y, w = res['y'], res['w']
    num = np.zeros_like(ff)
    den = np.zeros_like(ff)
    np.add.at(num, (pr[y == 1], ip[y == 1], ie[y == 1]), w[y == 1])
    np.add.at(den, (pr[y == 0], ip[y == 0], ie[y == 0]), w[y == 0])
    ok = den > 0
    ff[ok] = np.clip(num[ok] / den[ok], 1e-4, 20.0)
    ff[~ok] = w[y == 1].sum() / w[y == 0].sum()      # fall back to inclusive
    return ff


def apply_binned(ff, res):
    pr = (res['tau1decayMode'] >= 5).astype(int)
    ip = np.clip(np.digitize(res['tau1jetPt'], PT_E) - 1, 0, len(PT_E) - 2)
    ie = np.clip(np.digitize(res['abs_tau1Eta'], ETA_E) - 1, 0, len(ETA_E) - 2)
    return ff[pr, ip, ie]


# ---------------------------------------------------------------- reporting
def closure(w_ff, res, mask=None):
    y, w = res['y'], res['w']
    m = np.ones_like(y, bool) if mask is None else mask
    pred = (w_ff[m & (y == 0)] * w[m & (y == 0)]).sum()
    obs = w[m & (y == 1)].sum()
    return pred / obs if obs != 0 else np.nan


DIFF = [('tau1Pt', np.array([20, 30, 40, 50, 65, 85, 120, 250.])),
        ('tau1jetPt', np.array([20, 34, 41, 50, 70, 100, 150, 300.])),
        ('abs_tau1Eta', np.array([0, 0.5, 0.9, 1.3, 1.7, 2.4])),
        ('nsmalljets', np.array([3.5, 4.5, 5.5, 6.5, 7.5, 12.5])),
        ('met', np.array([0, 40, 80, 120, 180, 400.]))]


def diff_table(w_ff, res, label):
    """Yield-weighted mean |1 - pred/obs| over the bins of each variable."""
    out = {}
    for var, edges in DIFF:
        v, rows = res[var], []
        for k in range(len(edges) - 1):
            m = (v >= edges[k]) & (v < edges[k + 1])
            y, w = res['y'], res['w']
            obs = w[m & (y == 1)].sum()
            if obs <= 0 or not (m & (y == 0)).any():
                continue
            pred = (w_ff[m & (y == 0)] * w[m & (y == 0)]).sum()
            rows.append((obs, abs(1 - pred / obs)))
        if rows:
            a = np.array(rows)
            out[var] = float((a[:, 1] * a[:, 0]).sum() / a[:, 0].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='poster', choices=sorted(mc.FEATURE_SETS))
    ap.add_argument('--kfold', type=int, default=4)
    args = ap.parse_args()
    names = mc.FEATURE_SETS[args.features]

    print('=== loading ttbar MC fake taus (tau1genPartFlav != 5) ===')
    src = load('1tau1l')
    tgt = load('1tau0l')
    for nm, r in (('1tau1l', src), ('1tau0l', tgt)):
        y, w = r['y'], r['w']
        print('  %-7s %8d entries   net pass %10.1f   net fail %10.1f   '
              'inclusive FF %.4f'
              % (nm, y.size, w[y == 1].sum(), w[y == 0].sum(),
                 w[y == 1].sum() / w[y == 0].sum()))

    Xs = mc.select_features(src, names)
    Xt = mc.select_features(tgt, names)
    print('features (%d): %s' % (len(names), ', '.join(names)))

    results = {}

    # --- 1. transfer: measure in 1tau1l, predict 1tau0l --------------------
    print('\n=== transfer 1tau1l -> 1tau0l ===')
    m_src = MuffinModel(names).fit(Xs, src['y'], src['w'])
    w_muf = m_src.predict(Xt)
    ff_bin = measure_binned(src)
    w_bin = apply_binned(ff_bin, tgt)
    incl = src['w'][src['y'] == 1].sum() / src['w'][src['y'] == 0].sum()
    w_incl = np.full(tgt['y'].size, incl)
    for tag, wf in (('MUFFIN', w_muf), ('binned', w_bin), ('inclusive-only', w_incl)):
        results[('transfer', tag)] = (closure(wf, tgt), diff_table(wf, tgt, tag))

    # --- 2. reference: same-channel out-of-fold self-closure in 1tau0l -----
    print('=== reference: 1tau0l -> 1tau0l, out of fold ===')
    rng = np.random.default_rng(1234)
    fold = rng.integers(0, args.kfold, size=tgt['y'].size)
    w_cv = np.empty(tgt['y'].size)
    for k in range(args.kfold):
        tr, te = fold != k, fold == k
        w_cv[te] = MuffinModel(names).fit(Xt[tr], tgt['y'][tr], tgt['w'][tr]).predict(Xt[te])
    ff_bin_self = measure_binned(tgt)
    w_bin_self = apply_binned(ff_bin_self, tgt)
    incl_t = tgt['w'][tgt['y'] == 1].sum() / tgt['w'][tgt['y'] == 0].sum()
    for tag, wf in (('MUFFIN', w_cv), ('binned', w_bin_self),
                    ('inclusive-only', np.full(tgt['y'].size, incl_t))):
        results[('self', tag)] = (closure(wf, tgt), diff_table(wf, tgt, tag))

    # --- report -----------------------------------------------------------
    print('\n%-34s %9s | %s' % ('', 'integral', '  '.join('%-11s' % v for v, _ in DIFF)))
    print('-' * 100)
    for kind, title in (('transfer', '1tau1l -> 1tau0l  (the assumption)'),
                        ('self', '1tau0l -> 1tau0l  (reference)')):
        print('%s' % title)
        for tag in ('MUFFIN', 'binned', 'inclusive-only'):
            c, d = results[(kind, tag)]
            print('  %-32s %8.3f | %s'
                  % (tag, c, '  '.join('%9.1f%%' % (100 * d.get(v, np.nan))
                                       for v, _ in DIFF)))
    print('-' * 100)
    print('integral = predicted/observed pass yield; the rest is the '
          'yield-weighted mean |1 - pred/obs| per variable')

    # per njet x prong class, transfer only
    print('\n=== transfer, per class ===')
    nj6 = tgt['nsmalljets'] >= 6
    p3 = tgt['tau1decayMode'] >= 5
    print('  %-12s %9s %9s' % ('class', 'MUFFIN', 'binned'))
    for nm, msk in (('nj45 1p', ~nj6 & ~p3), ('nj45 3p', ~nj6 & p3),
                    ('nj6 1p', nj6 & ~p3), ('nj6 3p', nj6 & p3)):
        print('  %-12s %9.3f %9.3f'
              % (nm, closure(w_muf, tgt, msk), closure(w_bin, tgt, msk)))


if __name__ == '__main__':
    main()
