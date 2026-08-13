#!/usr/bin/env python3
"""Measure MUFFIN in 1tau1l itself, instead of carrying the 1tau0l one over.

Conventions follow taufr_1tau1l_measure.py:
  pool  kind_category_FR == 1 with a tight lepton (lep1passAnalysisWP)
  data  SingleMuon and SingleElectron, each restricted to its own lepton flavour
  pass  tau1idDeepTau2017v2p1VSjet >= 8      fail  pool && < 8
  sim   genuine taus (tau1genPartFlav == 5) subtracted with the 1tau1l weight
        chain (triggerSF_1tau1l_v1 x Muon1IdSF x Ele1IdSF), QCD excluded,
        ttbb overlap kill on the inclusive TT samples

The determination region is the channel's own MR -- jet4DeepFlavB < 0.035, the
balanced 60/40 split that taufr_1tau1l_measure.py calls mr60 -- and the VR is
its complement.  Measuring in the MR and validating in the VR keeps this
parallel to the 1tau0l measurement, and it means the agreement plots are NOT
closing by construction: an inclusive in-channel fit would reproduce the tight
region trivially and would show nothing.

    bash muffin/run.sh train_muffin_1tau1l.py --bootstrap 20
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
from train_muffin import closure_table  # noqa: E402

J4CUT_1T1L = 0.035          # mr60 / vr40 split of taufr_1tau1l_measure.py
INCL_TT = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic')

COLS = ['kind_category_FR', 'lep1passAnalysisWP', 'lep1Id', 'jet4DeepFlavB',
        'tau1idDeepTau2017v2p1VSjet', 'tau1decayMode', 'tau1Pt', 'tau1Eta',
        'tau1Phi', 'tau1jetPt', 'tau1jetDeepFlavB', 'tau1jetQGL',
        'nsmalljets', 'nbtags', 'met']
WCOLS = ['tau1genPartFlav', 'xsecWeight', 'genWeight', 'puWeight',
         'l1PreFiringWeight', 'btagWeight_shape', 'btagShapeR_weight',
         'tauIDSF_weight', 'triggerSF_1tau1l_v1', 'Muon1IdSF', 'Ele1IdSF',
         'genTtbarId']


def load(region, cache=True):
    """region: 'MR' (jet4DeepFlavB < 0.035), 'VR' (>=), or 'all'."""
    cpath = os.path.join(mc.OUTDIR, 'cache', '1tau1l_%s.npz' % region)
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        res = {k: z[k] for k in z.files if k != 'feature_names'}
        res['feature_names'] = list(z['feature_names'])
        print('  (cached) %s: %d entries' % (region, res['y'].size))
        return res

    names = list(mc.FEATURES)
    out = {k: [] for k in ('y', 'w', 'is_mc', 'nsmalljets', 'tau1decayMode',
                           'tau1Pt', 'tau1jetPt', 'abs_tau1Eta', 'met')}
    feats = {f: [] for f in names}

    def fill(fn, is_mc, flav=None):
        cols = COLS + (WCOLS if is_mc else [])
        bnm = os.path.basename(fn)
        incl = is_mc and any(p in bnm for p in INCL_TT) and 'TTbb_4f' not in bnm
        n = 0
        for a in uproot.iterate(fn + ':Events', cols, step_size='500 MB',
                                library='np'):
            m = (a['kind_category_FR'] == 1) & (a['lep1passAnalysisWP'] == 1)
            if flav is not None:
                m &= np.abs(a['lep1Id']) == flav
            if region == 'MR':
                m &= a['jet4DeepFlavB'] < J4CUT_1T1L
            elif region == 'VR':
                m &= a['jet4DeepFlavB'] >= J4CUT_1T1L
            if is_mc:
                m &= a['tau1genPartFlav'] == 5
                if incl:
                    m &= (a['genTtbarId'] % 100) < 51
            if not m.any():
                continue
            i = np.where(m)[0]
            if is_mc:
                w = -(mc.LUMI * a['xsecWeight'] * a['genWeight'] * a['puWeight']
                      * a['l1PreFiringWeight'] * a['btagWeight_shape']
                      * a['btagShapeR_weight'] * a['tauIDSF_weight']
                      * a['triggerSF_1tau1l_v1'] * a['Muon1IdSF'] * a['Ele1IdSF'])[i]
            else:
                w = np.ones(i.size)
            d = mc._derive(a, i)
            for f in names:
                feats[f].append(d[f])
            out['y'].append((a['tau1idDeepTau2017v2p1VSjet'][i] >= mc.TAU_PASS).astype(np.int8))
            out['w'].append(w)
            out['is_mc'].append(np.full(i.size, is_mc, bool))
            for k in ('nsmalljets', 'tau1decayMode', 'tau1Pt', 'tau1jetPt',
                      'abs_tau1Eta'):
                out[k].append(d[k])
            out['met'].append(a['met'][i])
            n += i.size
        return n

    for pd, fl in (('SingleMuon', 13), ('SingleElectron', 11)):
        print('  data %-16s %8d' % (pd, fill('%s/data/parts/%s_tree.root'
                                             % (mc.BASE, pd), False, fl)))
    for fn in sorted(glob.glob(mc.BASE + '/mc/parts/*_tree.root')):
        b = os.path.basename(fn)
        if b.startswith('QCD') or 'FakeTau' in b:
            continue
        fill(fn, True)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res['X'] = np.column_stack([np.concatenate(feats[f]).astype(np.float32)
                                for f in names])
    res['feature_names'] = names
    if cache:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, **res)
    return res


def classes(res):
    p3 = res['tau1decayMode'] >= 5
    nj6 = res['nsmalljets'] >= 6
    return [('nj45 1p', ~nj6 & ~p3), ('nj45 3p', ~nj6 & p3),
            ('nj6 1p', nj6 & ~p3), ('nj6 3p', nj6 & p3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='poster', choices=sorted(mc.FEATURE_SETS))
    ap.add_argument('--tag', default='1tau1l')
    ap.add_argument('--bootstrap', type=int, default=0)
    ap.add_argument('--kfold', type=int, default=4)
    args = ap.parse_args()
    names = mc.FEATURE_SETS[args.features]
    mdir = os.path.join(mc.OUTDIR, 'models')

    print('=== 1tau1l determination region (jet4DeepFlavB < %.3f) ===' % J4CUT_1T1L)
    dr = load('MR')
    mc.summarise(dr, 'MR')
    X, y, w = mc.select_features(dr, names), dr['y'], dr['w']
    print('features (%d): %s' % (len(names), ', '.join(names)))

    print('\n=== out-of-fold self-closure in the MR ===')
    rng = np.random.default_rng(1234)
    fold = rng.integers(0, args.kfold, size=y.size)
    w_cv = np.empty(y.size)
    for k in range(args.kfold):
        tr, te = fold != k, fold == k
        w_cv[te] = MuffinModel(names).fit(X[tr], y[tr], w[tr]).predict(X[te])
    print(closure_table(w_cv, y, w, classes(dr)))

    model = MuffinModel(names).fit(X, y, w)
    model.save(os.path.join(mdir, 'muffin_%s_nominal' % args.tag))
    print('\n  inclusive F_F %.4f   norm %.4f' % (model.inclusive_ff, model.norm))
    print('  feature ranking:')
    for i, (f, g) in enumerate(model.ranking()):
        print('    %2d. %-16s %9.1f' % (i + 1, f, g))

    print('\n=== MR -> VR closure (the validation) ===')
    vr = load('VR')
    mc.summarise(vr, 'VR')
    Xv = mc.select_features(vr, names)
    print(closure_table(model.predict(Xv), vr['y'], vr['w'], classes(vr)))

    if args.bootstrap:
        print('\n=== %d bootstrap replicas ===' % args.bootstrap)
        for b in range(args.bootstrap):
            pre = os.path.join(mdir, 'muffin_%s_boot%03d' % (args.tag, b))
            if os.path.exists(pre + '_cfg.json'):
                continue
            r = np.random.default_rng(9000 + b)
            MuffinModel(names).fit(X, y, w, seed=b,
                                   boot=r.poisson(1.0, size=w.size)).save(pre)
        print('  done')


if __name__ == '__main__':
    main()
