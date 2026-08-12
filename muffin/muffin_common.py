#!/usr/bin/env python3
"""MUFFIN (MUltivariate Fake-Factor INference) for the v29pre 1tau0l jet->tau_h
background -- shared configuration and data loading.

Method (Andreou, Colling, Winterbottom, LHCP poster; CMS-PAS-TAU-25-001):
the binned fake factor F_F = N_pass/N_fail is replaced by a continuous,
multi-dimensional per-object weight

    w_MUFFIN(z) = [p_pass^data(z) - p_pass^sim(z)] / [p_fail^data(z) - p_fail^sim(z)]

Implementation note (the reason this is only one classifier):
a binary classifier trained with SIGNED sample weights -- data with +1, genuine
(prompt) simulation with -w_analysis -- minimises

    -sum_i w_i [ y_i log s(z_i) + (1-y_i) log(1-s(z_i)) ]

whose per-z minimiser is s(z) = A(z)/(A(z)+B(z)) with A, B the *net* (data minus
simulation) unnormalised densities of the pass and fail classes.  Hence

    w_MUFFIN(z) = s(z) / (1 - s(z))

exactly, with no extra normalisation constant: the absolute yields are already
carried by the weights.  This is the same quantity the settled binned map
delivers as FR/(1-FR) (inclusive-loose denominator => FR/(1-FR) = pass/fail).

v29pre 1tau0l conventions are taken verbatim from measure_fr_flavour2d_v29pre.py
and taufr_1tau1l_measure.py:
  pool   kind_category_FR == 2, trigger plateau
  DR     pool && jet4DeepFlavB >= 0.1   (determination region, "MR")
  AR     pool && jet4DeepFlavB <  0.1   (validation region,    "VR")
  pass   tau1idDeepTau2017v2p1VSjet >= 8   (analysis Loose WP, v27+)
  fail   pool && tau1idDeepTau2017v2p1VSjet < 8
  sim    genuine taus only (tau1genPartFlav == 5), QCD excluded, ttbb overlap kill
"""
import glob
import os

import numpy as np
import uproot

# --------------------------------------------------------------------------
# Paths / conventions (v29pre)
# --------------------------------------------------------------------------
BASE = '/eos/user/r/rtu/TurbOutputMC2017_v29pre_ak8_option92_2017'
# the main checkout, for reading the settled binned map we benchmark against
REPO = '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework'
# The current 1tau0l baseline map (2026-08-07): x = tau1jetPt, y = |tau1Eta|
# with the original tttt-recipe bins [0, 0.8, 1.5, 2.4], njet integrated, both
# nj TH2s written with the same values.  The mother-jet DeepFlavB axis was
# dropped from the FR method, so the older fr_flavour2d_v29pre.root (y = jet
# DeepFlavB) is NOT the benchmark.
FR2D_REF = os.environ.get(
    'MUFFIN_FR2D',
    os.path.join(REPO, 'fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root'))

OUTDIR = os.environ.get('MUFFIN_OUTDIR',
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))

LUMI = 41500.0
J4CUT = 0.1                      # DR/AR split on the 4th-ranked jet b-tag
TAU_PASS = 8                     # tau1idDeepTau2017v2p1VSjet >= 8  (Loose)

INCL_TT = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic')

# --------------------------------------------------------------------------
# Input features -- the poster's list, verbatim:
#   1 decay mode of tau_h                      tau1decayMode
#   2 ratio of seeding jet to tau_h pT         tau1jetPt / tau1Pt
#   3 tau_h pT                                 tau1Pt
#   4 number of jets                           nsmalljets
#   5 number of b-tagged jets                  nbtags
#   6 eta, phi of tau_h                        tau1Eta, tau1Phi
#   7 era label                                dropped -- 2017 only
#
# tau1jetPt enters only through the ratio, as on the poster; the mother-jet
# DeepFlavB and QGL axes are NOT used -- the v29pre FR method dropped them.
#
# Deliberately NOT features: tau1idDeepTau*VSjet / tau1rawDeepTau*VSjet (they
# define pass/fail) and jet4DeepFlavB (it defines DR vs AR -- no support).
# The extra columns below the poster block are carried for plotting and for
# diagnostic feature sets only.
# --------------------------------------------------------------------------
FEATURES = [
    'tau1decayMode',
    'ptratio',            # tau1jetPt / tau1Pt
    'tau1Pt',
    'tau1jetPt',
    'nsmalljets',
    'nbtags',
    'tau1Eta',            # signed, as on the poster
    'abs_tau1Eta',        # symmetric detector -> |eta| is the cheaper axis
    'tau1Phi',
    'tau1jetDeepFlavB',
    'tau1jetQGL',
]
# feature sets selectable with MUFFIN_FEATURES / --features.
# 'poster' is the default and the one to use: the poster's list verbatim.
POSTER = ['tau1decayMode', 'ptratio', 'tau1Pt', 'nsmalljets', 'nbtags',
          'tau1Eta', 'tau1Phi']
FEATURE_SETS = {
    'poster': POSTER,
    # ---- diagnostics only, not for production ----
    # folded in eta: the detector is symmetric, so |eta| spends half the
    # statistics on the same physics
    'poster_abseta': ['tau1decayMode', 'ptratio', 'tau1Pt', 'nsmalljets',
                      'nbtags', 'abs_tau1Eta', 'tau1Phi'],
    # phi carries no physics for a fake factor: capacity spent on noise?
    'poster_nophi': ['tau1decayMode', 'ptratio', 'tau1Pt', 'nsmalljets',
                     'nbtags', 'tau1Eta'],
    # what the binned baseline map effectively uses (jetPt x |eta| x prong)
    'binlike': ['tau1decayMode', 'tau1jetPt', 'abs_tau1Eta'],
    # nbtags is the poster feature most correlated with jet4DeepFlavB, which
    # defines DR vs AR -- dropping it probes the extrapolation robustness
    'poster_nobtag': ['tau1decayMode', 'ptratio', 'tau1Pt', 'nsmalljets',
                      'tau1Eta', 'tau1Phi'],
}

# columns read from the trees
BASE_COLS = ['kind_category_FR', 'trigSF_pfHT', 'trigSF_caloHT', 'jet4DeepFlavB',
             'tau1idDeepTau2017v2p1VSjet', 'tau1decayMode', 'tau1Pt', 'tau1Eta',
             'tau1Phi', 'tau1jetPt', 'tau1jetEta', 'tau1jetDeepFlavB', 'tau1jetQGL',
             'nsmalljets', 'nbtags', 'met']
MC_COLS = ['tau1genPartFlav', 'xsecWeight', 'genWeight', 'puWeight',
           'l1PreFiringWeight', 'btagWeight_shape', 'btagShapeR_weight',
           'tauIDSF_weight', 'triggerSF_perfilter_2nBtag_v24c', 'triggerLumiSF',
           'genTtbarId']


def mc_weight(a):
    """Hadronic 1tau0l analysis weight -- identical to W_HAD in
    measure_fr_flavour2d_v29pre.py."""
    return (LUMI * a['xsecWeight'] * a['genWeight'] * a['l1PreFiringWeight']
            * a['puWeight'] * a['btagWeight_shape'] * a['btagShapeR_weight']
            * a['tauIDSF_weight'] * a['triggerSF_perfilter_2nBtag_v24c']
            * a['triggerLumiSF'])


def _derive(a, idx):
    """Build the feature dictionary for the selected entries."""
    d = {}
    tau_pt = a['tau1Pt'][idx]
    jet_pt = a['tau1jetPt'][idx]
    d['tau1decayMode'] = a['tau1decayMode'][idx]
    d['ptratio'] = np.where(tau_pt > 0, jet_pt / np.maximum(tau_pt, 1e-6), 1.0)
    d['tau1Pt'] = tau_pt
    d['tau1jetPt'] = jet_pt
    d['nsmalljets'] = a['nsmalljets'][idx]
    d['nbtags'] = a['nbtags'][idx]
    d['tau1Eta'] = a['tau1Eta'][idx]
    d['abs_tau1Eta'] = np.abs(a['tau1Eta'][idx])
    d['tau1Phi'] = a['tau1Phi'][idx]
    d['tau1jetDeepFlavB'] = a['tau1jetDeepFlavB'][idx]
    d['tau1jetQGL'] = a['tau1jetQGL'][idx]
    return d


def load_region(region, verbose=True, cache=True):
    """Load one region ('DR' | 'AR' | 'pool') as flat numpy arrays.

    Reading the 42 MC trees takes a couple of minutes, so the flattened arrays
    are cached under OUTDIR/cache; delete them after a re-production.

    Returns a dict with
      X      (n, nfeat_all) all features in FEATURES order
      y      1 = pass (VSjet >= 8), 0 = fail
      w      SIGNED weight: +1 for data, -w_analysis for genuine-tau simulation
      is_mc  bool
      met, tau1jetPt, tau1jetDeepFlavB, nsmalljets, tau1decayMode, tau1Pt,
      abs_tau1Eta  (kept for binning/plotting/closure)
    """
    assert region in ('DR', 'AR', 'pool')
    cpath = os.path.join(OUTDIR, 'cache', 'region_%s.npz' % region)
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        res = {k: z[k] for k in z.files if k != 'feature_names'}
        res['feature_names'] = list(z['feature_names'])
        if verbose:
            print('  (cached) %s: %d entries' % (cpath, res['y'].size))
        return res

    out = {k: [] for k in ('y', 'w', 'is_mc', 'met', 'tau1jetPt',
                           'tau1jetDeepFlavB', 'nsmalljets', 'tau1decayMode',
                           'tau1Pt', 'abs_tau1Eta')}
    feats = {f: [] for f in FEATURES}

    def _fill(fn, is_mc):
        cols = BASE_COLS + (MC_COLS if is_mc else [])
        incl_tt = is_mc and any(p in os.path.basename(fn) for p in INCL_TT) \
            and 'TTbb' not in os.path.basename(fn)
        n_kept = 0
        for a in uproot.iterate(fn + ':Events', cols, step_size='500 MB', library='np'):
            m = ((a['kind_category_FR'] == 2)
                 & (a['trigSF_pfHT'] >= 300) & (a['trigSF_caloHT'] >= 160))
            if region == 'DR':
                m &= a['jet4DeepFlavB'] >= J4CUT
            elif region == 'AR':
                m &= a['jet4DeepFlavB'] < J4CUT
            if is_mc:
                m &= a['tau1genPartFlav'] == 5          # genuine taus only
                if incl_tt:
                    m &= (a['genTtbarId'] % 100) < 51   # ttbar/ttbb overlap kill
            if not m.any():
                continue
            idx = np.where(m)[0]
            if is_mc:
                w = -mc_weight(a)[idx]                  # subtracted -> negative
            else:
                w = np.ones(idx.size)
            d = _derive(a, idx)
            for f in FEATURES:
                feats[f].append(d[f])
            out['y'].append((a['tau1idDeepTau2017v2p1VSjet'][idx] >= TAU_PASS).astype(np.int8))
            out['w'].append(w)
            out['is_mc'].append(np.full(idx.size, is_mc, bool))
            out['met'].append(a['met'][idx])
            out['tau1jetPt'].append(d['tau1jetPt'])
            out['tau1jetDeepFlavB'].append(d['tau1jetDeepFlavB'])
            out['nsmalljets'].append(d['nsmalljets'])
            out['tau1decayMode'].append(d['tau1decayMode'])
            out['tau1Pt'].append(d['tau1Pt'])
            out['abs_tau1Eta'].append(d['abs_tau1Eta'])
            n_kept += idx.size
        return n_kept

    n = _fill(BASE + '/data/parts/BTagCSV_tree.root', False)
    if verbose:
        print('  data BTagCSV            : %7d' % n)
    for fn in sorted(glob.glob(BASE + '/mc/parts/*_tree.root')):
        bnm = os.path.basename(fn)
        if bnm.startswith('QCD') or 'FakeTau' in bnm:
            continue                                    # QCD is the data-driven part
        n = _fill(fn, True)
        if verbose and n:
            print('  %-24s: %7d' % (bnm.replace('_tree.root', ''), n))

    res = {k: np.concatenate(v) for k, v in out.items()}
    res['X'] = np.column_stack([np.concatenate(feats[f]).astype(np.float32)
                                for f in FEATURES])
    res['feature_names'] = list(FEATURES)
    if cache:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, **res)
    return res


def select_features(res, names):
    """Column-slice res['X'] down to `names`."""
    cols = [res['feature_names'].index(f) for f in names]
    return res['X'][:, cols]


def summarise(res, tag):
    y, w, mc = res['y'], res['w'], res['is_mc']
    p, f = y == 1, y == 0
    print('[%s] entries %d  (data %d, sim %d)' % (tag, y.size, (~mc).sum(), mc.sum()))
    print('       pass: data %8.1f  sim %8.1f  net %8.1f'
          % (w[p & ~mc].sum(), -w[p & mc].sum(), w[p].sum()))
    print('       fail: data %8.1f  sim %8.1f  net %8.1f'
          % (w[f & ~mc].sum(), -w[f & mc].sum(), w[f].sum()))
    print('       inclusive F_F = net_pass/net_fail = %.4f'
          % (w[p].sum() / max(w[f].sum(), 1e-9)))
