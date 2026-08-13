#!/usr/bin/env python3
"""
2tau0l closure plotting with a data-driven Fake-tau background built from the
2-tau INCLUSION-EXCLUSION fake factor -- with the MUFFIN fake factor.

Copy of closure_v29pre_2tau0l_eta.py plus USE_MUFFIN=1, which replaces the
per-tau transfer factor FF_i with the MUFFIN weight (muffin/README.md) and puts
BOTH Data/Pred curves in the ratio pad -- MUFFIN as the points, the binned map
as a red line.  The inclusion-exclusion algebra, regions, samples, weights and
binning are untouched; only FF_i changes.

Both fake factors were measured in 1tau0l and are applied here unchanged, so
this is a CROSS-CHANNEL TRANSFER TEST for both of them on equal terms: the
question is not which models 2tau0l best from scratch, but which one carries
better from the channel it was fitted in.

Adapted from closure_v26_in_v26_option92_1tau0l_NHiggs_20bin_split.py.

Region (2tau0l):
  - pool / loose-WP region: kind_category_FR == 0  (both saved taus at pool WP VSjet>=2)
  - SR / observed:          kind_category_analysis == 0  with BOTH taus tight

Per tau i in {1,2}:
  - tight_i = tau{i}idDeepTau2017v2p1VSjet >= 8   (Loose WP)
  - anti_i  = tau{i}idDeepTau2017v2p1VSjet >= 2 && < 8
  - FF{i}   = tauFR_weight(tau{i}jetPt, tau{i}jetEta, tau{i}decayMode, nsmalljets)
              == FR/(1-FR)  (loose->tight transfer factor)

2-tau inclusion-exclusion fake template (predicts the (tight,tight) fake yield):
  FAKE = sum_data[ tight1 & anti2 ] * FF2
       + sum_data[ anti1 & tight2 ] * FF1
       - sum_data[ anti1 & anti2 ] * FF1*FF2
       (MINUS the same three terms on prompt MC, to remove real-tau events
        already in the prompt stack -- the anti-ID tau being a real tau)
  Per-bin clamp to >= 0.

Prompt-MC subtraction (real tau in the anti-ID slot):
  - (tight1 & anti2): subtract prompt-MC with tau2genPartFlav==5, weight mc_weight*FF2
  - (anti1 & tight2): subtract prompt-MC with tau1genPartFlav==5, weight mc_weight*FF1
  - (anti1 & anti2):  subtract prompt-MC with (tau1genPartFlav==5 || tau2genPartFlav==5),
                      weight mc_weight*FF1*FF2
  (The (anti1&anti2) term uses the OR convention -- the same loose-prompt
   subtraction spirit as the 1tau0l plotter; documented here as the choice.)

Observed side:
  - Data (obs) = data with kind_category_analysis==0 && tight1 && tight2
  - Prompt-MC stack (per group) = MC with tight1 && tight2 && tau1genPartFlav==5
    && tau2genPartFlav==5 (both real), v26 weight chain. QCD MC omitted.
  - Total pred = prompt-MC stack + Fake template. Data/Pred + Stat(+)FR band.
  - Signal x1000 overlays (tight1 && tight2 && kc_analysis==0).

Only the three SPANet score branches are plotted (full range, no cut, lin+log):
  ProbHHH4b2tau, ProbHHH4b2tau_Disco, ProbHHH4b2tau_Disco_ep60

Usage:
  cmssw-el7 -- bash -c 'cd .../CMSSW_12_5_2/src && eval $(scramv1 runtime -sh) && \
      python3 hhh-analysis-framework/closure_v26_2tau0l.py [--variable ProbHHH4b2tau]'
"""
from __future__ import print_function
import os
import sys
import glob
import array
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
BASEDIR = '/eos/user/r/rtu/TurbOutputMC2017_v29pre_ak8_option92_2017'
MERGED = 'parts'
LUMI = 41500.0  # pb^-1 for 2017
OUTDIR = os.environ.get('CLOSURE_OUTDIR',
    '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/plots_v29pre_2tau0l_eta_MRmap')

# 1tau0l-measured eta-axis fr2d map (fr2d_{nj}_{pr} TH2, x=jetPt, y=|tau eta|),
# applied cross-channel to 2tau0l (same hadronic trigger + cuts -> same fake
# composition; this closure IS the transfer test).
FR2D_ROOT = os.environ.get('FR2D_ROOT',
    '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root')

# MUFFIN: the 1tau0l-trained multivariate fake factor, applied per tau.
# USE_MUFFIN=1 keeps the binned map too -- it is the comparison curve.
# SIDEBAND=1: close on the single-tight sideband instead of (tight,tight).
# Same fake factor, same events, 526 observed instead of 111.
SIDEBAND = os.environ.get('SIDEBAND', '') == '1'
USE_MUFFIN = os.environ.get('USE_MUFFIN', '') == '1'
MUFFIN_HEADER = os.environ.get(
    'MUFFIN_HEADER', os.path.join(THIS_DIR, 'muffin', 'out', 'muffin_poster.h'))
if USE_MUFFIN and 'CLOSURE_OUTDIR' not in os.environ:
    OUTDIR = OUTDIR + '_muffin'
if SIDEBAND and 'CLOSURE_OUTDIR' not in os.environ:
    OUTDIR = OUTDIR + '_sideband'

NBINS_UNIFORM = int(os.environ.get('NBINS_UNIFORM', '20'))
# BINS_JSON: {var: [edges]} overriding the default binning (statistics-optimised
# edges from optimize_bins_2tau0l.py). HISTOS_ONLY: fill + write the ROOT file
# but skip canvas drawing (fast pass used to derive those edges).
BINS_JSON = os.environ.get('BINS_JSON', '')
HISTOS_ONLY = os.environ.get('HISTOS_ONLY', '') == '1'

# 1tau0l-plotter convention: for per-object quantities a stored 0 means "this
# object is not in the event", so those entries must not be histogrammed. Only
# event-level quantities and globals where 0 is physically reachable are exempt
# (mt2_ll piles up at exactly 0 for signal; dzeta/pzeta_miss cross 0).
NO_ZERO_FILTER_VARS = set([
    'ht', 'met', 'metphi', 'nsmalljets', 'nfatjets', 'nbtags', 'nbtags_loose',
    'tau1decayMode', 'tau2decayMode', 'tau1Charge', 'tau2Charge',
    'min_dphi_jet_met', 'met_over_ht', 'met_significance',
    'sphericity', 'aplanarity', 'sphericity_lin', 'shapeC', 'shapeD',
    'mt2_bb', 'dzeta', 'pzeta_vis', 'pzeta_miss', 'mt_tot', 'mt2_ll',
    'tau1Mt', 'tau2Mt',
    'higgs3_mass_manu', 'higgs3_pt_manu', 'higgs3_eta_manu', 'higgs3_phi_manu',
    'deltaR_taupair', 'deltaPhi_taupair_MET',
])


def get_mc_groups():
    """Prompt-tau MC process groups (stacked, bottom -> top).
    QCD intentionally omitted -- the data-driven fake template covers the QCD
    fake-tau contribution; including QCD MC would double-count."""
    import ROOT
    return [
        ('VV',    ['WWTo', 'WZTo', 'ZZTo'],            ROOT.kOrange + 1),
        ('Vjets', ['DYJetsTo', 'WJetsTo', 'ZJetsTo'],  ROOT.kGreen + 2),
        ('ttX',   ['TTWJets', 'TTZTo', 'ttHJet'],      ROOT.kMagenta - 4),
        ('ttbb',  ['TTbb_4f_TTTo'],                    ROOT.kAzure + 7),
        ('ttbar', ['TTTo2L2Nu', 'TTToHadronic',
                   'TTToSemiLeptonic'],                ROOT.kAzure - 9),
    ]


def linbin(lo, hi, n):
    step = (hi - lo) / n
    return array.array('d', [lo + i * step for i in range(n + 1)])


def cont(lo, hi):
    return linbin(lo, hi, NBINS_UNIFORM)


def build_plot_vars():
    import array
    pv = [
        ('ht',               'H_{T} [GeV]',        cont(330, 1500)),
        ('met',              'MET [GeV]',          cont(0, 300)),
        ('metphi',           'MET #phi',           cont(-3.14, 3.14)),
        ('nsmalljets',       'N_{jets} (small R)', array.array('d', [4, 5, 6, 7, 8, 9, 11])),
        ('nfatjets',         'N_{fatjets}',        array.array('d', [0, 1, 2, 3, 4])),
        ('nbtags',           'N_{b-tags medium}',  array.array('d', [2.5, 3.5, 4.5, 5.5, 8.5])),
        ('nbtags_loose',     'N_{b-tags loose}',   array.array('d', [2.5, 3.5, 4.5, 5.5, 6.5, 10.5])),
    ]
    for i in range(1, 9):
        pv += [
            ('jet%dMass'      % i, 'jet_{%d} mass [GeV]' % i, cont(0, 100)),
            ('jet%dPt'        % i, 'jet_{%d} p_{T} [GeV]' % i, cont(20, 500)),
            ('jet%dbRegCorr'  % i, 'jet_{%d} bRegCorr'    % i, cont(0.5, 1.5)),
            ('jet%dEta'       % i, 'jet_{%d} #eta'        % i, cont(-2.5, 2.5)),
            ('jet%dPhi'       % i, 'jet_{%d} #phi'        % i, cont(-3.14, 3.14)),
            ('jet%dDeepFlavB' % i, 'jet_{%d} DeepFlavB'   % i, cont(0, 1)),
        ]
    for i in range(1, 4):
        pv += [
            ('fatJet%dPt'                 % i, 'fj_{%d} p_{T} [GeV]'  % i, cont(0, 700)),
            ('fatJet%dEta'                % i, 'fj_{%d} #eta'         % i, cont(-2.5, 2.5)),
            ('fatJet%dPhi'                % i, 'fj_{%d} #phi'         % i, cont(-3.14, 3.14)),
            ('fatJet%dPNetXbb'            % i, 'fj_{%d} PNetXbb'      % i, cont(0, 1)),
            ('fatJet%dPNetXjj'            % i, 'fj_{%d} PNetXjj'      % i, cont(0, 1)),
            ('fatJet%dMassSD_UnCorrected' % i, 'fj_{%d} SDmass [GeV]' % i, cont(0, 300)),
        ]
    for i in range(1, 3):
        pv += [
            ('tau%dMass'      % i, '#tau_{%d} mass [GeV]'  % i, cont(0, 3.0)),
            ('tau%dPt'        % i, '#tau_{%d} p_{T} [GeV]' % i, cont(0, 200)),
            ('tau%dEta'       % i, '#tau_{%d} #eta'        % i, cont(-2.4, 2.4)),
            ('tau%dPhi'       % i, '#tau_{%d} #phi'        % i, cont(-3.14, 3.14)),
            ('tau%dCharge'    % i, '#tau_{%d} charge'      % i, array.array('d', [-1.5, -0.5, 0.5, 1.5])),
            ('tau%ddecayMode' % i, '#tau_{%d} decayMode'   % i, array.array('d', [-0.5, 0.5, 1.5, 2.5, 9.5, 10.5, 11.5])),
            ('tau%dMt'        % i, '#tau_{%d} M_{T}(#tau,MET) [GeV]' % i, cont(0, 200)),
        ]
    pv += [
        ('higgs3_mass_manu',     '#tau-pair mass [GeV]',      cont(0, 300)),
        ('higgs3_pt_manu',       '#tau-pair p_{T} [GeV]',     cont(0, 500)),
        ('higgs3_eta_manu',      '#tau-pair #eta',            cont(-3.0, 3.0)),
        ('higgs3_phi_manu',      '#tau-pair #phi',            cont(-3.14, 3.14)),
        ('deltaR_taupair',       '#DeltaR(#tau,#tau)',        cont(0, 5)),
        ('deltaPhi_taupair_MET', '#Delta#phi(#tau#tau, MET)', cont(-3.14, 3.14)),
    ]
    for i in range(1, 8):
        for j in range(i + 1, 9):
            tag = 'jet%djet%d' % (i, j)
            pv += [
                ('mass' + tag, 'm(j_{%d}j_{%d}) [GeV]'     % (i, j), cont(0, 800)),
                ('pt'   + tag, 'p_{T}(j_{%d}j_{%d}) [GeV]' % (i, j), cont(0, 800)),
                ('eta'  + tag, '#eta(j_{%d}j_{%d})'        % (i, j), cont(-5, 5)),
                ('phi'  + tag, '#phi(j_{%d}j_{%d})'        % (i, j), cont(-3.14, 3.14)),
                ('dr'   + tag, '#DeltaR(j_{%d}j_{%d})'     % (i, j), cont(0, 5)),
            ]
    pv += [
        ('min_dphi_jet_met', 'min #Delta#phi(4 jets, MET)', cont(0, 3.15)),
        ('met_over_ht',      'MET / H_{T}',                 cont(0, 1.0)),
        ('met_significance', 'MET significance',            cont(0, 100)),
        ('sphericity',       'sphericity',                  cont(0, 1)),
        ('aplanarity',       'aplanarity',                  cont(0, 0.5)),
        ('sphericity_lin',   'sphericity (linear)',         cont(0, 1)),
        ('shapeC',           'shape C',                     cont(0, 1)),
        ('shapeD',           'shape D',                     cont(0, 1)),
        ('mt2_bb',           'M_{T2}(bb) [GeV]',            cont(0, 300)),
        ('dzeta',            'D_{#zeta} [GeV]',             cont(-200, 200)),
        ('pzeta_vis',        'p_{#zeta}^{vis} [GeV]',       cont(0, 300)),
        ('pzeta_miss',       'p_{#zeta}^{miss} [GeV]',      cont(-100, 300)),
        ('mt_tot',           'm_{T}^{tot} [GeV]',           cont(0, 500)),
        ('mt2_ll',           'M_{T2}(#tau#tau) [GeV]',      cont(0, 150)),
    ]
    return pv


# =========================================================================
def make_df(subdir, exclude_patterns=(), include_patterns=None):
    import ROOT
    parts_dir = os.path.join(BASEDIR, subdir, MERGED)
    files = sorted(glob.glob(os.path.join(parts_dir, '*_tree.root')))
    # FakeTau (FR) method: QCD is captured by the data-driven fake template, so it
    # must NOT enter the MC prompt subtraction in MR/VR (genPartFlav==5 in QCD is
    # spurious and would double-count). Drop QCD from every 'mc' input.
    if subdir == 'mc':
        exclude_patterns = tuple(exclude_patterns) + ('QCD',)
    files = [f for f in files
             if not any(p in os.path.basename(f) for p in exclude_patterns)]
    if include_patterns is not None:
        files = [f for f in files
                 if any(p in os.path.basename(f) for p in include_patterns)]
    if not files:
        return None, []
    vec = ROOT.std.vector('string')()
    for f in files:
        vec.push_back(f)
    return ROOT.RDataFrame('Events', vec), files


def common_defines(df):
    cols = set(str(c) for c in df.GetColumnNames())
    if 'min_dphi_jet_met' not in cols:
        df = df.Define('min_dphi_jet_met',
            'std::min(std::min(fabs(TVector2::Phi_mpi_pi(jet1Phi-metphi)), fabs(TVector2::Phi_mpi_pi(jet2Phi-metphi))),'
            ' std::min(fabs(TVector2::Phi_mpi_pi(jet3Phi-metphi)), fabs(TVector2::Phi_mpi_pi(jet4Phi-metphi))))')
    if 'met_over_ht' not in cols:
        df = df.Define('met_over_ht', 'met / std::max((double)ht, 1e-6)')
    if 'nbtags_loose' not in cols:
        # nbtags (stored) counts DeepFlavB > MEDIUM (0.3040 in 2017) and the
        # hadronic selection already requires >=3 of them, so it only ever takes
        # 3/4/5 -- truncated and uninformative. The LOOSE count (>0.0532) is the
        # meaningful b-content axis for 2tau0l. Summed over the stored jet1..10
        # (btag-sorted, so these are the 10 highest-btag jets).
        _wpl = 0.0532
        df = df.Define('nbtags_loose', '(int)(' + ' + '.join(
            '(jet%dDeepFlavB > %ff)' % (i, _wpl) for i in range(1, 11)) + ')')
    # Per-tau tight / anti (loose-not-tight) at pool WP
    df = df.Define('tight1', 'tau1idDeepTau2017v2p1VSjet >= 8')
    df = df.Define('tight2', 'tau2idDeepTau2017v2p1VSjet >= 8')
    df = df.Define('anti1',  'tau1idDeepTau2017v2p1VSjet >= 2 && tau1idDeepTau2017v2p1VSjet < 8')
    df = df.Define('anti2',  'tau2idDeepTau2017v2p1VSjet >= 2 && tau2idDeepTau2017v2p1VSjet < 8')
    # Fake factors FF = FR/(1-FR) for each tau, via the SAME positional helper
    df = df.Define('FF1',      ('tauFR_weight_2d(tau1jetPt, ' + ('(float)std::abs(tau1Eta)' if os.environ.get('FR2D_YVAR', 'abseta') == 'abseta' else 'tau1jetDeepFlavB') + ', tau1decayMode, nsmalljets)'))
    df = df.Define('FF1_up',   ('tauFR_weight_2d_up(tau1jetPt, ' + ('(float)std::abs(tau1Eta)' if os.environ.get('FR2D_YVAR', 'abseta') == 'abseta' else 'tau1jetDeepFlavB') + ', tau1decayMode, nsmalljets)'))
    df = df.Define('FF1_down', ('tauFR_weight_2d_down(tau1jetPt, ' + ('(float)std::abs(tau1Eta)' if os.environ.get('FR2D_YVAR', 'abseta') == 'abseta' else 'tau1jetDeepFlavB') + ', tau1decayMode, nsmalljets)'))
    df = df.Define('FF2',      ('tauFR_weight_2d(tau2jetPt, ' + ('(float)std::abs(tau2Eta)' if os.environ.get('FR2D_YVAR', 'abseta') == 'abseta' else 'tau2jetDeepFlavB') + ', tau2decayMode, nsmalljets)'))
    df = df.Define('FF2_up',   ('tauFR_weight_2d_up(tau2jetPt, ' + ('(float)std::abs(tau2Eta)' if os.environ.get('FR2D_YVAR', 'abseta') == 'abseta' else 'tau2jetDeepFlavB') + ', tau2decayMode, nsmalljets)'))
    df = df.Define('FF2_down', ('tauFR_weight_2d_down(tau2jetPt, ' + ('(float)std::abs(tau2Eta)' if os.environ.get('FR2D_YVAR', 'abseta') == 'abseta' else 'tau2jetDeepFlavB') + ', tau2decayMode, nsmalljets)'))
    # Combined FF for the double-anti term
    df = df.Define('FF12',      'FF1 * FF2')
    df = df.Define('FF12_up',   'FF1_up * FF2_up')
    df = df.Define('FF12_down', 'FF1_down * FF2_down')
    if USE_MUFFIN:
        # The very same fake factor, evaluated on each tau in turn: MUFFIN is a
        # per-object weight, so the inclusion-exclusion algebra below needs no
        # change at all.  The feature list is read off the exported header.
        for i in (1, 2):
            args = ', '.join(muffin_branch_expr(f, i) for f in muffin_features())
            df = df.Define('FF%d_muf' % i, 'muffin_weight(%s)' % args)
            df = df.Define('FF%d_muf_rms' % i, 'muffin_weight_rms(%s)' % args)
            df = df.Define('FF%d_muf_up' % i, 'FF%d_muf + FF%d_muf_rms' % (i, i))
            df = df.Define('FF%d_muf_down' % i,
                           'std::max(0., FF%d_muf - FF%d_muf_rms)' % (i, i))
        df = df.Define('FF12_muf',      'FF1_muf * FF2_muf')
        df = df.Define('FF12_muf_up',   'FF1_muf_up * FF2_muf_up')
        df = df.Define('FF12_muf_down', 'FF1_muf_down * FF2_muf_down')
    return df


def add_mc_weight(df, with_ttbb_kill=True, scale=1.0):
    if with_ttbb_kill:
        df = df.Define('ttbb_overlap_kill', '((genTtbarId % 100) < 51) ? 1.0f : 0.0f')
    else:
        df = df.Define('ttbb_overlap_kill', '1.0f')
    df = df.Define('mc_weight',
        '(float)(%f * ttbb_overlap_kill * xsecWeight * genWeight * l1PreFiringWeight * puWeight'
        ' * triggerLumiSF * triggerSF_perfilter_2nBtag_v24c * btagWeight_shape * btagShapeR_weight'
        ' * tauIDSF_weight * Muon1IdSF * Ele1IdSF * %f)' % (scale, LUMI))
    # MC-weighted FF products for the prompt subtraction in each application region
    df = df.Define('mc_FF1',      'mc_weight * FF1')
    df = df.Define('mc_FF1_up',   'mc_weight * FF1_up')
    df = df.Define('mc_FF1_down', 'mc_weight * FF1_down')
    df = df.Define('mc_FF2',      'mc_weight * FF2')
    df = df.Define('mc_FF2_up',   'mc_weight * FF2_up')
    df = df.Define('mc_FF2_down', 'mc_weight * FF2_down')
    df = df.Define('mc_FF12',      'mc_weight * FF12')
    df = df.Define('mc_FF12_up',   'mc_weight * FF12_up')
    df = df.Define('mc_FF12_down', 'mc_weight * FF12_down')
    if USE_MUFFIN:
        for tag in ('FF1_muf', 'FF2_muf', 'FF12_muf'):
            for suf in ('', '_up', '_down'):
                df = df.Define('mc_%s%s' % (tag, suf), 'mc_weight * %s%s' % (tag, suf))
    return df


def muffin_features():
    """Feature names, in call order, as recorded by the exported header."""
    tag = '// features (in order): '
    with open(MUFFIN_HEADER) as fh:
        for line in fh:
            if line.startswith(tag):
                return [f.strip() for f in line[len(tag):].split(',')]
    raise RuntimeError('no feature list in %s' % MUFFIN_HEADER)


def muffin_branch_expr(feature, i):
    """The 1tau0l feature, re-pointed at tau `i` of the 2tau0l event."""
    return {
        'tau1decayMode':    'tau%ddecayMode' % i,
        'ptratio':          '(tau%djetPt/std::max(tau%dPt,1.e-6f))' % (i, i),
        'tau1Pt':           'tau%dPt' % i,
        'tau1jetPt':        'tau%djetPt' % i,
        'nsmalljets':       'nsmalljets',
        'nbtags':           'nbtags',
        'tau1Eta':          'tau%dEta' % i,
        'abs_tau1Eta':      'std::fabs(tau%dEta)' % i,
        'tau1Phi':          'tau%dPhi' % i,
        'tau1jetDeepFlavB': 'tau%djetDeepFlavB' % i,
        'tau1jetQGL':       'tau%djetQGL' % i,
    }[feature]


def declare_muffin():
    """JIT-declare the exported MUFFIN evaluator."""
    import ROOT
    with open(MUFFIN_HEADER) as fh:
        src = fh.read()
    if not ROOT.gInterpreter.Declare(src):
        raise RuntimeError('muffin_weight Declare failed (%s)' % MUFFIN_HEADER)
    print('  muffin_weight{,_rms} declared from %s' % MUFFIN_HEADER)


def declare_fr2d():
    """Read fr_flavour2d_v26.root and JIT-declare tauFR_weight_2d{,_up,_down}
    (jetPt, tau1jetDeepFlavB, decayMode, nsmalljets) = FR/(1-FR) from the 2-D table."""
    import ROOT
    f = ROOT.TFile.Open(FR2D_ROOT)
    h0 = f.Get('fr2d_nj45_1p')
    npt, nb = h0.GetNbinsX(), h0.GetNbinsY()
    pte = [h0.GetXaxis().GetBinLowEdge(1)] + [h0.GetXaxis().GetBinUpEdge(i) for i in range(1, npt + 1)]
    be = [h0.GetYaxis().GetBinLowEdge(1)] + [h0.GetYaxis().GetBinUpEdge(i) for i in range(1, nb + 1)]

    def table(tag, shift):
        h = f.Get('fr2d_%s' % tag)
        rows = []
        for ix in range(1, npt + 1):
            cols = []
            for iy in range(1, nb + 1):
                v = h.GetBinContent(ix, iy) + shift * h.GetBinError(ix, iy)
                cols.append(max(0.0, min(0.999, v)))
            rows.append('{' + ', '.join('%.6f' % v for v in cols) + '}')
        return '{' + ', '.join(rows) + '}'

    def arr1(vals):
        return '{' + ', '.join('%.5f' % v for v in vals) + '}'

    lines = ['namespace FR2D {',
             '  static const double pte[%d] = %s;' % (npt + 1, arr1(pte)),
             '  static const double be[%d]  = %s;' % (nb + 1, arr1(be)),
             '  static const int npt = %d, nb = %d;' % (npt, nb),
             '}',
             'inline int _fr2d_ptb(float pt){ for(int i=0;i<FR2D::npt;i++) if(pt>=FR2D::pte[i]&&pt<FR2D::pte[i+1]) return i; return FR2D::npt-1; }',
             'inline int _fr2d_bb(float b){ if(b<0.f) return 0; for(int i=0;i<FR2D::nb;i++) if(b>=FR2D::be[i]&&b<FR2D::be[i+1]) return i; return FR2D::nb-1; }']
    for suf, shift in (('', 0), ('_up', +1), ('_down', -1)):
        lines += [
            'inline double tauFR_weight_2d%s(float jetPt, float b, int dm, int nsmalljets){' % suf,
            '  static const double fr_nj45_1p[%d][%d]=%s;' % (npt, nb, table('nj45_1p', shift)),
            '  static const double fr_nj45_3p[%d][%d]=%s;' % (npt, nb, table('nj45_3p', shift)),
            '  static const double fr_nj6_1p[%d][%d]=%s;'  % (npt, nb, table('nj6_1p', shift)),
            '  static const double fr_nj6_3p[%d][%d]=%s;'  % (npt, nb, table('nj6_3p', shift)),
            '  const bool is1p=(dm==0||dm==1||dm==2), is3p=(dm==10||dm==11);',
            '  if(!is1p&&!is3p) return 0.0;',
            '  const bool nj6=(nsmalljets>=6);',
            '  const int ip=_fr2d_ptb(jetPt), ib=_fr2d_bb(b);',
            '  double fr = is1p ? (nj6?fr_nj6_1p[ip][ib]:fr_nj45_1p[ip][ib])',
            '                   : (nj6?fr_nj6_3p[ip][ib]:fr_nj45_3p[ip][ib]);',
            '  if(fr<=0.0||fr>=1.0) return 0.0;',
            '  return fr/(1.0-fr);',
            '}']
    f.Close()
    ok = ROOT.gInterpreter.Declare('\n'.join(lines))
    if not ok:
        raise RuntimeError('tauFR_weight_2d Declare failed')
    print('  tauFR_weight_2d{,_up,_down} declared from %s (%dx%d per prong/njet)' % (FR2D_ROOT, npt, nb))



# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', default=None,
                        help='Plot only this single variable.')
    parser.add_argument('--list-vars', action='store_true',
                        help='Print all variable names and exit.')
    args = parser.parse_args()

    plot_vars = build_plot_vars()
    if BINS_JSON:
        import json as _json
        _bj = _json.load(open(BINS_JSON))
        plot_vars = [(v, ti, (array.array('d', _bj[v]) if v in _bj else e))
                     for v, ti, e in plot_vars]
        print('BINS_JSON: overrode edges for %d/%d variables'
              % (sum(1 for v, _, _ in plot_vars if v in _bj), len(plot_vars)))

    if args.list_vars:
        for v, _t, _e in plot_vars:
            print(v)
        return

    if args.variable is not None:
        plot_vars = [(v, t, e) for v, t, e in plot_vars if v == args.variable]
        if not plot_vars:
            print('ERROR: variable %s not found in spec' % args.variable, file=sys.stderr)
            sys.exit(2)

    os.makedirs(OUTDIR, exist_ok=True)
    print('OUTDIR:', OUTDIR)
    print('NUM VARS TO PLOT:', len(plot_vars))

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    sys.path.insert(0, THIS_DIR)
    MC_GROUPS = get_mc_groups()

    declare_fr2d()
    if USE_MUFFIN:
        declare_muffin()
        print('  USE_MUFFIN=1 -> per-tau MUFFIN fake factor in the '
              'inclusion-exclusion template, binned map overlaid; OUTDIR=%s' % OUTDIR)

    # ---- Data + aggregate MC (for the fake template subtraction) + overlays ----
    df_data, data_files = make_df('data', exclude_patterns=('SingleMuon', 'SingleElectron', 'FakeTau_'))  # BTagCSV only
    df_mc,   mc_files   = make_df('mc',   exclude_patterns=('FakeTau_',))

    print('=== Data files: %d ===' % len(data_files))
    print('=== MC files:   %d ===' % len(mc_files))

    df_data = common_defines(df_data)

    # Split aggregate MC into inclusive-TT (genTtbarId kill) and the rest (no kill)
    INCL_TT_PATTERNS = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic')
    incl_tt_files = [f for f in mc_files
                     if any(p in os.path.basename(f) for p in INCL_TT_PATTERNS)
                     and 'TTbb_4f_' not in os.path.basename(f)]
    rest_files = [f for f in mc_files if f not in set(incl_tt_files)]

    def _weighted_df(file_list, with_kill):
        if not file_list:
            return None
        v = ROOT.std.vector('string')()
        for f in file_list:
            v.push_back(f)
        return add_mc_weight(common_defines(ROOT.RDataFrame('Events', v)),
                             with_ttbb_kill=with_kill)

    df_mc_list = []
    _d_incl = _weighted_df(incl_tt_files, True)
    if _d_incl is not None:
        df_mc_list.append(_d_incl)
    _d_rest = _weighted_df(rest_files, False)
    if _d_rest is not None:
        df_mc_list.append(_d_rest)
    print('=== aggregate MC: %d inclusive-TT files (kill) + %d other files ==='
          % (len(incl_tt_files), len(rest_files)))

    # ---- Per-signal overlays (HHH and HH samples) ----
    overlay_specs = [
        ('HHH#rightarrow4b2#tau #times1000',  'HHHTo4B2Tau',            ROOT.kRed,        2),
    ]  # plot standard: only the signal of interest
    overlay_dfs = []
    for label, pat, color, lstyle in overlay_specs:
        sdf, sfiles = make_df('signal', include_patterns=(pat,))
        if sdf is None:
            print('  (no signal files for %s)' % pat)
            continue
        print('=== Overlay %-22s : %d files ===' % (pat, len(sfiles)))
        sdf = add_mc_weight(common_defines(sdf), with_ttbb_kill=False, scale=1000.0)
        overlay_dfs.append((label, pat, color, lstyle, sdf))

    # ---- Per-group prompt MC dataframes ----
    grp_df = {}
    for gname, patterns, _col in MC_GROUPS:
        gfiles = [f for f in mc_files
                  if any(p in os.path.basename(f) for p in patterns)]
        if not gfiles:
            print('  (no files for group %s)' % gname)
            continue
        gvec = ROOT.std.vector('string')()
        for f in gfiles:
            gvec.push_back(f)
        gdf = ROOT.RDataFrame('Events', gvec)
        gdf = add_mc_weight(common_defines(gdf), with_ttbb_kill=(gname == 'ttbar'))
        grp_df[gname] = gdf
    col_of = {g: c for g, _p, c in MC_GROUPS}

    reg_label = ('CR: 2#tau_{h}0l single-tight sideband' if SIDEBAND
                 else 'CR: 2#tau_{h}0l (inclusive)')

    # =====================================================================
    # Selections
    #   pool region: kind_category_FR == 0
    #   SR/observed: kind_category_analysis == 0
    # Three fake application regions in the pool:
    #   A: tight1 & anti2  -> weight FF2
    #   B: anti1 & tight2  -> weight FF1
    #   C: anti1 & anti2   -> weight -FF1*FF2
    # =====================================================================
    POOL = 'kind_category_FR == 0'
    # Loose WP: observed SR is the FR pool (kind_category_FR==0) with BOTH taus passing
    # the Loose tight (VSjet>=8). kind_category_analysis is Medium-WP locked (VSjet>=16)
    # and would only ever return the Medium subset (33 ev), so it must NOT be used here;
    # the Loose SR is defined consistently within the pool (143 ev).
    SR_OBS = 'kind_category_FR == 0 && tight1 && tight2'
    SEL_A = '%s && tight1 && anti2' % POOL
    SEL_B = '%s && anti1 && tight2' % POOL
    SEL_C = '%s && anti1 && anti2' % POOL

    if SIDEBAND:
        # Validate the fake factor where the statistics are: predict the
        # single-tight sideband (526 data events) from (anti,anti), instead of
        # (tight,tight) (111).  Promoting either tau of an (anti,anti) event
        # gives an exactly-one-tight event, so
        #     pred N(1T) = sum over (anti,anti) of [F(tau1) + F(tau2)]
        # which is exact for independent per-object pass probabilities.  That
        # maps onto the existing A + B - C machinery with both A and B pointing
        # at (anti,anti) -- A carries FF2 and subtracts MC where tau2 is real,
        # B carries FF1 and subtracts MC where tau1 is real -- and no C term.
        SR_OBS = '%s && tight1 && anti2' % POOL
        SEL_A = '%s && anti1 && anti2' % POOL
        SEL_B = '%s && anti1 && anti2' % POOL
        SEL_C = '%s && tau1Pt < 0' % POOL          # never true: drops the term

    # Data application-region dataframes
    df_data_A = df_data.Filter(SEL_A)
    df_data_B = df_data.Filter(SEL_B)
    df_data_C = df_data.Filter(SEL_C)
    df_data_obs = df_data.Filter(SR_OBS)

    # Prompt-MC subtraction dataframes (real tau in the anti-ID slot)
    #   A: subtract where tau2 is real (tau2 is the anti-ID slot)
    #   B: subtract where tau1 is real
    #   C: subtract where tau1 OR tau2 is real
    df_mc_A = [d.Filter('%s && tau2genPartFlav == 5' % SEL_A) for d in df_mc_list]
    df_mc_B = [d.Filter('%s && tau1genPartFlav == 5' % SEL_B) for d in df_mc_list]
    df_mc_C = [d.Filter('%s && (tau1genPartFlav == 5 || tau2genPartFlav == 5)' % SEL_C)
               for d in df_mc_list]

    # Per-group prompt stack for the observed region.  In the sideband the
    # observed region has one tight tau, so the prompt stack is the events whose
    # TIGHT tau is real -- the anti-ID one is what the fake factor promotes.
    _gen_obs = ('tau1genPartFlav == 5' if SIDEBAND
                else 'tau1genPartFlav == 5 && tau2genPartFlav == 5')
    df_grp_t = {g: grp_df[g].Filter('%s && %s' % (SR_OBS, _gen_obs))
                for g in grp_df}

    # Signal overlays (tight1 && tight2 && kc_analysis==0)
    df_overlay_t = [(label, pat, color, lstyle, sdf.Filter(SR_OBS))
                    for label, pat, color, lstyle, sdf in overlay_dfs]

    # ---- Book histograms ----
    booked = {}
    for var_name, var_title, edges in plot_vars:
        nbins = len(edges) - 1
        hpfx = '2tau0l_%s' % var_name

        entry = dict(title=var_title, edges=edges, label=reg_label)

        # per-variable zero filter (see NO_ZERO_FILTER_VARS)
        if var_name in NO_ZERO_FILTER_VARS:
            fOBS, fA, fB, fC = df_data_obs, df_data_A, df_data_B, df_data_C
            fmcA, fmcB, fmcC = df_mc_A, df_mc_B, df_mc_C
            fgrp, fov = df_grp_t, df_overlay_t
        else:
            _z = '%s != 0' % var_name
            fOBS = df_data_obs.Filter(_z)
            fA, fB, fC = df_data_A.Filter(_z), df_data_B.Filter(_z), df_data_C.Filter(_z)
            fmcA = [d.Filter(_z) for d in df_mc_A]
            fmcB = [d.Filter(_z) for d in df_mc_B]
            fmcC = [d.Filter(_z) for d in df_mc_C]
            fgrp = {g: d.Filter(_z) for g, d in df_grp_t.items()}
            fov = [(l, pt, co, ls, d.Filter(_z)) for l, pt, co, ls, d in df_overlay_t]

        # Observed data
        entry['data_obs'] = fOBS.Histo1D((hpfx + '_data_obs', '', nbins, edges), var_name)

        # Data application regions (weighted by the appropriate FF)
        entry['data_A']      = fA.Histo1D((hpfx + '_data_A', '', nbins, edges), var_name, 'FF2')
        entry['data_A_up']   = fA.Histo1D((hpfx + '_data_A_up', '', nbins, edges), var_name, 'FF2_up')
        entry['data_A_down'] = fA.Histo1D((hpfx + '_data_A_down', '', nbins, edges), var_name, 'FF2_down')
        entry['data_B']      = fB.Histo1D((hpfx + '_data_B', '', nbins, edges), var_name, 'FF1')
        entry['data_B_up']   = fB.Histo1D((hpfx + '_data_B_up', '', nbins, edges), var_name, 'FF1_up')
        entry['data_B_down'] = fB.Histo1D((hpfx + '_data_B_down', '', nbins, edges), var_name, 'FF1_down')
        entry['data_C']      = fC.Histo1D((hpfx + '_data_C', '', nbins, edges), var_name, 'FF12')
        entry['data_C_up']   = fC.Histo1D((hpfx + '_data_C_up', '', nbins, edges), var_name, 'FF12_up')
        entry['data_C_down'] = fC.Histo1D((hpfx + '_data_C_down', '', nbins, edges), var_name, 'FF12_down')

        # Prompt-MC subtraction terms (lists over the 2 aggregate-MC dataframes)
        if USE_MUFFIN:
            # exactly the same A/B/C terms, with FF_i -> the MUFFIN weight
            for suf, w2, w1, w12 in (('_muf', 'FF2_muf', 'FF1_muf', 'FF12_muf'),
                                     ('_muf_up', 'FF2_muf_up', 'FF1_muf_up', 'FF12_muf_up'),
                                     ('_muf_down', 'FF2_muf_down', 'FF1_muf_down', 'FF12_muf_down')):
                entry['data_A' + suf] = fA.Histo1D((hpfx + '_data_A' + suf, '', nbins, edges), var_name, w2)
                entry['data_B' + suf] = fB.Histo1D((hpfx + '_data_B' + suf, '', nbins, edges), var_name, w1)
                entry['data_C' + suf] = fC.Histo1D((hpfx + '_data_C' + suf, '', nbins, edges), var_name, w12)
                entry['mc_A' + suf] = [d.Histo1D((hpfx + ('_mc_A%s_%d' % (suf, k)), '', nbins, edges), var_name, 'mc_' + w2) for k, d in enumerate(fmcA)]
                entry['mc_B' + suf] = [d.Histo1D((hpfx + ('_mc_B%s_%d' % (suf, k)), '', nbins, edges), var_name, 'mc_' + w1) for k, d in enumerate(fmcB)]
                entry['mc_C' + suf] = [d.Histo1D((hpfx + ('_mc_C%s_%d' % (suf, k)), '', nbins, edges), var_name, 'mc_' + w12) for k, d in enumerate(fmcC)]
        entry['mc_A']      = [d.Histo1D((hpfx + ('_mc_A_%d' % k), '', nbins, edges), var_name, 'mc_FF2')      for k, d in enumerate(fmcA)]
        entry['mc_A_up']   = [d.Histo1D((hpfx + ('_mc_A_up_%d' % k), '', nbins, edges), var_name, 'mc_FF2_up')   for k, d in enumerate(fmcA)]
        entry['mc_A_down'] = [d.Histo1D((hpfx + ('_mc_A_down_%d' % k), '', nbins, edges), var_name, 'mc_FF2_down') for k, d in enumerate(fmcA)]
        entry['mc_B']      = [d.Histo1D((hpfx + ('_mc_B_%d' % k), '', nbins, edges), var_name, 'mc_FF1')      for k, d in enumerate(fmcB)]
        entry['mc_B_up']   = [d.Histo1D((hpfx + ('_mc_B_up_%d' % k), '', nbins, edges), var_name, 'mc_FF1_up')   for k, d in enumerate(fmcB)]
        entry['mc_B_down'] = [d.Histo1D((hpfx + ('_mc_B_down_%d' % k), '', nbins, edges), var_name, 'mc_FF1_down') for k, d in enumerate(fmcB)]
        entry['mc_C']      = [d.Histo1D((hpfx + ('_mc_C_%d' % k), '', nbins, edges), var_name, 'mc_FF12')      for k, d in enumerate(fmcC)]
        entry['mc_C_up']   = [d.Histo1D((hpfx + ('_mc_C_up_%d' % k), '', nbins, edges), var_name, 'mc_FF12_up')   for k, d in enumerate(fmcC)]
        entry['mc_C_down'] = [d.Histo1D((hpfx + ('_mc_C_down_%d' % k), '', nbins, edges), var_name, 'mc_FF12_down') for k, d in enumerate(fmcC)]

        # Per-group prompt-tight stack
        entry['grp_t'] = {}
        for g in fgrp:
            entry['grp_t'][g] = fgrp[g].Histo1D((hpfx + '_' + g, '', nbins, edges), var_name, 'mc_weight')

        # Signal overlays
        entry['overlays'] = []
        for label, pat, color, lstyle, fdf in fov:
            h = fdf.Histo1D((hpfx + '_ov_' + pat, '', nbins, edges), var_name, 'mc_weight')
            entry['overlays'].append((label, pat, color, lstyle, h))

        booked[var_name] = entry

    # ---- Materialize and draw ----
    print('\n=== Materializing histograms ===')
    root_out_name = 'closure_v26_2tau0l'
    if args.variable is not None:
        root_out_name += '_%s' % args.variable
    fout = ROOT.TFile(os.path.join(OUTDIR, root_out_name + '.root'), 'RECREATE')

    def _accumulate(hlist, name):
        """Sum a list of RResultPtr<TH1> into one owned TH1 (explicit, no Add-chaining)."""
        tot = None
        for hp in hlist:
            h = hp.GetPtr()
            if tot is None:
                tot = h.Clone(name); tot.SetDirectory(0)
            else:
                tot.Add(h)
        return tot

    for var_name, var_title, edges in plot_vars:
        bk = booked[var_name]
        nbins = len(edges) - 1

        # Observed data
        h_data = bk['data_obs'].GetPtr().Clone('2tau0l_%s_data' % var_name)
        h_data.SetDirectory(0)

        # Per-group prompt-tau tight stack
        h_grp = {}
        h_mc_p_total = None
        for g, _p, _c in MC_GROUPS:
            if g not in bk['grp_t']:
                continue
            h = bk['grp_t'][g].GetPtr().Clone('2tau0l_%s_%s' % (var_name, g))
            h.SetDirectory(0)
            h.SetFillColor(col_of[g]); h.SetLineColor(ROOT.kBlack); h.SetLineWidth(1)
            h_grp[g] = h
            if h_mc_p_total is None:
                h_mc_p_total = h.Clone('2tau0l_%s_mc_prompt_total' % var_name)
                h_mc_p_total.SetDirectory(0)
            else:
                h_mc_p_total.Add(h)

        # ---- Build the inclusion-exclusion fake template for a given variation ----
        def build_fake(suffix):
            # data terms
            dA = bk['data_A' + suffix].GetPtr()
            dB = bk['data_B' + suffix].GetPtr()
            dC = bk['data_C' + suffix].GetPtr()
            # mc subtraction terms (accumulate the per-df lists)
            mA = _accumulate(bk['mc_A' + suffix], '2tau0l_%s_mcA%s' % (var_name, suffix))
            mB = _accumulate(bk['mc_B' + suffix], '2tau0l_%s_mcB%s' % (var_name, suffix))
            mC = _accumulate(bk['mc_C' + suffix], '2tau0l_%s_mcC%s' % (var_name, suffix))
            h = dA.Clone('2tau0l_%s_fake%s' % (var_name, suffix)); h.SetDirectory(0)
            # FAKE = (dA - mA) + (dB - mB) - (dC - mC)
            if mA is not None: h.Add(mA, -1.0)
            h.Add(dB, +1.0)
            if mB is not None: h.Add(mB, -1.0)
            h.Add(dC, -1.0)
            if mC is not None: h.Add(mC, +1.0)
            # clamp per bin to >= 0
            for i in range(1, h.GetNbinsX() + 1):
                if h.GetBinContent(i) < 0.0:
                    h.SetBinContent(i, 0.0)
            return h

        h_fake    = build_fake('')
        h_fake_up = build_fake('_up')
        h_fake_dn = build_fake('_down')
        # keep the binned-map template as the comparison, then swap MUFFIN's in
        # so the stack, the total and the band below are built from it
        h_fake_binned = None
        if USE_MUFFIN:
            h_fake_binned = h_fake.Clone('2tau0l_%s_fake_binned' % var_name)
            h_fake_binned.SetDirectory(0)
            h_fake_binned_up = h_fake_up.Clone('2tau0l_%s_fake_binned_up' % var_name)
            h_fake_binned_up.SetDirectory(0)
            h_fake_binned_dn = h_fake_dn.Clone('2tau0l_%s_fake_binned_dn' % var_name)
            h_fake_binned_dn.SetDirectory(0)
            h_fake    = build_fake('_muf')
            h_fake_up = build_fake('_muf_up')
            h_fake_dn = build_fake('_muf_down')
        h_fake.SetFillColor(ROOT.kGray); h_fake.SetLineColor(ROOT.kBlack); h_fake.SetLineWidth(1)

        # Yield summary
        tot_p = h_mc_p_total.Integral() if h_mc_p_total is not None else 0.0
        tot_fake = h_fake.Integral()
        tot_pred = tot_p + tot_fake
        print('\n--- %s ---' % var_name)
        print('  Data obs (tight,tight): %.0f' % h_data.Integral())
        print('  MC prompt total:        %.2f' % tot_p)
        print('  FakeTau (incl-excl):    %.2f' % tot_fake)
        print('  Pred (MC+Fake):         %.2f' % tot_pred)
        if tot_pred > 0:
            print('  Data/Pred:              %.3f' % (h_data.Integral() / tot_pred))
        if USE_MUFFIN and h_fake_binned is not None:
            tb = tot_p + h_fake_binned.Integral()
            print('  FakeTau (binned map):   %.2f' % h_fake_binned.Integral())
            if tb > 0:
                print('  Data/Pred (binned):     %.3f' % (h_data.Integral() / tb))

        # ----------------- Draw -----------------
        c = ROOT.TCanvas('c_2tau0l_%s' % var_name, '', 800, 800)
        pad1 = ROOT.TPad('pad1', '', 0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.05); pad1.SetTopMargin(0.08); pad1.SetLogy()
        pad1.Draw(); pad1.cd()

        hs = ROOT.THStack('hs_2tau0l_%s' % var_name, '')
        for g, _p, _c in MC_GROUPS:
            if g in h_grp:
                hs.Add(h_grp[g])
        hs.Add(h_fake)
        hs.Draw('HIST')
        hs.GetYaxis().SetTitle('Events'); hs.GetYaxis().SetTitleSize(0.05)
        hs.GetYaxis().SetTitleOffset(1.0); hs.GetYaxis().SetLabelSize(0.04)
        hs.GetXaxis().SetLabelSize(0)

        # Total prediction + Stat (+) FR band
        if h_mc_p_total is None:
            h_tot = h_fake.Clone('2tau0l_%s_tot' % var_name)
            h_tot_up = h_fake_up.Clone('2tau0l_%s_tot_up' % var_name)
            h_tot_dn = h_fake_dn.Clone('2tau0l_%s_tot_dn' % var_name)
        else:
            h_tot = h_mc_p_total.Clone('2tau0l_%s_tot' % var_name); h_tot.Add(h_fake)
            h_tot_up = h_mc_p_total.Clone('2tau0l_%s_tot_up' % var_name); h_tot_up.Add(h_fake_up)
            h_tot_dn = h_mc_p_total.Clone('2tau0l_%s_tot_dn' % var_name); h_tot_dn.Add(h_fake_dn)
        h_tot.SetDirectory(0)
        for i in range(1, h_tot.GetNbinsX() + 1):
            stat = h_tot.GetBinError(i)
            fr_err = max(abs(h_tot_up.GetBinContent(i) - h_tot.GetBinContent(i)),
                         abs(h_tot_dn.GetBinContent(i) - h_tot.GetBinContent(i)))
            h_tot.SetBinError(i, (stat ** 2 + fr_err ** 2) ** 0.5)
        h_tot.SetFillStyle(3354); h_tot.SetFillColor(ROOT.kBlack)
        h_tot.SetMarkerSize(0); h_tot.SetLineWidth(0)
        h_tot.Draw('E2 SAME')

        h_data.SetLineColor(ROOT.kBlack); h_data.SetMarkerColor(ROOT.kBlack)
        h_data.SetMarkerStyle(20); h_data.SetMarkerSize(1.0); h_data.SetLineWidth(1)
        h_data.Draw('E1 SAME')

        # Signal overlays
        overlay_hists = []
        for label, pat, color, lstyle, hptr in bk['overlays']:
            h = hptr.GetPtr().Clone('2tau0l_%s_ov_%s' % (var_name, pat))
            h.SetDirectory(0)
            h.SetLineColor(color); h.SetLineWidth(2); h.SetLineStyle(lstyle)
            h.SetMarkerSize(0); h.SetFillStyle(0)
            h.Draw('HIST SAME')
            overlay_hists.append((label, h))

        leg = ROOT.TLegend(0.44, 0.66, 0.93, 0.90)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.023); leg.SetTextFont(42)
        leg.SetNColumns(2); leg.SetColumnSeparation(0.04)
        leg.AddEntry(h_data, 'Data', 'lep')
        leg.AddEntry(h_fake, 'Fake #tau (MUFFIN)' if USE_MUFFIN
                     else 'Fake #tau (FR method)', 'f')
        for g, _p, _c in reversed(MC_GROUPS):
            if g in h_grp:
                leg.AddEntry(h_grp[g], g, 'f')
        leg.AddEntry(h_tot, 'Stat. #oplus FR unc.', 'f')
        for label, h in overlay_hists:
            leg.AddEntry(h, label, 'l')
        leg.Draw()

        lt = ROOT.TLatex(); lt.SetNDC()
        lt.SetTextFont(61); lt.SetTextSize(0.06); lt.DrawLatex(0.12, 0.93, 'CMS')
        lt.SetTextFont(52); lt.SetTextSize(0.045); lt.DrawLatex(0.22, 0.93, 'Internal')
        lt.SetTextFont(42); lt.SetTextSize(0.04)
        lt.DrawLatex(0.62, 0.93, '41.5 fb^{-1} (13 TeV)')
        lt.DrawLatex(0.15, 0.83, reg_label)

        # Ratio pad
        c.cd()
        pad2 = ROOT.TPad('pad2', '', 0, 0.0, 1, 0.3)
        pad2.SetTopMargin(0.02); pad2.SetBottomMargin(0.35)
        pad2.SetLeftMargin(0.12); pad2.SetRightMargin(0.05); pad2.Draw(); pad2.cd()

        h_ratio = h_data.Clone('2tau0l_%s_ratio' % var_name)
        h_ratio.Divide(h_tot)
        h_ratio.SetMarkerStyle(20); h_ratio.SetMarkerSize(0.8)
        h_ratio.GetXaxis().SetTitle(var_title); h_ratio.GetXaxis().SetTitleSize(0.12)
        h_ratio.GetXaxis().SetTitleOffset(1.0); h_ratio.GetXaxis().SetLabelSize(0.10)
        h_ratio.GetYaxis().SetTitle('Data / Pred.'); h_ratio.GetYaxis().SetTitleSize(0.12)
        h_ratio.GetYaxis().SetTitleOffset(0.45); h_ratio.GetYaxis().SetLabelSize(0.10)
        h_ratio.GetYaxis().SetRangeUser(0, 2.5); h_ratio.GetYaxis().SetNdivisions(505)
        h_ratio.SetTitle(''); h_ratio.Draw('E1')

        h_r_unc = h_tot.Clone('2tau0l_%s_ratio_unc' % var_name)
        for i in range(1, h_r_unc.GetNbinsX() + 1):
            v = h_r_unc.GetBinContent(i); e = h_r_unc.GetBinError(i)
            h_r_unc.SetBinContent(i, 1.0)
            h_r_unc.SetBinError(i, e / v if v > 0 else 0)
        h_r_unc.SetFillStyle(3354); h_r_unc.SetFillColor(ROOT.kGray + 2)
        h_r_unc.SetMarkerSize(0); h_r_unc.SetLineWidth(0)
        h_r_unc.Draw('E2 SAME')

        ln = ROOT.TLine(edges[0], 1, edges[-1], 1)
        ln.SetLineStyle(2); ln.SetLineColor(ROOT.kGray + 2); ln.Draw()

        # Same events, same MC-prompt stack, same inclusion-exclusion algebra --
        # only FF_i differs, so the gap between the curves is the difference
        # between the two fake factors carried over from 1tau0l.
        h_ratio_bin = None
        legr = None
        if USE_MUFFIN and h_fake_binned is not None:
            if h_mc_p_total is None:
                h_tot_bin = h_fake_binned.Clone('2tau0l_%s_tot_binned' % var_name)
            else:
                h_tot_bin = h_mc_p_total.Clone('2tau0l_%s_tot_binned' % var_name)
                h_tot_bin.Add(h_fake_binned)
            h_tot_bin.SetDirectory(0)
            h_ratio_bin = h_data.Clone('2tau0l_%s_ratio_binned' % var_name)
            h_ratio_bin.SetDirectory(0); h_ratio_bin.SetStats(0)
            for ib in range(1, h_ratio_bin.GetNbinsX() + 1):
                pb = h_tot_bin.GetBinContent(ib)
                h_ratio_bin.SetBinContent(ib, h_data.GetBinContent(ib) / pb if pb > 0 else 0)
                h_ratio_bin.SetBinError(ib, 0.0)
            h_ratio_bin.SetLineColor(ROOT.kRed + 1); h_ratio_bin.SetLineWidth(2)
            h_ratio_bin.SetMarkerSize(0); h_ratio_bin.SetFillStyle(0)
            h_ratio_bin.Draw('HIST SAME')
            legr = ROOT.TLegend(0.14, 0.78, 0.62, 0.97)
            legr.SetBorderSize(0); legr.SetFillStyle(0)
            legr.SetTextSize(0.085); legr.SetTextFont(42); legr.SetNColumns(2)
            legr.AddEntry(h_ratio, 'MUFFIN', 'lep')
            legr.AddEntry(h_ratio_bin, 'binned F_{F}', 'l')
            legr.Draw()

        h_ratio.Draw('E1 SAME')

        # Save log + linear
        _ymax = max(h_data.GetMaximum(), hs.GetMaximum())
        if HISTOS_ONLY:
            _ymax = -1  # skip canvas output; histograms are still written below
        for scale in ([] if HISTOS_ONLY else ['log', 'lin']):
            pad1.cd()
            if scale == 'log':
                pad1.SetLogy(1); hs.SetMinimum(0.5); hs.SetMaximum(_ymax * 5 if _ymax > 0 else 10)
            else:
                pad1.SetLogy(0); hs.SetMinimum(0.0); hs.SetMaximum(_ymax * 1.5 if _ymax > 0 else 10)
            pad1.Modified(); pad1.Update()
            for ext in ['png', 'pdf']:
                c.SaveAs(os.path.join(OUTDIR, 'closure_2tau0l_%s_%s.%s' % (var_name, scale, ext)))

        fout.cd()
        h_data.Write(); h_fake.Write(); h_tot.Write(); h_ratio.Write()
        if USE_MUFFIN and h_ratio_bin is not None:
            h_fake_binned.Write(); h_ratio_bin.Write()
            h_fake_binned_up.Write(); h_fake_binned_dn.Write()
            h_fake_up.Write(); h_fake_dn.Write()
        if h_mc_p_total is not None:
            h_mc_p_total.Write()
        for g in h_grp:
            h_grp[g].Write()
        for _label, h in overlay_hists:
            h.Write()

    fout.Close()
    print('\nDone. Plots saved to %s' % OUTDIR)


if __name__ == '__main__':
    main()
