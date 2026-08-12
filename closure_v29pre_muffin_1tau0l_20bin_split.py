#!/usr/bin/env python3
"""
20-bin per-process closure plotting for the v29pre option-92 1tau0l selection,
with the MUFFIN fake factor.

Copy of closure_v29pre_in_v29pre_option92_1tau0l_NHiggs_20bin_split.py with one
addition: USE_MUFFIN=1 builds the stacked fake-tau template from the MUFFIN
per-event weight (muffin/README.md) instead of the binned map, and the ratio pad
then carries BOTH Data/Pred curves -- MUFFIN as the points, the binned map as a
red line -- so the two fake factors can be compared variable by variable in the
format the analysis already uses.  Everything else (regions, samples, weights,
variable list, binning, styling) is untouched.

  USE_MUFFIN=1 USE_FR2D=1 TAU_TIGHT_WP=loose \
  FR2D_ROOT=.../fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root FR2D_YVAR=abseta \
  EXTRA_CUT='jet4DeepFlavB < 0.1'   # VR;  EXTRA_CUT=1 for inclusive
(sed-copy of closure_v26_...: v29pre parts/, fr_flavour2d_v29pre map, trigSF v24c,
 SingleElectron excluded from data, v28 CLS5 score branches added, V27 default on.)
(sed/edit of the v25 plotter; v26 production used the v25_2fj SPANet models.)

v26 vs v25 changes (branch differences):
  - BASEDIR -> TurbOutputMC2017_v26_ak8_option92_2017
  - MERGED  -> parts_SPANET_v25_2fj_merged
  - btagShapeR_weight is a REAL branch -> used directly in mc_weight
    (v25 computed it inline via btagShapeR(nsmalljets))
  - is_inclusive_TT branch is ABSENT -> TTbb-overlap kill applied via
    genTtbarId%100 < 51 on the inclusive TTTo* samples only (sample-name based)
  - FR map reused from v25 (plots_tauFR_2017_v25_2nBtag_fixed_ttbb_HH_prob4b2tauLT0p8)

Differences vs closure_v24FRprob0p8_in_v24_option92_1tau0l_NHiggs.py:
  - Continuous variables use 20 UNIFORM bins (categorical/integer vars keep their original edges)
  - "MC prompt tau" stack is broken down into per-process groups
        QCD, VV, Vjets, ttX, ttbb, ttbar
    (with `tau1genPartFlav == 5` applied per group)
  - FR fake-tau template still uses the aggregate MC for the loose-not-tight subtraction
  - Optional `--variable VAR` flag plots only one variable (for condor parallelization)
  - Optional `--list-vars` flag prints all variable names and exits

Usage:
  cmssw-el7 -- bash -c 'cd .../CMSSW_12_5_2/src && eval $(scramv1 runtime -sh) && \
      python3 hhh-analysis-framework/closure_v24FRprob0p8_in_v24_option92_1tau0l_NHiggs_20bin_split.py \
          [--variable ht] [--list-vars]'
"""
from __future__ import print_function
import os
import sys
import glob
import array
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
BASEDIR = os.environ.get('CLOSURE_BASEDIR', '/eos/user/r/rtu/TurbOutputMC2017_v29pre_ak8_option92_2017')
MERGED = os.environ.get('CLOSURE_MERGED', 'parts')
# V27=1: parts have the CLS5 branches only (no old 11-class / Higgs-cat branches),
# so common_defines must NOT build IndexMaxProb/IndexMaxCat from them. Also skips
# the NHiggs regions (no SPANet Higgs-number head applied to v27).
V27 = os.environ.get('V27', '1') == '1'  # v29pre parts: CLS5 postfixed scores only
LUMI = 41500.0  # pb^-1 for 2017
OUTDIR = os.environ.get('CLOSURE_OUTDIR',
    '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/plots_v29pre_in_v29pre_option92_1tau0l_inclusive_full_withOverlay_20bin_split')
CLOSURE_REGION = os.environ.get('CLOSURE_REGION', '')  # '0H','1H','2H','inclusive' (default inclusive)
EXTRA_CUT = os.environ.get('EXTRA_CUT', '1')           # extra event cut, e.g. 'ProbHHH4b2tau > 0.3'
# Base region selection (default = 1tau0l Fakeable). Override for cross-region FR validation,
# e.g. the tt-enriched 1tau1l-hadronic control:
#   'is_1tau1l_hadronic==1 && lep1passAnalysisWP==1 && (lep1Id*tau1Charge)>0'
REGION_BASE = os.environ.get('REGION_BASE', 'kind_category_FR == 2')
# MCFAKES=1: source-split fake model — the data-driven template covers ONLY QCD
# fakes (AR subtraction removes ALL non-QCD MC, genuine AND fake tau), while
# tt/W fake taus enter the stack from MC (they carry real nu -> fixes the
# high-MET tail without any MET-binned FR). Conversion-pattern a la lepton FR.
MCFAKES = os.environ.get('MCFAKES', '') == '1'
if MCFAKES and 'CLOSURE_OUTDIR' not in os.environ:
    OUTDIR = OUTDIR + '_mcfakes'
# PURE_MC: data vs full MC stack with fake taus from MC truth (no data-driven FR
# template); QCD MC included as a process group. Cross-check, NOT the analysis method.
PURE_MC = os.environ.get('PURE_MC', '') == '1'
# NO_TRIGSF=1 drops the trigger SF (triggerLumiSF * triggerSF_perfilter_2nBtag_v24c)
# from the MC prompt weight -> cross-check of the trigger-SF impact on the closure.
NO_TRIGSF = os.environ.get('NO_TRIGSF', '') == '1'

# Settled "new" FR: direct flavour-aware 2-D FR(jetPt x tau1jetDeepFlavB), per prong/njet,
# eta-inclusive. USE_FR2D=1 replaces the 1-D FR entirely. Uses the FIXED WP-specific maps
# from measure_fr2d_wp_v26.py (INCLUSIVE-loose denom -> stores true fake rate f; declare_fr2d
# applies f/(1-f)). TAU_TIGHT_WP selects the FakeTau target WP: medium (VSjet>=16, default)
# or loose (VSjet>=8); the loose-not-tight anti-ID window is [2, _WP_TIGHT).
USE_FR2D = os.environ.get('USE_FR2D', '') == '1'
TAU_TIGHT_WP = os.environ.get('TAU_TIGHT_WP', 'medium').lower()
_WP_TIGHT = {'medium': 16, 'loose': 8, 'tight': 32}[TAU_TIGHT_WP]
# FR_AST=1: "as-if-tight" jet bookkeeping for the anti-ID leg. The tight region's
# jet collection has the tau's mother jet CLEANED (analysisTaus veto) while the
# anti-ID region's never does, so template/denominator jet counts sit one (mother)
# jet high -- the v27 nsmalljets closure slope (D/P ~1.5 at njet=4 -> ~0.7 high;
# template offset +0.41 jets vs +0.14 in v26). With FR_AST=1 anti-ID events get
# nsmalljets/ht/nbtags/nfatjets REDEFINED to mother-jet-subtracted values and the
# analysis cuts re-applied, so the FR njet-class lookup, the region cuts and every
# plot fill become consistent with the tight region automatically. MUST be used
# for BOTH the FR measurement and the application: a one-sided correction INVERTS
# the slope (validated on BTagCSV, claude_tmp/fix_njet_closure_v2.py). Default OFF.
FR_AST = os.environ.get('FR_AST', '') == '1'
if FR_AST and 'CLOSURE_OUTDIR' not in os.environ:
    print('FR_AST=1: anti-ID leg uses as-if-tight jet bookkeeping')
FR2D_ROOT = os.environ.get('FR2D_ROOT',
    '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/fr_flavour2d_v29pre.root')
if USE_FR2D and 'CLOSURE_OUTDIR' not in os.environ:
    OUTDIR = OUTDIR + '_fr2d' + ('' if TAU_TIGHT_WP == 'medium' else '_%sWP' % TAU_TIGHT_WP)

# MUFFIN (MUltivariate Fake-Factor INference): continuous multi-dimensional fake
# factor, see muffin/README.md.  USE_MUFFIN=1 requires USE_FR2D=1 -- the binned
# map is still evaluated, to be drawn alongside in the ratio pad.
USE_MUFFIN = os.environ.get('USE_MUFFIN', '') == '1'
MUFFIN_HEADER = os.environ.get(
    'MUFFIN_HEADER', os.path.join(THIS_DIR, 'muffin', 'out', 'muffin_poster.h'))
if USE_MUFFIN and 'CLOSURE_OUTDIR' not in os.environ:
    OUTDIR = OUTDIR + '_muffin'

# Standalone CDF/quantile mapping of ProbHHH4b2tau -> decorrelate from the tau fake-quality
# axis tau1rawDeepTau2017v2p1VSjet (training-free alternative to DisCo). USE_CDFMAP=1 adds the
# branch ProbHHH4b2tau_cdfmap = F(score | rawVSjet), F = conditional CDF of the data fake
# template (data - prompt MC), built inclusive over kc_FR==2. See project_disco_fr_resolution.
USE_CDFMAP = os.environ.get('USE_CDFMAP', '') == '1'
CDFMAP_ROOT = os.environ.get('CDFMAP_ROOT',
    '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/cdfmap_ddt_loose_v26.root')
# Partial decorrelation strength: ProbHHH4b2tau_cdfmap = (1-a)*s + a*M^-1(F(s|x)).
# a=1 full decorrelation (max closure, max signal loss); a~0.5 = sweet spot (closure saturated,
# ~75% of the signal AUC recovered). See alpha_scan_results.txt.
CDFMAP_ALPHA = float(os.environ.get('CDFMAP_ALPHA', '1.0'))
CDFMAP_SCORE = os.environ.get('CDFMAP_SCORE', 'ProbHHH4b2tau_v28common_ep281')
if USE_CDFMAP and 'CLOSURE_OUTDIR' not in os.environ:
    OUTDIR = OUTDIR + '_cdfmap_a%03d' % int(round(CDFMAP_ALPHA * 100))

# Categorical variable names: skip the 20-bin uniform override and keep their original edges.
CATEGORICAL_VARS = set([
    'nsmalljets', 'nfatjets', 'kind_category_FR',
    'lep1Id', 'lep2Id', 'lep3Id',
    'tau1Charge', 'tau2Charge', 'tau3Charge',
    'tau1decayMode', 'tau2decayMode', 'tau3decayMode',
    'IndexMaxProb', 'IndexMaxCat',
])

NBINS_UNIFORM = 20

# For variables in this set we do NOT apply the `var != 0` filter, either because
# 0 is a legitimate physics value (decayMode==0 means 1-prong tau; metphi, eta, ...)
# or because the value is always defined (event-level / SPANet outputs).
# For all other variables we drop events where the variable equals exactly 0,
# since for per-object quantities (jetN/lepN/tauN pT, mass, btag, ...) a value
# of 0 means that the Nth physics object is not present in the event.
NO_ZERO_FILTER_VARS = set([
    # Event-level
    'ht', 'met', 'metphi',
    'nsmalljets', 'nfatjets', 'kind_category_FR',
    # Tau decay mode: 0 = 1-prong (very common!), do NOT filter
    'tau1decayMode', 'tau2decayMode', 'tau3decayMode',
    # SPANet outputs (always defined for the analyzed events)
    'h1_spanet_mass', 'h2_spanet_mass', 'hh_mass',
    'ProbHHH4b2tau', 'ProbHHH4b2tau_cdfmap', 'ProbHHH6b', 'ProbQCD', 'ProbTTHard', 'ProbTTSemi', 'ProbTTlep',
    'ProbWJets', 'ProbZJets', 'ProbVV', 'ProbHH4b', 'ProbHH2b2tau',
    'Prob0rh0bh', 'Prob1rh0bh', 'Prob2rh0bh', 'Prob0rh1bh', 'Prob1rh1bh', 'Prob0rh2bh',
    'IndexMaxProb', 'IndexMaxCat',
] + ['%s%s' % (c, p) for p in ('_v28common_ep281', '_v28qcdfrac_ep289', '_v29wmirror_ep201')
     for c in ('ProbHHH4b2tau', 'ProbTT', 'ProbQCD', 'ProbOtherH', 'ProbVjetsVVttX')]
  + [os.environ.get('CDFMAP_SCORE', 'ProbHHH4b2tau_v28common_ep281') + '_cdfmap'])


def get_mc_groups():
    """MC process groups (stacked, bottom -> top). ROOT colors resolved lazily.
    Small processes at the bottom: VV -> Vjets -> ttX -> ttbb -> ttbar (top).
    The FR fake-tau template is drawn on top of the whole MC stack.

    NOTE: QCD is intentionally removed because the FR fake-tau template already
    captures the QCD-with-fake-tau contribution in this data-driven method;
    including QCD MC here would double-count."""
    import ROOT
    return [
        ('VV',    ['WWTo', 'WZTo', 'ZZTo'],            ROOT.kOrange + 1),
        ('Vjets', ['DYJetsTo', 'WJetsTo', 'ZJetsTo'],  ROOT.kGreen + 2),
        ('ttX',   ['TTWJets', 'TTZTo', 'ttHJet'],      ROOT.kMagenta - 4),
        ('ttbb',  ['TTbb_4f_TTTo'],                    ROOT.kAzure + 7),
        ('ttbar', ['TTTo2L2Nu', 'TTToHadronic',
                   'TTToSemiLeptonic'],                ROOT.kAzure - 9),
    ] + ([('QCD', ['QCD'], ROOT.kYellow - 7)] if PURE_MC else [])


def _declare_cpp_helpers():
    import ROOT
    ROOT.gInterpreter.Declare('''
int get_max_prob(float ProbHHH6b, float ProbQCD, float ProbTTHard, float ProbWJets,
                 float ProbZJets, float ProbTTSemi, float ProbHHH4b2tau, float ProbTTlep,
                 float ProbVV, float ProbHH4b, float ProbHH2b2tau){
    std::vector<float> probs = {ProbHHH6b, ProbQCD, ProbTTHard, ProbWJets, ProbZJets,
                                 ProbTTSemi, ProbHHH4b2tau, ProbTTlep, ProbVV, ProbHH4b, ProbHH2b2tau};
    auto it = std::max_element(probs.begin(), probs.end());
    return std::distance(probs.begin(), it) + 1;
}

int get_max_cat(float Prob0rh0bh, float Prob1rh0bh, float Prob2rh0bh,
                float Prob0rh1bh, float Prob1rh1bh, float Prob0rh2bh){
    std::vector<float> probs = {Prob0rh0bh, Prob1rh0bh, Prob2rh0bh,
                                 Prob0rh1bh, Prob1rh1bh, Prob0rh2bh};
    auto it = std::max_element(probs.begin(), probs.end());
    return std::distance(probs.begin(), it) + 1;
}

bool is_1prong(float dm) {
    int idm = (int)dm;
    return (idm == 0 || idm == 1 || idm == 2);
}
''')


# =========================================================================
# Build the variable specification.
# Continuous vars: 20 uniform bins between [lo, hi]; categorical vars: keep original edges.
# =========================================================================
def _declare_ast_helpers():
    """FR_AST: fatjets overlapping the (anti-ID) tau are removed by the
    analysisTaus DR<1.0 veto in the tight region but kept in the anti-ID one."""
    import ROOT
    ROOT.gInterpreter.Declare(r"""
#ifndef FR_AST_HELPERS
#define FR_AST_HELPERS
inline int n_fj_overlap(int nfj, float e1, float p1, float e2, float p2,
                        float e3, float p3, float te, float tp){
    float es[3] = {e1, e2, e3}, ps[3] = {p1, p2, p3};
    int n = 0;
    int lim = nfj < 3 ? nfj : 3;
    for (int i = 0; i < lim; i++) {
        float dphi = ps[i] - tp;
        while (dphi >  (float)M_PI) dphi -= 2.f * (float)M_PI;
        while (dphi < -(float)M_PI) dphi += 2.f * (float)M_PI;
        float deta = es[i] - te;
        if (std::sqrt(deta * deta + dphi * dphi) < 1.0f) n++;
    }
    return n;
}
#endif
""")


def linbin(lo, hi, n):
    step = (hi - lo) / n
    return array.array('d', [lo + i * step for i in range(n + 1)])


def cont(lo, hi):
    # SCORE_ZOOM="lo,hi" re-ranges the uniform binning (still NBINS_UNIFORM bins) so a
    # score tail can be inspected at fine granularity, e.g. SCORE_ZOOM=0.9,1.0 -> 20 bins
    # of width 0.005. Use with CLOSURE_OUTDIR so the full-range plots are not overwritten.
    z = os.environ.get('SCORE_ZOOM', '')
    if z:
        zlo, zhi = (float(v) for v in z.split(','))
        return linbin(zlo, zhi, NBINS_UNIFORM)
    return linbin(lo, hi, NBINS_UNIFORM)


def build_plot_vars():
    pv = []

    # Global
    pv += [
        ('ht',               'H_{T} [GeV]',        cont(330, 1500)),
        ('met',              'MET [GeV]',          cont(0, 300)),
        ('metphi',           'MET #phi',           cont(-3.14, 3.14)),
        ('nsmalljets',       'N_{jets} (small R)', array.array('d', [4, 5, 6, 7, 8, 9, 11])),
        ('nfatjets',         'N_{fatjets}',        array.array('d', [0, 1, 2, 3, 4])),
        ('kind_category_FR', 'kind_category_FR',   array.array('d', [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])),
    ]

    # Per-jet (1-8)
    for i in range(1, 9):
        pv += [
            ('jet%dMass'      % i, 'jet_{%d} mass [GeV]' % i, cont(0, 100)),
            ('jet%dPt'        % i, 'jet_{%d} p_{T} [GeV]' % i, cont(20, 500)),
            ('jet%dbRegCorr'  % i, 'jet_{%d} bRegCorr'    % i, cont(0.5, 1.5)),
            ('jet%dEta'       % i, 'jet_{%d} #eta'        % i, cont(-2.5, 2.5)),
            ('jet%dPhi'       % i, 'jet_{%d} #phi'        % i, cont(-3.14, 3.14)),
            ('jet%dDeepFlavB' % i, 'jet_{%d} DeepFlavB'   % i, cont(0, 1)),
        ]

    # Per-fatjet (1-3)
    for i in range(1, 4):
        pv += [
            ('fatJet%dPt'                 % i, 'fj_{%d} p_{T} [GeV]'  % i, cont(0, 700)),
            ('fatJet%dEta'                % i, 'fj_{%d} #eta'         % i, cont(-2.5, 2.5)),
            ('fatJet%dPhi'                % i, 'fj_{%d} #phi'         % i, cont(-3.14, 3.14)),
            ('fatJet%dPNetXbb'            % i, 'fj_{%d} PNetXbb'      % i, cont(0, 1)),
            ('fatJet%dPNetXjj'            % i, 'fj_{%d} PNetXjj'      % i, cont(0, 1)),
            ('fatJet%dMassSD_UnCorrected' % i, 'fj_{%d} SDmass [GeV]' % i, cont(0, 300)),
        ]

    # Per-lep (1-3)
    for i in range(1, 4):
        pv += [
            ('lep%dPt'  % i, 'lep_{%d} p_{T} [GeV]' % i, cont(0, 300)),
            ('lep%dEta' % i, 'lep_{%d} #eta'        % i, cont(-2.5, 2.5)),
            ('lep%dPhi' % i, 'lep_{%d} #phi'        % i, cont(-3.14, 3.14)),
            ('lep%dId'  % i, 'lep_{%d} Id'          % i,
             array.array('d', [-15.5, -13.5, -11.5, -10.5, 10.5, 11.5, 13.5, 15.5])),
        ]

    # Per-tau (1-3)
    for i in range(1, 4):
        pv += [
            ('tau%dMass'      % i, '#tau_{%d} mass [GeV]'  % i, cont(0, 3.0)),
            ('tau%dPt'        % i, '#tau_{%d} p_{T} [GeV]' % i, cont(0, 200)),
            ('tau%dEta'       % i, '#tau_{%d} #eta'        % i, cont(-2.4, 2.4)),
            ('tau%dPhi'       % i, '#tau_{%d} #phi'        % i, cont(-3.14, 3.14)),
            ('tau%dCharge'    % i, '#tau_{%d} charge'      % i, array.array('d', [-1.5, -0.5, 0.5, 1.5])),
            ('tau%ddecayMode' % i, '#tau_{%d} decayMode'   % i, array.array('d', [-0.5, 0.5, 1.5, 2.5, 9.5, 10.5, 11.5])),
        ]

    # tau mother-jet variables (FR-axis acceptance gates)
    pv += [
        ('tau1jetPt',        '#tau_{1} mother-jet p_{T} [GeV]', cont(0, 300)),
        ('tau1jetDeepFlavB', '#tau_{1} mother-jet DeepFlavB',   cont(0, 1)),
        ('tau1jetQGL',       '#tau_{1} mother-jet QGL',         cont(0, 1)),
        ('jet4DeepFlavB',    'jet_{4} DeepFlavB',               cont(0, 0.5)),
    ]

    # Tau-pair
    pv += [
        ('higgs3_mass_manu',     '#tau-pair mass [GeV]',      cont(0, 300)),
        ('higgs3_pt_manu',       '#tau-pair p_{T} [GeV]',     cont(0, 500)),
        ('higgs3_eta_manu',      '#tau-pair #eta',            cont(-3.0, 3.0)),
        ('higgs3_phi_manu',      '#tau-pair #phi',            cont(-3.14, 3.14)),
        ('deltaR_taupair',       '#DeltaR(#tau,#tau)',        cont(0, 5)),
        ('deltaPhi_taupair_MET', '#Delta#phi(#tau#tau, MET)', cont(-3.14, 3.14)),
    ]

    # Jet pairs (1-8 -> 28 unique pairs)
    for i in range(1, 8):
        for j in range(i + 1, 9):
            tag = 'jet%djet%d' % (i, j)
            pv += [
                ('mass' + tag, 'm(j_{%d}j_{%d}) [GeV]'    % (i, j), cont(0, 800)),
                ('pt'   + tag, 'p_{T}(j_{%d}j_{%d}) [GeV]' % (i, j), cont(0, 800)),
                ('eta'  + tag, '#eta(j_{%d}j_{%d})'        % (i, j), cont(-5, 5)),
                ('phi'  + tag, '#phi(j_{%d}j_{%d})'        % (i, j), cont(-3.14, 3.14)),
                ('dr'   + tag, '#DeltaR(j_{%d}j_{%d})'     % (i, j), cont(0, 5)),
            ]

    # SPANet outputs
    pv += [
        ('h1_spanet_mass', 'h_{1} SPANet mass [GeV]', cont(0, 250)),
        ('h2_spanet_mass', 'h_{2} SPANet mass [GeV]', cont(0, 250)),
        ('h1_spanet_mass_Disco_prod_ep228', 'h_{1} SPANet mass (DisCo prod) [GeV]', cont(0, 250)),
        ('h2_spanet_mass_Disco_prod_ep228', 'h_{2} SPANet mass (DisCo prod) [GeV]', cont(0, 250)),
        ('hh_mass',        'hh mass [GeV]',           cont(0, 1500)),
        ('ProbHHH4b2tau',  'SPANet ProbHHH4b2tau',    cont(0, 1)),
        (os.environ.get('CDFMAP_SCORE', 'ProbHHH4b2tau_v28common_ep281') + '_cdfmap',
         'CDF-mapped %s (decorr. rawVSjet)' % os.environ.get('CDFMAP_SCORE', 'ProbHHH4b2tau_v28common_ep281'), cont(0, 1)),
        ('ProbHHH4b2tau_Disco',      'SPANet ProbHHH4b2tau (DisCo l20 ep51)', cont(0, 1)),
        ('ProbHHH4b2tau_Disco_ep60', 'SPANet ProbHHH4b2tau (DisCo l20 ep60)', cont(0, 1)),
        ('ProbHHH4b2tau_Disco_prod_ep228', 'SPANet ProbHHH4b2tau (DisCo prod ep228)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_nodisco_ep53', 'SPANet ProbHHH4b2tau (1tau0l no-DisCo ep53)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mode_l10_ep56', 'SPANet ProbHHH4b2tau (1tau0l MoDe l10 ep56)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mode_l1_ep56', 'SPANet ProbHHH4b2tau (1tau0l MoDe l1 ep56)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mode_l10_b4_ep56', 'SPANet ProbHHH4b2tau (1tau0l MoDe l10 b4 ep56)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mode_l50_b8_ep55', 'SPANet ProbHHH4b2tau (1tau0l MoDe l50 b8 ep55)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_nomet_ep58', 'SPANet ProbHHH4b2tau (1tau0l noMET ep58)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mode_l20_genfake_warmup_ep62', 'SPANet ProbHHH4b2tau (1tau0l MoDe l20 genfake warmup ep62)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_disco_l20_genfake_warmup_ep66', 'SPANet ProbHHH4b2tau (1tau0l DisCo l20 genfake warmup ep66)', cont(0, 1)),
        ('ProbHHH4b2tau_2fj_1tau0l_resampling_ep56', 'SPANet ProbHHH4b2tau (1tau0l 2fj resampling ep56)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mirror5c_ep57', 'SPANet ProbHHH4b2tau (1tau0l mirror5c ep57)', cont(0, 1)),
        ('ProbHHH4b2tau_3ch_mirror5c_ep50', 'SPANet ProbHHH4b2tau (3ch mirror5c ep50)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mirror5c_ep299', 'SPANet ProbHHH4b2tau (1tau0l mirror5c ep299)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mirror5c_nopairetaphi_ep56', 'SPANet ProbHHH4b2tau (1tau0l mirror5c nopairetaphi ep56)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mirror11c_ep299', 'SPANet ProbHHH4b2tau (1tau0l mirror11c ep299)', cont(0, 1)),
        ('ProbHHH4b2tau_faketau_1tau0l_ep74', 'SPANet ProbHHH4b2tau (1tau0l faketau ep74)', cont(0, 1)),
        ('ProbFakeTau_faketau_1tau0l_ep74', 'SPANet ProbFakeTau (1tau0l faketau ep74)', cont(0, 1)),
        ('ProbHHH4b2tau_3ch_mirror11c_rawdeeptau_ep290', 'SPANet ProbHHH4b2tau (3ch mirror11c rawDeepTau ep290)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mirror5c_shares_ep286', 'SPANet ProbHHH4b2tau (1tau0l mirror5c shares ep286)', cont(0, 1)),
        ('ProbHHH4b2tau_1tau0l_mirror5c_benriched_shares_nopairetaphi_ep266', 'SPANet ProbHHH4b2tau (1tau0l mirror5c benriched shares nopairetaphi ep266)', cont(0, 1)),
        ('ProbHHH4b2tau_shares_attnpool_ep286', 'SPANet ProbHHH4b2tau (1tau0l mirror5c shares attnpool ep286)', cont(0, 1)),
        ('ProbHHH6b',      'SPANet ProbHHH6b',        cont(0, 1)),
        ('ProbHHH6b_1tau0l_nodisco_ep53', 'SPANet ProbHHH6b (1tau0l no-DisCo ep53)', cont(0, 1)),
        ('ProbQCD',        'SPANet ProbQCD',          cont(0, 1)),
        ('ProbTTHard',     'SPANet ProbTTHard',       cont(0, 1)),
        ('ProbTTSemi',     'SPANet ProbTTSemi',       cont(0, 1)),
        ('ProbTTlep',      'SPANet ProbTTlep',        cont(0, 1)),
        ('ProbWJets',      'SPANet ProbWJets',        cont(0, 1)),
        ('ProbZJets',      'SPANet ProbZJets',        cont(0, 1)),
        ('ProbVV',         'SPANet ProbVV',           cont(0, 1)),
        ('ProbHH4b',       'SPANet ProbHH4b',         cont(0, 1)),
        ('ProbHH2b2tau',   'SPANet ProbHH2b2tau',     cont(0, 1)),
        ('Prob0rh0bh',     'SPANet Prob0rh0bh',       cont(0, 1)),
        ('Prob1rh0bh',     'SPANet Prob1rh0bh',       cont(0, 1)),
        ('Prob2rh0bh',     'SPANet Prob2rh0bh',       cont(0, 1)),
        ('Prob0rh1bh',     'SPANet Prob0rh1bh',       cont(0, 1)),
        ('Prob1rh1bh',     'SPANet Prob1rh1bh',       cont(0, 1)),
        ('Prob0rh2bh',     'SPANet Prob0rh2bh',       cont(0, 1)),
        ('IndexMaxProb',   'SPANet argmax class',     array.array('d', [0.5 + i for i in range(12)])),
        ('IndexMaxCat',    'SPANet argmax category',  array.array('d', [0.5 + i for i in range(7)])),
    ]

    # ---- v27 additions -----------------------------------------------------
    # v27 SPANet inputs that are new/defined in 1tau0l: per-tau MT, the derived
    # min-dphi(jet,MET) and MET/HT, and the event-shape / MET-resolution scalars.
    # (The two-visible-leg Globals dzeta/pzeta_*/mt2_ll and the tau-pair block are
    #  sentinel/degenerate in 1tau0l -> not plotted.)
    for i in range(1, 4):
        pv += [('tau%dMt' % i, '#tau_{%d} M_{T}(#tau,MET) [GeV]' % i, cont(0, 200))]
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
    ]
    # v27 CLS5 SPANet classification scores, both models (qcd_free, split).
    for _pf, _lab in (('_v27_qcd_free_ep275', 'v27 qcd_free ep275'),
                      ('_v27_split_ep254',    'v27 split ep254')):
        for _cl in ('ProbHHH4b2tau', 'ProbTT', 'ProbQCD', 'ProbOtherH', 'ProbVjetsVVttX'):
            pv += [('%s%s' % (_cl, _pf), 'SPANet %s (%s)' % (_cl, _lab), cont(0, 1))]

    # v28 CLS5 SPANet classification scores, both models (common, qcd_frac), on v29pre.
    # + v29 weighted-mirror baseline (ep201, 2026-08-10).
    for _pf, _lab in (('_v28common_ep281', '1tau0l v28 common ep281'),
                      ('_v28qcdfrac_ep289', '1tau0l v28 qcd_frac ep289'),
                      ('_v29wmirror_ep201', '1tau0l v29 wmirror ep201')):
        for _cl in ('ProbHHH4b2tau', 'ProbTT', 'ProbQCD', 'ProbOtherH', 'ProbVjetsVVttX'):
            pv += [('%s%s' % (_cl, _pf), 'SPANet %s (%s)' % (_cl, _lab), cont(0, 1))]

    return pv


# =========================================================================
def make_df(subdir, exclude_patterns=(), include_patterns=None):
    """Build an RDataFrame over all *_tree.root in parts_SPANET_v13_merged/."""
    import ROOT
    parts_dir = os.path.join(BASEDIR, subdir, MERGED)
    files = sorted(glob.glob(os.path.join(parts_dir, '*_tree.root')))
    # FakeTau (FR) method: QCD is captured by the data-driven fake template, so it
    # must NOT enter the MC prompt subtraction in MR/VR (genPartFlav==5 in QCD is
    # spurious and would double-count). Drop QCD from every 'mc' input.
    if subdir == 'mc' and not PURE_MC:
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
    if V27:
        # v27 parts carry only the CLS5 classification branches (per-model, postfixed)
        # and NO old 11-class / Higgs-cat branches -> the argmax defines below would
        # fail. The SPANet Higgs-number head is not applied to v27, so IndexMaxCat is
        # undefined; set sentinels so downstream (inclusive-only) code stays valid.
        df = df.Define('IndexMaxProb', '-1').Define('IndexMaxCat', '-1')
    else:
        idx_expr = ('get_max_prob(ProbHHH6b, ProbQCD, ProbTTHard, ProbWJets, ProbZJets, '
                    'ProbTTSemi, ProbHHH4b2tau, ProbTTlep, ProbVV, ProbHH4b, ProbHH2b2tau)')
        cat_expr = ('get_max_cat(Prob0rh0bh, Prob1rh0bh, Prob2rh0bh, '
                    'Prob0rh1bh, Prob1rh1bh, Prob0rh2bh)')
        df = df.Define('IndexMaxProb', idx_expr).Define('IndexMaxCat', cat_expr)
    df = df.Define('is_tight',  'tau1idDeepTau2017v2p1VSjet >= %d' % _WP_TIGHT)
    df = df.Define('is_loose_not_tight',
                   'tau1idDeepTau2017v2p1VSjet >= 2 && tau1idDeepTau2017v2p1VSjet < %d' % _WP_TIGHT)
    if FR_AST:
        # as-if-tight bookkeeping for anti-ID events (tight events untouched:
        # their mother jet is already cleaned). ROOT 6.18 has no Redefine, so the
        # corrected values live in *_eff columns; the tauFR_w defines below and
        # the plot fills (fill-site redirect in main) consume them. For tight
        # events _eff == recorded, so one column serves both legs.
        _declare_ast_helpers()
        df = df.Define('_mj_counted', 'tau1jetPt > 25 && abs(tau1jetEta) < 2.4')
        df = df.Define('_mj_in_ht',   'tau1jetPt > 20 && abs(tau1jetEta) < 2.4')
        df = df.Define('nsmalljets_eff',
                       'is_loose_not_tight ? nsmalljets - (int)_mj_counted : nsmalljets')
        df = df.Define('ht_eff',
                       'is_loose_not_tight ? (float)(ht - (_mj_in_ht ? tau1jetPt : 0.f)) : ht')
        df = df.Define('nbtags_eff',
                       'is_loose_not_tight ? nbtags - (int)(_mj_counted && tau1jetDeepFlavB > 0.3040f) : nbtags')
        df = df.Define('nfatjets_eff',
                       'is_loose_not_tight ? nfatjets - n_fj_overlap(nfatjets, fatJet1Eta, fatJet1Phi, '
                       'fatJet2Eta, fatJet2Phi, fatJet3Eta, fatJet3Phi, tau1Eta, tau1Phi) : nfatjets')
        # re-apply the analysis cuts on the corrected values: anti-ID events whose
        # as-if-tight jets fail them would never enter the tight region.
        df = df.Filter('(!is_loose_not_tight) || '
                       '(nsmalljets_eff >= 4 && ht_eff >= 330 && nbtags_eff >= 3)',
                       'FR_AST anti-ID re-cut')
    if USE_CDFMAP:
        df = df.Define('%s_cdfmap' % CDFMAP_SCORE,
                       '(float)((1.0-%f)*%s + %f*prob_cdfmap((float)%s, tau1rawDeepTau2017v2p1VSjet))'
                       % (CDFMAP_ALPHA, CDFMAP_SCORE, CDFMAP_ALPHA, CDFMAP_SCORE))
    _njcol = 'nsmalljets_eff' if FR_AST else 'nsmalljets'
    if USE_FR2D:
        # FR2D_YVAR: 'dfb' (default) or 'abseta' (eta-axis maps, tttt-style test)
        _yv = ('tau1jetDeepFlavB' if os.environ.get('FR2D_YVAR', 'dfb') == 'dfb'
               else '(float)std::abs(tau1Eta)')
        df = df.Define('tauFR_w',       'tauFR_weight_2d(tau1jetPt, %s, tau1decayMode, %s)' % (_yv, _njcol))
        df = df.Define('tauFR_w_up',    'tauFR_weight_2d_up(tau1jetPt, %s, tau1decayMode, %s)' % (_yv, _njcol))
        df = df.Define('tauFR_w_down',  'tauFR_weight_2d_down(tau1jetPt, %s, tau1decayMode, %s)' % (_yv, _njcol))
    else:
        df = df.Define('tauFR_w',       'tauFR_weight(tau1jetPt, tau1jetEta, tau1decayMode, %s)' % _njcol)
        df = df.Define('tauFR_w_up',    'tauFR_weight_up(tau1jetPt, tau1jetEta, tau1decayMode, %s)' % _njcol)
        df = df.Define('tauFR_w_down',  'tauFR_weight_down(tau1jetPt, tau1jetEta, tau1decayMode, %s)' % _njcol)
    if USE_MUFFIN:
        # poster feature order: decay mode, seeding-jet/tau pT ratio, tau pT,
        # njet, nbtag, eta, phi.  The band uses the bootstrap spread of the
        # fake factor itself, not the binned map's up/down.
        _nbcol = 'nbtags_eff' if FR_AST else 'nbtags'
        _mufargs = ('tau1decayMode, (tau1jetPt/std::max(tau1Pt,1.e-6f)), tau1Pt, '
                    '%s, %s, tau1Eta, tau1Phi' % (_njcol, _nbcol))
        df = df.Define('tauFR_w_muf', 'muffin_weight(%s)' % _mufargs)
        df = df.Define('tauFR_w_muf_rms', 'muffin_weight_rms(%s)' % _mufargs)
        df = df.Define('tauFR_w_muf_up', 'tauFR_w_muf + tauFR_w_muf_rms')
        df = df.Define('tauFR_w_muf_down', 'std::max(0., tauFR_w_muf - tauFR_w_muf_rms)')
    return df


def add_mc_weight(df, with_ttbb_kill=True, scale=1.0):
    # v26: `is_inclusive_TT` is NOT a stored branch. The TTbb-overlap kill must
    # drop genTtbarId%100 >= 51 events ONLY on the inclusive TTTo* samples (not
    # TTbb_4f_TTTo, not non-TT). The caller therefore passes with_ttbb_kill=True
    # only for dataframes built purely from inclusive TTTo* files; for every
    # other group (incl. TTbb_4f and non-TT) it passes with_ttbb_kill=False.
    # (matches gotcha #3: genTtbarId%100 < 51 on the inclusive TTTo* samples.)
    if with_ttbb_kill:
        df = df.Define('ttbb_overlap_kill',
            '((genTtbarId % 100) < 51) ? 1.0f : 0.0f')
    else:
        df = df.Define('ttbb_overlap_kill', '1.0f')
    _trig = '' if NO_TRIGSF else ' * triggerLumiSF * triggerSF_perfilter_2nBtag_v24c'
    df = df.Define('mc_weight',
        '(float)(%f * ttbb_overlap_kill * xsecWeight * genWeight * l1PreFiringWeight * puWeight'
        '%s * btagWeight_shape * btagShapeR_weight'
        ' * tauIDSF_weight * Muon1IdSF * Ele1IdSF * %f)' % (scale, _trig, LUMI))
    df = df.Define('is_prompt_tau', 'tau1genPartFlav == 5')
    df = df.Define('mc_fr_w',      'mc_weight * tauFR_w')
    df = df.Define('mc_fr_w_up',   'mc_weight * tauFR_w_up')
    df = df.Define('mc_fr_w_down', 'mc_weight * tauFR_w_down')
    if USE_MUFFIN:
        df = df.Define('mc_fr_w_muf',      'mc_weight * tauFR_w_muf')
        df = df.Define('mc_fr_w_muf_up',   'mc_weight * tauFR_w_muf_up')
        df = df.Define('mc_fr_w_muf_down', 'mc_weight * tauFR_w_muf_down')
    return df


# =========================================================================
def get_regions():
    _all_regions = {
        '0H':        ('0H_1tau0l',        'IndexMaxCat == 1',
                      'CR: 1#tau_{h}0l (0 Higgs)'),
        '1H':        ('1H_1tau0l',        'IndexMaxCat == 2 || IndexMaxCat == 4',
                      'CR: 1#tau_{h}0l (1 Higgs)'),
        '2H':        ('2H_1tau0l',        'IndexMaxCat == 3 || IndexMaxCat == 5 || IndexMaxCat == 6',
                      'CR: 1#tau_{h}0l (2 Higgs)'),
        # 0+1H = the complement of 2H (IndexMaxCat in {1,2,4}); the low-NHiggs bucket kept
        # together so 2H can be compared against everything else.
        '0p1H':      ('0p1H_1tau0l',      'IndexMaxCat == 1 || IndexMaxCat == 2 || IndexMaxCat == 4',
                      'CR: 1#tau_{h}0l (0+1 Higgs)'),
        'inclusive': ('inclusive_1tau0l', '1', 'CR: 1#tau_{h}0l (NHiggs inclusive)'),
    }
    if CLOSURE_REGION:
        return [_all_regions[CLOSURE_REGION]]
    return [_all_regions['inclusive']]


# =========================================================================
def declare_cdfmap():
    """Read CDFMAP_ROOT and JIT-declare prob_cdfmap(score, rawVSjet) = s' = M^-1(F(score|rawVSjet)).
    TH2 'cdfmap': x-axis = input ProbHHH4b2tau (NS uniform bins on [0,1]), y-axis = rawVSjet
    (NX quantile edges; bin centers used as interpolation nodes). BILINEAR interpolation in
    (input-score, rawVSjet), clamped at the axis ends -> smooth, no combing."""
    import ROOT
    f = ROOT.TFile.Open(CDFMAP_ROOT)
    h = f.Get('cdfmap')
    ns, nx = h.GetNbinsX(), h.GetNbinsY()
    yc = [h.GetYaxis().GetBinCenter(i) for i in range(1, nx + 1)]     # rawVSjet node centers
    rows = []
    for iy in range(1, nx + 1):
        cols = [h.GetBinContent(ix, iy) for ix in range(1, ns + 1)]
        rows.append('{' + ', '.join('%.6f' % v for v in cols) + '}')
    f.Close()
    yc_s = '{' + ', '.join('%.6f' % v for v in yc) + '}'
    cdf_s = '{' + ', '.join(rows) + '}'
    ok = ROOT.gInterpreter.Declare('''
namespace CDFM {
  static const int NS=%d, NX=%d;
  static const double yc[%d]=%s;          // rawVSjet node centers
  static const double M[%d][%d]=%s;       // output score s' on [input-score bin][rawVSjet node]
}
inline double _cdfm_row(int iy, float s){          // linear interp over input-score within a row
  double t = s*CDFM::NS - 0.5; if(t<0) t=0; if(t>CDFM::NS-1) t=CDFM::NS-1;
  int i0=(int)t; int i1=(i0+1<CDFM::NS)?i0+1:i0; double f=t-i0;
  return CDFM::M[iy][i0]*(1-f) + CDFM::M[iy][i1]*f;
}
inline double prob_cdfmap(float s, float x){       // bilinear: interp over rawVSjet nodes too
  if(x<=CDFM::yc[0])            return _cdfm_row(0, s);
  if(x>=CDFM::yc[CDFM::NX-1])   return _cdfm_row(CDFM::NX-1, s);
  int j=0; while(j<CDFM::NX-1 && !(x>=CDFM::yc[j] && x<CDFM::yc[j+1])) j++;
  double g=(x-CDFM::yc[j])/(CDFM::yc[j+1]-CDFM::yc[j]);
  return _cdfm_row(j, s)*(1-g) + _cdfm_row(j+1, s)*g;
}
''' % (ns, nx, nx, yc_s, nx, ns, cdf_s))
    if not ok:
        raise RuntimeError('prob_cdfmap Declare failed')
    print('  prob_cdfmap (DDT, bilinear) declared from %s (NS=%d x NX=%d)' % (CDFMAP_ROOT, ns, nx))


# =========================================================================
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


def declare_muffin():
    """JIT-declare the exported MUFFIN evaluator (muffin/export_muffin_cpp.py)."""
    import ROOT
    with open(MUFFIN_HEADER) as fh:
        src = fh.read()
    if not ROOT.gInterpreter.Declare(src):
        raise RuntimeError('muffin_weight Declare failed (%s)' % MUFFIN_HEADER)
    print('  muffin_weight{,_rms} declared from %s' % MUFFIN_HEADER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', default=None,
                        help='Plot only this single variable (used for condor parallelization).')
    parser.add_argument('--list-vars', action='store_true',
                        help='Print all variable names (one per line) and exit.')
    parser.add_argument('--chunk', default=None,
                        help='I/N: process round-robin slice plot_vars[I::N] in ONE event loop (condor chunking).')
    args = parser.parse_args()

    plot_vars = build_plot_vars()
    if not USE_CDFMAP:
        plot_vars = [pv for pv in plot_vars if not pv[0].endswith('_cdfmap')]

    if args.list_vars:
        for v, _t, _e in plot_vars:
            print(v)
        return

    if args.variable is not None:
        _want = [s.strip() for s in args.variable.split(',') if s.strip()]
        plot_vars = [(v, t, e) for v, t, e in plot_vars if v in _want]
        if not plot_vars:
            print('ERROR: variable(s) %s not found in spec' % args.variable, file=sys.stderr)
            sys.exit(2)

    if args.chunk is not None:
        ci, cn = (int(x) for x in args.chunk.split('/'))
        plot_vars = plot_vars[ci::cn]

    os.makedirs(OUTDIR, exist_ok=True)
    print('OUTDIR:', OUTDIR)
    print('NUM VARS TO PLOT:', len(plot_vars))

    # Lazy ROOT + calibrations imports so --list-vars works outside cmssw-el7
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    _declare_cpp_helpers()
    sys.path.insert(0, THIS_DIR)

    MC_GROUPS = get_mc_groups()

    # Settled "new" FR method: replace the 1-D FR with the direct 2-D FR(jetPt x DeepFlavB).
    if USE_MUFFIN and not USE_FR2D:
        raise RuntimeError('USE_MUFFIN=1 needs USE_FR2D=1: the binned map is the '
                           'comparison curve in the ratio pad')
    if USE_FR2D:
        declare_fr2d()
        print('  USE_FR2D=1 -> fr2d DEEP weighting; OUTDIR=%s' % OUTDIR)
    if USE_MUFFIN:
        declare_muffin()
        print('  USE_MUFFIN=1 -> stack uses the MUFFIN fake factor, binned map '
              'overlaid in the ratio pad; OUTDIR=%s' % OUTDIR)
    else:
        # 1-D FR path needs correctionlib (calibrations); only import when actually used.
        import calibrations
        # Load the v26 measured FR map (derived on this same v26 option-92 production).
        calibrations.tauFR_init(
            '2017',
            fr_root_path='/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/plots_tauFR_2017_v26_2nBtag_fixed_ttbb_HH_prob4b2tauLT0p8/tau_fakerate_2017.root')

    if USE_CDFMAP:
        declare_cdfmap()
        print('  USE_CDFMAP=1 -> ProbHHH4b2tau_cdfmap added; OUTDIR=%s' % OUTDIR)

    # v26: btagShapeR_weight IS a stored branch in the merged files, so it is
    # used directly in mc_weight (no inline btagShapeR(nsmalljets) helper needed).

    # ---- Data + aggregate MC (for FR template) + overlays ----
    df_data, data_files = make_df('data', exclude_patterns=('SingleMuon', 'SingleElectron', 'FakeTau_'))  # BTagCSV only
    df_mc,   mc_files   = make_df('mc',   exclude_patterns=('FakeTau_',))

    print('=== Data files: %d ===' % len(data_files))
    print('=== MC files:   %d ===' % len(mc_files))

    df_data = common_defines(df_data)

    # Robustness: cross-region control trees (e.g. 1tau1l-hadronic) carry a reduced
    # branch set. Drop plot vars whose branch is absent to avoid JIT Filter failures.
    _avail = set(str(c) for c in df_data.GetColumnNames())
    _missing = [v for v, _t, _e in plot_vars if v not in _avail]
    if _missing:
        print('  skipping %d vars absent from input tree: %s%s'
              % (len(_missing), ', '.join(_missing[:8]), ' ...' if len(_missing) > 8 else ''))
        plot_vars = [(v, t, e) for v, t, e in plot_vars if v in _avail]

    # v26: no `is_inclusive_TT` branch -> split aggregate MC into inclusive-TT
    # files (genTtbarId-kill applied) and the rest (no kill), so the FR
    # loose-not-tight prompt-MC subtraction handles the TTbb overlap correctly.
    # `df_mc_list` = [(weighted_df, files), ...]; their loose histos are summed.
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
    # Each entry: (legend label, file pattern, color, linestyle)
    overlay_specs = [
        ('HHH#rightarrow4b2#tau #times1000',  'HHHTo4B2Tau',            ROOT.kRed,        2),
    ]  # only the signal of interest; HH/HHH6b overlays removed
    overlay_dfs = []
    for label, pat, color, lstyle in overlay_specs:
        sdf, sfiles = make_df('signal', include_patterns=(pat,))
        if sdf is None:
            print('  (no signal files for %s)' % pat)
            continue
        print('=== Overlay %-22s : %d files ===' % (pat, len(sfiles)))
        sdf = add_mc_weight(common_defines(sdf), with_ttbb_kill=False, scale=1000.0)
        overlay_dfs.append((label, pat, color, lstyle, sdf))

    # ---- Per-group MC dataframes (prompt-tau filter applied later in tight selection) ----
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
        # v26: TTbb-overlap kill (genTtbarId%100 < 51) applies ONLY to the
        # inclusive TTTo* samples == the 'ttbar' group. 'ttbb' (TTbb_4f_TTTo)
        # and all non-TT groups must NOT be killed.
        gdf = add_mc_weight(common_defines(gdf), with_ttbb_kill=(gname == 'ttbar'))
        grp_df[gname] = gdf
    col_of = {g: c for g, _p, c in MC_GROUPS}

    closure_regions = get_regions()

    # ---- Book histograms ----
    booked = {}  # (reg, var) -> dict
    for reg_name, reg_sel, reg_label in closure_regions:
        # Numerator (data_t / MC-prompt-tight) region MUST match the tight WP of the
        # template. kind_category_analysis==2 is built with the analysis = Medium tau WP,
        # so it pins the numerator to VSjet>=16. At a non-medium tight WP (e.g. loose,
        # VSjet>=8) the template extrapolates to >=8 while kind_category_analysis==2 would
        # cap data_t at >=16 -> ~2.5x overprediction (flat Data/Pred~0.4). For non-medium WP
        # use the Fakeable categorization (kind_category_FR==2) + is_tight only, WP-consistent
        # with sel_loose and the FR map (matches the FR-measurement closure).
        if TAU_TIGHT_WP == 'medium':
            sel_tight = '(%s) && kind_category_analysis == 2 && kind_category_FR == 2 && (%s)' % (reg_sel, EXTRA_CUT)
        else:
            sel_tight = '(%s) && %s && (%s)' % (reg_sel, REGION_BASE, EXTRA_CUT)
        sel_loose = '(%s) && %s && is_loose_not_tight && (%s)' % (reg_sel, REGION_BASE, EXTRA_CUT)

        df_data_t = df_data.Filter(sel_tight + ' && is_tight')
        df_data_l = df_data.Filter(sel_loose)
        _taugate = '' if MCFAKES else ' && is_prompt_tau'
        df_mc_l   = [d.Filter(sel_loose + (_taugate if MCFAKES else ' && is_prompt_tau')) for d in df_mc_list]
        if MCFAKES:
            df_mc_l = [d.Filter(sel_loose) for d in df_mc_list]

        # Per-group tight-prompt MC filters
        df_grp_t = {g: grp_df[g].Filter(sel_tight + ' && is_tight'
                                        + ('' if (PURE_MC or MCFAKES) else ' && is_prompt_tau'))
                    for g in grp_df}

        # Per-overlay (signal) tight filters; no prompt-tau cut needed
        df_overlay_t = [(label, pat, color, lstyle, sdf.Filter(sel_tight + ' && is_tight'))
                        for label, pat, color, lstyle, sdf in overlay_dfs]

        for var_name, var_title, edges in plot_vars:
            nbins = len(edges) - 1
            hpfx = '%s_%s' % (reg_name, var_name)

            # Per-variable filter: drop events where the variable is exactly 0
            # (interpreted as "this physics object is not present in the event")
            # for everything that isn't in NO_ZERO_FILTER_VARS.
            if var_name in NO_ZERO_FILTER_VARS:
                ddt, ddl, dml = df_data_t, df_data_l, df_mc_l
                dgrp = df_grp_t
                doverlay = df_overlay_t
            else:
                zf = '%s != 0' % var_name
                ddt = df_data_t.Filter(zf)
                ddl = df_data_l.Filter(zf)
                dml = [d.Filter(zf) for d in df_mc_l]
                dgrp = {g: f.Filter(zf) for g, f in df_grp_t.items()}
                doverlay = [(L, P, C, S, f.Filter(zf))
                            for L, P, C, S, f in df_overlay_t]

            # FR_AST: jet-derived variables are filled from the *_eff columns
            # (tight leg: identical to recorded; anti-ID leg: as-if-tight).
            fill_col = var_name
            if FR_AST and var_name in ('nsmalljets', 'ht', 'nfatjets'):
                fill_col = var_name + '_eff'
            entry = dict(
                title=var_title, edges=edges, label=reg_label,
                data_t      =ddt.Histo1D((hpfx + '_data_t',      '', nbins, edges), fill_col),
                data_l      =ddl.Histo1D((hpfx + '_data_l',      '', nbins, edges), fill_col, 'tauFR_w'),
                data_l_up   =ddl.Histo1D((hpfx + '_data_l_up',   '', nbins, edges), fill_col, 'tauFR_w_up'),
                data_l_down =ddl.Histo1D((hpfx + '_data_l_down', '', nbins, edges), fill_col, 'tauFR_w_down'),
                mc_l        =[d.Histo1D((hpfx + ('_mc_l_%d' % k),      '', nbins, edges), fill_col, 'mc_fr_w')      for k, d in enumerate(dml)],
                mc_l_up     =[d.Histo1D((hpfx + ('_mc_l_up_%d' % k),   '', nbins, edges), fill_col, 'mc_fr_w_up')   for k, d in enumerate(dml)],
                mc_l_down   =[d.Histo1D((hpfx + ('_mc_l_down_%d' % k), '', nbins, edges), fill_col, 'mc_fr_w_down') for k, d in enumerate(dml)],
                grp_t       ={},
                overlays    =[],
            )
            if USE_MUFFIN:
                # same events, same subtraction -- only the fake factor differs
                entry.update(
                    data_l_muf      =ddl.Histo1D((hpfx + '_data_l_muf',      '', nbins, edges), fill_col, 'tauFR_w_muf'),
                    data_l_muf_up   =ddl.Histo1D((hpfx + '_data_l_muf_up',   '', nbins, edges), fill_col, 'tauFR_w_muf_up'),
                    data_l_muf_down =ddl.Histo1D((hpfx + '_data_l_muf_down', '', nbins, edges), fill_col, 'tauFR_w_muf_down'),
                    mc_l_muf        =[d.Histo1D((hpfx + ('_mc_l_muf_%d' % k),      '', nbins, edges), fill_col, 'mc_fr_w_muf')      for k, d in enumerate(dml)],
                    mc_l_muf_up     =[d.Histo1D((hpfx + ('_mc_l_muf_up_%d' % k),   '', nbins, edges), fill_col, 'mc_fr_w_muf_up')   for k, d in enumerate(dml)],
                    mc_l_muf_down   =[d.Histo1D((hpfx + ('_mc_l_muf_down_%d' % k), '', nbins, edges), fill_col, 'mc_fr_w_muf_down') for k, d in enumerate(dml)],
                )
            for g in dgrp:
                entry['grp_t'][g] = dgrp[g].Histo1D(
                    (hpfx + '_' + g, '', nbins, edges), fill_col, 'mc_weight')
            for label, pat, color, lstyle, fdf in doverlay:
                h = fdf.Histo1D((hpfx + '_ov_' + pat, '', nbins, edges), fill_col, 'mc_weight')
                entry['overlays'].append((label, pat, color, lstyle, h))
            booked[(reg_name, var_name)] = entry

    # ---- Materialize and draw ----
    print('\n=== Materializing histograms ===')
    root_out_name = 'closure_v29pre_20bin_split'
    if args.variable is not None:
        root_out_name += '_%s' % args.variable.replace(',', '_')
    if args.chunk is not None:
        root_out_name += '_chunk%s' % args.chunk.replace('/', 'of')
    # Many-variable runs concatenate every name -> can exceed the 255-char filename
    # limit. Cap it with a short hash suffix (the per-variable PNGs are unaffected).
    if len(root_out_name) > 180:
        import hashlib
        root_out_name = 'closure_v26_20bin_split_multi_%s' % hashlib.md5(
            root_out_name.encode()).hexdigest()[:10]
    fout = ROOT.TFile(os.path.join(OUTDIR, root_out_name + '.root'), 'RECREATE')

    for reg_name, reg_sel, reg_label in closure_regions:
        print('\n--- %s ---' % reg_name)
        for var_name, var_title, edges in plot_vars:
            bk = booked[(reg_name, var_name)]

            # Data tight
            h_data = bk['data_t'].GetPtr().Clone('%s_%s_data' % (reg_name, var_name))
            h_data.SetDirectory(0)

            # Per-group prompt-tau tight
            h_grp = {}
            h_mc_p_total = None
            for g, _p, _c in MC_GROUPS:
                if g not in bk['grp_t']:
                    continue
                h = bk['grp_t'][g].GetPtr().Clone('%s_%s_%s' % (reg_name, var_name, g))
                h.SetDirectory(0)
                h.SetFillColor(col_of[g]); h.SetLineColor(ROOT.kBlack); h.SetLineWidth(1)
                h_grp[g] = h
                if h_mc_p_total is None:
                    h_mc_p_total = h.Clone('%s_%s_mc_prompt_total' % (reg_name, var_name))
                    h_mc_p_total.SetDirectory(0)
                else:
                    h_mc_p_total.Add(h)

            # v26: aggregate loose-prompt MC is split across the inclusive-TT /
            # rest dataframes -> sum the per-df histogram lists into one each.
            def _sum_list(hlist, name):
                tot = None
                for hp in hlist:
                    h = hp.GetPtr()
                    if tot is None:
                        tot = h.Clone(name); tot.SetDirectory(0)
                    else:
                        tot.Add(h)
                return tot

            h_mc_l    = _sum_list(bk['mc_l'],      '%s_%s_mc_l'      % (reg_name, var_name))
            h_mc_l_up = _sum_list(bk['mc_l_up'],   '%s_%s_mc_l_up'   % (reg_name, var_name))
            h_mc_l_dn = _sum_list(bk['mc_l_down'], '%s_%s_mc_l_down' % (reg_name, var_name))

            # FR fake-tau template = data_loose - mc_prompt_loose
            h_fake = bk['data_l'].GetPtr().Clone('%s_%s_faketau' % (reg_name, var_name))
            h_fake.Add(h_mc_l, -1.0); h_fake.SetDirectory(0)
            h_fake_up = bk['data_l_up'].GetPtr().Clone('%s_%s_faketau_up' % (reg_name, var_name))
            h_fake_up.Add(h_mc_l_up, -1.0); h_fake_up.SetDirectory(0)
            h_fake_dn = bk['data_l_down'].GetPtr().Clone('%s_%s_faketau_dn' % (reg_name, var_name))
            h_fake_dn.Add(h_mc_l_dn, -1.0); h_fake_dn.SetDirectory(0)
            if PURE_MC:
                # No data-driven FR template; fakes are inside the MC groups (MC truth).
                for _h in (h_fake, h_fake_up, h_fake_dn):
                    _h.Reset()

            # MUFFIN: keep the binned template as the comparison curve, then swap
            # the MUFFIN one into h_fake, so the stack, the total and the band
            # that follow are all built from it without touching that code.
            h_fake_binned = None
            if USE_MUFFIN:
                h_fake_binned = h_fake.Clone('%s_%s_faketau_binned' % (reg_name, var_name))
                h_fake_binned.SetDirectory(0)
                h_mc_l_muf    = _sum_list(bk['mc_l_muf'],      '%s_%s_mc_l_muf'      % (reg_name, var_name))
                h_mc_l_muf_up = _sum_list(bk['mc_l_muf_up'],   '%s_%s_mc_l_muf_up'   % (reg_name, var_name))
                h_mc_l_muf_dn = _sum_list(bk['mc_l_muf_down'], '%s_%s_mc_l_muf_down' % (reg_name, var_name))
                for _key, _mc, _dst in (('data_l_muf',      h_mc_l_muf,    h_fake),
                                        ('data_l_muf_up',   h_mc_l_muf_up, h_fake_up),
                                        ('data_l_muf_down', h_mc_l_muf_dn, h_fake_dn)):
                    _dst.Reset()
                    _dst.Add(bk[_key].GetPtr())
                    _dst.Add(_mc, -1.0)
            h_fake.SetFillColor(ROOT.kGray); h_fake.SetLineColor(ROOT.kBlack); h_fake.SetLineWidth(1)

            # Print yield summary for HT
            if var_name == 'ht':
                tot_p = h_mc_p_total.Integral() if h_mc_p_total is not None else 0.0
                tot_pred = tot_p + h_fake.Integral()
                print('  Data tight:        %.0f' % h_data.Integral())
                print('  MC prompt total:   %.1f' % tot_p)
                print('  FakeTau (data-MC): %.1f' % h_fake.Integral())
                print('  Pred (MC+Fake):    %.1f' % tot_pred)
                if tot_pred > 0:
                    print('  Data/Pred:         %.3f' % (h_data.Integral() / tot_pred))
                if USE_MUFFIN and h_fake_binned is not None:
                    tot_pred_b = tot_p + h_fake_binned.Integral()
                    print('  FakeTau (binned):  %.1f' % h_fake_binned.Integral())
                    if tot_pred_b > 0:
                        print('  Data/Pred (binned):%.3f' % (h_data.Integral() / tot_pred_b))

            # ----------------- Draw -----------------
            c = ROOT.TCanvas('c_%s_%s' % (reg_name, var_name), '', 800, 800)
            pad1 = ROOT.TPad('pad1', '', 0, 0.3, 1, 1.0)
            pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.12)
            pad1.SetRightMargin(0.05); pad1.SetTopMargin(0.08); pad1.SetLogy()
            pad1.Draw(); pad1.cd()

            hs = ROOT.THStack('hs_%s_%s' % (reg_name, var_name), '')
            # bottom -> top: small MC first (VV -> Vjets -> ttX -> ttbb -> ttbar),
            # then fake-tau template on top.
            for g, _p, _c in MC_GROUPS:
                if g in h_grp:
                    hs.Add(h_grp[g])
            hs.Add(h_fake)
            hs.Draw('HIST')
            hs.GetYaxis().SetTitle('Events'); hs.GetYaxis().SetTitleSize(0.05)
            hs.GetYaxis().SetTitleOffset(1.0); hs.GetYaxis().SetLabelSize(0.04)
            hs.GetXaxis().SetLabelSize(0)
            ymax = max(h_data.GetMaximum(), hs.GetMaximum()) * 5
            hs.SetMinimum(0.5); hs.SetMaximum(ymax)

            # Total prediction = MC_prompt_total + Fake; band has stat (sqrt sum of variances)
            # plus FR-template uncertainty (from up/down on fake).
            if h_mc_p_total is None:
                h_tot = h_fake.Clone('%s_%s_tot' % (reg_name, var_name))
                h_tot_up = h_fake_up.Clone('%s_%s_tot_up' % (reg_name, var_name))
                h_tot_dn = h_fake_dn.Clone('%s_%s_tot_dn' % (reg_name, var_name))
            else:
                h_tot = h_mc_p_total.Clone('%s_%s_tot' % (reg_name, var_name)); h_tot.Add(h_fake)
                h_tot_up = h_mc_p_total.Clone('%s_%s_tot_up' % (reg_name, var_name)); h_tot_up.Add(h_fake_up)
                h_tot_dn = h_mc_p_total.Clone('%s_%s_tot_dn' % (reg_name, var_name)); h_tot_dn.Add(h_fake_dn)
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

            # Signal overlays (HHH + HH; ×1000)
            overlay_hists = []
            for label, pat, color, lstyle, hptr in bk['overlays']:
                h = hptr.GetPtr().Clone('%s_%s_ov_%s' % (reg_name, var_name, pat))
                h.SetDirectory(0)
                h.SetLineColor(color); h.SetLineWidth(2); h.SetLineStyle(lstyle)
                h.SetMarkerSize(0); h.SetFillStyle(0)
                h.Draw('HIST SAME')
                overlay_hists.append((label, h))

            # Legend (top-of-stack first: fake -> ttbar -> ... -> VV)
            # compact 2-column box in the top-right so it does not overlap the stack
            leg = ROOT.TLegend(0.44, 0.66, 0.93, 0.90)
            leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.023); leg.SetTextFont(42)
            leg.SetNColumns(2); leg.SetColumnSeparation(0.04)
            leg.AddEntry(h_data, 'Data', 'lep')
            if not PURE_MC:
                leg.AddEntry(h_fake, 'Fake #tau (MUFFIN)' if USE_MUFFIN
                             else 'Fake #tau (FR method)', 'f')
            for g, _p, _c in reversed(MC_GROUPS):
                if g in h_grp:
                    leg.AddEntry(h_grp[g], g, 'f')
            leg.AddEntry(h_tot, 'Stat. unc.' if PURE_MC else 'Stat. #oplus FR unc.', 'f')
            for label, h in overlay_hists:
                leg.AddEntry(h, label, 'l')
            leg.Draw()

            lt = ROOT.TLatex(); lt.SetNDC()
            lt.SetTextFont(61); lt.SetTextSize(0.06); lt.DrawLatex(0.12, 0.93, 'CMS')
            lt.SetTextFont(52); lt.SetTextSize(0.045); lt.DrawLatex(0.22, 0.93, 'Internal')
            lt.SetTextFont(42); lt.SetTextSize(0.04)
            lt.DrawLatex(0.62, 0.93, '41.5 fb^{-1} (13 TeV)')
            _lbl = reg_label + (' [pure MC]' if PURE_MC else '')
            if EXTRA_CUT.strip() not in ('1', ''):
                _lbl += '  [%s]' % EXTRA_CUT
            lt.DrawLatex(0.15, 0.83, _lbl)

            # Ratio pad
            c.cd()
            pad2 = ROOT.TPad('pad2', '', 0, 0.0, 1, 0.3)
            pad2.SetTopMargin(0.02); pad2.SetBottomMargin(0.35)
            pad2.SetLeftMargin(0.12); pad2.SetRightMargin(0.05); pad2.Draw(); pad2.cd()

            h_ratio = h_data.Clone('%s_%s_ratio' % (reg_name, var_name))
            # Ratio points carry ONLY the data statistical error (data_err/pred); the
            # prediction (stat (+) FR) uncertainty is shown by the band, so do NOT
            # propagate it into the points (TH1::Divide would double-count it).
            for ib in range(1, h_ratio.GetNbinsX() + 1):
                d = h_data.GetBinContent(ib); de = h_data.GetBinError(ib)
                p = h_tot.GetBinContent(ib)
                h_ratio.SetBinContent(ib, d / p if p > 0 else 0)
                h_ratio.SetBinError(ib, de / p if p > 0 else 0)
            h_ratio.SetMarkerStyle(20); h_ratio.SetMarkerSize(0.8)
            h_ratio.GetXaxis().SetTitle(var_title); h_ratio.GetXaxis().SetTitleSize(0.12)
            h_ratio.GetXaxis().SetTitleOffset(1.0); h_ratio.GetXaxis().SetLabelSize(0.10)
            h_ratio.GetYaxis().SetTitle('Data / Pred.'); h_ratio.GetYaxis().SetTitleSize(0.12)
            h_ratio.GetYaxis().SetTitleOffset(0.45); h_ratio.GetYaxis().SetLabelSize(0.10)
            h_ratio.GetYaxis().SetRangeUser(0, 2.5); h_ratio.GetYaxis().SetNdivisions(505)
            h_ratio.SetTitle(''); h_ratio.Draw('E1')

            h_r_unc = h_tot.Clone('%s_%s_ratio_unc' % (reg_name, var_name))
            for i in range(1, h_r_unc.GetNbinsX() + 1):
                v = h_r_unc.GetBinContent(i); e = h_r_unc.GetBinError(i)
                h_r_unc.SetBinContent(i, 1.0)
                h_r_unc.SetBinError(i, e / v if v > 0 else 0)
            h_r_unc.SetFillStyle(3354); h_r_unc.SetFillColor(ROOT.kGray + 2)
            h_r_unc.SetMarkerSize(0); h_r_unc.SetLineWidth(0)
            h_r_unc.Draw('E2 SAME')

            ln = ROOT.TLine(edges[0], 1, edges[-1], 1)
            ln.SetLineStyle(2); ln.SetLineColor(ROOT.kGray + 2); ln.Draw()

            # Binned-map comparison. Same data, same MC-prompt stack, same
            # anti-ID events and same prompt subtraction -- only the fake factor
            # differs, so the distance between the two curves IS the difference
            # between the two methods.
            h_ratio_bin = None
            legr = None
            if USE_MUFFIN and h_fake_binned is not None:
                if h_mc_p_total is None:
                    h_tot_bin = h_fake_binned.Clone('%s_%s_tot_binned' % (reg_name, var_name))
                else:
                    h_tot_bin = h_mc_p_total.Clone('%s_%s_tot_binned' % (reg_name, var_name))
                    h_tot_bin.Add(h_fake_binned)
                h_tot_bin.SetDirectory(0)
                h_ratio_bin = h_data.Clone('%s_%s_ratio_binned' % (reg_name, var_name))
                h_ratio_bin.SetDirectory(0)
                for ib in range(1, h_ratio_bin.GetNbinsX() + 1):
                    d = h_data.GetBinContent(ib)
                    pb = h_tot_bin.GetBinContent(ib)
                    h_ratio_bin.SetBinContent(ib, d / pb if pb > 0 else 0)
                    h_ratio_bin.SetBinError(ib, 0.0)   # the points carry the data error
                h_ratio_bin.SetStats(0)
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

            # Save both log-y (emphasized) and linear-y versions per variable
            _ymax = max(h_data.GetMaximum(), hs.GetMaximum())
            for scale in ['log', 'lin']:
                pad1.cd()
                if scale == 'log':
                    # LOGY_MIN lowers the log-y floor (default 0.5). Needed when zooming into
                    # a score tail where the x1000 signal overlay falls below the default floor
                    # and would otherwise be clipped off the canvas entirely.
                    # NOTE: after the first hs.Draw() ROOT paints the axis from the stack's
                    # internal fHistogram, so SetMinimum on the THStack alone does NOT move
                    # the drawn axis -- it must be set on GetHistogram() as well.
                    _ymin = float(os.environ.get('LOGY_MIN', '0.5'))
                    pad1.SetLogy(1)
                    hs.SetMinimum(_ymin); hs.SetMaximum(_ymax * 5)
                    if hs.GetHistogram():
                        hs.GetHistogram().SetMinimum(_ymin)
                        hs.GetHistogram().SetMaximum(_ymax * 5)
                else:
                    pad1.SetLogy(0); hs.SetMinimum(0.0); hs.SetMaximum(_ymax * 1.5)
                    if hs.GetHistogram():
                        hs.GetHistogram().SetMinimum(0.0)
                        hs.GetHistogram().SetMaximum(_ymax * 1.5)
                pad1.Modified(); pad1.Update()
                for ext in ['png', 'pdf']:
                    c.SaveAs(os.path.join(OUTDIR, 'closure_%s_%s_%s.%s' % (reg_name, var_name, scale, ext)))

            fout.cd()
            h_data.Write(); h_fake.Write(); h_tot.Write(); h_ratio.Write()
            if USE_MUFFIN and h_ratio_bin is not None:
                h_fake_binned.Write(); h_ratio_bin.Write()
            if h_mc_p_total is not None:
                h_mc_p_total.Write()
            for g in h_grp:
                h_grp[g].Write()
            # Persist the HH/HHH signal overlays so re-plotters can read them.
            for _label, h in overlay_hists:
                h.Write()

    fout.Close()
    print('\nDone. Plots saved to %s' % OUTDIR)


if __name__ == '__main__':
    main()
