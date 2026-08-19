#!/usr/bin/env python3
"""How much of the *other* HHH decay modes leaks into the 4b2tau selection?

Only two HHH samples exist on disk (HHHTo6B and HHHTo4B2Tau, both c3=0/d4=0),
so the one mode we can measure directly is HHH->6b: it has no genuine tau at
all, and enters the tau channels purely through a jet->tau_h fake.  This is a
*net extra signal*, NOT covered by the data-driven fake-tau estimate: the
MUFFIN/FR weights are measured on data whose HHH content is negligible
(sigma_HHH ~ 0.1 fb), so a 6b signal event with a fake tau in the SR is simply
unaccounted for.

The xsecWeights of the two samples differ by exactly BR(6b)/BR(4b2tau) = 3.10,
so the yields below are directly comparable.

    python3 studies/other_hhh_decay_leakage.py
"""
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'muffin'))
import muffin_common as mc  # noqa: E402

BASE = '/eos/user/r/rtu/TurbOutputMC2017_v29_ak8_option92_2017/signal/parts/'
SAMPLES = [('HHH->4b2tau', 'HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8_tree.root'),
           ('HHH->6b',     'HHHTo6B_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8_tree.root')]
SCORE = os.environ.get('SCORE', 'ProbHHH4b2tau_v29fns_ep115')
# kind_category_analysis: 2 = 1tau_h0l, 0/1 = the other two tau channels,
# 6 = the 0-tau (6b-analysis) category
CHANNELS = {2: '1tau0l', 0: 'cat0 (2tau0l/1tau1l)', 1: 'cat1 (2tau0l/1tau1l)'}


def load(fn):
    cols = ['kind_category_analysis', 'trigSF_pfHT', 'trigSF_caloHT', mc.TAU_ID_BR,
            'tau1genPartFlav', 'nbtags', SCORE] + mc.MC_COLS
    a = uproot.open(BASE + fn)['Events'].arrays(list(dict.fromkeys(cols)), library='np')
    a['_w'] = mc.mc_weight(a)
    a['_trig'] = (a['trigSF_pfHT'] >= 300) & (a['trigSF_caloHT'] >= 160)
    a['_pass'] = a[mc.TAU_ID_BR] >= 8          # Loose VSjet, the analysis WP
    return a


def main():
    d = {tag: load(fn) for tag, fn in SAMPLES}

    print('== per analysis category, Loose VSjet + trigger plateau + nbtags>=4')
    print('%-24s %12s %12s %10s' % ('category', 'N(4b2tau)', 'N(6b)', '6b/4b2tau'))
    for cat, name in CHANNELS.items():
        y = {}
        for tag, a in d.items():
            m = (a['kind_category_analysis'] == cat) & a['_trig'] & a['_pass'] & (a['nbtags'] >= 4)
            y[tag] = (a['_w'][m].sum(), int(m.sum()))
        print('%-24s %12.5f %12.5f %9.1f%%   (raw 6b: %d)'
              % (name, y['HHH->4b2tau'][0], y['HHH->6b'][0],
                 100 * y['HHH->6b'][0] / max(y['HHH->4b2tau'][0], 1e-12), y['HHH->6b'][1]))

    print('\n== 1tau0l, vs the MVA cut (%s)' % SCORE)
    print('%-10s %12s %12s %10s %10s' % ('cut', 'N(4b2tau)', 'N(6b)', '6b/4b2tau', 'raw 6b'))
    for cut in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
        y = {}
        for tag, a in d.items():
            m = ((a['kind_category_analysis'] == 2) & a['_trig'] & a['_pass']
                 & (a['nbtags'] >= 4) & (a[SCORE] > cut))
            y[tag] = (a['_w'][m].sum(), int(m.sum()))
        print('%-10.2f %12.5f %12.5f %9.1f%% %10d'
              % (cut, y['HHH->4b2tau'][0], y['HHH->6b'][0],
                 100 * y['HHH->6b'][0] / max(y['HHH->4b2tau'][0], 1e-12), y['HHH->6b'][1]))

    a = d['HHH->6b']
    m = (a['kind_category_analysis'] == 2) & a['_trig'] & a['_pass'] & (a['nbtags'] >= 4)
    v, c = np.unique(a['tau1genPartFlav'][m], return_counts=True)
    print('\n6b tau1genPartFlav in 1tau0l (raw): %s  -> the leakage is jet fakes (flav 0)'
          % dict(zip(v.tolist(), c.tolist())))


if __name__ == '__main__':
    main()
