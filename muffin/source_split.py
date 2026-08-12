#!/usr/bin/env python3
"""How much can splitting the fake factor by source (ttbar vs QCD) buy?

The premise of the source-split plan is that 1tau0l and 2tau0l differ in their
fake composition, so a single averaged fake factor measured in 1tau0l is wrong
when carried to 2tau0l.  The size of that effect is

    Delta(FF) = (f_QCD[2tau0l] - f_QCD[1tau0l]) * (FF_QCD - FF_ttbar)

so it is worth measuring both factors before building anything: if the two
component fake factors are similar, or the two compositions are, splitting buys
nothing no matter how well it is done.

Neither factor needs QCD MC (whose statistics are too small to be useful):
  FF_ttbar  from ttbar MC, which is plentiful
  FF_QCD    from data minus ALL MC, i.e. QCD inferred as the remainder
  f_ttbar   from ttbar MC over the data yield, per ID class

Taus are counted per object: in 2tau0l each event contributes both of its taus
as fake candidates, since the fake factor is a per-object weight.

    bash muffin/run.sh source_split.py
"""
import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402

INCL_TT = ('TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic')
TT = INCL_TT + ('TTbb_4f_TTTo',)

PLATEAU = None   # the pool branches already carry the channel's trigger cuts
COLS_BASE = ['kind_category_FR', 'trigSF_pfHT', 'trigSF_caloHT']
WCOLS = ['xsecWeight', 'genWeight', 'puWeight', 'l1PreFiringWeight',
         'btagWeight_shape', 'btagShapeR_weight', 'tauIDSF_weight',
         'triggerSF_perfilter_2nBtag_v24c', 'triggerLumiSF', 'genTtbarId']


def tau_cols(i):
    return ['tau%didDeepTau2017v2p1VSjet' % i, 'tau%dgenPartFlav' % i,
            'tau%djetPt' % i, 'tau%dEta' % i, 'tau%ddecayMode' % i]


def accumulate(channel, ntau):
    """Per-object pass/fail yields, split into data / ttbar-fake / other-MC-fake
    / MC-genuine, for the given channel's FR pool."""
    kc = 2 if channel == '1tau0l' else 0
    acc = {k: np.zeros(2) for k in ('data', 'tt_fake', 'oth_fake', 'genuine')}

    def fill(fn, is_mc):
        cols = COLS_BASE + sum((tau_cols(i) for i in range(1, ntau + 1)), [])
        if is_mc:
            cols = cols + WCOLS
        bnm = os.path.basename(fn)
        is_tt = any(p in bnm for p in TT)
        incl = any(p in bnm for p in INCL_TT) and 'TTbb_4f' not in bnm
        for a in uproot.iterate(fn + ':Events', cols, step_size='500 MB',
                                library='np'):
            ev = ((a['kind_category_FR'] == kc)
                  & (a['trigSF_pfHT'] >= 300) & (a['trigSF_caloHT'] >= 160))
            if is_mc:
                w = mc.LUMI * (a['xsecWeight'] * a['genWeight'] * a['puWeight']
                               * a['l1PreFiringWeight'] * a['btagWeight_shape']
                               * a['btagShapeR_weight'] * a['tauIDSF_weight']
                               * a['triggerSF_perfilter_2nBtag_v24c']
                               * a['triggerLumiSF'])
                if incl:
                    ev = ev & ((a['genTtbarId'] % 100) < 51)
            else:
                w = np.ones(ev.size)
            if not ev.any():
                continue
            for i in range(1, ntau + 1):
                v = a['tau%didDeepTau2017v2p1VSjet' % i]
                cand = ev & (v >= 2)                 # in the loose pool
                if not cand.any():
                    continue
                idx = np.where(cand)[0]
                p = (v[idx] >= mc.TAU_PASS).astype(int)   # 1 = pass, 0 = fail
                ww = w[idx]
                if not is_mc:
                    np.add.at(acc['data'], p, ww)
                else:
                    gen = a['tau%dgenPartFlav' % i][idx] == 5
                    np.add.at(acc['genuine'], p[gen], ww[gen])
                    key = 'tt_fake' if is_tt else 'oth_fake'
                    np.add.at(acc[key], p[~gen], ww[~gen])

    fill(mc.BASE + '/data/parts/BTagCSV_tree.root', False)
    for fn in sorted(glob.glob(mc.BASE + '/mc/parts/*_tree.root')):
        if os.path.basename(fn).startswith('QCD'):
            continue                                  # QCD is what we infer
        fill(fn, True)
    return acc


def report(channel, ntau):
    a = accumulate(channel, ntau)
    # QCD = data - genuine - non-QCD fakes, per ID class
    qcd = a['data'] - a['genuine'] - a['tt_fake'] - a['oth_fake']
    fake_tot = a['tt_fake'] + a['oth_fake'] + qcd
    print('\n=== %s : per-tau yields in the FR pool ===' % channel)
    print('  %-18s %10s %10s   FF = pass/fail' % ('', 'fail', 'pass'))
    for k, lbl in (('data', 'data'), ('genuine', 'MC genuine tau'),
                   ('tt_fake', 'ttbar fake'), ('oth_fake', 'other MC fake')):
        v = a[k]
        print('  %-18s %10.1f %10.1f   %s'
              % (lbl, v[0], v[1],
                 '%.4f' % (v[1] / v[0]) if v[0] > 0 else '-'))
    print('  %-18s %10.1f %10.1f   %.4f   <- inferred'
          % ('QCD fake', qcd[0], qcd[1], qcd[1] / qcd[0] if qcd[0] > 0 else np.nan))
    print('  %-18s %10.1f %10.1f   %.4f'
          % ('all fakes', fake_tot[0], fake_tot[1], fake_tot[1] / fake_tot[0]))
    f_tt = a['tt_fake'][0] / fake_tot[0]
    f_qcd = qcd[0] / fake_tot[0]
    print('  composition of the FAIL (anti-ID) population:'
          '  ttbar %.3f   QCD %.3f   other %.3f'
          % (f_tt, f_qcd, a['oth_fake'][0] / fake_tot[0]))
    return dict(ff_tt=a['tt_fake'][1] / a['tt_fake'][0],
                ff_qcd=qcd[1] / qcd[0], f_qcd=f_qcd, f_tt=f_tt,
                ff_all=fake_tot[1] / fake_tot[0])


def main():
    one = report('1tau0l', 1)
    two = report('2tau0l', 2)
    d_ff = one['ff_qcd'] - one['ff_tt']
    d_f = two['f_qcd'] - one['f_qcd']
    print('\n=== how much can a source split buy? ===')
    print('  FF_QCD - FF_ttbar        (in 1tau0l)  %+.4f  (%.1f%% of FF_all)'
          % (d_ff, 100 * d_ff / one['ff_all']))
    print('  f_QCD[2tau0l] - f_QCD[1tau0l]         %+.4f' % d_f)
    print('  => composition-induced FF shift       %+.4f  = %+.1f%% of FF_all'
          % (d_ff * d_f, 100 * d_ff * d_f / one['ff_all']))
    print('\n  measured inclusive FF:  1tau0l %.4f   2tau0l %.4f   (%.1f%% apart)'
          % (one['ff_all'], two['ff_all'],
             100 * (two['ff_all'] / one['ff_all'] - 1)))
    print('  source-split prediction for 2tau0l:   %.4f'
          % (two['f_tt'] * one['ff_tt'] + two['f_qcd'] * one['ff_qcd']
             + (1 - two['f_tt'] - two['f_qcd']) * one['ff_all']))


if __name__ == '__main__':
    main()
