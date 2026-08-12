#!/usr/bin/env python3
"""Per-bin uncertainty of the two fake factors, for a few example variables.

Reads the ROOT file written by closure_v29pre_muffin_1tau0l_20bin_split.py with
USE_MUFFIN=1, which stores both fake-tau templates and both up/down bands:

  *_faketau            MUFFIN template            *_faketau_up/_dn  its bootstrap band
  *_faketau_binned     binned-map template        *_faketau_binned_up/_dn  its map-error band

and prints, per bin of the variable, the fake-tau yield of each method with its
own uncertainty, plus Data/Pred for each.  The uncertainty quoted is on the
FAKE-TAU COMPONENT, which is what the fake factor controls (the MC-prompt part
is common to both methods).

    bash muffin/run.sh perbin_table.py out/perbin_vr tau1Pt ht jet1Pt
"""
import os
import sys

import numpy as np
import uproot


def band(f, base, suf):
    """(nominal, |up-nom|, |dn-nom|) for one template."""
    nom = f[base + suf].values()
    up = f[base + suf + '_up'].values()
    dn = f[base + suf + '_dn'].values()
    return nom, np.abs(up - nom), np.abs(nom - dn)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'out/perbin_vr'
    if not os.path.isabs(d):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), d)
    variables = sys.argv[2:] or ['tau1Pt', 'ht', 'jet1Pt']
    path = os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith('.root'))[0])
    f = uproot.open(path)
    keys = set(k.split(';')[0] for k in f.keys())

    for var in variables:
        base = 'inclusive_1tau0l_%s_' % var
        if base + 'faketau' not in keys:
            print('!! %s not in %s' % (var, path))
            continue
        edges = f[base + 'data'].axis().edges()
        data = f[base + 'data'].values()
        prompt = (f[base + 'mc_prompt_total'].values()
                  if base + 'mc_prompt_total' in keys else np.zeros_like(data))
        mu, mu_up, mu_dn = band(f, base, 'faketau')
        bi, bi_up, bi_dn = band(f, base, 'faketau_binned')

        print('\n=== %s ===' % var)
        print('%-18s %8s %9s %-18s %-18s %8s %8s'
              % ('range', 'data', 'MC prompt', 'FakeTau MUFFIN',
                 'FakeTau binned', 'D/P muf', 'D/P bin'))
        print('-' * 100)
        for i in range(len(data)):
            if data[i] < 25:      # bins with a handful of events say nothing
                continue
            em = 0.5 * (mu_up[i] + mu_dn[i])
            eb = 0.5 * (bi_up[i] + bi_dn[i])
            pm, pb = prompt[i] + mu[i], prompt[i] + bi[i]
            print('%7.0f-%-9.0f %8.0f %9.1f  %8.1f +-%5.1f (%4.1f%%)  %8.1f +-%5.1f (%4.1f%%) %8.3f %8.3f'
                  % (edges[i], edges[i + 1], data[i], prompt[i],
                     mu[i], em, 100 * em / max(mu[i], 1e-9),
                     bi[i], eb, 100 * eb / max(bi[i], 1e-9),
                     data[i] / max(pm, 1e-9), data[i] / max(pb, 1e-9)))
        # yield-weighted means over the printed bins
        m = data >= 25
        wm = data[m] / data[m].sum()
        rel_m = 0.5 * (mu_up[m] + mu_dn[m]) / np.maximum(mu[m], 1e-9)
        rel_b = 0.5 * (bi_up[m] + bi_dn[m]) / np.maximum(bi[m], 1e-9)
        rm = data[m] / np.maximum(prompt[m] + mu[m], 1e-9)
        rb = data[m] / np.maximum(prompt[m] + bi[m], 1e-9)
        print('-' * 100)
        print('%-18s %8s %9s  mean unc %5.1f%%          mean unc %5.1f%%       '
              '|1-r| %5.1f%% %5.1f%%'
              % ('yield-weighted', '', '', 100 * (rel_m * wm).sum(),
                 100 * (rel_b * wm).sum(), 100 * (np.abs(1 - rm) * wm).sum(),
                 100 * (np.abs(1 - rb) * wm).sum()))


if __name__ == '__main__':
    main()
