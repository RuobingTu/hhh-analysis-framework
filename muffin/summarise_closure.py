#!/usr/bin/env python3
"""Rank the variables by how much MUFFIN changes the closure.

Reads the ROOT file written by closure_v29pre_muffin_1tau0l_20bin_split.py,
which stores both Data/Pred curves per variable (`*_ratio` = MUFFIN,
`*_ratio_binned` = the binned map), and reports for each variable the
yield-weighted mean |1 - Data/Pred| over its bins -- bins weighted by the data
they hold, so the tails do not dominate.

    bash muffin/run.sh summarise_closure.py out/closure_muffin_vr
"""
import os
import sys

import numpy as np
import uproot


MIN_BIN = float(os.environ.get('MIN_BIN', '25'))     # min data events per bin
SANE = float(os.environ.get('SANE', '0.25'))         # skip vars the binned map
                                                     # itself misses by more than this


def curves(f, key):
    h = f[key]
    v = h.values()
    e = h.errors()
    return v, e


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'out/closure_muffin_vr'
    if not os.path.isabs(d):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), d)
    roots = [f for f in os.listdir(d) if f.endswith('.root')]
    if not roots:
        sys.exit('no ROOT file in %s' % d)
    path = os.path.join(d, sorted(roots)[0])
    f = uproot.open(path)
    keys = set(k.split(';')[0] for k in f.keys())

    rows = []
    for k in sorted(keys):
        if not k.endswith('_ratio'):
            continue
        base = k[:-len('_ratio')]
        kb = base + '_ratio_binned'
        kd = base + '_data'
        if kb not in keys or kd not in keys:
            continue
        rm, _ = curves(f, k)
        rb, _ = curves(f, kb)
        dat, _ = curves(f, kd)
        # bins with a handful of events carry no information about the fake
        # factor, and variables where BOTH methods are wildly off (sparse
        # fatjet/jet-pair tails) would otherwise dominate any average
        m = (dat >= MIN_BIN) & (rm > 0) & (rb > 0)
        if m.sum() < 3 or dat[m].sum() <= 0:
            continue
        w = dat[m] / dat[m].sum()
        am = float((np.abs(1 - rm[m]) * w).sum())
        ab = float((np.abs(1 - rb[m]) * w).sum())
        rows.append((base.replace('inclusive_1tau0l_', ''), am, ab, ab - am,
                     float(dat[m].sum())))

    rows.sort(key=lambda r: -r[3])
    print('%s\n%d variables with both curves\n' % (path, len(rows)))
    print('yield-weighted mean |1 - Data/Pred|   (gain > 0 = MUFFIN closer to 1)')
    print('%-52s %8s %8s %8s' % ('variable', 'MUFFIN', 'binned', 'gain'))
    print('-' * 80)
    for name, am, ab, g, _n in rows[:20]:
        print('%-52s %7.2f%% %7.2f%% %+7.2f%%' % (name, 100 * am, 100 * ab, 100 * g))
    print('   ... %d variables in between ...' % max(0, len(rows) - 40))
    for name, am, ab, g, _n in rows[-20:]:
        print('%-52s %7.2f%% %7.2f%% %+7.2f%%' % (name, 100 * am, 100 * ab, 100 * g))

    print('-' * 80)
    for label, sel in (('ALL variables', rows),
                       ('variables the binned map models to better than %.0f%%' % (100 * SANE),
                        [r for r in rows if r[2] < SANE])):
        if not sel:
            continue
        a = np.array([(r[1], r[2]) for r in sel])
        better = int((a[:, 0] < a[:, 1]).sum())
        print('%-52s %7.2f%% %7.2f%% %+7.2f%%   MUFFIN better in %d/%d (%.0f%%)'
              % (label, 100 * a[:, 0].mean(), 100 * a[:, 1].mean(),
                 100 * (a[:, 1] - a[:, 0]).mean(), better, len(sel),
                 100.0 * better / len(sel)))


if __name__ == '__main__':
    main()
