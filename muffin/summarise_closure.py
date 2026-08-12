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


def _short(base):
    """Strip the channel/region prefix the plotters put on every histogram."""
    for pfx in ('inclusive_1tau0l_', '2tau0l_'):
        if base.startswith(pfx):
            return base[len(pfx):]
    return base


def curves(f, key):
    h = f[key]
    v = h.values()
    e = h.errors()
    return v, e


def _resolve(d):
    """Accept either a directory or the ROOT file itself."""
    if not os.path.isabs(d):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), d)
    if d.endswith('.root'):
        return d
    # a run still in progress has an open, empty file -- take the largest
    cands = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.root')]
    return max(cands, key=os.path.getsize)


def collect(d):
    """{variable: (muffin, binned)} of yield-weighted mean |1 - Data/Pred|."""
    path = _resolve(d)
    f = uproot.open(path)
    keys = set(k.split(';')[0] for k in f.keys())
    out = {}
    for k in sorted(keys):
        if not k.endswith('_ratio'):
            continue
        base = k[:-len('_ratio')]
        if base + '_ratio_binned' not in keys or base + '_data' not in keys:
            continue
        rm, _ = curves(f, k)
        rb, _ = curves(f, base + '_ratio_binned')
        dat, _ = curves(f, base + '_data')
        m = (dat >= MIN_BIN) & (rm > 0) & (rb > 0)
        if m.sum() < 3 or dat[m].sum() <= 0:
            continue
        w = dat[m] / dat[m].sum()
        out[_short(base)] = (
            float((np.abs(1 - rm[m]) * w).sum()), float((np.abs(1 - rb[m]) * w).sum()))
    return out


def compare(d1, d2):
    """Two MUFFIN trainings side by side, against the same binned baseline."""
    a, b = collect(d1), collect(d2)
    common = sorted(set(a) & set(b))
    rows = [(v, a[v][0], b[v][0], a[v][1], a[v][0] - b[v][0]) for v in common]
    rows.sort(key=lambda r: -r[4])
    print('%-46s %9s %9s %9s %8s' % ('variable', os.path.basename(d1)[-12:],
                                     os.path.basename(d2)[-12:], 'binned', 'gain'))
    print('-' * 88)
    def _row(r):
        name, m1, m2, bb, g = r
        print('%-46s %8.2f%% %8.2f%% %8.2f%% %+7.2f%%'
              % (name, 100 * m1, 100 * m2, 100 * bb, 100 * g))
    for r in rows[:15]:
        _row(r)
    print('   ... %d in between ...' % max(0, len(rows) - 30))
    for r in rows[-15:]:
        _row(r)
    arr = np.array([(r[1], r[2], r[3]) for r in rows])
    print('-' * 88)
    print('%-46s %8.2f%% %8.2f%% %8.2f%%'
          % ('MEAN over %d variables' % len(rows), 100 * arr[:, 0].mean(),
             100 * arr[:, 1].mean(), 100 * arr[:, 2].mean()))
    phi = [r for r in rows if 'Phi' in r[0] or r[0].startswith('phi')]
    if phi:
        p = np.array([(r[1], r[2], r[3]) for r in phi])
        print('%-46s %8.2f%% %8.2f%% %8.2f%%'
              % ('  of which phi-type (%d)' % len(phi), 100 * p[:, 0].mean(),
                 100 * p[:, 1].mean(), 100 * p[:, 2].mean()))


def chi2_mode(d):
    """Low-statistics comparison: chi2 of each prediction against the data,
    using the data Poisson error only (identical for both methods, so the two
    chi2 are directly comparable).  With ~10 events per bin the |1 - r| metric
    just measures the data statistics, not the fake factor."""
    path = _resolve(d)
    f = uproot.open(path)
    keys = set(k.split(';')[0] for k in f.keys())
    rows = []
    for k in sorted(keys):
        if not k.endswith('_ratio'):
            continue
        base = k[:-len('_ratio')]
        if base + '_ratio_binned' not in keys or base + '_data' not in keys:
            continue
        dat = f[base + '_data'].values()
        rm = f[base + '_ratio'].values()
        rb = f[base + '_ratio_binned'].values()
        m = (dat > 0) & (rm > 0) & (rb > 0)
        if m.sum() < 3:
            continue
        pm, pb = dat[m] / rm[m], dat[m] / rb[m]     # pred = data / ratio
        var = np.maximum(dat[m], 1.0)               # Poisson on the data
        cm = float(((dat[m] - pm) ** 2 / var).sum() / m.sum())
        cb = float(((dat[m] - pb) ** 2 / var).sum() / m.sum())
        rows.append((_short(base), cm, cb, cb - cm, int(m.sum()),
                     float(dat[m].sum()), float(pm.sum()), float(pb.sum())))
    rows.sort(key=lambda r: -r[3])
    print('%s\n%d variables\n' % (path, len(rows)))
    print('chi2/ndf against the data (Poisson errors); integral data/pred')
    print('%-40s %8s %8s %6s %9s %9s' % ('variable', 'MUFFIN', 'binned', 'nbin',
                                         'D/P muf', 'D/P bin'))
    print('-' * 86)
    for r in rows[:12] + [None] + rows[-12:]:
        if r is None:
            print('   ... %d in between ...' % max(0, len(rows) - 24))
            continue
        n, cm, cb, _g, nb, dd, pm, pb = r
        print('%-40s %8.2f %8.2f %6d %9.3f %9.3f'
              % (n, cm, cb, nb, dd / max(pm, 1e-9), dd / max(pb, 1e-9)))
    a = np.array([(r[1], r[2]) for r in rows])
    print('-' * 86)
    print('%-40s %8.2f %8.2f   MUFFIN better in %d/%d (%.0f%%)'
          % ('MEAN chi2/ndf', a[:, 0].mean(), a[:, 1].mean(),
             int((a[:, 0] < a[:, 1]).sum()), len(rows),
             100.0 * (a[:, 0] < a[:, 1]).sum() / len(rows)))


def main():
    if os.environ.get('CHI2'):
        return chi2_mode(sys.argv[1])
    if len(sys.argv) > 2:
        return compare(sys.argv[1], sys.argv[2])
    d = sys.argv[1] if len(sys.argv) > 1 else 'out/closure_muffin_vr'
    path = _resolve(d)
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
        rows.append((_short(base), am, ab, ab - am,
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
