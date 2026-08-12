#!/usr/bin/env python3
"""Total yields: data vs (prompt MC + fake tau), for both fake factors.

Reads the closure ROOT files and integrates.  Every variable in those files
carries its own zero-filter and axis range, so the totals differ slightly from
one to the next; the variable that retains the most data is used, and the spread
over the ten most-populated variables is quoted as the bookkeeping ambiguity so
it is not mistaken for a physics uncertainty.

The quoted uncertainty on each prediction is that method's own band, integrated:
MUFFIN's bootstrap RMS and the map's bin errors, both as coherent envelopes.

    bash muffin/run.sh total_yields.py \\
        out/closure_muffin_vr out/closure_muffin_incl out/closure_muffin_2tau0l
"""
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarise_closure import _resolve, _short  # noqa: E402


def totals(path):
    f = uproot.open(path)
    keys = set(k.split(';')[0] for k in f.keys())
    rows = []
    for k in sorted(keys):
        if not k.endswith('_data'):
            continue
        b = k[:-len('_data')]
        # 1tau0l names the templates _faketau/_faketau_binned, 2tau0l
        # _fake_muf/_fake_binned; the up/down suffixes differ too
        schemes = [('_faketau', '_faketau_up', '_faketau_dn',
                    '_faketau_binned', '_faketau_binned_up', '_faketau_binned_dn'),
                   ('_fake_muf', '_fake_muf_up', '_fake_muf_down',
                    '_fake_binned', '_fake_binned_up', '_fake_binned_dn')]
        sch = next((x for x in schemes
                    if b + x[0] in keys and b + x[3] in keys), None)
        if sch is None:
            continue
        fk, fk_up, fk_dn, fkb, fkb_up, fkb_dn = sch
        d = f[k].values().sum()
        prompt = (f[b + '_mc_prompt_total'].values().sum()
                  if b + '_mc_prompt_total' in keys else 0.0)
        mu = f[b + fk].values().sum()
        bi = f[b + fkb].values().sum()

        def bandsum(suf_up, suf_dn, nom):
            if b + suf_up not in keys or b + suf_dn not in keys:
                return float('nan')     # not stored by that run
            u = f[b + suf_up].values().sum()
            v = f[b + suf_dn].values().sum()
            return 0.5 * (abs(u - nom) + abs(nom - v))
        emu = bandsum(fk_up, fk_dn, mu)
        ebi = bandsum(fkb_up, fkb_dn, bi)
        rows.append((_short(b), d, prompt, mu, emu, bi, ebi))
    rows.sort(key=lambda r: -r[1])
    return rows


def report(label, path):
    rows = totals(path)
    if not rows:
        print('%s: nothing to read in %s' % (label, path))
        return
    name, d, p, mu, emu, bi, ebi = rows[0]
    top = np.array([r[1] for r in rows[:10]])
    print('\n=== %s ===' % label)
    print('  (from "%s", the variable retaining the most data; the ten most '
          'populated\n   span %.0f-%.0f data events, i.e. %.1f%% zero-filter '
          'ambiguity)' % (name, top.min(), top.max(),
                          100 * (top.max() - top.min()) / max(top.max(), 1)))
    print('  %-26s %10s' % ('observed data', '%.0f' % d))
    print('  %-26s %10s' % ('prompt MC (genuine tau)', '%.1f' % p))
    for tag, fk, e in (('MUFFIN', mu, emu), ('binned map', bi, ebi)):
        tot = p + fk
        r = d / tot if tot > 0 else 0
        # the band is on the fake component only
        rel = e / tot if tot > 0 else 0
        if np.isnan(e):
            print('  %-26s fake %8.1f  (band not stored)  total %8.1f   '
                  'data/pred %6.3f' % (tag, fk, tot, r))
        else:
            print('  %-26s fake %8.1f +- %6.1f   total %8.1f   data/pred %6.3f +- %.3f'
                  % (tag, fk, e, tot, r, r * rel))
    print('  %-26s %+.1f events (%+.1f%% of the total prediction)'
          % ('MUFFIN - binned', mu - bi, 100 * (mu - bi) / max(p + bi, 1e-9)))


def main():
    args = sys.argv[1:] or ['out/closure_muffin_vr', 'out/closure_muffin_incl',
                            'out/closure_muffin_2tau0l']
    for a in args:
        report(os.path.basename(a.rstrip('/')), _resolve(a))


if __name__ == '__main__':
    main()
