#!/usr/bin/env python3
"""Is the njet slope in the 2tau0l sideband closure a jet-bookkeeping artefact?

In 1tau0l the anti-ID tau's mother jet stays in the jet collection while the
tight tau's is removed by the analysisTaus overlap veto, so the anti-ID region
sits one jet high (the documented FR_AST effect).  If 2tau0l vetoes both channel
taus regardless of their ID -- as the framework notes say -- that offset should
NOT exist here, and the njet slope must have another cause.

Direct test: the mean jet count in the three tau-ID regions.  Bookkeeping alone
would put (anti,anti) two jets above (tight,tight) and one above single-tight.
Anything much smaller means the veto is doing its job.

The tau's mother jet is also identified explicitly (tau_jetPt matched against
the jet collection) to see whether it is present in each region.

    bash muffin/run.sh njet_veto_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sideband_2tau0l import load  # noqa: E402


def main():
    res = load()
    d = ~res['is_mc']
    TT_ = res['t1_pass'] & res['t2_pass'] & d
    ONE = res['t1_pass'] & res['t2_anti'] & d
    LL = res['t1_anti'] & res['t2_anti'] & d
    nj, nb, ht = res['nsmalljets'], res['nbtags'], res['ht']

    print('=== mean jet counts per tau-ID region (data) ===')
    print('  %-16s %6s %10s %10s %10s   anti-ID taus'
          % ('region', 'N', '<njet>', '<nbtag>', '<HT>'))
    ref = None
    for lbl, m, na in (('(tight,tight)', TT_, 0), ('(tight,anti)', ONE, 1),
                       ('(anti,anti)', LL, 2)):
        n = int(m.sum())
        print('  %-16s %6d %10.3f %10.3f %10.1f   %d'
              % (lbl, n, nj[m].mean(), nb[m].mean(), ht[m].mean(), na))
        if ref is None:
            ref = (nj[m].mean(), nb[m].mean())
    print('\n  offsets vs (tight,tight):')
    for lbl, m, na in (('(tight,anti)', ONE, 1), ('(anti,anti)', LL, 2)):
        e = np.sqrt(nj[m].var() / max(m.sum(), 1) + nj[TT_].var() / max(TT_.sum(), 1))
        print('    %-14s d<njet> = %+.3f +- %.3f   (bookkeeping alone would give %+d)'
              % (lbl, nj[m].mean() - ref[0], e, na))

    # is the tau's mother jet inside the counted collection?  a counted jet has
    # pT > 25 and |eta| < 2.4; compare the tau-jet kinematics between regions
    print('\n=== is the anti-ID tau\'s mother jet countable? ===')
    for t, lbl in ((1, 'tau1'), (2, 'tau2')):
        jp, et = res['t%d_jetPt' % t], np.abs(res['t%d_eta' % t])
        countable = (jp > 25) & (et < 2.4)
        for rl, m in (('(tight,tight)', TT_), ('(tight,anti)', ONE),
                      ('(anti,anti)', LL)):
            print('  %-6s in %-14s: mother jet passes (pT>25, |eta|<2.4) for '
                  '%5.1f%% of events' % (lbl, rl, 100 * countable[m].mean()))

    # the njet slope of the closure: where does the sideband sit in njet?
    print('\n=== njet spectra (data) ===')
    print('  njet   (tight,tight)   (tight,anti)   (anti,anti)')
    for k in range(4, 9):
        sel = (nj == k) if k < 8 else (nj >= 8)
        print('  %-5s %10d %14d %14d'
              % (k if k < 8 else '>=8', int((TT_ & sel).sum()),
                 int((ONE & sel).sum()), int((LL & sel).sum())))


if __name__ == '__main__':
    main()
