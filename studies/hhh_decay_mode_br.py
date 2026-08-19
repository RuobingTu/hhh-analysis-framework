#!/usr/bin/env python3
"""SM branching fractions of the HHH final states, m_H = 125 GeV.

Which modes other than 4b2tau can populate a "4 b-jets + >=1 tau_h" selection?
The answer drives what has to be simulated (or argued away) in the note.
"""
from collections import Counter
from itertools import combinations_with_replacement as cwr
from math import factorial

BR = {'bb': 0.5824, 'WW': 0.2137, 'gg': 0.0817, 'tautau': 0.06272,
      'cc': 0.02891, 'ZZ': 0.02619, 'gaga': 0.00227, 'Zga': 0.00153,
      'mumu': 0.000218}
W_TO_TAUH = 0.1138 * 0.6482      # W -> tau nu, tau -> hadrons
Z_TO_TAUTAU = 0.03370

rows = []
for c in cwr(BR, 3):
    n = Counter(c)
    mult = factorial(3)
    for v in n.values():
        mult //= factorial(v)
    p = mult
    for m in c:
        p *= BR[m]
    rows.append(('+'.join(sorted(c)), p))
rows.sort(key=lambda x: -x[1])

print('%-24s %8s' % ('HHH final state', 'BR [%]'))
for name, p in rows[:12]:
    print('%-24s %8.3f' % (name, 100 * p))

b, w, t, z = BR['bb'], BR['WW'], BR['tautau'], BR['ZZ']
print('\n-- modes that can give 4 b-jets + >=1 genuine tau_h')
print('%-34s %8s %10s' % ('', 'BR [%]', 'vs 4b2tau'))
ref = 3 * b * b * t
for name, val in [('4b2tau (our signal)', ref),
                  ('4b + WW, >=1 W->tau_h nu', 3 * b * b * w * (1 - (1 - W_TO_TAUH) ** 2)),
                  ('2b + 2tau + WW (2 b-tags only)', 6 * b * t * w),
                  ('4b + ZZ, >=1 Z->tautau', 3 * b * b * z * (1 - (1 - Z_TO_TAUTAU) ** 2)),
                  ('2b + 4tau (2 b-tags only)', 3 * b * t * t)]:
    print('%-34s %8.3f %9.2f' % (name, 100 * val, val / ref))
print('\n-- modes with NO genuine tau (enter only via jet->tau_h fakes)')
for name, val in [('6b', b ** 3), ('4b + gg', 3 * b * b * BR['gg']),
                  ('4b + cc', 3 * b * b * BR['cc'])]:
    print('%-34s %8.3f %9.2f' % (name, 100 * val, val / ref))
