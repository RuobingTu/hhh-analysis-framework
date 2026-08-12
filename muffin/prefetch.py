#!/usr/bin/env python3
"""Fill the flat-array cache for a region so later runs start instantly.

    bash muffin/run.sh prefetch.py AR
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muffin_common as mc  # noqa: E402

for region in (sys.argv[1:] or ['DR', 'AR']):
    print('=== %s ===' % region)
    res = mc.load_region(region)
    mc.summarise(res, region)
