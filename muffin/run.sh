#!/bin/bash
# Run any MUFFIN script inside the LCG view that provides xgboost + uproot + ROOT.
#   bash muffin/run.sh train_muffin.py --features full --bootstrap 20
#   bash muffin/run.sh closure_muffin.py --features full
set -e
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
source $LCG
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 -u "$HERE/$@"
