#!/bin/bash
# 1tau1l agreement with the MUFFIN fake-tau transfer factor (measured in 1tau0l).
#   bash muffin/run_agreement_muffin_1tau1l.sh --variable ht,nsmalljets
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

export USE_MUFFIN=1
export USE_FR2D=1                       # needed to reach the declare/define path
export TAU_TIGHT_WP=loose               # the WP MUFFIN is trained at
export FR2D_ROOT=${FR2D_ROOT:-/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root}
export FR2D_YVAR=abseta
export FAKETAU=1
export MUFFIN_HEADER=${MUFFIN_HEADER:-$HERE/out/muffin_poster.h}
export CLOSURE_OUTDIR=${CLOSURE_OUTDIR:-$HERE/out/agreement_muffin_1tau1l}

exec python3 -u "$REPO/agreement_v29pre_muffin_1tau1l.py" "$@"
