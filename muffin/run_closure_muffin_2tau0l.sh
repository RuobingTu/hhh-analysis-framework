#!/bin/bash
# 2tau0l closure with the MUFFIN fake factor, binned map overlaid in the ratio.
#
#   bash muffin/run_closure_muffin_2tau0l.sh              # all variables
#   bash muffin/run_closure_muffin_2tau0l.sh --variable ht
#
# Both fake factors were measured in 1tau0l and are applied here unchanged, so
# this is a cross-channel transfer test for both, on equal terms.
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

export USE_MUFFIN=1
# 2tau0l holds ~110 tight-tight data events, so 20 uniform bins leave most of
# them empty; use the statistics-optimised edges the channel already has.
export BINS_JSON=${BINS_JSON:-/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/bins_2tau0l_v29pre.json}
# the 2tau0l script already defaults to the 1tau0l eta map with FR2D_YVAR=abseta
export FR2D_YVAR=${FR2D_YVAR:-abseta}
MUFFIN_TAG=${MUFFIN_TAG:-poster}
SUFFIX=""
[ "$MUFFIN_TAG" = "poster" ] || SUFFIX="_${MUFFIN_TAG#poster_}"
export MUFFIN_HEADER=${MUFFIN_HEADER:-$HERE/out/muffin_$MUFFIN_TAG.h}
[ "${SIDEBAND:-}" = "1" ] && SUFFIX="${SUFFIX}_sideband"
export CLOSURE_OUTDIR=${CLOSURE_OUTDIR:-$HERE/out/closure_muffin${SUFFIX}_2tau0l}

exec python3 -u "$REPO/closure_v29pre_muffin_2tau0l.py" "$@"
