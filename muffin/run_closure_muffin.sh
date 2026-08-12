#!/bin/bash
# Full-variable closure plots with the MUFFIN fake factor, in the analysis'
# own plot format, with the binned map's Data/Pred overlaid in the ratio pad.
#
#   bash muffin/run_closure_muffin.sh vr    [--variable ht]
#   bash muffin/run_closure_muffin.sh incl  [--chunk 0/8]
#
# vr   = validation region, jet4DeepFlavB < 0.1  (MUFFIN is NOT trained here)
# incl = inclusive, no jet4DeepFlavB cut         (contains the training region)
set -e
REGION=${1:?usage: run_closure_muffin.sh vr|incl [extra args]}
shift || true

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

case "$REGION" in
  vr)   export EXTRA_CUT='jet4DeepFlavB < 0.1' ;;
  incl) export EXTRA_CUT='1' ;;
  *) echo "region must be vr or incl" >&2; exit 2 ;;
esac

# MUFFIN on; the binned map is still evaluated -- it is the comparison curve.
export USE_MUFFIN=1
export USE_FR2D=1
# pass = VSjet >= 8, i.e. the WP MUFFIN was trained at (anti-ID window [2, 8))
export TAU_TIGHT_WP=loose
# the current 1tau0l baseline map: x = tau1jetPt, y = |tau1Eta|, njet integrated
export FR2D_ROOT=${FR2D_ROOT:-/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/fr_flavour2d_1tau0l_all-mr-eta_nonj_2017.root}
export FR2D_YVAR=abseta
# MUFFIN_TAG picks which trained model to apply (poster, poster_nophi, ...).
# The closure script reads the feature list off the header, so a different
# feature set needs nothing else changed here.
MUFFIN_TAG=${MUFFIN_TAG:-poster}
SUFFIX=""
[ "$MUFFIN_TAG" = "poster" ] || SUFFIX="_${MUFFIN_TAG#poster_}"
export MUFFIN_HEADER=${MUFFIN_HEADER:-$HERE/out/muffin_$MUFFIN_TAG.h}
export CLOSURE_OUTDIR=${CLOSURE_OUTDIR:-$HERE/out/closure_muffin${SUFFIX}_$REGION}

exec python3 -u "$REPO/closure_v29pre_muffin_1tau0l_20bin_split.py" "$@"
