#!/bin/bash
# Collect the beamer figures from the plot directories in the main checkout.
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework
FIG=$HERE/beamer/figs
mkdir -p "$FIG"

VARS="ht nsmalljets nfatjets tau1Pt jet1Pt"
for v in $VARS; do
  cp -f "$SRC/plots_v29pre_1tau0l_muffin_inclusive/closure_inclusive_1tau0l_${v}_log.png" "$FIG/1tau0l_${v}.png"
  cp -f "$SRC/plots_v29pre_2tau0l_muffin_vs_map/closure_2tau0l_${v}_log.png"              "$FIG/2tau0l_${v}.png"
  cp -f "$SRC/plots_v29pre_2tau0l_muffin_sideband/closure_2tau0l_${v}_log.png"            "$FIG/2tau0lsb_${v}.png"
  cp -f "$SRC/plots_v29pre_1tau1l_fakelep_20bin_split/closure_inclusive_1tau1l_${v}_log.png" "$FIG/1tau1lfl_${v}.png"
done
# 1tau1l also carries the lepton
cp -f "$SRC/plots_v29pre_1tau1l_fakelep_20bin_split/closure_inclusive_1tau1l_lep1Pt_log.png" "$FIG/1tau1lfl_lep1Pt.png"
# the SPANet score slide
cp -f "$SRC/plots_v29pre_1tau0l_muffin_inclusive/closure_inclusive_1tau0l_ProbHHH4b2tau_v29wmirror_ep201_log.png" \
      "$FIG/1tau0l_spanet.png"

# the 1tau1l MUFFIN run writes into the worktree
M=$HERE/out/agreement_muffin_1tau1l
if [ -d "$M" ]; then
  for v in $VARS lep1Pt; do
    f="$M/closure_inclusive_1tau1l_${v}_log.png"
    [ -f "$f" ] && cp -f "$f" "$FIG/1tau1l_${v}.png"
  done
fi
ls "$FIG" | wc -l
