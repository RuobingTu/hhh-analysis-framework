#!/bin/bash
# ===========================================================================
# Post-skimm hadd: merge _partN output files back into single per-sample ROOT files
# ===========================================================================
#
# After splitting large input files and running skimm_tree.py, the output
# directories contain per-part files (e.g., HHHTo4B2Tau_part0.root, _part1.root, ...).
# This script hadds them back into a single file per sample per category.
#
# Usage:
#   bash hadd_skimm_parts.sh           # hadd all categories
#   bash hadd_skimm_parts.sh --check   # check which categories need hadd
#   bash hadd_skimm_parts.sh --cat ProbHHH4b2tau_0bh0h_1tau1l_inclusive  # single category

set -e

BASE=/eos/user/r/rtu/TurbOutputMC2017_v9_with_corr_ak8_option92_2017/test_skimm

# Samples that were split (base names without _partN)
SPLIT_SAMPLES=(
    "HHHTo4B2Tau"
    "TTTo2L2Nu"
)

hadd_category() {
    local catdir="$1"
    local catname=$(basename "$catdir")

    for sample in "${SPLIT_SAMPLES[@]}"; do
        parts=($(ls "${catdir}/${sample}_part"*.root 2>/dev/null || true))
        nparts=${#parts[@]}

        if [ ${nparts} -eq 0 ]; then
            # No parts found — either already hadded or sample not present
            continue
        fi

        merged="${catdir}/${sample}.root"

        # The first Snapshot in skimm_tree.py overwrites the merged file with only
        # the last part's data. Remove the broken merged file before hadd.
        if [ -f "$merged" ]; then
            rm "$merged"
            echo "  Removed broken ${catname}/${sample}.root (only had last part)"
        fi

        echo "  HADD ${catname}/${sample}.root from ${nparts} parts..."
        hadd -f "$merged" "${parts[@]}"

        # Verify merged file
        if [ -f "$merged" ]; then
            # Remove part files after successful merge
            for p in "${parts[@]}"; do
                rm "$p"
                echo "    Removed: $(basename $p)"
            done
        else
            echo "  ERROR: hadd failed for ${catname}/${sample}"
        fi
    done
}

# --check mode
if [ "$1" == "--check" ]; then
    echo "=== Categories needing hadd ==="
    need=0
    ok=0
    for catdir in "${BASE}"/ProbHHH4b2tau_*_SR; do
        catname=$(basename "$catdir")
        for sample in "${SPLIT_SAMPLES[@]}"; do
            nparts=$(ls "${catdir}/${sample}_part"*.root 2>/dev/null | wc -l)
            if [ ${nparts} -gt 0 ]; then
                merged="${catdir}/${sample}.root"
                if [ -f "$merged" ]; then
                    echo "  WARN: ${catname}/${sample} has both merged and ${nparts} parts"
                else
                    echo "  NEED: ${catname}/${sample} (${nparts} parts)"
                fi
                ((need++))
            fi
        done
    done
    if [ ${need} -eq 0 ]; then
        echo "  All categories already hadded!"
    fi
    exit 0
fi

# --cat mode: single category
if [ "$1" == "--cat" ]; then
    catdir="${BASE}/${2}_SR"
    if [ ! -d "$catdir" ]; then
        echo "ERROR: ${catdir} not found"
        exit 1
    fi
    echo "Processing: $2"
    hadd_category "$catdir"
    exit 0
fi

# Default: process all categories
echo "=== Hadd split parts for all categories ==="
for catdir in "${BASE}"/ProbHHH4b2tau_*_SR; do
    catname=$(basename "$catdir")
    hadd_category "$catdir"
done
echo ""
echo "Done!"
