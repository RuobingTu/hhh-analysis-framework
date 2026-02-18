#!/bin/bash
# ===========================================================================
# Submit Condor Jobs to Run skimm_tree.py in Parallel (40 categories)
# ===========================================================================
#
# Each category is independent, so we submit 40 condor jobs in parallel.
# Wall time: ~30 min per job (longlunch = 2 hour limit).
#
# Run from native lxplus9 shell, NOT inside cmssw-el7.
#
# Usage:
#   bash submit_skimm_v9.sh           # Submit all 40 jobs
#   bash submit_skimm_v9.sh --check   # Check output status
#   bash submit_skimm_v9.sh --resubmit 5 12 27  # Resubmit specific failed jobs

set -e

BASE=/eos/user/r/rtu/TurbOutputMC2017_v9_with_corr_ak8_option92_2017/test_skimm
FRAMEWORK=/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework
YEAR=2017
JOBDIR=${FRAMEWORK}/jobs_skimm_v9

# All 40 categories
CATEGORIES=(
    # 24 fine-grained: 6 topologies x 4 channels
    ProbHHH4b2tau_0bh0h_2tau0l_inclusive
    ProbHHH4b2tau_0bh0h_1tau1l_inclusive
    ProbHHH4b2tau_0bh0h_1tau0l_inclusive
    ProbHHH4b2tau_0bh0h_0tau2l_inclusive
    ProbHHH4b2tau_0bh1h_2tau0l_inclusive
    ProbHHH4b2tau_0bh1h_1tau1l_inclusive
    ProbHHH4b2tau_0bh1h_1tau0l_inclusive
    ProbHHH4b2tau_0bh1h_0tau2l_inclusive
    ProbHHH4b2tau_0bh2h_2tau0l_inclusive
    ProbHHH4b2tau_0bh2h_1tau1l_inclusive
    ProbHHH4b2tau_0bh2h_1tau0l_inclusive
    ProbHHH4b2tau_0bh2h_0tau2l_inclusive
    ProbHHH4b2tau_1bh0h_2tau0l_inclusive
    ProbHHH4b2tau_1bh0h_1tau1l_inclusive
    ProbHHH4b2tau_1bh0h_1tau0l_inclusive
    ProbHHH4b2tau_1bh0h_0tau2l_inclusive
    ProbHHH4b2tau_1bh1h_2tau0l_inclusive
    ProbHHH4b2tau_1bh1h_1tau1l_inclusive
    ProbHHH4b2tau_1bh1h_1tau0l_inclusive
    ProbHHH4b2tau_1bh1h_0tau2l_inclusive
    ProbHHH4b2tau_2bh0h_2tau0l_inclusive
    ProbHHH4b2tau_2bh0h_1tau1l_inclusive
    ProbHHH4b2tau_2bh0h_1tau0l_inclusive
    ProbHHH4b2tau_2bh0h_0tau2l_inclusive
    # 4 NHiggs inclusive
    ProbHHH4b2tau_3Higgs_inclusive
    ProbHHH4b2tau_2Higgs_inclusive
    ProbHHH4b2tau_1Higgs_inclusive
    ProbHHH4b2tau_0Higgs_inclusive
    # 12 NHiggs x channel
    ProbHHH4b2tau_3Higgs_sum_2tau0l
    ProbHHH4b2tau_3Higgs_sum_1tau1l
    ProbHHH4b2tau_3Higgs_sum_0tau2l
    ProbHHH4b2tau_2Higgs_sum_2tau0l
    ProbHHH4b2tau_2Higgs_sum_1tau1l
    ProbHHH4b2tau_2Higgs_sum_1tau0l
    ProbHHH4b2tau_2Higgs_sum_0tau2l
    ProbHHH4b2tau_1Higgs_sum_2tau0l
    ProbHHH4b2tau_1Higgs_sum_1tau1l
    ProbHHH4b2tau_1Higgs_sum_1tau0l
    ProbHHH4b2tau_1Higgs_sum_0tau2l
    ProbHHH4b2tau_0Higgs_sum_1tau0l
)

NCATS=${#CATEGORIES[@]}
echo "Total categories: ${NCATS}"

# --check mode: verify output
if [ "$1" == "--check" ]; then
    echo "=== Skimm output status ==="
    ok=0
    fail=0
    for cat in "${CATEGORIES[@]}"; do
        sig="${BASE}/${cat}_SR/HHHTo4B2Tau.root"
        if [ -f "$sig" ]; then
            echo "  OK: ${cat}"
            ((ok++))
        else
            echo "  MISSING: ${cat}"
            ((fail++))
        fi
    done
    echo ""
    echo "OK: ${ok}/${NCATS}, Missing: ${fail}/${NCATS}"
    if [ ${fail} -gt 0 ]; then
        echo ""
        echo "Check errors with: grep -l 'Error\|Traceback' ${JOBDIR}/logs/*.err"
    fi
    exit 0
fi

# --resubmit mode: resubmit specific failed jobs by index
if [ "$1" == "--resubmit" ]; then
    shift
    if [ $# -eq 0 ]; then
        echo "Usage: bash submit_skimm_v9.sh --resubmit <idx1> <idx2> ..."
        exit 1
    fi
    RESUB_LIST=${JOBDIR}/resubmit_joblist.txt
    > ${RESUB_LIST}
    for idx in "$@"; do
        echo "skimm_${idx}" >> ${RESUB_LIST}
        echo "  Will resubmit: [${idx}] ${CATEGORIES[$idx]}"
    done
    cat > ${JOBDIR}/resubmit.sub << EOF
universe              = vanilla
+JobFlavour           = "longlunch"
request_memory        = 6000
RequestCpus           = 1
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64"
should_transfer_files = NO
executable            = ${JOBDIR}/\$(Item).sh
output                = ${JOBDIR}/logs/\$(Item).out
error                 = ${JOBDIR}/logs/\$(Item).err
log                   = ${JOBDIR}/logs/\$(Item).log
queue Item from ${RESUB_LIST}
EOF
    echo ""
    echo "Submit with:"
    echo "  cd ${JOBDIR} && condor_submit resubmit.sub"
    exit 0
fi

# --- Main submission ---

# Create job directory
if [ -d "$JOBDIR" ]; then
    echo "Job directory $JOBDIR already exists. Remove it first or rename it."
    exit 1
fi
mkdir -p ${JOBDIR}/logs

# Generate worker scripts and joblist
JOBLIST=${JOBDIR}/joblist.txt
> ${JOBLIST}

for i in $(seq 0 $((NCATS - 1))); do
    cat="${CATEGORIES[$i]}"
    name="skimm_${i}"
    echo "${name}" >> ${JOBLIST}

    cat > ${JOBDIR}/${name}.sh << WORKEREOF
#!/bin/bash
set -e
echo "Starting category: ${cat} (job ${i}/${NCATS})"
echo "Hostname: \$(hostname)"
echo "Date: \$(date)"

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src
eval \$(scramv1 runtime -sh)
cd hhh-analysis-framework
export MYROOT=\$(pwd)

echo "Running skimm_tree.py for ${cat}..."
python3 skimm_tree.py \
    --base_folder ${BASE} \
    --category ${cat} \
    --year ${YEAR} \
    --do_SR \
    --skip_do_histograms \
    --skip_do_plots

echo "Done: ${cat}"
echo "Date: \$(date)"
WORKEREOF
    chmod +x ${JOBDIR}/${name}.sh
done

echo "Generated ${NCATS} worker scripts"

# Create condor submit file
cat > ${JOBDIR}/skimm.sub << EOF
universe              = vanilla
+JobFlavour           = "longlunch"
request_memory        = 6000
RequestCpus           = 1
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64"
should_transfer_files = NO

executable            = ${JOBDIR}/\$(Item).sh
output                = ${JOBDIR}/logs/\$(Item).out
error                 = ${JOBDIR}/logs/\$(Item).err
log                   = ${JOBDIR}/logs/\$(Item).log

queue Item from ${JOBLIST}
EOF

echo ""
echo "Job list:"
for i in $(seq 0 $((NCATS - 1))); do
    printf "  [%2d] %s\n" "$i" "${CATEGORIES[$i]}"
done
echo ""
echo "Submit with:"
echo "  cd ${JOBDIR} && condor_submit skimm.sub"
