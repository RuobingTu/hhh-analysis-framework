#!/bin/bash
# ===========================================================================
# Submit Condor Jobs to Merge SPANet v9 Inference Output Pieces
# ===========================================================================
#
# Merges parts_SPANET_v9/{sample}_*_N.root -> parts_SPANET_v9_merged/{sample}_tree.root
# One condor job per sample (30 total: 4 signal + 26 mc)
#
# Run from native lxplus9 shell, NOT inside cmssw-el7
#
# Usage:
#   bash submit_merge_v9.sh           # Submit merge jobs
#   bash submit_merge_v9.sh --check   # Check output status

set -e

BASEDIR=/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/spanet-inference/condor-run-boosted
OUTBASE=/eos/user/r/rtu/TurbOutputMC2017_v9_with_corr_ak8_option92_2017
JOBDIR=${BASEDIR}/jobs_merge_v9

if [ "$1" == "--check" ]; then
    echo "=== Merge output status ==="
    for subdir in signal mc; do
        echo "--- ${subdir} ---"
        outdir=${OUTBASE}/${subdir}/parts_SPANET_v9_merged
        if [ -d "$outdir" ]; then
            for f in ${outdir}/*_tree.root; do
                [ -f "$f" ] && echo "  OK: $(basename $f) ($(du -h $f | cut -f1))"
            done
        else
            echo "  (not yet created)"
        fi
    done
    exit 0
fi

# Create job directory
if [ -d "$JOBDIR" ]; then
    echo "Removing existing job directory $JOBDIR"
    rm -rf "$JOBDIR"
fi
mkdir -p ${JOBDIR}/logs

# Build job list: "sample_name subdir"
JOBLIST=${JOBDIR}/joblist.txt
> ${JOBLIST}

for subdir in signal mc; do
    piecesdir=${OUTBASE}/${subdir}/parts_SPANET_v9
    if [ ! -d "$piecesdir" ]; then
        echo "WARNING: $piecesdir does not exist, skipping"
        continue
    fi
    # Extract unique sample names (strip _N.root suffix)
    for sample in $(ls ${piecesdir}/*.root | xargs -n1 basename | sed 's/_[0-9]*\.root$//' | sort -u); do
        npieces=$(ls ${piecesdir}/${sample}_*.root 2>/dev/null | wc -l)
        echo "${sample} ${subdir} ${npieces}" >> ${JOBLIST}
    done
done

NJOBS=$(wc -l < ${JOBLIST})
echo "Generated ${NJOBS} merge jobs:"
cat ${JOBLIST} | awk '{printf "  [%d] %-80s %s (%s pieces)\n", NR-1, $1, $2, $3}'

# Create the merge worker script
cat > ${JOBDIR}/merge_worker.sh << 'WORKEREOF'
#!/bin/bash
set -e
JOBID=$1
JOBLIST=$2
OUTBASE=$3

LINE=$(sed -n "$((JOBID+1))p" ${JOBLIST})
SAMPLE=$(echo $LINE | awk '{print $1}')
SUBDIR=$(echo $LINE | awk '{print $2}')

PIECESDIR=${OUTBASE}/${SUBDIR}/parts_SPANET_v9
OUTDIR=${OUTBASE}/${SUBDIR}/parts_SPANET_v9_merged
mkdir -p ${OUTDIR}

OUTFILE=${OUTDIR}/${SAMPLE}_tree.root
INFILES=$(ls ${PIECESDIR}/${SAMPLE}_*.root 2>/dev/null | tr '\n' ' ')
NFILES=$(echo $INFILES | wc -w)

echo "Merging ${NFILES} pieces for ${SAMPLE} (${SUBDIR})"
echo "Output: ${OUTFILE}"

if [ "$NFILES" -eq 0 ]; then
    echo "ERROR: No pieces found"
    exit 1
fi

if [ "$NFILES" -eq 1 ]; then
    echo "Single file, copying directly"
    cp ${INFILES} ${OUTFILE}
else
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd /afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src
    eval $(scramv1 runtime -sh)

    # Two-stage merge for large samples (>200 pieces)
    if [ "$NFILES" -gt 200 ]; then
        echo "Large sample: two-stage merge"
        TMPDIR=$(mktemp -d)
        BATCH=0
        COUNT=0
        BATCHFILES=""
        for f in ${INFILES}; do
            BATCHFILES="${BATCHFILES} ${f}"
            COUNT=$((COUNT + 1))
            if [ "$COUNT" -eq 200 ]; then
                echo "  Stage 1: batch ${BATCH} (${COUNT} files)"
                hadd -f ${TMPDIR}/batch_${BATCH}.root ${BATCHFILES}
                BATCH=$((BATCH + 1))
                BATCHFILES=""
                COUNT=0
            fi
        done
        if [ "$COUNT" -gt 0 ]; then
            echo "  Stage 1: batch ${BATCH} (${COUNT} files)"
            hadd -f ${TMPDIR}/batch_${BATCH}.root ${BATCHFILES}
        fi
        echo "  Stage 2: merging batches"
        hadd -f ${OUTFILE} ${TMPDIR}/batch_*.root
        rm -rf ${TMPDIR}
    else
        hadd -f ${OUTFILE} ${INFILES}
    fi
fi

echo "Done: $(ls -lh ${OUTFILE} | awk '{print $5}') ${OUTFILE}"
WORKEREOF
chmod +x ${JOBDIR}/merge_worker.sh

# Create condor submit file
cat > ${JOBDIR}/merge_v9.sub << EOF
universe              = vanilla
+JobFlavour           = "longlunch"
request_memory        = 4000
RequestCpus           = 1

# Run inside SLC7 singularity so hadd from CMSSW_12_5_2 (SLC7) works on EL9 nodes
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64"

executable            = ${JOBDIR}/merge_worker.sh
arguments             = \$(ProcId) ${JOBDIR}/joblist.txt ${OUTBASE}

log                   = ${JOBDIR}/logs/merge_\$(ProcId).log
output                = ${JOBDIR}/logs/merge_\$(ProcId).out
error                 = ${JOBDIR}/logs/merge_\$(ProcId).err

should_transfer_files = NO
Notification          = never

queue ${NJOBS}
EOF

echo ""
echo "Submit with:"
echo "  condor_submit ${JOBDIR}/merge_v9.sub"
