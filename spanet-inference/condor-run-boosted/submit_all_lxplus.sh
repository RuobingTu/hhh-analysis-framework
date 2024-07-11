universe              = vanilla
requirements          = (Arch == "X86_64") && (OpSys == "LINUX")
request_memory = 6000
RequestCpus =2
environment = "SINGULARITY_BINDPATH='/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security/vomses:/etc/vomses,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security'"
+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el7:x86_64"
use_x509userproxy     = true
x509userproxy         = /afs/cern.ch/user/r/rtu/private/x509up_u150678

executable_path= jobs/$(Item).sh
Log_path        = log/$(Item).log
Output_path     = output/$(Item).out
Error_path      = error/$(Item).error

Executable= $(executable_path)

should_transfer_files = YES
Notification = never
Arguments  = $(ClusterId) $(ProcId)
Log        = $(Log_path)
Output     = $(Output_path)
Error      = $(Error_path)
queue Item from /afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/spanet-inference/condor-run-boosted/job_list.txt
