# Scritp to prepare jobs for plotting with make_histograms_rdataframe.py

import os, glob  
import ROOT
import math


current_path = '/afs/cern.ch/user/r/rtu/CMSSW_12_5_2/src/hhh-analysis-framework/spanet-inference'
script = current_path + '/' + 'predict_spanet_classification_pnet_all_vars_ruobing_5-7.py'

pwd = current_path + '/' + 'condor-run-boosted/'
jobs_path = 'jobs'
job_list_name = 'job_list.txt'

#submit="Universe   = vanilla\nrequirements = (Arch == 'X86_64')\nrequirements = (OpSysAndVer =?= 'CentOS7')\nrequest_memory = 2048\nRequestCpus = 1\nExecutable = %s\nshould_transfer_files = YES\nNotification = never\nArguments  = $(ClusterId) $(ProcId)\nLog        = log/job_%s_%s_%s.log\nOutput     = output/job_%s_%s_%s.out\nError      = error/job_%s_%s_%s.error\nQueue 1"
submi_cmd = '''universe              = vanilla
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
queue Item from %s%s
''' % (pwd, job_list_name)

# create condor_submit script
submit_all = 'submit_all_lxplus.sh'
with open(submit_all,'w') as f:
    f.write(submi_cmd)

#manual_all = 'manual_run_all.sh'
#jobs = []
#manual_jobs = []
#for year in ['2016','2016APV','2017','2018']:
job_cmd = '#! /bin/bash\nsource /cvmfs/cms.cern.ch/cmsset_default.sh\ncmsrel CMSSW_12_5_2\ncd CMSSW_12_5_2/src\nxrdcp -f root://eosuser.cern.ch//eos/user/r/rtu/PNet_spanet_v621.onnx .\nxrdcp -f root://eosuser.cern.ch//eos/user/r/rtu/PNet_spanet_category_v621.onnx .\ncmsenv\nulimit -s unlimited\nexport MYROOT=$(pwd)\nexport PYTHONPATH=$PYTHONPATH:$MYROOT \n%s'
#job_cmd = '%s'
year = '2017'
version = 'v20'

# create jobs file
batch_size = 3000
if year in ['2016','2016APV','2017','2018']:
    #path_to_samples = '/eos/user/r/rtu/eos-triple-h_weight_maybeWrong/%s-parts-no-lhe/mva-inputs-%s/inclusive-weights/'%(version,year)
    #path_to_samples = "/eos/user/r/rtu/Turb507Outputdata2017_ak8_option92_2017/1/v20-parts-no-lhe/mva-inputs-2017/inclusive-weights/"
    path_to_samples = "/eos/user/r/rtu/Turb607OutputMC2017_ak8_option92_2017/v20-parts-no-lhe/mva-inputs-2017/inclusive-weights/"
    files = glob.glob(path_to_samples+'*.root')
    samples = [os.path.basename(s).replace('.root','') for s in files]
    for i in range(len(files)):
        f_in = samples[i]
        f = files[i]
        df = ROOT.RDataFrame("Events", f)
        entries = df.Count().GetValue()
        n_batches = math.floor(float(entries)/batch_size)
        for j in range(n_batches+1):
            filename = 'job_%s_%s_%d.sh'%(f_in,year,j)
            cmd = 'python3 %s --f_in %s --year %s --batch_size %d --batch_number %d'%(script,f_in,year,batch_size,j)

            print("Writing %s"%filename)
            with open(jobs_path + '/' + filename, 'w') as f:
                f.write(job_cmd%cmd)
            '''
            manual_jobs.append(filename)

            submit_file = 'submit_%s_%s_%s'%(f_in,year,j)
            print('Writing %s/%s'%(jobs_path, submit_file))
            with open(jobs_path + '/' + submit_file, 'w') as f:
                f.write(submit%(jobs_path+'/'+filename,f_in,year,j,f_in,year,j,f_in,year,j))
            jobs.append(submit_file)
            '''
# create jobs list used for submitting job
def get_files_with_suffix(directory, suffix):
    file_list = [f[:-3] for f in os.listdir(directory) if f.endswith(suffix)]
    return file_list

joblist = get_files_with_suffix("%s/%s" % (pwd, jobs_path), ".sh")


with open("%s/%s" % (pwd, job_list_name), 'w') as file:
    for item in joblist:
        line_txt = "%s" % (item)
        file.write(line_txt + '\n')

'''
cmd = '#!/bin/bash\n'
for j in jobs:
    cmd += 'condor_submit %s/%s \n'%(jobs_path, j)


with open(submit_all, 'w') as f:
    f.write(cmd)


cmd = '#!/bin/bash\n'
for f in manual_jobs:
    cmd += 'source %s/%s/%s\n'%(pwd,jobs_path,f)

with open(manual_all,'w') as f:
    f.write(cmd)
'''
