# Script to make histograms using rdataframe

import os, ROOT
from utils import labels, binnings, cuts, wps, tags, luminosities, hlt_paths, triggersCorrections, add_bdt, bdts_xml
from calibrations import btag_init, addBTagSF, addBTagEffSF
from truthtagging import tt_init, addTTWeight
from array import array


# argument parser
import argparse
parser = argparse.ArgumentParser(description='Args')
parser.add_argument('-v','--version', default='v24') # version of NanoNN production
parser.add_argument('--year', default='2018') # year
parser.add_argument('--region', default = 'inclusive') # region: nFJ0, nFJ1, nFJ2, inclsuive, nFJ1p(= 1+2+3 fatjets)
parser.add_argument('--tag', default = '0ptag') # n b-tags on AK4 jets using DeepJet
parser.add_argument('--wp', default = 'loose') # b-tagging working point: loose, medium, tight
parser.add_argument('--f_in', default = 'GluGluToHHHTo6B_SM') # input samples
#parser.add_argument('--inputs_path', default = '/afs/cern.ch/work/m/mstamenk/public/forPKU/') # path of inputs after NanoNN
parser.add_argument('--inputs_path', default = '/isilon/data/users/mstamenk/eos-triple-h/') # path of inputs after NanoNN
parser.add_argument('--outputs_path', help='Please specify output path for histograms / mva inputs location to be stored', default = '/isilon/data/users/mstamenk/eos-triple-h/') # output path for histograms and mva inputs
parser.add_argument('--doMVAInputs', action = 'store_true') # store MVA inputs
parser.add_argument('--doHistograms', action = 'store_true') # store histograms
parser.add_argument('--addBDT', action = 'store_true') # store histograms

args = parser.parse_args()

if not args.outputs_path:
    print("Exitting script: error")
    print("Please provide output path --outputs_path /path/to/store/outputs/")
    exit()

# set batch to speed up the run
ROOT.gROOT.SetBatch(ROOT.kTRUE)
#ROOT.ROOT.EnableImplicitMT()


# get inputs 
version = args.version
path = args.inputs_path + '/' + 'samples-%s-%s-nanoaod'%(version,args.year) + '/'
#path = args.inputs_path + '/' + 'samples-%s-%s-%s-wp-%s-%s'%(version,args.region,args.wp,args.tag,args.year) + '/'
#path = args.inputs_path + '/' + 'mva-inputs-HLT-selection-%s-inclusive-loose-wp-0ptag-%s'%(version,args.year) + '/'
print(path)
f_in = args.f_in

# data frame with all the events 
df = ROOT.ROOT.RDataFrame('Events', path+'/' + f_in + '.root')


lumi = luminosities[args.year]
wp = args.wp
tag = args.tag
region = args.region
typename = 'HLT-fit-inputs-BTagSFTest'

if args.doHistograms:
    print("Running doHistograms: preparing output folder")
    out_path = args.outputs_path + '/' + args.version + '/'  + 'histograms-%s-%s-%s-%s-wp-%s-%s'%(typename,version,region,wp,tag,args.year)
    print("Histograms will be saved at: " + out_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    f_out = ROOT.TFile(out_path + '/' + 'histograms_%s.root'%(f_in), 'recreate')
    print("Writing in %s"%(out_path + '/' + 'histograms_%s.root'%(f_in)))

if args.doMVAInputs:
    mva_training_samples = args.outputs_path + '/' + args.version + '/' + 'mva-inputs-%s-%s-%s-%s-wp-%s-%s'%(typename,version,region,wp,tag,args.year)
    print("Running doMVAInputs: preparing folder")
    print("Writing in %s"%mva_training_samples)
    if not os.path.isdir(mva_training_samples):
        os.makedirs(mva_training_samples)

# Tagging selection
cutTag = tags[tag]
print(cutTag)
#if tag == '0ptag':
#    cutTag = tags[tag]
#    wp = 'noWP'
#else:
    #cutTag = tags[tag]%(wps[wp], wps[wp], wps[wp], wps[wp], wps[wp], wps[wp])

cutRegion = cuts[region]

if 'JetHT' in f_in or 'BTagCSV' in f_in:
    cutWeight = '1'
    cutWeightBTagSF = '1'
else: 
    if '2016' in args.year:
        ROOT.gInterpreter.Declare(triggersCorrections['2016'][0])
        if 'triggerSF' not in df.GetColumnNames():
            df = df.Define('triggerSF', triggersCorrections['2016'][1] )
        cutWeight = '(%f * weight * xsecWeight * l1PreFiringWeight * puWeight * genWeight * triggerSF)'%(lumi)
        cutWeightBTagSF = '(%f * weight * xsecWeight * l1PreFiringWeight * puWeight * genWeight * triggerSF * bcand1BTagSF * bcand2BTagSF * bcand3BTagSF * bcand4BTagSF * bcand5BTagSF * bcand6BTagSF * jet7BTagSF * jet8BTagSF * jet9BTagSF * jet10BTagSF)'%(lumi)
    else:
        ROOT.gInterpreter.Declare(triggersCorrections[args.year][0])
        if 'triggerSF' not in df.GetColumnNames():
            df = df.Define('triggerSF', triggersCorrections[args.year][1] )
        cutWeightBTagSF = '(%f * weight * xsecWeight * l1PreFiringWeight * puWeight * genWeight * triggerSF * bcand1BTagSF * bcand2BTagSF * bcand3BTagSF * bcand4BTagSF * bcand5BTagSF * bcand6BTagSF * jet7BTagSF * jet8BTagSF * jet9BTagSF * jet10BTagSF)'%(lumi)
        cutWeight = '(%f * weight * xsecWeight * l1PreFiringWeight * puWeight * genWeight * triggerSF)'%(lumi)


# HLT selection
cutHLT = hlt_paths[args.year]

# cut jets pT and eta
cutJets = '(bcand1Pt > 20 && TMath::Abs(bcand1Eta) < 2.5) && (bcand2Pt > 20 && TMath::Abs(bcand2Eta) < 2.5) && (bcand3Pt > 20 && TMath::Abs(bcand3Eta) < 2.5) && (bcand4Pt > 20 && TMath::Abs(bcand4Eta) < 2.5) && (bcand5Pt > 20 && TMath::Abs(bcand5Eta) < 2.5) && (bcand6Pt > 20 && TMath::Abs(bcand6Eta) < 2.5)'

cutJetIds = '(bcand1JetId == 6 && bcand2JetId == 6 && bcand3JetId == 6 && bcand4JetId == 6 && bcand5JetId == 6 && bcand6JetId == 6)'
if '2016' in args.year:
    cutPuId = '( (( jet1PuId == 3 || jet1PuId ==7 ) && jet1Pt < 50  ) || jet1Pt > 50) && ( (( jet2PuId == 3 || jet2PuId ==7 ) && jet2Pt < 50  ) || jet2Pt > 50) && ( (( jet3PuId == 3 || jet3PuId ==7 ) && jet3Pt < 50  ) || jet3Pt > 50) && ( (( jet4PuId == 3 || jet4PuId ==7 ) && jet4Pt < 50  ) || jet4Pt > 50) && ( (( jet5PuId == 3 || jet5PuId == 7 ) && jet5Pt < 50  ) || jet5Pt > 50) && ( (( jet6PuId == 3 || jet6PuId ==7 ) && jet6Pt < 50  ) || jet6Pt > 50)'
else:
    cutPuId = '( (( jet1PuId == 6 || jet1PuId ==7 ) && jet1Pt < 50  ) || jet1Pt > 50) && ( (( jet2PuId == 6 || jet2PuId ==7 ) && jet2Pt < 50  ) || jet2Pt > 50) && ( (( jet6PuId == 6 || jet6PuId ==7 ) && jet6Pt < 50  ) || jet6Pt > 50) && ( (( jet4PuId == 6 || jet4PuId ==7 ) && jet4Pt < 50  ) || jet4Pt > 50) && ( (( jet5PuId == 6 || jet5PuId == 7 ) && jet5Pt < 50  ) || jet5Pt > 50) && ( (( jet6PuId == 6 || jet6PuId ==7 ) && jet6Pt < 50  ) || jet6Pt > 50)'

# AK4 selection on leading jet in Higgs pairs
cutSignalJets = '(bcand1Pt > 40 && bcand3Pt > 40 && bcand5Pt > 40)'


# Event selection
df_hlt = df.Filter(cutHLT, "Pass HLT trigger")
df_jets = df_hlt.Filter(cutJets, "Pt and Eta, jet id and pu id cuts on b-candidates")
#if 'v22' in args.version or 'v23' in args.version:
df_jets = df_jets.Filter(cutJetIds, "jet id")
df_jets = df_jets.Filter(cutPuId, "pu id")
df_region = df_jets.Filter(cutRegion, "Pass region cut")
df_tags = df_region.Filter(cutTag, "Pass b-tagging")
df = df_tags.Filter(cutSignalJets,"Pass leading jet pT > 40")
#df = df.Filter('h_fit_mass > 80 && h_fit_mass < 150', "Pass mass cut window")

# Add b-tagging SFs
#if args.year == '2016' or args.year == '2016PostAPV':
if '2016APV' in args.year:
    btag_init('2016preVFP')
elif '2016' in args.year:
    btag_init('2016postVFP')
else:
    btag_init(args.year)

df = addBTagSF(df,f_in)
df = addBTagEffSF(df,f_in,'loose')
df = addBTagEffSF(df,f_in,'medium')
df = addBTagEffSF(df,f_in,'tight')

# Add TT weight
if 'JetHT' not in f_in and 'BTagCSV' not in f_in:
    tt_init(args.year,args.f_in)

#df = addTTWeight(df,f_in,'loose')
#df = addTTWeight(df,f_in,'medium')
#df = addTTWeight(df,f_in,'tight')

# Define new variables
if 'hhh_t3_pt' not in df.GetColumnNames():
    df = df.Define('hhh_t3_pt','h1_t3_pt + h2_t3_pt + h3_t3_pt')
if 'eventWeight' not in df.GetColumnNames():
    df = df.Define('eventWeight',cutWeight)

if 'eventWeightBTagSF' not in df.GetColumnNames():
    df = df.Define('eventWeightBTagSF',cutWeightBTagSF)

# Normalisation ratio
weight_before = df.Sum('eventWeight')
weight_after = df.Sum('eventWeightBTagSF')

ratio = weight_before.GetValue() / weight_after.GetValue()
df = df.Define('ratioPerEvent', '%f'%ratio)
df = df.Define('eventWeightBTagCorrected', 'eventWeightBTagSF * ratioPerEvent')


print("Finished addng b-tag SF and new variables")



if args.addBDT:
    #xml = bdts_xml[args.year]
    df = add_bdt(df,args.year)



# Print report on event selection
report = df.Report()
report.Print()

if args.doMVAInputs:
    if 'QCD' in f_in:
        cut6b = 'bcand1HadronFlavour == 5 && bcand2HadronFlavour == 5 && bcand3HadronFlavour == 5 && bcand4HadronFlavour == 5 && bcand5HadronFlavour == 5 && bcand6HadronFlavour == 5'
        cutNon6b = 'bcand1HadronFlavour != 5 || bcand2HadronFlavour != 5 || bcand3HadronFlavour != 5 || bcand4HadronFlavour != 5 || bcand5HadronFlavour != 5 || bcand6HadronFlavour != 5' 
        #df.Snapshot('Events', mva_training_samples + '/' + f_in + '.root')
        df.Filter(cut6b).Snapshot('Events', mva_training_samples + '/' + f_in.replace("QCD",'QCD6B') + '.root')
        df.Filter(cutNon6b).Snapshot('Events', mva_training_samples + '/' + f_in + '.root')
    
    else:
        df.Snapshot('Events', mva_training_samples + '/' + f_in + '.root')

if args.doHistograms:
    # Define histograms to be produced
    histograms = []
    variables = df.GetColumnNames()
    variables = [v for v in labels if 'h_fit_mass' not in v and 'match' not in v and 'Match' not in v] # can be done better
    print("Will produce histograms for following variables:")
    print(variables)

    # Rdataframes require first histogram to be produced and then the rest is nested in the loop of the first one
    binning = binnings['h_fit_mass'].replace('(','').replace(')','').split(',')
    bins = int(binning[0])
    xmin = float(binning[1])
    xmax = float(binning[2])

    h = df.Histo1D(('h_fit_mass','h_fit_mass',bins,xmin,xmax),"h_fit_mass", 'eventWeight') # booking the rdataframe loop
    h.SetTitle('h_fit_mass')
    h.SetName('h_fit_mass')

    histograms.append(h)
    for var in variables: # booking all variables to be produced
        binning = binnings[var].replace('(','').replace(')','').split(',')
        bins = int(binning[0])
        xmin = float(binning[1])
        xmax = float(binning[2])
        h_tmp = df.Histo1D((var,var,bins,xmin,xmax),var, 'eventWeight')
        h_tmp.SetTitle('%s'%(var))
        h_tmp.SetName('%s'%(var))
        histograms.append(h_tmp)
    h.Draw() # run one loop for all variables

    # writing output histograms
    f_out.cd()
    for h in histograms:
        h.Write()
    f_out.Close()

