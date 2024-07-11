import ROOT, os
#ROOT.EnableImplicitMT()


import  onnxruntime 

import numpy as np

# argument parser
import argparse
parser = argparse.ArgumentParser(description='Args')
parser.add_argument('--f_in', default = 'GluGluToHHHTo6B_SM') # input samples
parser.add_argument('-v','--version', default='v31-merged-selection-no-lhe') # version of NanoNN production
parser.add_argument('--year', default='2017') # year
parser.add_argument('--batch_size', default='40') # year
parser.add_argument('--batch_number',default = '0')
args = parser.parse_args()


# Method declarations

def get_best(ls, index):
    tmp_ls = ls[index]
    #ret = str(ls[index])
    #if len(ret) == 1:
    #    ret = str(ls[index]*10)
    ret = f'{tmp_ls%10}{tmp_ls//10}'
    return ret

def get_best_lh(ls, index):
    tmp_ls = ls[index]
    #ret = str(ls[index])
    #if len(ret) == 1:
    #    ret = str(ls[index]*10)
    ret = f'{tmp_ls%4}{tmp_ls//4}'
    return ret

def get_maximas(arr_in):
    arr = np.triu(arr_in[0:10,0:10])
    np.fill_diagonal(arr, 0)
    #max_values = np.partition(arr.flatten(), -5)[-5:]
    #max_indices = np.argpartition(arr.flatten(), -5)[-5:]
    max_indices = np.argsort(arr.flatten())[::-1]
    max_values = arr.flatten()[max_indices]
    return max_values,max_indices

def get_maximas_lh(arr_in):
    arr = np.triu(arr_in[-4:,-4:])
    np.fill_diagonal(arr, 0)
    #max_values = np.partition(arr.flatten(), -5)[-5:]
    #max_indices = np.argpartition(arr.flatten(), -5)[-5:]
    max_indices = np.argsort(arr.flatten())[::-1]
    max_values = arr.flatten()[max_indices]
    max_indices = max_indices[0]
    max_values = max_values[0]
    return max_values,max_indices

def convertIndex(index):
    if len(str(index)) == 1:
        ret = str(index*10)
    else:
        ret = str(index)
    return ret

def convertIndex_lh(index, max_h3):
    if max_h3!=0:
        ret = f'{index%4}{index//4}'
    else:
        ret = False
    return ret

def remove_elements(index, m, ind):
    tmp_index = convertIndex(index)
    ind_ret = [i for i in ind if tmp_index[0] not in convertIndex(i) and tmp_index[1] not in convertIndex(i)]
    m_ret = [m[i] for i in range(len(m)) if ind.count(i) == 1]
    return m_ret, ind_ret

def pair_higgs(max_h1, index_h1, max_h2, index_h2, h1Det, h2Det):
    higgs = []

    m_h1 = h1Det
    m_h2 = h2Det

    if m_h1 > m_h2:
        higgs.append(index_h1[0])
        m_prime_2, index_prime_2 = remove_elements(index_h1[0], max_h2, index_h2)

    else:
        higgs.append(index_h2[0]) 
        m_prime_2, index_prime_2 = remove_elements(index_h2[0], max_h1, index_h1)

    higgs.append(index_prime_2[0])
    return higgs

def find_boosted_higgs(bh1,bh2,bh1Det,bh2Det):
    boosted_h = []
    for higgs in [bh1,bh2]:
        IfNotMatch = True
        for i in range(10,14):
            if higgs[i] > 0.5:
                boosted_h.append(i-10)
                IfNotMatch = False
        if IfNotMatch:
            boosted_h.append(-1)
    boosted_h = list(set(boosted_h))
    if len(boosted_h)<2: boosted_h.extend([-1]*(2-len(boosted_h)))
    return boosted_h

def process(i):
    max_h1, index_h1 = get_maximas(output_values[2][i])
    max_h2, index_h2 = get_maximas(output_values[3][i])
    max_h3, index_h3 = get_maximas_lh(output_values[4][i])
    h3_idx = convertIndex_lh(index_h3, max_h3)

    h1Det = output_values[7][i]
    h2Det = output_values[8][i]
    h3Det = output_values[9][i]

    bh1Det = output_values[5][i]
    bh2Det = output_values[6][i]

    bh1 = output_values[0][i]
    bh2 = output_values[1][i]
    #bh3 = output_values[5][i]

    boosted_higgs = find_boosted_higgs(bh1,bh2, bh1Det, bh2Det)

    #higgses = boosted_higgs + pair_higgs(max_h1.tolist(),index_h1.tolist(), max_h2.tolist(),index_h2.tolist(), max_h3.tolist(),index_h3.tolist(),h1Det,h2Det,h3Det)
    higgses = pair_higgs(max_h1.tolist(),index_h1.tolist(), max_h2.tolist(),index_h2.tolist(),h1Det,h2Det)
    return higgses[0], higgses[1], h3_idx, boosted_higgs[0], boosted_higgs[1]

# return df.Define(name, [&](ULong64_t e) { return v[e]; }, {"rdfentry_"});
# return df.Define(name, [&](ULong64_t e) { return v[e]; }, {"rdfentry_"});
ROOT.gInterpreter.Declare('''
ROOT::RDF::RNode AddArray(ROOT::RDF::RNode df, ROOT::RVec<double> &v, const std::string &name) {
    return df.Define(name, [&](unsigned int e) { return v[e]; }, {"counter"});
}

ROOT::RDF::RNode AddBoolArray(ROOT::RDF::RNode df, ROOT::RVec<Long64_t> &v, const std::string &name) {
    unsigned rdf_entry = 0;
    return df.Define(name, [&](unsigned int e) { return v[e]; }, {"counter"});
}

unsigned counter = 0;
''')

# Open RDF and onnx session

#session = onnxruntime.InferenceSession("/HEP/mstamenk/hhh-6b-producer/master/CMSSW_12_5_2/src/hhh-master/hhh-analysis-framework/spanet-inference/spanet_classification_test.onnx")

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
providers=[ 'CPUExecutionProvider']

session = onnxruntime.InferenceSession("PNet_spanet_v621.onnx",sess_options, providers=providers)
session_category = onnxruntime.InferenceSession("PNet_spanet_category_v621.onnx",sess_options, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_name_category = session_category.get_inputs()[0].name
output_name_category = session_category.get_outputs()[0].name

regime = 'inclusive-weights'
# path = '/users/mstamenk/scratch/mstamenk/%s/mva-inputs-%s/%s/'%(args.version,args.year,regime)
# path = 'root://eosuser.cern.ch//eos/user/r/rtu/eos-triple-h_weight_maybeWrong/v20-parts-no-lhe/mva-inputs-2017/inclusive-weights/'
# path = "root://eosuser.cern.ch//eos/user/r/rtu/Turb507Outputdata2017_ak8_option92_2017/v20-parts-no-lhe/mva-inputs-2017/inclusive-weights/"
path = "root://eosuser.cern.ch//eos/user/r/rtu/Turb607OutputMC2017_ak8_option92_2017/v20-parts-no-lhe/mva-inputs-2017/inclusive-weights/"

f_in = args.f_in
path_f_in = path + '%s.root'%f_in

df = ROOT.RDataFrame("Events", path_f_in)
entries = df.Count().GetValue()

event_min = int(args.batch_size) * int(args.batch_number)
event_max = event_min + int(args.batch_size)

print(entries, event_min, event_max)
if event_max > entries:
    event_max = entries 

if event_min > entries:
    print("Error %d out of range, max events %d"%(event_min,entries))
    exit()

df = df.Range(event_min,event_max)
df = df.Define('counter','counter++')

jet_vars = ["%sLogMass","%sLogPt", "%sbRegCorr", "%sEta","%sSinPhi","%sCosPhi", "%sPNetB"]
jetmask_var = ["%sPt"]
arrays = []
arrays_jetmask = []


for i in ['1','2','3','4','5','6','7','8','9','10']:
    df = df.Define('jet%sCosPhi'%i, 'TMath::Cos(jet%sPhi)'%i)
    df = df.Define('jet%sSinPhi'%i, 'TMath::Sin(jet%sPhi)'%i)
    df = df.Define('jet%sLogPt'%i, 'TMath::Log(jet%sPt+1)'%i)
    df = df.Define('jet%sLogMass'%i, 'TMath::Log(jet%sMass+1)'%i)
    # df = df.Define('jet%sPt'%i, 'jet%sPt'%i)
    # df = df.Define('jet%sPtCorr'%i, 'jet%sPt * jet%sbRegCorr'%(i,i))
    # df = df.Define('jet%sMass'%i, 'jet%sMass'%i)
    # df = df.Define('jet%sPtCorr'%i, 'jet%sbRegCorr'%i)
    if 'JetHT' in args.f_in or 'BTagCSV' in args.f_in or 'SingleMuon' in args.f_in:
        df = df.Define('jet%sHiggsMatchedIndex'%i,'-1')
    
    column = [el%'jet%s'%i for el in jet_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    arrays.append(np_arr)

    column_mask = [el%'jet%s'%i for el in jetmask_var]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    arrays_jetmask.append(np_arr_mask)

# Boosted arrays
boosted_arrays = []
boosted_arrays_mask = []
# fatjet_vars = ['fatJet%sPt', 'fatJet%sEta','fatJet%sSinPhi','fatJet%sCosPhi','fatJet%sPNetXbb','fatJet%sPNetXjj','fatJet%sPNetQCD','fatJet%sMassSD_UnCorrected']
fatjet_vars = ['fatJet%sLogPt', 'fatJet%sEta','fatJet%sSinPhi','fatJet%sCosPhi','fatJet%sPNetXbb','fatJet%sPNetXjj','fatJet%sMassSD_UnCorrected']
fatjetmask_var = ['fatJet%sPt', 'fatJet%sMassSD_UnCorrected']
for i in ['1','2','3','4']:
    df = df.Define('fatJet%sCosPhi'%i, 'TMath::Cos(fatJet%sPhi)'%i)
    df = df.Define('fatJet%sSinPhi'%i, 'TMath::Sin(fatJet%sPhi)'%i)
    df = df.Define('fatJet%sLogPt'%i, 'TMath::Log(fatJet%sPt+1)'%i)
    #if 'JetHT' in args.f_in or 'BTagCSV' in args.f_in or 'SingleMuon' in args.f_in:
    #    df = df.Define('fatJet%sHiggsMatchedIndex'%i,'-1')
    

    column = [el%i for el in fatjet_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    boosted_arrays.append(np_arr)

    column_mask = [el%i for el in fatjetmask_var]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    boosted_arrays_mask.append(np_arr_mask)


lep_arrays = []
lep_arrays_mask = [] 
lep_vars = ['lep%sLogPt', 'lep%sEta','lep%sSinPhi','lep%sCosPhi', 'lep%sId']
lep_vars_mask = ['lep%sPt', 'lep%sEta']

for i in ['1','2']:
    df = df.Define('lep%sCosPhi'%i, 'TMath::Cos(lep%sPhi)'%i)
    df = df.Define('lep%sSinPhi'%i, 'TMath::Sin(lep%sPhi)'%i)
    df = df.Define('lep%sLogPt'%i, 'TMath::Log(lep%sPt+1)'%i)
    #if 'JetHT' in args.f_in or 'BTagCSV' in args.f_in or 'SingleMuon' in args.f_in:
    #    df = df.Define('fatJet%sHiggsMatchedIndex'%i,'-1')
    

    column = [el%i for el in lep_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    lep_arrays.append(np_arr)

    column_mask = [el%i for el in lep_vars_mask]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    lep_arrays_mask.append(np_arr_mask)


tau_arrays = []
tau_arrays_mask = []
tau_vars = ['tau%srawDeepTau2017v2p1VSjet', 'tau%sLogMass', 'tau%sLogPt', 'tau%sEta','tau%sSinPhi','tau%sCosPhi', 'tau%sCharge']
tau_vars_mask = ['tau%sMass', 'tau%sPt']
for i in ['1','2','3','4']:
    df = df.Define('tau%sCosPhi'%i, 'TMath::Cos(tau%sPhi)'%i)
    df = df.Define('tau%sSinPhi'%i, 'TMath::Sin(tau%sPhi)'%i)
    df = df.Define('tau%sLogPt'%i, 'TMath::Log(tau%sPt+1)'%i)
    df = df.Define('tau%sLogMass'%i, 'TMath::Log(tau%sMass+1)'%i)
    #if 'JetHT' in args.f_in or 'BTagCSV' in args.f_in or 'SingleMuon' in args.f_in:
    #    df = df.Define('fatJet%sHiggsMatchedIndex'%i,'-1')
    
    column = [el%i for el in tau_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    tau_arrays.append(np_arr)

    column_mask = [el%i for el in tau_vars_mask]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    tau_arrays_mask.append(np_arr_mask)

Jets_arrays = {}
Jets_arrays_mask = {}
Higgs_vars = ['logbRegcorrptjet%sjet%s','logptjet%sjet%s','etajet%sjet%s','sinphijet%sjet%s','cosphijet%sjet%s','drjet%sjet%s']
Higgs_vars_mask = ['massjet%sjet%s','ptjet%sjet%s']
# Higgs_vars = ['massjet%sjet%s', 'ptjet%sjet%s','etajet%sjet%s','phijet%sjet%s', 'drjet%sjet%s']
for i in ['1','2','3','4','5','6','7','8','9','10']:
    name = 'Jet%s'%i
    Higgs_list = []
    Higgs_list_mask = []
    for j in ['2','3','4','5','6','7','8','9','10']:
        if i == j: continue
        if int(j) < int(i): continue
        df = df.Define('cosphijet%sjet%s'%(i,j), 'TMath::Cos(phijet%sjet%s)'%(i,j))
        df = df.Define('sinphijet%sjet%s'%(i,j), 'TMath::Sin(phijet%sjet%s)'%(i,j))
        df = df.Define('logbRegcorrptjet%sjet%s'%(i,j), 'TMath::Log(bRegcorrptjet%sjet%s+1)'%(i,j))
        df = df.Define('logptjet%sjet%s'%(i,j), 'TMath::Log(ptjet%sjet%s+1)'%(i,j))
        #if 'JetHT' in args.f_in or 'BTagCSV' in args.f_in or 'SingleMuon' in args.f_in:
        #    df = df.Define('fatJet%sHiggsMatchedIndex'%i,'-1')
        
        column = [el%(i,j) for el in Higgs_vars]
        np_dict = df.AsNumpy(column)
        np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
        Higgs_list.append(np_arr)

        column_mask = [el%(i,j) for el in Higgs_vars_mask]
        np_dict_mask = df.AsNumpy(column_mask)
        np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
        Higgs_list_mask.append(np_arr_mask)

    Jets_arrays[name] = Higgs_list
    Jets_arrays_mask[name] = Higgs_list_mask






met_arrays = []
met_vars = ['logmet']
df = df.Define('logmet','TMath::Log(met+1)')
column = [el for el in met_vars]
np_dict = df.AsNumpy(column)
np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
met_arrays.append(np_arr)

ht_arrays = []
ht_vars = ['loght']
df = df.Define('loght','TMath::Log(ht+1)')
column = [el for el in ht_vars]
np_dict = df.AsNumpy(column)
np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
ht_arrays.append(np_arr)


# 4 vectors
jet_4vec = ["%sPt", "%sEta","%sPhi","%sMass","%sHiggsMatchedIndex", "%sbRegCorr"]
array_4vec = []
for i in ['1','2','3','4','5','6','7','8','9','10']:
    column_4vec = [el%'jet%s'%i for el in jet_4vec]
    np_4vec = df.AsNumpy(column_4vec)
    np_arr_4vec = np.vstack(np_4vec[col] for col in column_4vec).T
    array_4vec.append(np_arr_4vec)

jets = []
for i in range(len(array_4vec[0])):
    jets_tmp = []
    for j in range(10):
        jet = ROOT.TLorentzVector()
        jet.SetPtEtaPhiM(array_4vec[j][i][0], array_4vec[j][i][1], array_4vec[j][i][2], array_4vec[j][i][3])
        jet_corrected = ROOT.TLorentzVector()
        jet_corrected.SetPtEtaPhiE(jet.Pt()*array_4vec[j][i][5],jet.Eta(),jet.Phi(),jet.E()*array_4vec[j][i][5])
        edfj = jet.Px()
        jet_corrected.HiggsMatchedIndex = array_4vec[j][i][4]
        jets_tmp.append(jet_corrected)
    jets.append(jets_tmp)


taus_4vec = ["%sPt", "%sEta","%sPhi","%sMass"]
tau_truth_4vec = []
array_taus_4vec = []
for i in ['1','2','3','4']:
    column_4vec = [el%'tau%s'%i for el in taus_4vec]
    np_4vec = df.AsNumpy(column_4vec)
    np_arr_4vec = np.vstack(np_4vec[col] for col in column_4vec).T
    array_taus_4vec.append(np_arr_4vec)
for i in range(len(array_taus_4vec[0])):
    taus_tmp = []
    for j in range(4):
        _tau = ROOT.TLorentzVector()
        _tau.SetPtEtaPhiM(array_4vec[j][i][0], array_4vec[j][i][1], array_4vec[j][i][2], array_4vec[j][i][3])
        taus_tmp.append(_tau)
    tau_truth_4vec.append(taus_tmp)

fatjet_4vec = ["%sPt", "%sEta","%sPhi","%sMass","%sHiggsMatchedIndex"]
array_fj_4vec = []
for i in ['1','2','3','4']:
    column_4vec = [el%'fatJet%s'%i for el in fatjet_4vec]
    np_4vec = df.AsNumpy(column_4vec)
    np_arr_4vec = np.vstack(np_4vec[col] for col in column_4vec).T
    array_fj_4vec.append(np_arr_4vec)

fatjets = []
for i in range(len(array_fj_4vec[0])):
    jets_tmp = []
    for j in range(4):
        jet = ROOT.TLorentzVector()
        jet.SetPtEtaPhiM(array_fj_4vec[j][i][0], array_fj_4vec[j][i][1], array_fj_4vec[j][i][2], array_fj_4vec[j][i][3])
        jet.HiggsMatchedIndex = array_fj_4vec[j][i][4]
        jets_tmp.append(jet)
    fatjets.append(jets_tmp)


# Inputs dictionaries
Jets_data = np.transpose(arrays,(1,0,2))
Jets_data_mask = np.transpose(arrays_jetmask,(1,0,2))
MIN_PT = 20
Jets_Pt = Jets_data_mask[:,:,0]

Jets_mask = Jets_Pt > MIN_PT

BoostedJets_data = np.transpose(boosted_arrays, (1,0,2))
BoostedJets_data_mask = np.transpose(boosted_arrays_mask, (1,0,2))
MIN_FJPT = 200
MIN_FJMASS = 70
BoostedJets_Pt = BoostedJets_data_mask[:,:,0]
#BoostedJets_Mass = BoostedJets_data[:,:,5]
BoostedJets_mask = BoostedJets_Pt > MIN_FJPT
#BoostedJets_mask = BoostedJets_Mass > MIN_FJMASS

Leptons_data = np.transpose(lep_arrays, (1,0,2))
Leptons_data_mask = np.transpose(lep_arrays_mask, (1,0,2))
Leptons_Pt = Leptons_data_mask[:,:,0]
Leptons_mask = Leptons_Pt > 3


Taus_data = np.transpose(tau_arrays, (1,0,2))
Taus_data_mask = np.transpose(tau_arrays_mask, (1,0,2))
Taus_Pt = Taus_data_mask[:,:,1]
Taus_mask = Taus_Pt > 20

MET_data = np.transpose(met_arrays, (1,0,2))
MET_mask = MET_data[:,:,0] > -999

HT_data = np.transpose(ht_arrays, (1,0,2))
HT_mask = MET_data[:,:,0] > -999

Jet1_data = np.transpose(Jets_arrays['Jet1'],(1,0,2))
Jet1_data_mask = np.transpose(Jets_arrays_mask['Jet1'],(1,0,2))
Jet1_Mass = Jet1_data_mask[:,:,0]
Jet1_mask = Jet1_Mass > 20

Jet2_data = np.transpose(Jets_arrays['Jet2'],(1,0,2))
Jet2_data_mask = np.transpose(Jets_arrays_mask['Jet2'],(1,0,2))
Jet2_Mass = Jet2_data_mask[:,:,0]
Jet2_mask = Jet2_Mass > 20

Jet3_data = np.transpose(Jets_arrays['Jet3'],(1,0,2))
Jet3_data_mask = np.transpose(Jets_arrays_mask['Jet3'],(1,0,2))
Jet3_Mass = Jet3_data_mask[:,:,0]
Jet3_mask = Jet3_Mass > 20

Jet4_data = np.transpose(Jets_arrays['Jet4'], (1, 0, 2))
Jet4_data_mask = np.transpose(Jets_arrays_mask['Jet4'], (1, 0, 2))
Jet4_Mass = Jet4_data_mask[:, :, 0]
Jet4_mask = Jet4_Mass > 20

Jet5_data = np.transpose(Jets_arrays['Jet5'], (1, 0, 2))
Jet5_data_mask = np.transpose(Jets_arrays_mask['Jet5'], (1, 0, 2))
Jet5_Mass = Jet5_data_mask[:, :, 0]
Jet5_mask = Jet5_Mass > 20

Jet6_data = np.transpose(Jets_arrays['Jet6'], (1, 0, 2))
Jet6_data_mask = np.transpose(Jets_arrays_mask['Jet6'], (1, 0, 2))
Jet6_Mass = Jet6_data_mask[:, :, 0]
Jet6_mask = Jet6_Mass > 20

Jet7_data = np.transpose(Jets_arrays['Jet7'], (1, 0, 2))
Jet7_data_mask = np.transpose(Jets_arrays_mask['Jet7'], (1, 0, 2))
Jet7_Mass = Jet7_data_mask[:, :, 0]
Jet7_mask = Jet7_Mass > 20

Jet8_data = np.transpose(Jets_arrays['Jet8'], (1, 0, 2))
Jet8_data_mask = np.transpose(Jets_arrays_mask['Jet8'], (1, 0, 2))
Jet8_Mass = Jet8_data_mask[:, :, 0]
Jet8_mask = Jet8_Mass > 20

Jet9_data = np.transpose(Jets_arrays['Jet9'], (1, 0, 2))
Jet9_data_mask = np.transpose(Jets_arrays_mask['Jet9'], (1, 0, 2))
Jet9_Mass = Jet9_data_mask[:, :, 0]
Jet9_mask = Jet9_Mass > 20


input_dict = {"Jets_data": Jets_data, "Jets_mask": Jets_mask, "FJets_data":BoostedJets_data, "FJets_mask": BoostedJets_mask, 'Lep_data': Leptons_data, 'Lep_mask': Leptons_mask, 'Taus_data' : Taus_data, 'Taus_mask': Taus_mask, "MET_data" : MET_data,'MET_mask': MET_mask, 'HT_data': HT_data,'HT_mask' : HT_mask, 'Jet1_data' : Jet1_data, 'Jet1_mask': Jet1_mask, 'Jet2_data' : Jet2_data, 'Jet2_mask': Jet2_mask, 'Jet3_data' : Jet3_data, 'Jet3_mask': Jet3_mask, 'Jet4_data' : Jet4_data, 'Jet4_mask': Jet4_mask, 'Jet5_data' : Jet5_data, 'Jet5_mask': Jet5_mask, 'Jet6_data' : Jet6_data, 'Jet6_mask': Jet6_mask, 'Jet7_data' : Jet7_data, 'Jet7_mask': Jet7_mask, 'Jet8_data' : Jet8_data, 'Jet8_mask': Jet8_mask,'Jet9_data' : Jet9_data, 'Jet9_mask': Jet9_mask}
output_nodes = session.get_outputs()
output_nodes_category = session_category.get_outputs()
output_names = [node.name for node in output_nodes]
output_names_category = [node.name for node in output_nodes_category]
output_values = session.run(output_names, input_dict)
output_values_category = session_category.run(output_names_category, input_dict)
h1_mass = []
h2_mass = []
h3_mass = []

h1_pt = []
h2_pt = []
h3_pt = []

h1_eta = []
h2_eta = []
h3_eta = []

h1_phi = []
h2_phi = []
h3_phi = []

h1_match = []
h2_match = []
h3_match = []

prob_hhh = [] # 1
prob_qcd = [] # 2
prob_tt = [] # 3
prob_ttHard = []
prob_ttlep = []
prob_ttSemi = []

prob_vjets = [] # 4
prob_vv = [] # 5
prob_hhh4b2tau = [] # 6
prob_hh4b = [] # 7
prob_hh2b2tau = [] # 8
prob_dy = [] # 9

prob_0rh0bh0th = []
prob_1rh0bh0th = []
prob_2rh0bh0th = []
prob_0rh1bh0th = []
prob_1rh1bh0th = []
prob_0rh2bh0th = []
prob_0rh0bh1th = []
prob_1rh0bh1th = []
prob_2rh0bh1th = []
prob_0rh1bh1th = []
prob_1rh1bh1th = []
prob_0rh2bh1th = []


# print(output_values)

dummy_particle = []
for i in range(len(array_fj_4vec[0])):
    jets_tmp = []
    for j in range(4):
        jet = ROOT.TLorentzVector()
        jet.SetPtEtaPhiM(0,0,0,0)
        jet.HiggsMatchedIndex = False
        jets_tmp.append(jet)
    dummy_particle.append(jets_tmp)


for i in range(len(output_values[0])):
    best = process(i)

    #print(best)

    jets_tmp = jets[i]
    tau_tmp = tau_truth_4vec[i]
    fjets_tmp = fatjets[i]
    dummy_particle_tmp = dummy_particle[i]

    h1_index = get_best(best,0)
    h2_index = get_best(best,1)
    h3_index = best[2]
    h1 = jets_tmp[int(h1_index[0])] + jets_tmp[int(h1_index[1])]
    h1.HiggsMatchedIndex = jets_tmp[int(h1_index[0])].HiggsMatchedIndex ==  jets_tmp[int(h1_index[1])].HiggsMatchedIndex and jets_tmp[int(h1_index[1])].HiggsMatchedIndex > 0
    h2 = jets_tmp[int(h2_index[0])] + jets_tmp[int(h2_index[1])]
    h2.HiggsMatchedIndex = jets_tmp[int(h2_index[0])].HiggsMatchedIndex ==  jets_tmp[int(h2_index[1])].HiggsMatchedIndex and jets_tmp[int(h2_index[1])].HiggsMatchedIndex > 0
    print(h3_index)
    if not h3_index:
        h3 = dummy_particle_tmp[0]
    else:
        h3 = tau_tmp[int(h3_index[0])] + tau_tmp[int(h3_index[1])]

    bh1_index = best[3]
    bh2_index = best[4]
    if bh1_index==-1:
        bh1 = dummy_particle_tmp[0]
    else:
        bh1 = fjets_tmp[bh1_index]
    if bh2_index==-1:
        bh2 = dummy_particle_tmp[0]
    else:
        bh2 = fjets_tmp[bh2_index]

    '''
    if len(h1_index) == 2:
        h1 = jets_tmp[int(h1_index[0])] + jets_tmp[int(h1_index[1])]
        h1.HiggsMatchedIndex = jets_tmp[int(h1_index[0])].HiggsMatchedIndex ==  jets_tmp[int(h1_index[1])].HiggsMatchedIndex and jets_tmp[int(h1_index[1])].HiggsMatchedIndex > 0        
    elif len(h1_index) == 3:
        h1 = fjets_tmp[int(h1_index)-110]
        h1.HiggsMatchedIndex = fjets_tmp[int(h1_index)-110].HiggsMatchedIndex > 0


    if len(h2_index) == 2:
        h2 = jets_tmp[int(h2_index[0])] + jets_tmp[int(h2_index[1])]
        h2.HiggsMatchedIndex = jets_tmp[int(h2_index[0])].HiggsMatchedIndex ==  jets_tmp[int(h2_index[1])].HiggsMatchedIndex and jets_tmp[int(h2_index[1])].HiggsMatchedIndex > 0
    elif len(h2_index) == 3:
        h2 = fjets_tmp[int(h2_index)-110]
        h2.HiggsMatchedIndex = fjets_tmp[int(h2_index)-110].HiggsMatchedIndex > 0

    if len(h3_index) == 2:
        h3 = jets_tmp[int(h3_index[0])] + jets_tmp[int(h3_index[1])]
        h3.HiggsMatchedIndex = jets_tmp[int(h3_index[0])].HiggsMatchedIndex ==  jets_tmp[int(h3_index[1])].HiggsMatchedIndex and jets_tmp[int(h3_index[1])].HiggsMatchedIndex > 0
    elif len(h3_index) == 3:
        h3 = fjets_tmp[int(h2_index)-110]
        h3.HiggsMatchedIndex = fjets_tmp[int(h3_index)-110].HiggsMatchedIndex > 0
    '''
    higgses = [h1,h2,h3]
    #higgses.sort(key= lambda x: x.Pt(), reverse=True)

    h1 = higgses[0]
    h2 = higgses[1]
    h3 = higgses[2]

    h1_mass.append(h1.M())
    h1_pt.append(h1.Pt())
    h1_eta.append(h1.Eta())
    h1_phi.append(h1.Phi())

    h2_mass.append(h2.M())
    h2_pt.append(h2.Pt())
    h2_eta.append(h2.Eta())
    h2_phi.append(h2.Phi())
    
    h3_mass.append(h3.M())
    h3_pt.append(h3.Pt())
    h3_eta.append(h3.Eta())
    h3_phi.append(h3.Phi())
    


    h1_match.append(int(h1.HiggsMatchedIndex))
    h2_match.append(int(h2.HiggsMatchedIndex))
    test_mappings = {'HHHTo4B2Tau' : 1, 
            'HHHTo6B' : 2,
            'GluGluToHHTo4B'  : 3,
            'GluGluToHHTo2B2Tau': 4,
            'QCD_HT': 5,
            'TTTo2L2Nu': 6,
            'TTToSemiLeptonic': 7,
            'TTToHadronic': 8,
            'WJets': 9,
            'ZJets': 10,
            'WWTo': 11,
            'WZTo': 11,
            'ZZTo': 11,
    }
    const_n = 10     #depend on the output struction of SPANET

    prob_hhh.append(float(output_values[const_n][i][2])) # based on mapping in SPANET training
    prob_qcd.append(float(output_values[const_n][i][5]))
    prob_ttHard.append(float(output_values[const_n][i][8]))
    prob_ttSemi.append(float(output_values[const_n][i][7]))
    prob_ttlep.append(float(output_values[const_n][i][6]))
    prob_vjets.append(float(output_values[const_n][i][9]))
    prob_vv.append(float(output_values[const_n][i][11]))
    prob_hhh4b2tau.append(float(output_values[const_n][i][1]))
    prob_hh4b.append(float(output_values[const_n][i][3]))
    prob_hh2b2tau.append(float(output_values[const_n][i][4]))
    #prob_dy.append(float(output_values[12][i][9]))

    #prob_3bh0h.append(float(output_values2[12][i][1])) # based on mapping in SPANET training
    #prob_2bh1h.append(float(output_values2[12][i][2]))
    #prob_1bh2h.append(float(output_values2[12][i][3]))
    #prob_0bh3h.append(float(output_values2[12][i][4]))
    #prob_2bh0h.append(float(output_values2[12][i][5]))
    #prob_1bh1h.append(float(output_values2[12][i][6]))
    #prob_0bh2h.append(float(output_values2[12][i][7]))
    #prob_1bh0h.append(float(output_values2[12][i][8]))
    #prob_0bh1h.append(float(output_values2[12][i][9]))
    #prob_0bh0h.append(float(output_values2[12][i][0]))

    prob_0rh0bh0th.append(float(output_values_category[2][i][0]))
    prob_1rh0bh0th.append(float(output_values_category[2][i][1]))
    prob_2rh0bh0th.append(float(output_values_category[2][i][2]))
    prob_0rh1bh0th.append(float(output_values_category[2][i][3]))
    prob_1rh1bh0th.append(float(output_values_category[2][i][4]))
    prob_0rh2bh0th.append(float(output_values_category[2][i][5]))
    prob_0rh0bh1th.append(float(output_values_category[2][i][6]))
    prob_1rh0bh1th.append(float(output_values_category[2][i][7]))
    prob_2rh0bh1th.append(float(output_values_category[2][i][8]))
    prob_0rh1bh1th.append(float(output_values_category[2][i][9]))
    prob_1rh1bh1th.append(float(output_values_category[2][i][10]))
    prob_0rh2bh1th.append(float(output_values_category[2][i][11]))
#test = [i *0.2 for i in range(10)]


arr_h1_mass = ROOT.VecOps.AsRVec(np.array(h1_mass))
arr_h1_pt = ROOT.VecOps.AsRVec(np.array(h1_pt))
arr_h1_eta = ROOT.VecOps.AsRVec(np.array(h1_eta))
arr_h1_phi = ROOT.VecOps.AsRVec(np.array(h1_phi))
arr_h1_match = ROOT.VecOps.AsRVec(np.array(h1_match))


arr_h2_mass = ROOT.VecOps.AsRVec(np.array(h2_mass))
arr_h2_pt = ROOT.VecOps.AsRVec(np.array(h2_pt))
arr_h2_eta = ROOT.VecOps.AsRVec(np.array(h2_eta))
arr_h2_phi = ROOT.VecOps.AsRVec(np.array(h2_phi))
arr_h2_match = ROOT.VecOps.AsRVec(np.array(h2_match))

arr_h3_mass = ROOT.VecOps.AsRVec(np.array(h3_mass))
arr_h3_pt = ROOT.VecOps.AsRVec(np.array(h3_pt))
arr_h3_eta = ROOT.VecOps.AsRVec(np.array(h3_eta))
arr_h3_phi = ROOT.VecOps.AsRVec(np.array(h3_phi))



arr_prob_hhh = ROOT.VecOps.AsRVec(np.array(prob_hhh))
arr_prob_qcd = ROOT.VecOps.AsRVec(np.array(prob_qcd))
arr_prob_ttHard = ROOT.VecOps.AsRVec(np.array(prob_ttHard))
arr_prob_ttlep = ROOT.VecOps.AsRVec(np.array(prob_ttlep))
arr_prob_ttSemi = ROOT.VecOps.AsRVec(np.array(prob_ttSemi))
arr_prob_vjets = ROOT.VecOps.AsRVec(np.array(prob_vjets))
arr_prob_vv = ROOT.VecOps.AsRVec(np.array(prob_vv))
arr_prob_hhh4b2tau = ROOT.VecOps.AsRVec(np.array(prob_hhh4b2tau))
arr_prob_hh4b = ROOT.VecOps.AsRVec(np.array(prob_hh4b))
arr_prob_hh2b2tau = ROOT.VecOps.AsRVec(np.array(prob_hh2b2tau))
#arr_prob_dy = ROOT.VecOps.AsRVec(np.array(prob_dy))

arr_prob_0rh0bh0th = ROOT.VecOps.AsRVec(np.array(prob_0rh0bh0th))
arr_prob_1rh0bh0th = ROOT.VecOps.AsRVec(np.array(prob_1rh0bh0th))
arr_prob_2rh0bh0th = ROOT.VecOps.AsRVec(np.array(prob_2rh0bh0th))
arr_prob_0rh1bh0th = ROOT.VecOps.AsRVec(np.array(prob_0rh1bh0th))
arr_prob_1rh1bh0th = ROOT.VecOps.AsRVec(np.array(prob_1rh1bh0th))
arr_prob_0rh2bh0th = ROOT.VecOps.AsRVec(np.array(prob_0rh2bh0th))
arr_prob_0rh0bh1th = ROOT.VecOps.AsRVec(np.array(prob_0rh0bh1th))
arr_prob_1rh0bh1th = ROOT.VecOps.AsRVec(np.array(prob_1rh0bh1th))
arr_prob_2rh0bh1th = ROOT.VecOps.AsRVec(np.array(prob_2rh0bh1th))
arr_prob_0rh1bh1th = ROOT.VecOps.AsRVec(np.array(prob_0rh1bh1th))
arr_prob_1rh1bh1th = ROOT.VecOps.AsRVec(np.array(prob_1rh1bh1th))
arr_prob_0rh2bh1th = ROOT.VecOps.AsRVec(np.array(prob_0rh2bh1th))

df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_mass, "h1_spanet_boosted_mass")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_pt, "h1_spanet_boosted_pt")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_eta, "h1_spanet_boosted_eta")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_phi, "h1_spanet_boosted_phi")
df = ROOT.AddBoolArray(ROOT.RDF.AsRNode(df), arr_h1_match, "h1_spanet_boosted_match")

df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_mass, "h2_spanet_boosted_mass")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_pt, "h2_spanet_boosted_pt")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_eta, "h2_spanet_boosted_eta")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_phi, "h2_spanet_boosted_phi")
df = ROOT.AddBoolArray(ROOT.RDF.AsRNode(df), arr_h2_match, "h2_spanet_boosted_match")
'''
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h3_mass, "h3_spanet_boosted_mass")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h3_pt, "h3_spanet_boosted_pt")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h3_eta, "h3_spanet_boosted_eta")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h3_phi, "h3_spanet_boosted_phi")
df = ROOT.AddBoolArray(ROOT.RDF.AsRNode(df), arr_h3_match, "h3_spanet_boosted_match")
'''

df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hhh, "ProbHHH")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_qcd, "ProbQCD")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_ttHard, "ProbTTHard")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_ttSemi, "ProbTTSemi")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_ttlep, "ProbTTlep")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_vjets, "ProbVJets")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_vv, "ProbVV")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hhh4b2tau, "ProbHHH4b2tau")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hh4b, "ProbHH4b")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hh2b2tau, "ProbHH2b2tau")
#df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_dy, "ProbDY")

df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh0bh0th, "Prob0rh0bh0th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_1rh0bh0th, "Prob1rh0bh0th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_2rh0bh0th, "Prob2rh0bh0th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh1bh0th, "Prob0rh1bh0th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_1rh1bh0th, "Prob1rh1bh0th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh2bh0th, "Prob0rh2bh0th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh0bh1th, "Prob0rh0bh1th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_1rh0bh1th, "Prob1rh0bh1th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_2rh0bh1th, "Prob2rh0bh1th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh1bh1th, "Prob0rh1bh1th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_1rh1bh1th, "Prob1rh1bh1th")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh2bh1th, "Prob0rh2bh1th")

print("Saving output")
output_path = path.replace('%s'%args.year,'%s-spanet-boosted-classification'%args.year)
output_path = "/eos/user/r/rtu/Turb607OutputMC2017_ak8_option92_2017/v20-parts-no-lhe/mva-inputs-2017/inclusive-weights_SPANET/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# output_name = args.f_in + '_%s'%args.batch_number + '.root'
output_name = f_in + '_%s'%args.batch_number + '.root'
print(output_path,output_name)

df.Snapshot('Events',output_path + '/' + output_name)