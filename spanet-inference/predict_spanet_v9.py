"""SPANet v9 inference script.

Runs both category (Process ID, 12 classes) and classification (Higgs matching, 7 classes) models.

Key changes from v711:
- HT container: 2 features (ht + kind_category_analysis) instead of 1
- Jets btag: PNetCat (int 0-10) instead of PNetB (float 0-1)
- Leptons: 2 slots instead of 3
- Both model outputs (process ID & Higgs matching) at ONNX output index 8
- Model naming: "category" model outputs Process ID (EVENT/signal),
                "classification" model outputs Higgs matching (EVENT/classfication_signal)

ONNX input structure (both models identical):
  Jets_data:     [batch, 10, 7]  (Mass, Pt, bRegCorr, Eta, SinPhi, CosPhi, PNetCat)
  FJets_data:    [batch, 4, 7]   (Pt, Eta, SinPhi, CosPhi, PNetXbb, PNetXjj, MassSD_UnCorrected)
  Lep_data:      [batch, 2, 5]   (Pt, Eta, SinPhi, CosPhi, Id)
  Jet1-9_data:   [batch, N, 6]   (mass, pt, eta, sinphi, cosphi, dr)
  Taus_data:     [batch, 4, 7]   (rawDeepTau, Mass, Pt, Eta, SinPhi, CosPhi, Charge)
  TauPair_data:  [batch, 1, 6]   (mass, pt, eta, sinphi, cosphi, deltaPhi_MET)
  MET_data:      [batch, 1, 1]   (met)
  HT_data:       [batch, 1, 2]   (ht, kind_category_analysis)

ONNX output structure (both models, 9 outputs):
  [0] bh1_assignment_probability  [1] bh2_assignment_probability
  [2] h1_assignment_probability   [3] h2_assignment_probability
  [4] bh1_detection_probability   [5] bh2_detection_probability
  [6] h1_detection_probability    [7] h2_detection_probability
  [8] EVENT/signal (category, 12 cls) or EVENT/classfication_signal (classification, 7 cls)
"""
import ROOT, os
import onnxruntime
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='SPANet v9 inference')
parser.add_argument('--f_in', default='HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8_tree')
parser.add_argument('--path', default='/eos/user/r/rtu/TurbOutputMC2017_v9_with_corr_ak8_option92_2017/mc/parts/')
parser.add_argument('--output_dir', default='')
parser.add_argument('--model_dir', default='/eos/user/r/rtu/')
parser.add_argument('--year', default='2017')
parser.add_argument('--batch_size', default='40')
parser.add_argument('--batch_number', default='0')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions (jet pairing logic, unchanged)
# ---------------------------------------------------------------------------

def get_best(ls, index):
    tmp_ls = ls[index]
    ret = f'{tmp_ls%10}{tmp_ls//10}'
    return ret


def get_maximas(arr_in):
    arr = np.triu(arr_in[0:10, 0:10])
    np.fill_diagonal(arr, 0)
    max_indices = np.argsort(arr.flatten())[::-1]
    max_values = arr.flatten()[max_indices]
    return max_values, max_indices


def convertIndex(index):
    if len(str(index)) == 1:
        ret = str(index * 10)
    else:
        ret = str(index)
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
        higgs.append(index_prime_2[0])
    else:
        m_prime_2, index_prime_2 = remove_elements(index_h2[0], max_h1, index_h1)
        higgs.append(index_prime_2[0])
        higgs.append(index_h2[0])
    return higgs


def find_boosted_higgs(bh1, bh2, bh1Det, bh2Det):
    boosted_h = []
    for higgs in [bh1, bh2]:
        IfNotMatch = True
        for i in range(10, 14):
            if higgs[i] > 0.5:
                boosted_h.append(i - 10)
                IfNotMatch = False
        if IfNotMatch:
            boosted_h.append(-1)
    boosted_h = list(set(boosted_h))
    if len(boosted_h) < 2:
        boosted_h.extend([-1] * (2 - len(boosted_h)))
    return boosted_h


def process(i):
    """Extract jet pairing from classification model (best assignment accuracy)."""
    max_h1, index_h1 = get_maximas(output_values_cls[2][i])
    max_h2, index_h2 = get_maximas(output_values_cls[3][i])
    h3_idx = 0  # placeholder

    h1Det = output_values_cls[6][i]
    h2Det = output_values_cls[7][i]
    bh1Det = output_values_cls[4][i]
    bh2Det = output_values_cls[5][i]
    bh1 = output_values_cls[0][i]
    bh2 = output_values_cls[1][i]

    boosted_higgs = find_boosted_higgs(bh1, bh2, bh1Det, bh2Det)
    higgses = pair_higgs(max_h1.tolist(), index_h1.tolist(),
                         max_h2.tolist(), index_h2.tolist(), h1Det, h2Det)
    return higgses[0], higgses[1], h3_idx, boosted_higgs[0], boosted_higgs[1]


# ---------------------------------------------------------------------------
# C++ helpers for RDataFrame
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ONNX sessions
# ---------------------------------------------------------------------------
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
providers = ['CPUExecutionProvider']

model_cls_path = os.path.join(args.model_dir, 'PNet_spanet_classification_v9_vlog.onnx')
model_cat_path = os.path.join(args.model_dir, 'PNet_spanet_category_v9_vlog.onnx')

session_cls = onnxruntime.InferenceSession(model_cls_path, sess_options, providers=providers)
session_cat = onnxruntime.InferenceSession(model_cat_path, sess_options, providers=providers)

print("Classification model (Higgs matching):", model_cls_path)
print("Category model (Process ID):", model_cat_path)


# ---------------------------------------------------------------------------
# Open ROOT file
# ---------------------------------------------------------------------------
f_in = args.f_in
path_f_in = os.path.join(args.path, '%s.root' % f_in)

df = ROOT.RDataFrame("Events", path_f_in)
entries = df.Count().GetValue()

event_min = int(args.batch_size) * int(args.batch_number)
event_max = event_min + int(args.batch_size)

print(entries, event_min, event_max)
if event_max > entries:
    event_max = entries

if event_min > entries:
    print("Error %d out of range, max events %d" % (event_min, entries))
    exit()

df = df.Range(event_min, event_max)
df = df.Define('counter', 'counter++')


# ---------------------------------------------------------------------------
# 1. AK4 Jets (10 jets x 7 features)
#    Changed: PNetB -> PNetCat
# ---------------------------------------------------------------------------
jet_vars = ["%sMass", "%sPt", "%sbRegCorr", "%sEta", "%sSinPhi", "%sCosPhi", "%sPNetCat"]
jetmask_var = ["%sPt"]
arrays = []
arrays_jetmask = []

for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
    df = df.Define('jet%sCosPhi' % i, 'TMath::Cos(jet%sPhi)' % i)
    df = df.Define('jet%sSinPhi' % i, 'TMath::Sin(jet%sPhi)' % i)
    if 'JetHT' in args.f_in or 'BTagCSV' in args.f_in or 'SingleMuon' in args.f_in:
        df = df.Define('jet%sHiggsMatchedIndex' % i, '-1')

    column = [el % 'jet%s' % i for el in jet_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    arrays.append(np_arr)

    column_mask = [el % 'jet%s' % i for el in jetmask_var]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    arrays_jetmask.append(np_arr_mask)


# ---------------------------------------------------------------------------
# 2. AK8 FatJets (4 fat jets x 7 features) — unchanged
# ---------------------------------------------------------------------------
boosted_arrays = []
boosted_arrays_mask = []
fatjet_vars = ['fatJet%sPt', 'fatJet%sEta', 'fatJet%sSinPhi', 'fatJet%sCosPhi',
               'fatJet%sPNetXbb', 'fatJet%sPNetXjj', 'fatJet%sMassSD_UnCorrected']
fatjetmask_var = ['fatJet%sPt', 'fatJet%sMassSD_UnCorrected']

for i in ['1', '2', '3', '4']:
    df = df.Define('fatJet%sCosPhi' % i, 'TMath::Cos(fatJet%sPhi)' % i)
    df = df.Define('fatJet%sSinPhi' % i, 'TMath::Sin(fatJet%sPhi)' % i)

    column = [el % i for el in fatjet_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    boosted_arrays.append(np_arr)

    column_mask = [el % i for el in fatjetmask_var]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    boosted_arrays_mask.append(np_arr_mask)


# ---------------------------------------------------------------------------
# 3. Leptons (2 leptons x 5 features) — changed from 3 to 2
# ---------------------------------------------------------------------------
lep_arrays = []
lep_arrays_mask = []
lep_vars = ['lep%sPt', 'lep%sEta', 'lep%sSinPhi', 'lep%sCosPhi', 'lep%sId']
lep_vars_mask = ['lep%sPt', 'lep%sEta']

for i in ['1', '2']:  # was ['1','2','3']
    df = df.Define('lep%sCosPhi' % i, 'TMath::Cos(lep%sPhi)' % i)
    df = df.Define('lep%sSinPhi' % i, 'TMath::Sin(lep%sPhi)' % i)

    column = [el % i for el in lep_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    lep_arrays.append(np_arr)

    column_mask = [el % i for el in lep_vars_mask]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    lep_arrays_mask.append(np_arr_mask)


# ---------------------------------------------------------------------------
# 4. Taus (4 taus x 7 features) — unchanged
# ---------------------------------------------------------------------------
tau_arrays = []
tau_arrays_mask = []
tau_vars = ['tau%srawDeepTau2017v2p1VSjet', 'tau%sMass', 'tau%sPt', 'tau%sEta',
            'tau%sSinPhi', 'tau%sCosPhi', 'tau%sCharge']
tau_vars_mask = ['tau%sMass', 'tau%sPt']

for i in ['1', '2', '3', '4']:
    df = df.Define('tau%sCosPhi' % i, 'TMath::Cos(tau%sPhi)' % i)
    df = df.Define('tau%sSinPhi' % i, 'TMath::Sin(tau%sPhi)' % i)

    column = [el % i for el in tau_vars]
    np_dict = df.AsNumpy(column)
    np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
    tau_arrays.append(np_arr)

    column_mask = [el % i for el in tau_vars_mask]
    np_dict_mask = df.AsNumpy(column_mask)
    np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
    tau_arrays_mask.append(np_arr_mask)


# ---------------------------------------------------------------------------
# 5. TauPair (1 pair x 6 features) — unchanged
# ---------------------------------------------------------------------------
tau_pair_arrays = []
tau_pair_arrays_mask = []
tau_pair_vars = ['higgs3_mass_manu', 'higgs3_pt_manu', 'higgs3_eta_manu',
                 'higgs3_Sinphi_manu', 'higgs3_Cosphi_manu', 'deltaPhi_taupair_MET']
tau_pair_vars_mask = ['higgs3_mass_manu', 'higgs3_pt_manu']

df = df.Define('higgs3_Cosphi_manu', 'TMath::Cos(higgs3_phi_manu)')
df = df.Define('higgs3_Sinphi_manu', 'TMath::Sin(higgs3_phi_manu)')

column = tau_pair_vars
np_dict = df.AsNumpy(column)
np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
tau_pair_arrays.append(np_arr)

column_mask = tau_pair_vars_mask
np_dict_mask = df.AsNumpy(column_mask)
np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
tau_pair_arrays_mask.append(np_arr_mask)


# ---------------------------------------------------------------------------
# 6. Jet pairs (Jet1-Jet9, variable number per jet x 6 features) — unchanged
# ---------------------------------------------------------------------------
Jets_arrays = {}
Jets_arrays_mask = {}
Higgs_vars = ['massjet%sjet%s', 'ptjet%sjet%s', 'etajet%sjet%s',
              'sinphijet%sjet%s', 'cosphijet%sjet%s', 'drjet%sjet%s']
Higgs_vars_mask = ['massjet%sjet%s', 'ptjet%sjet%s']

for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
    name = 'Jet%s' % i
    Higgs_list = []
    Higgs_list_mask = []
    for j in ['2', '3', '4', '5', '6', '7', '8', '9', '10']:
        if i == j:
            continue
        if int(j) < int(i):
            continue
        df = df.Define('cosphijet%sjet%s' % (i, j), 'TMath::Cos(phijet%sjet%s)' % (i, j))
        df = df.Define('sinphijet%sjet%s' % (i, j), 'TMath::Sin(phijet%sjet%s)' % (i, j))

        column = [el % (i, j) for el in Higgs_vars]
        np_dict = df.AsNumpy(column)
        np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
        Higgs_list.append(np_arr)

        column_mask = [el % (i, j) for el in Higgs_vars_mask]
        np_dict_mask = df.AsNumpy(column_mask)
        np_arr_mask = np.vstack(np_dict_mask[col] for col in column_mask).T.astype(np.float32)
        Higgs_list_mask.append(np_arr_mask)

    Jets_arrays[name] = Higgs_list
    Jets_arrays_mask[name] = Higgs_list_mask


# ---------------------------------------------------------------------------
# 7. MET (1 x 1 feature) — unchanged
# ---------------------------------------------------------------------------
met_arrays = []
met_vars = ['met']
column = [el for el in met_vars]
np_dict = df.AsNumpy(column)
np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
met_arrays.append(np_arr)


# ---------------------------------------------------------------------------
# 8. HT (1 x 2 features) — changed: added kind_category_analysis
# ---------------------------------------------------------------------------
ht_arrays = []
ht_vars = ['ht', 'kind_category_analysis']  # was just ['ht']
column = [el for el in ht_vars]
np_dict = df.AsNumpy(column)
np_arr = np.vstack(np_dict[col] for col in column).T.astype(np.float32)
ht_arrays.append(np_arr)


# ---------------------------------------------------------------------------
# 4-vectors for truth matching
# ---------------------------------------------------------------------------
jet_4vec = ["%sPt", "%sEta", "%sPhi", "%sMass", "%sHiggsMatchedIndex", "%sbRegCorr"]
array_4vec = []
for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
    column_4vec = [el % 'jet%s' % i for el in jet_4vec]
    np_4vec = df.AsNumpy(column_4vec)
    np_arr_4vec = np.vstack(np_4vec[col] for col in column_4vec).T
    array_4vec.append(np_arr_4vec)

jets = []
for i in range(len(array_4vec[0])):
    jets_tmp = []
    for j in range(10):
        jet = ROOT.TLorentzVector()
        jet.SetPtEtaPhiM(array_4vec[j][i][0], array_4vec[j][i][1],
                          array_4vec[j][i][2], array_4vec[j][i][3])
        jet_corrected = ROOT.TLorentzVector()
        jet_corrected.SetPtEtaPhiE(jet.Pt() * array_4vec[j][i][5], jet.Eta(),
                                    jet.Phi(), jet.E() * array_4vec[j][i][5])
        jet_corrected.HiggsMatchedIndex = array_4vec[j][i][4]
        jets_tmp.append(jet_corrected)
    jets.append(jets_tmp)

taus_4vec = ["%sPt", "%sEta", "%sPhi", "%sMass"]
tau_truth_4vec = []
array_taus_4vec = []
for i in ['1', '2', '3', '4']:
    column_4vec = [el % 'tau%s' % i for el in taus_4vec]
    np_4vec = df.AsNumpy(column_4vec)
    np_arr_4vec = np.vstack(np_4vec[col] for col in column_4vec).T
    array_taus_4vec.append(np_arr_4vec)
for i in range(len(array_taus_4vec[0])):
    taus_tmp = []
    for j in range(4):
        _tau = ROOT.TLorentzVector()
        _tau.SetPtEtaPhiM(array_4vec[j][i][0], array_4vec[j][i][1],
                           array_4vec[j][i][2], array_4vec[j][i][3])
        taus_tmp.append(_tau)
    tau_truth_4vec.append(taus_tmp)

fatjet_4vec = ["%sPt", "%sEta", "%sPhi", "%sMass", "%sHiggsMatchedIndex"]
array_fj_4vec = []
for i in ['1', '2', '3', '4']:
    column_4vec = [el % 'fatJet%s' % i for el in fatjet_4vec]
    np_4vec = df.AsNumpy(column_4vec)
    np_arr_4vec = np.vstack(np_4vec[col] for col in column_4vec).T
    array_fj_4vec.append(np_arr_4vec)

fatjets = []
for i in range(len(array_fj_4vec[0])):
    jets_tmp = []
    for j in range(4):
        jet = ROOT.TLorentzVector()
        jet.SetPtEtaPhiM(array_fj_4vec[j][i][0], array_fj_4vec[j][i][1],
                          array_fj_4vec[j][i][2], array_fj_4vec[j][i][3])
        jet.HiggsMatchedIndex = array_fj_4vec[j][i][4]
        jets_tmp.append(jet)
    fatjets.append(jets_tmp)


# ---------------------------------------------------------------------------
# Construct input tensors
# ---------------------------------------------------------------------------
Jets_data = np.transpose(arrays, (1, 0, 2))
Jets_data_mask = np.transpose(arrays_jetmask, (1, 0, 2))
MIN_PT = 1
Jets_Pt = Jets_data_mask[:, :, 0]
Jets_mask = Jets_Pt > MIN_PT

BoostedJets_data = np.transpose(boosted_arrays, (1, 0, 2))
BoostedJets_data_mask = np.transpose(boosted_arrays_mask, (1, 0, 2))
MIN_FJPT = 1
BoostedJets_Pt = BoostedJets_data_mask[:, :, 0]
BoostedJets_mask = BoostedJets_Pt > MIN_FJPT

Leptons_data = np.transpose(lep_arrays, (1, 0, 2))
Leptons_data_mask = np.transpose(lep_arrays_mask, (1, 0, 2))
Leptons_Pt = Leptons_data_mask[:, :, 0]
Leptons_mask = Leptons_Pt > 3

Taus_data = np.transpose(tau_arrays, (1, 0, 2))
Taus_data_mask = np.transpose(tau_arrays_mask, (1, 0, 2))
Taus_Pt = Taus_data_mask[:, :, 1]
Taus_mask = Taus_Pt > 10

Taus_pair_data = np.transpose(tau_pair_arrays, (1, 0, 2))
Taus_pair_data_mask = np.transpose(tau_pair_arrays_mask, (1, 0, 2))
Taus_pair_Pt = Taus_pair_data_mask[:, :, 1]
Taus_pair_mask = Taus_pair_Pt > 0

MET_data = np.transpose(met_arrays, (1, 0, 2))
MET_mask = MET_data[:, :, 0] > -999

HT_data = np.transpose(ht_arrays, (1, 0, 2))  # shape: [batch, 1, 2]
HT_mask = MET_data[:, :, 0] > -999  # always True

# Jet pair tensors
Jet1_data = np.transpose(Jets_arrays['Jet1'], (1, 0, 2))
Jet1_data_mask = np.transpose(Jets_arrays_mask['Jet1'], (1, 0, 2))
Jet1_Mass = Jet1_data_mask[:, :, 0]
Jet1_mask = Jet1_Mass > 20

Jet2_data = np.transpose(Jets_arrays['Jet2'], (1, 0, 2))
Jet2_data_mask = np.transpose(Jets_arrays_mask['Jet2'], (1, 0, 2))
Jet2_Mass = Jet2_data_mask[:, :, 0]
Jet2_mask = Jet2_Mass > 20

Jet3_data = np.transpose(Jets_arrays['Jet3'], (1, 0, 2))
Jet3_data_mask = np.transpose(Jets_arrays_mask['Jet3'], (1, 0, 2))
Jet3_Mass = Jet3_data_mask[:, :, 0]
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


# ---------------------------------------------------------------------------
# Run inference (both models)
# ---------------------------------------------------------------------------
input_dict = {
    "Jets_data": Jets_data, "Jets_mask": Jets_mask,
    "FJets_data": BoostedJets_data, "FJets_mask": BoostedJets_mask,
    "Lep_data": Leptons_data, "Lep_mask": Leptons_mask,
    "Jet1_data": Jet1_data, "Jet1_mask": Jet1_mask,
    "Jet2_data": Jet2_data, "Jet2_mask": Jet2_mask,
    "Jet3_data": Jet3_data, "Jet3_mask": Jet3_mask,
    "Jet4_data": Jet4_data, "Jet4_mask": Jet4_mask,
    "Jet5_data": Jet5_data, "Jet5_mask": Jet5_mask,
    "Jet6_data": Jet6_data, "Jet6_mask": Jet6_mask,
    "Jet7_data": Jet7_data, "Jet7_mask": Jet7_mask,
    "Jet8_data": Jet8_data, "Jet8_mask": Jet8_mask,
    "Jet9_data": Jet9_data, "Jet9_mask": Jet9_mask,
    "Taus_data": Taus_data, "Taus_mask": Taus_mask,
    "TauPair_data": Taus_pair_data, "TauPair_mask": Taus_pair_mask,
    "MET_data": MET_data, "MET_mask": MET_mask,
    "HT_data": HT_data, "HT_mask": HT_mask,
}

# Classification model -> Higgs matching (7 classes) + jet assignment
output_nodes_cls = session_cls.get_outputs()
output_names_cls = [node.name for node in output_nodes_cls]
output_values_cls = session_cls.run(output_names_cls, input_dict)
print("Classification model outputs:", output_names_cls)

# Category model -> Process ID (12 classes)
output_nodes_cat = session_cat.get_outputs()
output_names_cat = [node.name for node in output_nodes_cat]
output_values_cat = session_cat.run(output_names_cat, input_dict)
print("Category model outputs:", output_names_cat)

# Output index for classification/category outputs (both at index 8)
CLS_IDX = 8  # EVENT/classfication_signal (Higgs matching, 7 classes)
CAT_IDX = 8  # EVENT/signal (Process ID, 12 classes)


# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------
h1_mass, h1_pt, h1_eta, h1_phi, h1_match = [], [], [], [], []
h2_mass, h2_pt, h2_eta, h2_phi, h2_match = [], [], [], [], []
h3_mass, h3_pt, h3_eta, h3_phi = [], [], [], []

# Process ID probabilities (from category model, 12 classes)
# Index mapping: 0=Other, 1=HHH4b2tau, 2=HHH6b, 3=HH4b, 4=HH2b2tau, 5=QCD,
#                6=TTTo2L2Nu, 7=TTToSemiLep, 8=TTToHadronic, 9=WJets, 10=ZJets, 11=VV
prob_hhh4b2tau = []
prob_hhh = []  # HHH6b
prob_hh4b = []
prob_hh2b2tau = []
prob_qcd = []
prob_ttlep = []
prob_ttSemi = []
prob_ttHard = []
prob_vjets = []  # WJets
prob_zjets = []  # ZJets (new)
prob_vv = []

# Higgs matching probabilities (from classification model, 7 classes)
# Index mapping: 0=NoHiggs, 1=1rh0bh, 2=2rh0bh, 3=0rh1bh, 4=1rh1bh, 5=unused, 6=0rh2bh
prob_0rh0bh = []  # No Higgs matched
prob_1rh0bh = []  # 1 resolved
prob_2rh0bh = []  # 2 resolved
prob_0rh1bh = []  # 1 boosted
prob_1rh1bh = []  # 1 resolved + 1 boosted
prob_0rh2bh = []  # 2 boosted

dummy_particle = []
for i in range(len(array_fj_4vec[0])):
    jets_tmp = []
    for j in range(4):
        jet = ROOT.TLorentzVector()
        jet.SetPtEtaPhiM(0, 0, 0, 0)
        jet.HiggsMatchedIndex = False
        jets_tmp.append(jet)
    dummy_particle.append(jets_tmp)


for i in range(len(output_values_cls[0])):
    best = process(i)

    jets_tmp = jets[i]
    tau_tmp = tau_truth_4vec[i]
    fjets_tmp = fatjets[i]
    dummy_particle_tmp = dummy_particle[i]

    h1_index = get_best(best, 0)
    h2_index = get_best(best, 1)
    h3_index = best[2]
    h1 = jets_tmp[int(h1_index[0])] + jets_tmp[int(h1_index[1])]
    h1.HiggsMatchedIndex = (jets_tmp[int(h1_index[0])].HiggsMatchedIndex == jets_tmp[int(h1_index[1])].HiggsMatchedIndex
                            and jets_tmp[int(h1_index[1])].HiggsMatchedIndex > 0) or \
                           (5 in [jets_tmp[int(h1_index[0])].HiggsMatchedIndex,
                                  jets_tmp[int(h1_index[1])].HiggsMatchedIndex]
                            and jets_tmp[int(h1_index[0])].HiggsMatchedIndex > 0
                            and jets_tmp[int(h1_index[1])].HiggsMatchedIndex > 0)
    h2 = jets_tmp[int(h2_index[0])] + jets_tmp[int(h2_index[1])]
    h2.HiggsMatchedIndex = (jets_tmp[int(h2_index[0])].HiggsMatchedIndex == jets_tmp[int(h2_index[1])].HiggsMatchedIndex
                            and jets_tmp[int(h2_index[1])].HiggsMatchedIndex > 0) or \
                           (5 in [jets_tmp[int(h2_index[0])].HiggsMatchedIndex,
                                  jets_tmp[int(h2_index[1])].HiggsMatchedIndex]
                            and jets_tmp[int(h2_index[0])].HiggsMatchedIndex > 0
                            and jets_tmp[int(h2_index[1])].HiggsMatchedIndex > 0)

    if not h3_index:
        h3 = dummy_particle_tmp[0]
    else:
        h3 = tau_tmp[int(h3_index[0])] + tau_tmp[int(h3_index[1])]

    bh1_index = best[3]
    bh2_index = best[4]
    if bh1_index == -1:
        bh1 = dummy_particle_tmp[0]
    else:
        bh1 = fjets_tmp[bh1_index]
    if bh2_index == -1:
        bh2 = dummy_particle_tmp[0]
    else:
        bh2 = fjets_tmp[bh2_index]

    higgses = [h1, h2, h3]
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

    # Process ID from category model (index 8, 12 classes)
    prob_hhh4b2tau.append(float(output_values_cat[CAT_IDX][i][1]))
    prob_hhh.append(float(output_values_cat[CAT_IDX][i][2]))
    prob_hh4b.append(float(output_values_cat[CAT_IDX][i][3]))
    prob_hh2b2tau.append(float(output_values_cat[CAT_IDX][i][4]))
    prob_qcd.append(float(output_values_cat[CAT_IDX][i][5]))
    prob_ttlep.append(float(output_values_cat[CAT_IDX][i][6]))
    prob_ttSemi.append(float(output_values_cat[CAT_IDX][i][7]))
    prob_ttHard.append(float(output_values_cat[CAT_IDX][i][8]))
    prob_vjets.append(float(output_values_cat[CAT_IDX][i][9]))
    prob_zjets.append(float(output_values_cat[CAT_IDX][i][10]))
    prob_vv.append(float(output_values_cat[CAT_IDX][i][11]))

    # Higgs matching from classification model (index 8, 7 classes)
    prob_0rh0bh.append(float(output_values_cls[CLS_IDX][i][0]))
    prob_1rh0bh.append(float(output_values_cls[CLS_IDX][i][1]))
    prob_2rh0bh.append(float(output_values_cls[CLS_IDX][i][2]))
    prob_0rh1bh.append(float(output_values_cls[CLS_IDX][i][3]))
    prob_1rh1bh.append(float(output_values_cls[CLS_IDX][i][4]))
    prob_0rh2bh.append(float(output_values_cls[CLS_IDX][i][6]))


# ---------------------------------------------------------------------------
# Attach results to RDataFrame and save
# ---------------------------------------------------------------------------
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

arr_prob_hhh4b2tau = ROOT.VecOps.AsRVec(np.array(prob_hhh4b2tau))
arr_prob_hhh = ROOT.VecOps.AsRVec(np.array(prob_hhh))
arr_prob_hh4b = ROOT.VecOps.AsRVec(np.array(prob_hh4b))
arr_prob_hh2b2tau = ROOT.VecOps.AsRVec(np.array(prob_hh2b2tau))
arr_prob_qcd = ROOT.VecOps.AsRVec(np.array(prob_qcd))
arr_prob_ttlep = ROOT.VecOps.AsRVec(np.array(prob_ttlep))
arr_prob_ttSemi = ROOT.VecOps.AsRVec(np.array(prob_ttSemi))
arr_prob_ttHard = ROOT.VecOps.AsRVec(np.array(prob_ttHard))
arr_prob_vjets = ROOT.VecOps.AsRVec(np.array(prob_vjets))
arr_prob_zjets = ROOT.VecOps.AsRVec(np.array(prob_zjets))
arr_prob_vv = ROOT.VecOps.AsRVec(np.array(prob_vv))

arr_prob_0rh0bh = ROOT.VecOps.AsRVec(np.array(prob_0rh0bh))
arr_prob_1rh0bh = ROOT.VecOps.AsRVec(np.array(prob_1rh0bh))
arr_prob_2rh0bh = ROOT.VecOps.AsRVec(np.array(prob_2rh0bh))
arr_prob_0rh1bh = ROOT.VecOps.AsRVec(np.array(prob_0rh1bh))
arr_prob_1rh1bh = ROOT.VecOps.AsRVec(np.array(prob_1rh1bh))
arr_prob_0rh2bh = ROOT.VecOps.AsRVec(np.array(prob_0rh2bh))

# Higgs candidates
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_mass, "h1_spanet_mass")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_pt, "h1_spanet_pt")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_eta, "h1_spanet_eta")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h1_phi, "h1_spanet_phi")
df = ROOT.AddBoolArray(ROOT.RDF.AsRNode(df), arr_h1_match, "h1_spanet_match")

df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_mass, "h2_spanet_mass")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_pt, "h2_spanet_pt")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_eta, "h2_spanet_eta")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_h2_phi, "h2_spanet_phi")
df = ROOT.AddBoolArray(ROOT.RDF.AsRNode(df), arr_h2_match, "h2_spanet_match")

# Process ID probabilities
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hhh4b2tau, "ProbHHH4b2tau")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hhh, "ProbHHH6b")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hh4b, "ProbHH4b")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_hh2b2tau, "ProbHH2b2tau")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_qcd, "ProbQCD")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_ttlep, "ProbTTlep")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_ttSemi, "ProbTTSemi")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_ttHard, "ProbTTHard")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_vjets, "ProbWJets")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_zjets, "ProbZJets")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_vv, "ProbVV")

# Higgs matching probabilities
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh0bh, "Prob0rh0bh")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_1rh0bh, "Prob1rh0bh")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_2rh0bh, "Prob2rh0bh")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh1bh, "Prob0rh1bh")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_1rh1bh, "Prob1rh1bh")
df = ROOT.AddArray(ROOT.RDF.AsRNode(df), arr_prob_0rh2bh, "Prob0rh2bh")


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
print("Saving output")
if args.output_dir:
    output_path = args.output_dir
else:
    # Auto-derive: replace /parts/ with /parts_SPANET_v9/
    output_path = args.path.replace('/parts/', '/parts_SPANET_v9/')
    output_path = output_path.replace('root://eosuser.cern.ch/', '')

if not os.path.isdir(output_path):
    os.makedirs(output_path)

output_name = f_in + '_%s' % args.batch_number + '.root'
output_full = os.path.join(output_path, output_name)
print(output_full)

df.Snapshot('Events', output_full)
print("Done!")
