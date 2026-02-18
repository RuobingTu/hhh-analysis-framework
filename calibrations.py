# Utility to add calibrations
# git clone ssh://git@gitlab.cern.ch:7999/cms-nanoAOD/jsonpog-integration.git

import os
import ROOT
import correctionlib
correctionlib.register_pyroot_binding()


# ============================================================================
# Tau ID Scale Factor Functions
# ============================================================================

computeTauIDSF = '''
// Compute Tau ID SF for a single tau
// Returns 1.0 if:
//   - tau pt <= 0 (no tau reconstructed in this slot)
//   - genPartFlav != 5 (not a real hadronic tau)
//   - invalid decay mode
//
// genPartFlav values:
//   0 = fake (jet -> tau)
//   1 = prompt electron
//   2 = prompt muon
//   3 = electron from tau decay
//   4 = muon from tau decay
//   5 = real hadronic tau (apply SF only to this!)
//
// This makes the function robust for all categories:
//   2tau0l: both taus get SF
//   1tau1l: hadronic tau gets SF, lepton slot returns 1.0
//   1tau0l: reconstructed tau gets SF, missing slot returns 1.0
//   0tau2l: both slots return 1.0

float computeTauIDSF_single(float pt, float eta, int decayMode, int genPartFlav,
                             const std::string& syst_vsjet, const std::string& syst_vsmu,
                             const std::string& syst_vsele, const std::string& wpVsJet) {
    // Check if tau exists (valid reconstruction)
    if (pt <= 0) return 1.0f;

    // Only apply SF to real hadronic taus in MC (genPartFlav == 5)
    // For fakes (genPartFlav == 0) or leptons (1-4), return 1.0
    if (genPartFlav != 5) return 1.0f;

    // Valid decay modes for DeepTau
    if (decayMode != 0 && decayMode != 1 && decayMode != 10 && decayMode != 11) return 1.0f;

    float sf_vsjet = taujson_vsjet->evaluate({pt, decayMode, genPartFlav, wpVsJet, "VVLoose", syst_vsjet, "dm"});
    float sf_vsmu = taujson_vsmu->evaluate({std::abs(eta), genPartFlav, "VLoose", syst_vsmu});
    float sf_vsele = taujson_vsele->evaluate({std::abs(eta), genPartFlav, "VVLoose", syst_vsele});

    return sf_vsjet * sf_vsmu * sf_vsele;
}
'''

def tau_init(year):
    """Initialize Tau ID SF from TauPOG JSON files."""
    if year in ['2016APV', '2016', '2017', '2018']:
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/TAU/%s_UL/tau.json.gz'%year)
    elif year == '2022':
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/TAU/2022_preEE/tau.json.gz')
    elif year == '2022EE':
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/TAU/2022_postEE/tau.json.gz')
    else:
        print(f"Warning: Unknown year {year} for Tau SF, using 2017_UL")
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/TAU/2017_UL/tau.json.gz')

    print(f"Loading Tau SF from: {sfDir}")
    ROOT.gInterpreter.Declare('auto taujson = correction::CorrectionSet::from_file("%s");'%sfDir)

    # For Run2 UL, use DeepTau2017v2p1
    if year in ['2016APV', '2016', '2017', '2018']:
        ROOT.gInterpreter.Declare('auto taujson_vsjet = taujson->at("DeepTau2017v2p1VSjet");')
        ROOT.gInterpreter.Declare('auto taujson_vsmu = taujson->at("DeepTau2017v2p1VSmu");')
        ROOT.gInterpreter.Declare('auto taujson_vsele = taujson->at("DeepTau2017v2p1VSe");')
    # For Run3, use DeepTau2018v2p5
    elif year in ['2022', '2022EE']:
        ROOT.gInterpreter.Declare('auto taujson_vsjet = taujson->at("DeepTau2018v2p5VSjet");')
        ROOT.gInterpreter.Declare('auto taujson_vsmu = taujson->at("DeepTau2018v2p5VSmu");')
        ROOT.gInterpreter.Declare('auto taujson_vsele = taujson->at("DeepTau2018v2p5VSe");')

    ROOT.gInterpreter.Declare(computeTauIDSF)


def addTauIDSF(df, f_in, wp='Medium'):
    """
    Add Tau ID SF to RDataFrame for selected taus.

    Works for all categories (robust to missing taus):
      - 2tau0l: SF = tau1SF * tau2SF
      - 1tau1l: SF = tau1SF * 1.0 (lepton has genPartFlav != 5)
      - 1tau0l: SF = tau1SF * 1.0 (missing tau has pt <= 0)
      - 0tau2l: SF = 1.0 * 1.0

    Requires columns: tau1Pt, tau1Eta, tau1decayMode, tau1genPartFlav (and tau2*)
    These are already output by hhh6bProducerPNetAK4.

    Args:
        df: RDataFrame
        f_in: Input filename (used to check if data)
        wp: Working point for VSjet ('VVLoose', 'VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight')
    """
    # Check if this is data (no SF for data)
    is_data = any(x in f_in for x in ['JetHT', 'BTagCSV', 'Tau', 'SingleMuon', 'EGamma', 'MuonEG'])

    if is_data:
        df = df.Define('tau1IDSF', '1.0f')
        df = df.Define('tau2IDSF', '1.0f')
        df = df.Define('tauIDSF_weight', '1.0f')
    else:
        # Compute SF for each tau (note: column names are lowercase from producer)
        df = df.Define('tau1IDSF',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "nom", "nom", "nom", "{wp}")')
        df = df.Define('tau2IDSF',
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "nom", "nom", "nom", "{wp}")')
        # Combined weight for 2 taus
        df = df.Define('tauIDSF_weight', 'tau1IDSF * tau2IDSF')

    return df


def addTauIDSF_systematics(df, f_in, wp='Medium'):
    """
    Add Tau ID SF with systematic variations.

    Works for all categories (robust to missing taus):
      - 2tau0l: SF = tau1SF * tau2SF
      - 1tau1l: SF = tau1SF * 1.0 (lepton has genPartFlav != 5)
      - 1tau0l: SF = tau1SF * 1.0 (missing tau has pt <= 0)
      - 0tau2l: SF = 1.0 * 1.0

    Adds branches for:
    - tauIDSF_weight (central)
    - tauIDSF_weight_vsjet_up/down
    - tauIDSF_weight_vsmu_up/down
    - tauIDSF_weight_vsele_up/down

    Args:
        df: RDataFrame
        f_in: Input filename (used to check if data)
        wp: Working point for VSjet
    """
    is_data = any(x in f_in for x in ['JetHT', 'BTagCSV', 'Tau', 'SingleMuon', 'EGamma', 'MuonEG'])

    if is_data:
        for syst in ['', '_vsjet_up', '_vsjet_down', '_vsmu_up', '_vsmu_down', '_vsele_up', '_vsele_down']:
            df = df.Define(f'tauIDSF_weight{syst}', '1.0f')
    else:
        # Central value (note: column names are lowercase from producer)
        df = df.Define('tauIDSF_weight',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "nom", "nom", "nom", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "nom", "nom", "nom", "{wp}")')

        # VSjet systematics
        df = df.Define('tauIDSF_weight_vsjet_up',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "up", "nom", "nom", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "up", "nom", "nom", "{wp}")')
        df = df.Define('tauIDSF_weight_vsjet_down',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "down", "nom", "nom", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "down", "nom", "nom", "{wp}")')

        # VSmu systematics
        df = df.Define('tauIDSF_weight_vsmu_up',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "nom", "up", "nom", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "nom", "up", "nom", "{wp}")')
        df = df.Define('tauIDSF_weight_vsmu_down',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "nom", "down", "nom", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "nom", "down", "nom", "{wp}")')

        # VSele systematics
        df = df.Define('tauIDSF_weight_vsele_up',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "nom", "nom", "up", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "nom", "nom", "up", "{wp}")')
        df = df.Define('tauIDSF_weight_vsele_down',
            f'computeTauIDSF_single(tau1Pt, tau1Eta, (int)tau1decayMode, tau1genPartFlav, "nom", "nom", "down", "{wp}") * '
            f'computeTauIDSF_single(tau2Pt, tau2Eta, (int)tau2decayMode, tau2genPartFlav, "nom", "nom", "down", "{wp}")')

    return df


# ============================================================================
# Lepton ID Scale Factor Functions
# ============================================================================

computeLeptonIDSF = '''
// Compute Lepton ID SF for a single lepton (electron or muon)
// Returns 1.0 if:
//   - lepton pt <= 0 (no lepton reconstructed in this slot)
//
// lepId values (PDG ID × charge):
//   ±11 = electron
//   ±13 = muon
//
// Electron SF = Reco SF × ID SF
//   - Reco SF: P(electron reconstructed from supercluster)
//   - ID SF: P(pass ID | reconstructed)
//
// Muon SF = ID SF (Reco SF ~1.0 for muons)
//
// This makes the function robust for all categories:
//   0tau2l: both leptons get SF
//   1tau1l: one lepton gets SF, other slot returns 1.0
//   2tau0l/1tau0l: both slots return 1.0

float computeElectronRecoSF_single(float pt, float eta, const std::string& valtype) {
    if (pt <= 0) return 1.0f;
    // Reco SF: use RecoBelow20 for pt<20, RecoAbove20 for pt>=20
    std::string recoWP = (pt < 20) ? "RecoBelow20" : "RecoAbove20";
    return elejson->evaluate({eleyear, valtype, recoWP, eta, pt});
}

float computeElectronIDSF_single(float pt, float eta, const std::string& valtype, const std::string& wp) {
    if (pt <= 0) return 1.0f;
    // ID SF: uses the specified working point (e.g., wp90noiso for mvaFall17V2noIso_WP90)
    return elejson->evaluate({eleyear, valtype, wp, eta, pt});
}

float computeElectronSF_single(float pt, float eta, const std::string& valtype, const std::string& wp) {
    // Total electron SF = Reco SF × ID SF
    if (pt <= 0) return 1.0f;
    float reco_sf = computeElectronRecoSF_single(pt, eta, valtype);
    float id_sf = computeElectronIDSF_single(pt, eta, valtype, wp);
    return reco_sf * id_sf;
}

float computeMuonIDSF_single(float pt, float eta, const std::string& valtype) {
    if (pt <= 0) return 1.0f;
    // Muon SF: eta range -2.4 to 2.4, pt typically 15-120 GeV
    // Clamp pt to valid range to avoid extrapolation errors
    float pt_clamped = std::min(std::max(pt, 15.0f), 120.0f);
    return muojson_id->evaluate({std::abs(eta), pt_clamped, valtype});
}

float computeLeptonIDSF_single(float pt, float eta, int lepId,
                                const std::string& ele_valtype, const std::string& mu_valtype,
                                const std::string& ele_wp) {
    if (pt <= 0) return 1.0f;

    int absId = std::abs(lepId);
    if (absId == 11) {
        // Electron: Reco SF × ID SF
        return computeElectronSF_single(pt, eta, ele_valtype, ele_wp);
    } else if (absId == 13) {
        // Muon: ID SF only
        return computeMuonIDSF_single(pt, eta, mu_valtype);
    }
    return 1.0f;  // Unknown lepton type
}
'''

def lepton_init(year):
    """Initialize Lepton ID SF from EGamma and Muon POG JSON files."""
    # Electron SF
    if year in ['2016APV', '2016', '2017', '2018']:
        eleDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/EGM/%s_UL/electron.json.gz'%year)
    elif year == '2022':
        eleDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/EGM/2022_Summer22/electron.json.gz')
    elif year == '2022EE':
        eleDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/EGM/2022_Summer22EE/electron.json.gz')
    else:
        print(f"Warning: Unknown year {year} for Electron SF, using 2017_UL")
        eleDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/EGM/2017_UL/electron.json.gz')

    # Muon SF
    if year in ['2016APV', '2016', '2017', '2018']:
        muoDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/MUO/%s_UL/muon_Z.json.gz'%year)
    elif year == '2022':
        muoDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/MUO/2022_Summer22/muon_Z.json.gz')
    elif year == '2022EE':
        muoDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/MUO/2022_Summer22EE/muon_Z.json.gz')
    else:
        print(f"Warning: Unknown year {year} for Muon SF, using 2017_UL")
        muoDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/MUO/2017_UL/muon_Z.json.gz')

    # Map year to JSON year string for electron SF
    ele_year_map = {
        '2016APV': '2016preVFP',
        '2016': '2016postVFP',
        '2017': '2017',
        '2018': '2018',
        '2022': '2022Re-recoBCD',
        '2022EE': '2022Re-recoE+PromptFG',
    }
    ele_year_str = ele_year_map.get(year, '2017')

    print(f"Loading Electron SF from: {eleDir}")
    print(f"Loading Muon SF from: {muoDir}")

    # Declare year string for electron SF evaluation
    ROOT.gInterpreter.Declare('std::string eleyear = "%s";'%ele_year_str)

    ROOT.gInterpreter.Declare('auto elejson_set = correction::CorrectionSet::from_file("%s");'%eleDir)
    ROOT.gInterpreter.Declare('auto elejson = elejson_set->at("UL-Electron-ID-SF");')

    ROOT.gInterpreter.Declare('auto muojson_set = correction::CorrectionSet::from_file("%s");'%muoDir)
    # Use MediumID for muons (matches producer selection: mu.mediumId)
    ROOT.gInterpreter.Declare('auto muojson_id = muojson_set->at("NUM_MediumID_DEN_TrackerMuons");')

    ROOT.gInterpreter.Declare(computeLeptonIDSF)


def addLeptonIDSF(df, f_in, ele_wp='wp90noiso'):
    """
    Add Lepton ID SF to RDataFrame for selected leptons.

    Works for all categories (robust to missing leptons):
      - 0tau2l: SF = lep1SF * lep2SF
      - 1tau1l: SF = lep1SF * 1.0 (only one lepton)
      - 2tau0l/1tau0l: SF = 1.0 * 1.0 (no leptons)

    Requires columns: lep1Pt, lep1Eta, lep1Id, lep2Pt, lep2Eta, lep2Id
    These are already output by hhh6bProducerPNetAK4.

    Args:
        df: RDataFrame
        f_in: Input filename (used to check if data)
        ele_wp: Electron WP ('wp90noiso' for mvaFall17V2noIso_WP90, 'wp90iso', 'Medium', etc.)
    """
    is_data = any(x in f_in for x in ['JetHT', 'BTagCSV', 'Tau', 'SingleMuon', 'EGamma', 'MuonEG'])

    if is_data:
        df = df.Define('lep1IDSF', '1.0f')
        df = df.Define('lep2IDSF', '1.0f')
        df = df.Define('leptonIDSF_weight', '1.0f')
    else:
        # Compute SF for each lepton
        df = df.Define('lep1IDSF',
            f'computeLeptonIDSF_single(lep1Pt, lep1Eta, lep1Id, "sf", "sf", "{ele_wp}")')
        df = df.Define('lep2IDSF',
            f'computeLeptonIDSF_single(lep2Pt, lep2Eta, lep2Id, "sf", "sf", "{ele_wp}")')
        # Combined weight for 2 leptons
        df = df.Define('leptonIDSF_weight', 'lep1IDSF * lep2IDSF')

    return df


def addLeptonIDSF_systematics(df, f_in, ele_wp='wp90noiso'):
    """
    Add Lepton ID SF with systematic variations.

    Works for all categories (robust to missing leptons):
      - 0tau2l: SF = lep1SF * lep2SF
      - 1tau1l: SF = lep1SF * 1.0
      - 2tau0l/1tau0l: SF = 1.0 * 1.0

    Adds branches for:
    - leptonIDSF_weight (central)
    - leptonIDSF_weight_up/down (combined systematic)

    Args:
        df: RDataFrame
        f_in: Input filename (used to check if data)
        ele_wp: Electron WP
    """
    is_data = any(x in f_in for x in ['JetHT', 'BTagCSV', 'Tau', 'SingleMuon', 'EGamma', 'MuonEG'])

    if is_data:
        for syst in ['', '_up', '_down']:
            df = df.Define(f'leptonIDSF_weight{syst}', '1.0f')
    else:
        # Central value
        df = df.Define('leptonIDSF_weight',
            f'computeLeptonIDSF_single(lep1Pt, lep1Eta, lep1Id, "sf", "sf", "{ele_wp}") * '
            f'computeLeptonIDSF_single(lep2Pt, lep2Eta, lep2Id, "sf", "sf", "{ele_wp}")')

        # Up systematic (electron sfup, muon systup)
        df = df.Define('leptonIDSF_weight_up',
            f'computeLeptonIDSF_single(lep1Pt, lep1Eta, lep1Id, "sfup", "systup", "{ele_wp}") * '
            f'computeLeptonIDSF_single(lep2Pt, lep2Eta, lep2Id, "sfup", "systup", "{ele_wp}")')

        # Down systematic (electron sfdown, muon systdown)
        df = df.Define('leptonIDSF_weight_down',
            f'computeLeptonIDSF_single(lep1Pt, lep1Eta, lep1Id, "sfdown", "systdown", "{ele_wp}") * '
            f'computeLeptonIDSF_single(lep2Pt, lep2Eta, lep2Id, "sfdown", "systdown", "{ele_wp}")')

    return df


def addBTagSF(df, f_in): # shape scale factors for MVA
    if 'JetHT' in f_in or 'BTagCSV' in f_in:
        df = df.Define('bcand1BTagSF', '1')
        df = df.Define('bcand2BTagSF', '1')
        df = df.Define('bcand3BTagSF', '1')
        df = df.Define('bcand4BTagSF', '1')
        df = df.Define('bcand5BTagSF', '1')
        df = df.Define('bcand6BTagSF', '1')
        df = df.Define('jet7BTagSF', '1')
        df = df.Define('jet8BTagSF', '1')
        df = df.Define('jet9BTagSF', '1')
        df = df.Define('jet10BTagSF', '1')
    else:
        df = df.Define('bcand1BTagSF', 'btvjson_shape->evaluate({"central",int(round(bcand1HadronFlavour)),std::abs(bcand1Eta),bcand1Pt,bcand1DeepFlavB})')
        df = df.Define('bcand2BTagSF', 'btvjson_shape->evaluate({"central",int(round(bcand2HadronFlavour)),std::abs(bcand2Eta),bcand2Pt,bcand2DeepFlavB})')
        df = df.Define('bcand3BTagSF', 'btvjson_shape->evaluate({"central",int(round(bcand3HadronFlavour)),std::abs(bcand3Eta),bcand3Pt,bcand3DeepFlavB})')
        df = df.Define('bcand4BTagSF', 'btvjson_shape->evaluate({"central",int(round(bcand4HadronFlavour)),std::abs(bcand4Eta),bcand4Pt,bcand4DeepFlavB})')
        df = df.Define('bcand5BTagSF', 'btvjson_shape->evaluate({"central",int(round(bcand5HadronFlavour)),std::abs(bcand5Eta),bcand5Pt,bcand5DeepFlavB})')
        df = df.Define('bcand6BTagSF', 'btvjson_shape->evaluate({"central",int(round(bcand6HadronFlavour)),std::abs(bcand6Eta),bcand6Pt,bcand6DeepFlavB})')
        df = df.Define('jet7BTagSF', 'btvjson_shape->evaluate({"central",int(round(jet7HadronFlavour)),std::abs(jet7Eta),jet7Pt,jet7DeepFlavB})')
        df = df.Define('jet8BTagSF', 'btvjson_shape->evaluate({"central",int(round(jet8HadronFlavour)),std::abs(jet8Eta),jet8Pt,jet8DeepFlavB})')
        df = df.Define('jet9BTagSF', 'btvjson_shape->evaluate({"central",int(round(jet9HadronFlavour)),std::abs(jet9Eta),jet9Pt,jet9DeepFlavB})')
        df = df.Define('jet10BTagSF', 'btvjson_shape->evaluate({"central",int(round(jet10HadronFlavour)),std::abs(jet10Eta),jet10Pt,jet10DeepFlavB})')

    return df


computeLooseBTAGSF = '''
    float computeLooseBTagsEffSFPerFlavour(int hadronFlavour, float eta, float pt){
        float sf;
        if (hadronFlavour == 0) { sf = btvjson_incl->evaluate({"central","L",hadronFlavour,eta,pt});  }
        else { sf = btvjson_comb->evaluate({"central","L",hadronFlavour,eta,pt});} 
        return sf;
    }
'''

computeMediumBTAGSF = '''
    float computeMediumBTagsEffSFPerFlavour(int hadronFlavour, float eta, float pt){
        float sf;
        if (hadronFlavour == 0) { sf = btvjson_incl->evaluate({"central","M",hadronFlavour,eta,pt});  }
        else {sf = btvjson_comb->evaluate({"central","M",hadronFlavour,eta,pt});} 
        return sf;
    }
'''

computeTightBTAGSF = '''
    float computeTightBTagsEffSFPerFlavour(int hadronFlavour, float eta, float pt){
        float sf;
        if (hadronFlavour == 0) { sf = btvjson_incl->evaluate({"central","T",hadronFlavour,eta,pt});  }
        else {sf = btvjson_comb->evaluate({"central","T",hadronFlavour,eta,pt});} 
        return sf;
    }
'''

def btag_init(year):
    if year in ['2016APV', '2016', '2017', '2018']:
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/BTV/%s_UL/btagging.json.gz'%year)
    elif year == '2022':
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/BTV/2022_Summer22/btagging.json.gz')
    elif year == '2022EE':
        sfDir = os.path.join(os.environ["CMSSW_BASE"], 'src/jsonpog-integration/POG/BTV/2022_Summer22EE/btagging.json.gz')
    ROOT.gInterpreter.Declare('auto btvjson = correction::CorrectionSet::from_file("%s");'%sfDir)
    if year in ['2016APV', '2016', '2017', '2018']:
        ROOT.gInterpreter.Declare('auto btvjson_shape = btvjson->at("deepJet_shape");')
        ROOT.gInterpreter.Declare('auto btvjson_comb = btvjson->at("deepJet_comb");') # bc
        ROOT.gInterpreter.Declare('auto btvjson_incl = btvjson->at("deepJet_incl");') # light
        ROOT.gInterpreter.Declare(computeLooseBTAGSF)
        ROOT.gInterpreter.Declare(computeMediumBTAGSF)
        ROOT.gInterpreter.Declare(computeTightBTAGSF)
    elif year in ['2022', '2022EE']:
        # No SFs available yet; just WPs
        ROOT.gInterpreter.Declare('float computeLooseBTagsEffSFPerFlavour(int hadronFlavour, float eta, float pt){return 1.0;}')
        ROOT.gInterpreter.Declare('float computeMediumBTagsEffSFPerFlavour(int hadronFlavour, float eta, float pt){return 1.0;}')
        ROOT.gInterpreter.Declare('float computeTightBTagsEffSFPerFlavour(int hadronFlavour, float eta, float pt){return 1.0;}')



def addBTagEffSF(df,f_in,wp):

    if 'JetHT' in f_in or 'BTagCSV' in f_in:
        if wp == 'loose':
            df = df.Define('jet1LooseBTagEffSF', '1')
            df = df.Define('jet2LooseBTagEffSF', '1')
            df = df.Define('jet3LooseBTagEffSF', '1')
            df = df.Define('jet4LooseBTagEffSF', '1')
            df = df.Define('jet5LooseBTagEffSF', '1')
            df = df.Define('jet6LooseBTagEffSF', '1')
        elif wp == 'medium':
            df = df.Define('jet1MediumBTagEffSF', '1')
            df = df.Define('jet2MediumBTagEffSF', '1')
            df = df.Define('jet3MediumBTagEffSF', '1')
            df = df.Define('jet4MediumBTagEffSF', '1')
            df = df.Define('jet5MediumBTagEffSF', '1')
            df = df.Define('jet6MediumBTagEffSF', '1')
        elif wp == 'tight':
            df = df.Define('jet1TightBTagEffSF', '1')
            df = df.Define('jet2TightBTagEffSF', '1')
            df = df.Define('jet3TightBTagEffSF', '1')
            df = df.Define('jet4TightBTagEffSF', '1')
            df = df.Define('jet5TightBTagEffSF', '1')
            df = df.Define('jet6TightBTagEffSF', '1')
    else: 
        if wp == 'loose':
            name = 'LooseBTagEffSF'
            script = 'computeLooseBTagsEffSFPerFlavour'
        elif wp == 'medium':
            name = 'MediumBTagEffSF'
            script = 'computeMediumBTagsEffSFPerFlavour'
        elif wp == 'tight':
            name = 'TightBTagEffSF'
            script = 'computeTightBTagsEffSFPerFlavour'
        df = df.Define('jet1%s'%name, "%s(int(round(jet1HadronFlavour)),std::abs(jet1Eta),jet1Pt)"%script)
        df = df.Define('jet2%s'%name, "%s(int(round(jet2HadronFlavour)),std::abs(jet2Eta),jet2Pt)"%script)
        df = df.Define('jet3%s'%name, "%s(int(round(jet3HadronFlavour)),std::abs(jet3Eta),jet3Pt)"%script)
        df = df.Define('jet4%s'%name, "%s(int(round(jet4HadronFlavour)),std::abs(jet4Eta),jet4Pt)"%script)
        df = df.Define('jet5%s'%name, "%s(int(round(jet5HadronFlavour)),std::abs(jet5Eta),jet5Pt)"%script)
        df = df.Define('jet6%s'%name, "%s(int(round(jet6HadronFlavour)),std::abs(jet6Eta),jet6Pt)"%script)
    return df



