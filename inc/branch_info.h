
#ifndef BDTRAIN_BRANCH_INFO_H
#define BDTRAIN_BRANCH_INFO_H

struct BranchInfo
{
   void Init() {
    labels["label_b"] = false;
    labels["label_bbar"] = false;
    labels["label_c"] = false;
    labels["label_cbar"] = false;
    labels["label_u"] = false;
    labels["label_ubar"] = false;
    labels["label_d"] = false;
    labels["label_dbar"] = false;
    labels["label_s"] = false;
    labels["label_sbar"] = false;
    labels["label_g"] = false;

    floats["is_signal"] = 0;

    floats["gen_match"] = 0;
    floats["genpart_pt"] = 0;
    floats["genpart_eta"] = 0;
    floats["genpart_phi"] = 0;
    floats["genpart_pid"] = 0;

    floats["jet_pt"] = 0;
    floats["jet_eta"] = 0;
    floats["jet_phi"] = 0;
    floats["jet_energy"] = 0;
    floats["jet_nparticles"] = 0;
    floats["btag"] = 0;
    floats["ctag"] = 0;

    // Initialize std::vector<float> for array_floats
    array_floats["part_px"] = std::vector<float>();
    array_floats["part_py"] = std::vector<float>();
    array_floats["part_pz"] = std::vector<float>();
    array_floats["part_energy"] = std::vector<float>();
    array_floats["part_pt"] = std::vector<float>();
    array_floats["part_deta"] = std::vector<float>();
    array_floats["part_dphi"] = std::vector<float>();
    array_floats["part_charge"] = std::vector<float>();
    array_floats["part_pid"] = std::vector<float>();
    array_floats["part_d0val"] = std::vector<float>();
    array_floats["part_d0err"] = std::vector<float>();
    array_floats["part_dzval"] = std::vector<float>();
    array_floats["part_dzerr"] = std::vector<float>();
    array_floats["part_deltaR"] = std::vector<float>();
    array_floats["part_logptrel"] = std::vector<float>();
    array_floats["part_logerel"] = std::vector<float>();
    array_floats["part_e_log"] = std::vector<float>();
    array_floats["part_pt_log"] = std::vector<float>();
    array_floats["part_d0"] = std::vector<float>();
    array_floats["part_dz"] = std::vector<float>();
    array_floats["part_vtxX"] = std::vector<float>();
    array_floats["part_vtxY"] = std::vector<float>();
    array_floats["part_vtxZ"] = std::vector<float>();

    // Initialize std::vector<bool> for array_labels
    array_labels["part_isChargedHadron"] = std::vector<bool>();
    array_labels["part_isNeutralHadron"] = std::vector<bool>();
    array_labels["part_isPhoton"] = std::vector<bool>();
    array_labels["part_isElectron"] = std::vector<bool>();
    array_labels["part_isMuon"] = std::vector<bool>();
    array_labels["part_isPion"] = std::vector<bool>();
    array_labels["part_isProton"] = std::vector<bool>();
    array_labels["part_isChargedKaon"] = std::vector<bool>();
    array_labels["part_isKLong"] = std::vector<bool>();
    array_labels["part_isKShort"] = std::vector<bool>();
    array_labels["part_isPi0"] = std::vector<bool>();

    ints["event_number"] = -1;
}
    void Clear() {
    for (auto &[name, var] : labels) var = false;
    for (auto &[name, var] : floats) var = 0;
    for (auto &[name, var] : ints) var = -1;
    for (auto &[name, vars] : array_floats) vars = std::vector<float>();
    for (auto &[name, vars] : array_labels) vars = std::vector<bool>();
    }

    std::map<TString, bool>  labels;
    std::map<TString, float> floats;
    std::map<TString, int>   ints;
    std::map<TString, std::vector<float>> array_floats;
    std::map<TString, std::vector<bool>>  array_labels;


};

#endif //BDTRAIN_BRANCH_INFO_H
