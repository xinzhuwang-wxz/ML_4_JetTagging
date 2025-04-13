//
// Created by Bamboo on 2025/4/11.
//

#ifndef BDTRAIN_BDTRAIN_H
#define BDTRAIN_BDTRAIN_H

#include "TChain.h"
#include "TString.h"
#include "TFile.h"
#include "TClonesArray.h"
#include <cstdlib>

#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "TLorentzVector.h"
#include "TSystem.h"

#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <cmath>

#include "branch_info.h"
#include "config_reader.h"




class Bdtrain {

public:

        Bdtrain(const TString &config_path);
        ~Bdtrain() {
            delete chain_train_;
            delete chain_val_;
            delete chain_test_;
            delete tree_train_;
            delete tree_val_;
            delete tree_test_;
        }

        void Init();
        void Branch(TTree* tree, BranchInfo& b_info);
        void SetBranchAddress(TChain* chain, BranchInfo& b_info);
//        void Fill_branch(TChain* chain, TTree* tree);
//        void Root_process(TChain* chain, TTree* tree, BranchInfo& b_info);
        void Chain2Csv();


        void Data_preprocess();
        void Save_root();
        void Training();
//        void Predict();
//        void Write_csv();
//        void Csv_root();
//        void Clear();


        TString config_path_;

        std::vector<TString> input_train_;
        std::vector<TString> input_val_;
        std::vector<TString> input_test_;

        TChain *chain_train_ = nullptr;
        TChain *chain_val_ = nullptr;
        TChain *chain_test_ = nullptr;

        TTree *tree_train_ = nullptr;
        TTree *tree_val_ = nullptr;
        TTree *tree_test_ = nullptr;

    };

#endif //BDTRAIN_BDTRAIN_H
