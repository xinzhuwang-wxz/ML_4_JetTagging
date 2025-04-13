//
// Created by Bamboo on 2025/4/11.
//

#include "Bdtrain.h"
#include "config_reader.h"
#include "branch_info.h"
#include "TSystem.h"

#include <cstdlib>  // for std::rand
#include <ctime>

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>


Bdtrain::Bdtrain(const TString &config_path) : config_path_(config_path) {}

void Bdtrain::Init() {

    std::srand(std::time(nullptr));

//    BranchInfo b_info;
//    b_info.Init();

    ConfigReader config_reader;
    config_reader.Load(config_path_);


    input_train_ = config_reader.InputsTrain();
    input_val_   = config_reader.InputsVal();
    input_test_  = config_reader.InputsTest();

    chain_train_ = new TChain("tree");
    chain_val_   = new TChain("tree");
    chain_test_  = new TChain("tree");

//    tree_train_ = new TTree("tree", "training tree");
//    Branch(tree_train_, b_info);
//    tree_val_   = new TTree("tree", "validation tree");
//    Branch(tree_val_, b_info);
//    tree_test_  = new TTree("tree", "test tree");
//    Branch(tree_test_, b_info);



//std::cout << "[INFO] Adding files to training chain..., the size is " << input_train_.size() << std::endl;
//std::cout << "[INFO] Adding files to validation chain..., the size is " << input_val_.size() << std::endl;

    for (const auto &input : input_train_) chain_train_->Add(input);
    for (const auto &input : input_val_)   chain_val_->Add(input);
    for (const auto &input : input_test_)  chain_test_->Add(input);


//    tree_train_ = chain_train_->CloneTree(-1);
//    tree_val_   = chain_val_->CloneTree(-1);
//    tree_test_  = chain_test_->CloneTree(-1);

    Long64_t nEntries_train = chain_train_->GetEntries();
    Long64_t nEntries_val   = chain_val_->GetEntries();
    Long64_t nEntries_test  = chain_test_->GetEntries();
//
//    std::cout << "[INFO] Number of entries in training chain: " << nEntries_train << std::endl;
//    std::cout << "[INFO] Number of entries in validation chain: " << nEntries_val << std::endl;
//    std::cout << "[INFO] Number of entries in testing chain: " << nEntries_test << std::endl;

        // Clone the structure of the first entry
    tree_train_ = chain_train_->CloneTree(0);
    tree_val_   = chain_val_->CloneTree(0);
    tree_test_  = chain_test_->CloneTree(0);

    // Fill the cloned trees
    for (Long64_t i = 0; i < nEntries_train; ++i) {
        chain_train_->GetEntry(i);
        if (std::rand() < 0.05 * RAND_MAX) {
            tree_train_->Fill();  // Only fill with 10% chance
        }
    }

    for (Long64_t i = 0; i < nEntries_val; ++i) {
        chain_val_->GetEntry(i);
        if (std::rand() < 0.05 * RAND_MAX) {
            tree_val_->Fill();
        }
    }

    for (Long64_t i = 0; i < nEntries_test; ++i) {
        chain_test_->GetEntry(i);
        if (std::rand() < 0.05 * RAND_MAX) {
            tree_test_->Fill();
        }
      }

    tree_train_->SetName("tree");
    tree_val_->SetName("tree");
    tree_test_->SetName("tree");


}




void Bdtrain::Chain2Csv() {
    if (gSystem->AccessPathName("data")) {
        gSystem->MakeDirectory("data");
    }

    TString script_path = "../core/chain2csv.py";
    TString config_path = "../config.yaml";

    gSystem->ExpandPathName(script_path);
    gSystem->ExpandPathName(config_path);

    TString cmd = TString::Format("python3 %s --config %s", script_path.Data(), config_path.Data());
    std::cout << "[INFO] Running command: " << cmd << std::endl;

    int ret = gSystem->Exec(cmd);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to run tchain2csv.py, exit code = " << ret << std::endl;
    }
}



void Bdtrain::Data_preprocess() {

}

void Bdtrain::Save_root() {
    if (gSystem->AccessPathName("data")) { // Check if the directory exists
        gSystem->MakeDirectory("data");
    }

    if (tree_train_ && tree_train_->GetEntries() > 0) {
        std::cout << "Saving tree_train_ with " << tree_train_->GetEntries() << " entries." << std::endl;
        tree_train_->SaveAs("data/train.root");
    } else {
        std::cerr << "[ERROR] tree_train_ is empty or invalid." << std::endl;
    }

    if (tree_val_ && tree_val_->GetEntries() > 0) {
        std::cout << "Saving tree_val_ with " << tree_val_->GetEntries() << " entries." << std::endl;
        tree_val_->SaveAs("data/val.root");
    } else {
        std::cerr << "[ERROR] tree_val_ is empty or invalid." << std::endl;
    }

    if (tree_test_ && tree_test_->GetEntries() > 0) {
        std::cout << "Saving tree_test_ with " << tree_test_->GetEntries() << " entries." << std::endl;
        tree_test_->SaveAs("data/test.root");
    } else {
        std::cerr << "[ERROR] tree_test_ is empty or invalid." << std::endl;
    }
}

void Bdtrain::Training(){


    if (gSystem->AccessPathName("data")) {
        std::cerr << "[ERROR] data directory does not exist. Please run Chain2Csv first." << std::endl;
        return;
    }

    TString script_path = "../core/training.py";
    TString config_path = "../config.yaml";

    gSystem->ExpandPathName(script_path);
    gSystem->ExpandPathName(config_path);

    TString cmd = TString::Format("python3 %s --config %s", script_path.Data(), config_path.Data());
    std::cout << "[INFO] Running command: " << cmd << std::endl;
//    std::cout << "current dir is " << gSystem->pwd() << std::endl;

    int ret = gSystem->Exec(cmd);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to run training.py, exit code = " << ret << std::endl;
    }
}





