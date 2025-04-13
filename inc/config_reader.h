#ifndef BDTRAIN_CONFIG_READER_H
#define BDTRAIN_CONFIG_READER_H

#include "TString.h"
#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <vector>

class ConfigReader
{
public:
    void Load(const TString &config_name);
    void Print();

    std::vector<TString> InputsTrain() const { return inputs_train_; }
    std::vector<TString> InputsVal() const { return inputs_val_; }
    std::vector<TString> InputsTest() const { return inputs_test_; }

private:
    YAML::Node config_;

    std::vector<TString> inputs_train_;
    std::vector<TString> inputs_val_;
    std::vector<TString> inputs_test_;
};

#endif // BDTRAIN_CONFIG_READER_H

