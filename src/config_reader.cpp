//
// Created by Bamboo on 2025/4/11.
//

#include "config_reader.h"

#include <string>
#include <regex>
#include <iostream>

void ConfigReader::Load(const TString &config_name)
{
    config_ = YAML::LoadFile(config_name.Data());

    try
    {
        auto inputs_train = config_["input"]["train"].as<std::vector<std::vector<std::string>>>();
        for (const auto &input : inputs_train)
            for (const auto &i : input)
                inputs_train_.push_back(i);


        auto inputs_val = config_["input"]["val"].as<std::vector<std::string>>();
        for (const auto &input : inputs_val)
            inputs_val_.push_back(input);

        auto inputs_test = config_["input"]["test"].as<std::vector<std::string>>();
        for (const auto &input : inputs_test)
            inputs_test_.push_back(input);
    }
    catch(YAML::BadConversion &e)
    {
        std::cerr << "[WARNING] ==> In " << config_name << ", " << e.msg << std::endl << std::endl;
        return;
    }
    catch(YAML::InvalidNode &e)
    {
        std::cerr << "[WARNING] ==> In " << config_name << ", " << e.msg << std::endl << std::endl;
        return;
    }
}

void ConfigReader::Print()
{
    std::cout << "--- inputs:" << std::endl;
    std::cout << "-- train:" << std::endl;
    for (const auto &input : inputs_train_)
        std::cout << "   " << input << std::endl;

    std::cout << "-- val:" << std::endl;
    for (const auto &input : inputs_val_)
        std::cout << "   " << input << std::endl;

    std::cout << "-- test:" << std::endl;
    for (const auto &input : inputs_test_)
        std::cout << "   " << input << std::endl;
}
