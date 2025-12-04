/**
 *  @file gray_levels_features.hpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#pragma once

#include "features.hpp"

class GrayLevelsFeatures : public FeaturesExtractor
{
public:
    /**
     * @brief Create and set the default parameters.
     */
    GrayLevelsFeatures();
    ~GrayLevelsFeatures();

    virtual const std::string &get_extractor_name() const override;
    virtual const std::string &get_extractor_help() const override;
    virtual cv::Mat extract_features(const cv::Mat &img) override;

    // This extractor does not need override these methods:
    // virtual void train(const cv::Mat& samples) override;
    // virtual bool save_model(std::string const& fname) const;
    // virtual bool load_model(std::string const& fname);
};
