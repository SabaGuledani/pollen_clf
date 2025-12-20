/**
 *  @file lbp_hog_extractor.hpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#pragma once

#include "features.hpp"

class LbpHogExtractor : public FeaturesExtractor
{
public:
    /**
     * @brief Create and set the default parameters.
     */
    LbpHogExtractor();
    ~LbpHogExtractor();

    virtual const std::string &get_extractor_name() const override;
    virtual const std::string &get_extractor_help() const override;
    virtual cv::Mat extract_features(const cv::Mat &img) override;

    // This extractor does not need override these methods:
    // virtual void train(const cv::Mat& samples) override;
    // virtual bool save_model(std::string const& fname) const;
    // virtual bool load_model(std::string const& fname);
};

/**
 * @brief Extract combined LBP + HOG features from an image.
 * 
 * Function to extract combined Local Binary Pattern (LBP) and Histogram of 
 * Oriented Gradients (HOG) features from grayscale image.
 * 
 * @param img the input grayscale image.
 * @return the extracted features as a row vector (concatenated LBP + HOG features).
 * @pre !img.empty()
 * @pre img.channels() == 1
 * @post ret_v.type() == CV_32FC1
 * @post ret_v.rows == 1
 */
cv::Mat fsiv_extract_lbp_hog_features(const cv::Mat &img);

