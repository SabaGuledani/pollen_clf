/**
 *  @file hog_extractor.hpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#pragma once

#include "features.hpp"

class HogExtractor : public FeaturesExtractor
{
public:
    /**
     * @brief Create and set the default parameters.
     */
    HogExtractor();
    ~HogExtractor();

    virtual const std::string &get_extractor_name() const override;
    virtual const std::string &get_extractor_help() const override;
    virtual cv::Mat extract_features(const cv::Mat &img) override;

};

/**
 * @brief Extract HOG features from an image.
 * 
 * Function to extract Histogram of Oriented Gradients (HOG) features from grayscale image.
 * 
 * @param img the input grayscale image.
 * @return the extracted features as a row vector (HOG descriptor).
 * @pre !img.empty()
 * @pre img.channels() == 1
 * @post ret_v.type() == CV_32FC1
 * @post ret_v.rows == 1
 */
cv::Mat fsiv_extract_hog_features(const cv::Mat &img);

