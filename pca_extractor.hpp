/**
 *  @file pca_extractor.hpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#pragma once

#include "features.hpp"
#include <opencv2/core.hpp>

class PcaExtractor : public FeaturesExtractor
{
public:
    /**
     * @brief Create and set the default parameters.
     */
    PcaExtractor();
    ~PcaExtractor();

    virtual const std::string &get_extractor_name() const override;
    virtual const std::string &get_extractor_help() const override;
    virtual cv::Mat extract_features(const cv::Mat &img) override;
    virtual void train(const Dataset &dt) override;
    virtual bool save_model(std::string const &fname) const override;
    virtual bool load_model(std::string const &fname) override;

private:
    cv::PCA pca_;
    bool trained_;
    cv::Ptr<FeaturesExtractor> base_extractor_;
    FeaturesExtractor::FEATURE_IDS base_extractor_id_;
};

/**
 * @brief Extract PCA-reduced features from an image.
 * 
 * Function to extract features using PCA dimensionality reduction.
 * Requires training before use.
 * 
 * @param img the input grayscale image.
 * @return the extracted features as a row vector (PCA-reduced features).
 * @pre !img.empty()
 * @pre img.channels() == 1
 * @pre PCA must be trained before use
 * @post ret_v.type() == CV_32FC1
 * @post ret_v.rows == 1
 */
cv::Mat fsiv_extract_pca_features(const cv::Mat &img);

