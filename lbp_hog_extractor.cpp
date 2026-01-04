/**
 *  @file lbp_hog_extractor.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include "lbp_hog_extractor.hpp"
#include "lbp_extractor.hpp"
#include "hog_extractor.hpp"

static std::string name_{"LBP+HOG Combined Feature Extractor"};
static std::string help_{
    "  This extractor computes combined LBP and HOG \n"
    "  Parameters: [lbp_radius] [lbp_neighbors] [lbp_grid_rows] [lbp_grid_cols] "
    "[hog_win_width] [hog_win_height] [hog_block_size] [hog_block_stride] [hog_cell_size] [hog_nbins]\n"
    "    LBP parameters:\n"
    "    - lbp_radius: LBP radius (default: 1.0)\n"
    "    - lbp_neighbors: Number of neighbors (default: 8)\n"
    "    - lbp_grid_rows: Number of grid rows for spatial histogram (default: 1)\n"
    "    - lbp_grid_cols: Number of grid cols for spatial histogram (default: 1)\n"
    "    HOG parameters:\n"
    "    - hog_win_width: Window width (default: image width)\n"
    "    - hog_win_height: Window height (default: image height)\n"
    "    - hog_block_size: Block size in pixels (default: 16)\n"
    "    - hog_block_stride: Block stride in pixels (default: 8)\n"
    "    - hog_cell_size: Cell size in pixels (default: 8)\n"
    "    - hog_nbins: Number of orientation bins (default: 9)\n"};

const std::string &
LbpHogExtractor::get_extractor_name() const
{
    return name_;
}

const std::string &
LbpHogExtractor::get_extractor_help() const
{
    return help_;
}

LbpHogExtractor::LbpHogExtractor()
{
    type_ = FSIV_LBP_HOG;
}

LbpHogExtractor::~LbpHogExtractor() {}

cv::Mat
LbpHogExtractor::extract_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    // Apply histogram equalization to enhance contrast
    cv::Mat img_preprocessed;
    if (img.type() == CV_8UC1)
    {
        // histogram equalization for 8-bit images
        cv::equalizeHist(img, img_preprocessed);
    }
    else
    {
        // if not 8bit
        // convert to 8-bit first then equalize
        cv::Mat img_8bit;
        img.convertTo(img_8bit, CV_8UC1);
        cv::equalizeHist(img_8bit, img_preprocessed);
    }
    
    // parse params
    // LBP params
    float lbp_radius = (params_.size() > 0) ? params_[0] : 1.0f;
    int lbp_neighbors = (params_.size() > 1) ? static_cast<int>(params_[1]) : 8;
    int lbp_grid_rows = (params_.size() > 2) ? static_cast<int>(params_[2]) : 1;
    int lbp_grid_cols = (params_.size() > 3) ? static_cast<int>(params_[3]) : 1;
    
    // HOG params
    int hog_win_width = (params_.size() > 4) ? static_cast<int>(params_[4]) : img_preprocessed.cols;
    int hog_win_height = (params_.size() > 5) ? static_cast<int>(params_[5]) : img_preprocessed.rows;
    int hog_block_size = (params_.size() > 6) ? static_cast<int>(params_[6]) : 16;
    int hog_block_stride = (params_.size() > 7) ? static_cast<int>(params_[7]) : 8;
    int hog_cell_size = (params_.size() > 8) ? static_cast<int>(params_[8]) : 8;
    int hog_nbins = (params_.size() > 9) ? static_cast<int>(params_[9]) : 9;
    
    // extract lpb features
    LbpExtractor lbp_extractor;
    std::vector<float> lbp_params = {lbp_radius, static_cast<float>(lbp_neighbors), 
                                     static_cast<float>(lbp_grid_rows), static_cast<float>(lbp_grid_cols)};
    lbp_extractor.set_params(lbp_params);
    cv::Mat lbp_features = lbp_extractor.extract_features(img_preprocessed);
    
    // extract hog features
    HogExtractor hog_extractor;
    std::vector<float> hog_params = {static_cast<float>(hog_win_width), static_cast<float>(hog_win_height),
                                      static_cast<float>(hog_block_size), static_cast<float>(hog_block_stride),
                                      static_cast<float>(hog_cell_size), static_cast<float>(hog_nbins)};
    hog_extractor.set_params(hog_params);
    cv::Mat hog_features = hog_extractor.extract_features(img_preprocessed);
    
    // l2 normalization
    cv::Mat lbp_normalized, hog_normalized;
    cv::normalize(lbp_features, lbp_normalized, 1.0, 0.0, cv::NORM_L2);
    cv::normalize(hog_features, hog_normalized, 1.0, 0.0, cv::NORM_L2);
    
    // concat lbp and hog 
    cv::Mat combined_features;
    cv::hconcat(lbp_normalized, hog_normalized, combined_features);
    
    CV_Assert(combined_features.rows == 1);
    CV_Assert(combined_features.type() == CV_32FC1);
    CV_Assert(combined_features.cols > 0);
    return combined_features;
}

cv::Mat fsiv_extract_lbp_hog_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    LbpHogExtractor extractor;
    return extractor.extract_features(img);
}

