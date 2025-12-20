/**
 *  @file hog_extractor.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include "hog_extractor.hpp"

static std::string name_{"HOG Feature Extractor"};
static std::string help_{
    "  This extractor computes Histogram of Oriented Gradients (HOG) features from the input image.\n"
    "  Parameters: [win_width] [win_height] [block_size] [block_stride] [cell_size] [nbins]\n"
    "    - win_width: Window width (default: image width)\n"
    "    - win_height: Window height (default: image height)\n"
    "    - block_size: Block size in pixels (default: 16)\n"
    "    - block_stride: Block stride in pixels (default: 8)\n"
    "    - cell_size: Cell size in pixels (default: 8)\n"
    "    - nbins: Number of orientation bins (default: 9)\n"};

const std::string &
HogExtractor::get_extractor_name() const
{
    return name_;
}

const std::string &
HogExtractor::get_extractor_help() const
{
    return help_;
}

HogExtractor::HogExtractor()
{
    type_ = FSIV_HOG;
}

HogExtractor::~HogExtractor() {}

cv::Mat
HogExtractor::extract_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    // Get parameters with defaults
    int win_width = (params_.size() > 0) ? static_cast<int>(params_[0]) : img.cols;
    int win_height = (params_.size() > 1) ? static_cast<int>(params_[1]) : img.rows;
    int block_size = (params_.size() > 2) ? static_cast<int>(params_[2]) : 16;
    int block_stride = (params_.size() > 3) ? static_cast<int>(params_[3]) : 8;
    int cell_size = (params_.size() > 4) ? static_cast<int>(params_[4]) : 8;
    int nbins = (params_.size() > 5) ? static_cast<int>(params_[5]) : 9;
    
    // Ensure valid parameters
    if (win_width <= 0) win_width = img.cols;
    if (win_height <= 0) win_height = img.rows;
    if (block_size <= 0) block_size = 16;
    if (block_stride <= 0) block_stride = 8;
    if (cell_size <= 0) cell_size = 8;
    if (nbins <= 0) nbins = 9;
    
    // Ensure window size doesn't exceed image size
    win_width = std::min(win_width, img.cols);
    win_height = std::min(win_height, img.rows);
    
    // Convert to uchar if needed
    cv::Mat img_uchar;
    if (img.type() != CV_8UC1)
    {
        img.convertTo(img_uchar, CV_8UC1);
    }
    else
    {
        img_uchar = img;
    }
    
    // Create HOG descriptor
    cv::HOGDescriptor hog(
        cv::Size(win_width, win_height),  // winSize
        cv::Size(block_size, block_size),  // blockSize
        cv::Size(block_stride, block_stride), // blockStride
        cv::Size(cell_size, cell_size),    // cellSize
        nbins                              // nbins
    );
    
    // Compute HOG features
    std::vector<float> descriptors;
    std::vector<cv::Point> locations;
    
    // Resize image to window size if needed
    cv::Mat img_resized;
    if (img_uchar.cols != win_width || img_uchar.rows != win_height)
    {
        cv::resize(img_uchar, img_resized, cv::Size(win_width, win_height));
    }
    else
    {
        img_resized = img_uchar;
    }
    
    hog.compute(img_resized, descriptors, cv::Size(0, 0), cv::Size(0, 0), locations);
    
    // Convert to Mat (row vector)
    cv::Mat features(1, static_cast<int>(descriptors.size()), CV_32FC1);
    for (size_t i = 0; i < descriptors.size(); ++i)
    {
        features.at<float>(0, static_cast<int>(i)) = descriptors[i];
    }
    
    CV_Assert(features.rows == 1);
    CV_Assert(features.type() == CV_32FC1);
    CV_Assert(features.cols > 0);
    return features;
}

cv::Mat fsiv_extract_hog_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    // just use the HogExtractor class to extract features
    HogExtractor extractor;
    return extractor.extract_features(img);
}

