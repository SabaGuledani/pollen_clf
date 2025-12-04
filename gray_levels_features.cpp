/**
 *  @file gray_levels_features.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <opencv2/imgproc.hpp>
#include "gray_levels_features.hpp"

static std::string name_{"Gray Levels Feature Extractor"};
static std::string help_{
    "  This extractor normalizes the gray levels of the input image to the "
    "range [0, 1] and\n"
    "  returns the normalized pixel values as a row vector.\n"
    "  Parameters: None.\n"};

const std::string &
GrayLevelsFeatures::get_extractor_name() const
{
    return name_;
}

const std::string &
GrayLevelsFeatures::get_extractor_help() const
{
    return help_;
}

GrayLevelsFeatures::GrayLevelsFeatures()
{
    type_ = FSIV_01_GREY_LEVELS;
}

GrayLevelsFeatures::~GrayLevelsFeatures() {}

cv::Mat
GrayLevelsFeatures::extract_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    cv::Mat features;
    // TODO
    // Hint: use cv:normalize with norm MINMAX to normalize the input to full range [0, 1].
    // Remember: the output type must be CV_32F.
    // Hint: use cv::Mat::reshape() method to pass from WxH to 1xW*H row vector.

    // normalize image to range 0-1, convert to float
    cv::normalize(img, features, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    
    // reshape to row vector (1 row, all pixels in one row)
    features = features.reshape(1, 1);

    //
    CV_Assert(features.rows == 1);
    CV_Assert(features.type() == CV_32FC1);
    CV_Assert(features.cols > 0);
    return features;
}

cv::Mat fsiv_extract_01_normalized_graylevels(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    // just use the GrayLevelsFeatures class to extract features
    GrayLevelsFeatures extractor;
    return extractor.extract_features(img);
}
