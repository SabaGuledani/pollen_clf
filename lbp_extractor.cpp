/**
 *  @file lbp_extractor.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "lbp_extractor.hpp"

static std::string name_{"LBP Histogram Feature Extractor"};
static std::string help_{
    "  This extractor computes Local Binary Pattern (LBP) features from the input image\n"
    "  and returns a histogram of LBP values.\n"
    "  Parameters: [radius] [neighbors] [grid_rows] [grid_cols]\n"
    "    - radius: LBP radius (default: 1.0)\n"
    "    - neighbors: Number of neighbors (default: 8)\n"
    "    - grid_rows: Number of grid rows for spatial histogram (default: 1, no grid)\n"
    "    - grid_cols: Number of grid cols for spatial histogram (default: 1, no grid)\n"};

const std::string &
LbpExtractor::get_extractor_name() const
{
    return name_;
}

const std::string &
LbpExtractor::get_extractor_help() const
{
    return help_;
}

LbpExtractor::LbpExtractor()
{
    type_ = FSIV_LBP_HISTOGRAM;
}

LbpExtractor::~LbpExtractor() {}

/**
 * @brief Compute LBP value for a pixel
 */
static int compute_lbp_value(const cv::Mat &img, int x, int y, float radius, int neighbors)
{
    int lbp_value = 0;
    float center_value = img.at<uchar>(y, x);
    
    for (int i = 0; i < neighbors; ++i)
    {
        float angle = 2.0f * M_PI * i / neighbors;
        int neighbor_x = cvRound(x + radius * cos(angle));
        int neighbor_y = cvRound(y + radius * sin(angle));
        
        // Handle boundary conditions
        if (neighbor_x >= 0 && neighbor_x < img.cols && 
            neighbor_y >= 0 && neighbor_y < img.rows)
        {
            float neighbor_value = img.at<uchar>(neighbor_y, neighbor_x);
            if (neighbor_value >= center_value)
            {
                lbp_value |= (1 << i);
            }
        }
    }
    
    return lbp_value;
}

cv::Mat
LbpExtractor::extract_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    // get params
    float radius = (params_.size() > 0) ? params_[0] : 1.0f;
    int neighbors = (params_.size() > 1) ? static_cast<int>(params_[1]) : 8;
    int grid_rows = (params_.size() > 2) ? static_cast<int>(params_[2]) : 1;
    int grid_cols = (params_.size() > 3) ? static_cast<int>(params_[3]) : 1;
    
    // ensure valid params
    if (radius <= 0) radius = 1.0f;
    if (neighbors <= 0 || neighbors > 31) neighbors = 8; // Max 31 bits for int
    if (grid_rows <= 0) grid_rows = 1;
    if (grid_cols <= 0) grid_cols = 1;
    
    // convert to uchar if needed
    cv::Mat img_uchar;
    if (img.type() != CV_8UC1)
    {
        img.convertTo(img_uchar, CV_8UC1);
    }
    else
    {
        img_uchar = img;
    }
    
    // compute LBP image
    cv::Mat lbp_img = cv::Mat::zeros(img_uchar.size(), CV_8UC1);
    int border = static_cast<int>(ceil(radius));
    
    for (int y = border; y < img_uchar.rows - border; ++y)
    {
        for (int x = border; x < img_uchar.cols - border; ++x)
        {
            int lbp_value = compute_lbp_value(img_uchar, x, y, radius, neighbors);
            lbp_img.at<uchar>(y, x) = static_cast<uchar>(lbp_value);
        }
    }
    
    // compute histograms
    int hist_size = 1 << neighbors; // 2 in power of neighbors bins
    float range[] = {0, static_cast<float>(hist_size)};
    const float* hist_range = {range};
    
    cv::Mat features;
    
    if (grid_rows == 1 && grid_cols == 1)
    {
        // if rows and cols are 1, compute single histogram for entire image
        
        cv::Mat hist;
        cv::calcHist(&lbp_img, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);
        
        // normalize hist
        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
        
        // convert to row vector
        hist.convertTo(features, CV_32F);
        features = features.reshape(1, 1);
    }
    else
    {
        // if rows and cols are not 1 divide and compute 
        // divide image into grid and compute histogram per cell
        int cell_height = img_uchar.rows / grid_rows;
        int cell_width = img_uchar.cols / grid_cols;
        
        std::vector<cv::Mat> histograms;
        
        for (int gr = 0; gr < grid_rows; ++gr)
        {
            for (int gc = 0; gc < grid_cols; ++gc)
            {
                int y_start = gr * cell_height;
                int y_end = (gr == grid_rows - 1) ? img_uchar.rows : (gr + 1) * cell_height;
                int x_start = gc * cell_width;
                int x_end = (gc == grid_cols - 1) ? img_uchar.cols : (gc + 1) * cell_width;
                
                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);
                cv::Mat cell = lbp_img(roi);
                
                cv::Mat hist;
                cv::calcHist(&cell, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);
                
                // normalize hist
                cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
                
                histograms.push_back(hist);
            }
        }
        
        // concatenate all histograms into a single row vector
        features = cv::Mat(1, hist_size * grid_rows * grid_cols, CV_32F);
        int offset = 0;
        for (const auto &hist : histograms)
        {
            cv::Mat hist_float;
            hist.convertTo(hist_float, CV_32F);
            hist_float.reshape(1, 1).copyTo(features.colRange(offset, offset + hist_size));
            offset += hist_size;
        }
    }
    
    CV_Assert(features.rows == 1);
    CV_Assert(features.type() == CV_32FC1);
    CV_Assert(features.cols > 0);
    return features;
}

cv::Mat fsiv_extract_lbp_histogram(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    LbpExtractor extractor;
    return extractor.extract_features(img);
}

