/**
 *  @file pca_extractor.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/pca.hpp>
#include "pca_extractor.hpp"
#include "lbp_hog_extractor.hpp"

static std::string name_{"PCA Feature Extractor"};
static std::string help_{
    "  This extractor applies Principal Component Analysis (PCA) to reduce\n"
    "  dimensionality of features from a base extractor (e.g., LBP+HOG).\n"
    "  Parameters: [base_extractor_id] [variance_retained]\n"
    "    - base_extractor_id: ID of base extractor to use (0=GrayLevels, 1=LBP, 2=HOG, 3=LBP+HOG, default: 3)\n"
    "    - variance_retained: Fraction of variance to retain (0.0-1.0, default: 0.95)\n"
    "  Note: This extractor requires training before use.\n"};

const std::string &
PcaExtractor::get_extractor_name() const
{
    return name_;
}

const std::string &
PcaExtractor::get_extractor_help() const
{
    return help_;
}

PcaExtractor::PcaExtractor()
{
    type_ = FSIV_PCA;
    trained_ = false;
    base_extractor_id_ = FSIV_LBP_HOG; // Default to LBP+HOG
    base_extractor_ = FeaturesExtractor::create(base_extractor_id_);
}

PcaExtractor::~PcaExtractor() {}

void
PcaExtractor::train(const Dataset &dt)
{
    CV_Assert(dt.size() > 0);
    
    // Parse parameters
    int base_id = (params_.size() > 0) ? static_cast<int>(params_[0]) : static_cast<int>(FSIV_LBP_HOG);
    float variance_retained = (params_.size() > 1) ? params_[1] : 0.95f;
    
    // Ensure valid parameters
    if (base_id < 0 || base_id >= FSIV_NEXT_FEATURE_ID)
    {
        base_id = static_cast<int>(FSIV_LBP_HOG);
    }
    if (variance_retained <= 0.0f || variance_retained > 1.0f)
    {
        variance_retained = 0.95f;
    }
    
    base_extractor_id_ = static_cast<FeaturesExtractor::FEATURE_IDS>(base_id);
    base_extractor_ = FeaturesExtractor::create(base_extractor_id_);
    
    // Extract features from all training samples
    std::cout << "Extracting base features for PCA training..." << std::endl;
    
    // First, get feature dimension from first sample
    cv::Mat first_sample = dt.get_sample(0);
    if (first_sample.empty())
    {
        throw std::runtime_error("Error: First training sample is empty.");
    }
    cv::Mat first_feature = base_extractor_->extract_features(first_sample);
    int feature_dim = first_feature.cols;
    
    // Allocate matrix for all features
    cv::Mat all_features(dt.size(), feature_dim, CV_32F);
    first_feature.copyTo(all_features.row(0));
    
    // Extract features from remaining samples
    for (size_t i = 1; i < dt.size(); ++i)
    {
        cv::Mat sample = dt.get_sample(i);
        if (sample.empty())
        {
            std::cerr << "Warning: sample " << i << " is empty. Using zeros." << std::endl;
            all_features.row(static_cast<int>(i)).setTo(0);
            continue;
        }
        
        cv::Mat feature = base_extractor_->extract_features(sample);
        if (feature.cols != feature_dim)
        {
            std::cerr << "Warning: sample " << i << " has different feature dimension. Expected " 
                      << feature_dim << ", got " << feature.cols << ". Skipping." << std::endl;
            all_features.row(static_cast<int>(i)).setTo(0);
            continue;
        }
        
        feature.copyTo(all_features.row(static_cast<int>(i)));
        
        if ((i + 1) % 1000 == 0)
        {
            std::cout << "Extracted features from " << (i + 1) << " / " << dt.size() << " samples..." << std::endl;
        }
    }
    
    // Train PCA
    std::cout << "Training PCA to retain " << (variance_retained * 100.0f) << "% variance..." << std::endl;
    std::cout << "Original feature dimension: " << all_features.cols << std::endl;
    
    // Compute PCA
    // cv::PCA constructor: PCA(data, mean, flags, retainedVariance)
    // When retainedVariance is between 0 and 1, PCA automatically selects components
    pca_ = cv::PCA(all_features, cv::Mat(), cv::PCA::DATA_AS_ROW, static_cast<double>(variance_retained));
    
    std::cout << "PCA trained. Reduced feature dimension: " << pca_.eigenvectors.rows << std::endl;
    
    // Calculate actual variance retained
    if (pca_.eigenvalues.rows > 0 && pca_.eigenvalues.rows == pca_.eigenvectors.rows)
    {
        double total_variance = cv::sum(pca_.eigenvalues)[0];
        double retained_variance = cv::sum(pca_.eigenvalues.rowRange(0, pca_.eigenvectors.rows))[0];
        double variance_ratio = total_variance > 0 ? (retained_variance / total_variance) : 0.0;
        std::cout << "Actual variance retained: " << (variance_ratio * 100.0) << "%" << std::endl;
    }
    
    trained_ = true;
}

cv::Mat
PcaExtractor::extract_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    if (!trained_)
    {
        throw std::runtime_error("Error: PCA extractor must be trained before use. Call train() first.");
    }
    
    CV_Assert(base_extractor_ != nullptr);
    
    // Extract features using base extractor
    cv::Mat base_features = base_extractor_->extract_features(img);
    
    // Apply PCA transformation
    cv::Mat pca_features;
    pca_.project(base_features, pca_features);
    
    CV_Assert(pca_features.rows == 1);
    CV_Assert(pca_features.type() == CV_32FC1);
    CV_Assert(pca_features.cols > 0);
    return pca_features;
}

bool
PcaExtractor::save_model(std::string const &fname) const
{
    // First call parent to save feature ID and params
    if (!FeaturesExtractor::save_model(fname))
    {
        return false;
    }
    
    CV_Assert(trained_);
    
    cv::FileStorage f(fname, cv::FileStorage::APPEND);
    if (!f.isOpened())
    {
        return false;
    }
    
    // Save PCA-specific data
    f << "fsiv_pca_base_extractor_id" << static_cast<int>(base_extractor_id_);
    f << "fsiv_pca_trained" << trained_;
    
    if (trained_)
    {
        f << "fsiv_pca_mean" << pca_.mean;
        f << "fsiv_pca_eigenvectors" << pca_.eigenvectors;
        f << "fsiv_pca_eigenvalues" << pca_.eigenvalues;
    }
    
    return true;
}

bool
PcaExtractor::load_model(std::string const &fname)
{
    // First call parent to load feature ID and params
    if (!FeaturesExtractor::load_model(fname))
    {
        return false;
    }
    
    cv::FileStorage f(fname, cv::FileStorage::READ);
    if (!f.isOpened())
    {
        return false;
    }
    
    // Load PCA-specific data
    int base_id;
    f["fsiv_pca_base_extractor_id"] >> base_id;
    base_extractor_id_ = static_cast<FeaturesExtractor::FEATURE_IDS>(base_id);
    base_extractor_ = FeaturesExtractor::create(base_extractor_id_);
    
    f["fsiv_pca_trained"] >> trained_;
    
    if (trained_)
    {
        f["fsiv_pca_mean"] >> pca_.mean;
        f["fsiv_pca_eigenvectors"] >> pca_.eigenvectors;
        f["fsiv_pca_eigenvalues"] >> pca_.eigenvalues;
    }
    
    return true;
}

cv::Mat fsiv_extract_pca_features(const cv::Mat &img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels() == 1);
    
    // just use the PcaExtractor class to extract features
    PcaExtractor extractor;
    return extractor.extract_features(img);
}

