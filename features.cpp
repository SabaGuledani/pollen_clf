/**
 *  @file features.cpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#include <iostream>
#include <exception>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "features.hpp"

// TODO: add the include for each extractor to use.
// Hint: use gray_levels_features.hpp and gray_levels_features.cpp as model to
//   make yours.
#include "gray_levels_features.hpp"
// #include "xxxxxx.hpp"

// Remember: update CMakeLists.txt with the new files.
//

FeaturesExtractor::FEATURE_IDS
FeaturesExtractor::get_extractor_type() const
{
    return type_;
}

cv::Ptr<FeaturesExtractor> FeaturesExtractor::create(FEATURE_IDS id)
{
    cv::Ptr<FeaturesExtractor> extractor;
    switch (id)
    {
    case FSIV_01_GREY_LEVELS:
    {
        extractor = cv::makePtr<GrayLevelsFeatures>();
        break;
    }

        // TODO: add here 'cases' for your features.
        // case FSIV_XXXXX: {
        //    extractor = cv::makePtr<FeatureExtractor>(new XXXXX());
        //    break;
        // }
        //

    default:
    {
        throw std::runtime_error("Error: unknown feature id: " + std::to_string(int(id)));
        break;
    }
    }
    CV_Assert(extractor != nullptr);
    extractor->type_ = id;
    return extractor;
}

std::tuple<cv::Mat, cv::Mat>
fsiv_extract_features(const Dataset &dt,
                      cv::Ptr<FeaturesExtractor> &extractor)
{
    CV_Assert(dt.size() > 0);

    // Process the first image to get the features dimension.
    cv::Mat first_sample = dt.get_sample(0);
    if (first_sample.empty())
    {
        throw std::runtime_error("Error: first sample image is empty. Check image path: " + dt.get_sample_filename(0));
    }
    cv::Mat feature = extractor->extract_features(first_sample);

    // Allocate memory.
    cv::Mat X(dt.size(), feature.cols, CV_32F);
    cv::Mat y(dt.size(), 1, CV_32S);
    feature.copyTo(X.row(0));
    y.at<int>(0, 0) = dt.get_label(0);

    // Process the rest of dt.
    // Note: OpenMP disabled to avoid recursive termination on errors
    // #ifdef USE_OPENMP
    // #pragma omp parallel for
    // #endif
    for (size_t i = 1; i < dt.size(); ++i)
    {
        cv::Mat sample;
        int label;
        std::tie(sample, label) = dt[i];
        if (sample.empty())
        {
            std::cerr << "Warning: sample " << i << " is empty (file not found or corrupted). File: " << dt.get_sample_filename(i) << std::endl;
            std::cerr << "Skipping this sample and using zeros for features." << std::endl;
            // use zeros for missing images (same size as first feature)
            cv::Mat zero_features = cv::Mat::zeros(1, feature.cols, CV_32FC1);
            zero_features.copyTo(X.row(i));
            y.at<int>(int(i), 0) = label;
            continue;
        }
        try
        {
            extractor->extract_features(sample).copyTo(X.row(i));
            y.at<int>(int(i), 0) = label;
        }
        catch (cv::Exception &e)
        {
            std::cerr << "OpenCV error processing sample " << i << ": " << dt.get_sample_filename(i) << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
        }
        if (i % 1000 == 0)
        {
            std::cout << "Processed " << i << " / " << dt.size() << " samples..." << std::endl;
        }
    }
    return std::make_tuple(X, y);
}

void FeaturesExtractor::set_params(const std::vector<float> &new_p)
{
    if (new_p.size() != 0)
        params_ = new_p;
}

const std::vector<float> &
FeaturesExtractor::get_params() const
{
    return params_;
}

void FeaturesExtractor::train(const Dataset &dt)
{
    // do nothing.
    // Override this method in your class if it is needed.
    return;
}

bool FeaturesExtractor::save_model(std::string const &model_fname) const
{
    bool ret_v = false;
    cv::FileStorage f(model_fname, cv::FileStorage::APPEND);
    if (f.isOpened())
    {
        ret_v = true;
        f << "fsiv_feature_id" << int(type_);
        f << "fsiv_feature_params" << params_;
    }
    return ret_v;
}

bool FeaturesExtractor::load_model(std::string const &model_fname)
{
    cv::FileStorage f(model_fname, cv::FileStorage::READ);

    auto node = f["fsiv_feature_id"];
    if (node.empty() || !node.isInt())
        throw std::runtime_error("Could not load the 'fsiv_feature_id' "
                                 "label from file.");
    int loaded_type;
    node >> loaded_type;
    if (loaded_type != int(type_))
        throw std::runtime_error("Trainned model for a different "
                                 "feature extractor.");
    node = f["fsiv_feature_params"];
    if (node.empty())
        throw std::runtime_error("Could not load the 'fsiv_feature_params' "
                                 "label from file.");
    node >> params_;
    return true;
}

cv::Ptr<FeaturesExtractor>
FeaturesExtractor::create(const std::string &fname)
{
    cv::Ptr<FeaturesExtractor> extr;
    cv::FileStorage f;
    f.open(fname, cv::FileStorage::READ);
    if (f.isOpened())
    {
        auto node = f["fsiv_feature_id"];
        if (node.empty() || !node.isInt())
            throw std::runtime_error("Could not load the 'fsiv_feature_id' "
                                     "label from file.");
        int loaded_type;
        node >> loaded_type;
        extr = create(FEATURE_IDS(loaded_type));
        extr->load_model(fname);
    }
    return extr;
}