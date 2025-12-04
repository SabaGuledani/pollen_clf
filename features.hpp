/**
 *  @file features.hpp
 *  (C) 2022- FJMC fjmadrid@uco.es
 */
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "dataset.hpp"

/**
 * @brief Base class to define feature extractors.
 */
class FeaturesExtractor
{
public:
    /**
     * @brief Define feature extractors.
     */
    typedef enum
    {
        FSIV_01_GREY_LEVELS = 0,
        // TODO: Add new features to extract.
        // FSIV_MEAN_STDDEV_GREY_LEVELS = 1,
        // FSIV_LBP_HISTOGRAM = 2,
        // FSIV_HOG = 3,
        // FSIV_BOVW = 4,
        //....
        FSIV_NEXT_FEATURE_ID = 1 // Update this value when a new feature is added.
    } FEATURE_IDS;

    /**
     * @brief Get the extractor enum type.
     *
     * @return FEATURE_IDS type.
     */
    FEATURE_IDS get_extractor_type() const;

    /**
     * @brief Get extractor name.
     * @return the extractor name.
     */
    virtual const std::string &get_extractor_name() const = 0;

    /**
     * @brief Get extractor help message.
     * The help message should describe the extractor and its parameters.
     * @return the extractor help message.
     */
    virtual const std::string &get_extractor_help() const = 0;

    /**
     * @brief Virtual constructor for defined feature extractors.
     * @param id is the feature extractor type to create.
     * @return a shared ptr to the extractor.
     */
    static cv::Ptr<FeaturesExtractor> create(FEATURE_IDS id);

    /**
     * @brief Virtual constructor loading from a file storage.
     * @param fname is the file storage from which load the feature extractor.
     * @return a shared ptr to the extractor.
     */
    static cv::Ptr<FeaturesExtractor> create(const std::string &fname);

    /**
     * @brief Set extractor parameters.
     * @param params are the parameters.
     */
    void set_params(const std::vector<float> &params);

    /**
     * @brief Get extactor parameters.
     * @return the parameters.
     */
    const std::vector<float> &get_params() const;

    /**
     * @brief Train the extractor with samples.
     * @param dt is the Dataset used to train the feature extractor.
     * @warning By default this method does nothing. Override if your extractor need training.
     */
    virtual void train(const Dataset &dt);

    /**
     * @brief Extract features from an image.
     * @param img the input image.
     * @param features the output features.
     * @return the extracted features as a row vector.
     * @post features.type()==CV_32FC1
     * @post features.rows==1
     */
    virtual cv::Mat extract_features(const cv::Mat &img) = 0;

    /**
     * @brief Save the trained data for the feature extractor.
     *
     * At least the feature type id and the parameters are saved with
     * labels 'fsiv_feature_id' and 'fsiv_feature_params' labels.
     *
     * If you override this method use 'fsiv_xxxx' labels for your data.
     *
     * @param fname is the model filename.
     */
    virtual bool save_model(std::string const &fname) const;

    /**
     * @brief Load the trained data for the feature extractor.
     *
     * At least the feature type id and the parameters are load using
     * the labels 'fsiv_feature_id' and 'fsiv_feature_params'.
     *
     * @param f is the model filename.
     * @return true if success.
     */
    virtual bool load_model(std::string const &fname);

protected:
    FEATURE_IDS type_;
    std::vector<float> params_;
};

/**
 * @brief Extract features from a dataset.
 *
 * @param dt is are the dataset's samples (one sample per row).
 * @param extractor is the features extractor to use.
 * @return the extracted features [X,y] one row per dataset sample.
 * @pre dt.size()>0
 * @post ret_v.first.type()==CV_32FC1
 * @post ret_v.second.type()==CV_32SC1
 * @post ret_v.first.rows==dataset.size()
 * @post ret_v.second.rows==dataset.size()
 */
std::tuple<cv::Mat, cv::Mat> fsiv_extract_features(const Dataset &dt,
                                                   cv::Ptr<FeaturesExtractor> &extractor);

/**
 * @brief Outputs a parameters vector.
 * @param out is the output stream.
 * @param params is the parameter vector.
 * @return the output stream.
 */
inline std::ostream &
operator<<(std::ostream &out, const std::vector<float> &params)
{
    out << '[';
    for (auto &v : params)
        out << ' ' << v;
    out << " ]";
    return out;
}
