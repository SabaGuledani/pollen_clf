#pragma once

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

/**
 * @brief Create a KNN classifier.
 *
 * @param K specifies how many neighbors are used to class a new sample.
 * @return the created classifier.
 */
cv::Ptr<cv::ml::StatModel> fsiv_create_knn_classifier(int K);

/**
 * @brief Create a SVM classifier.
 *
 * @param Kernel is the kernel to use. @see SVM::KernelTypes.
 * @param C set the tolerance parameter.
 * @param degree set the degree if a polynomial kernel is used.
 * @param gamma set the gamma parameter if RBF kernel is used.
 * @return the created classifier.
 */
cv::Ptr<cv::ml::StatModel> fsiv_create_svm_classifier(int Kernel,
                                                      float C,
                                                      float degree,
                                                      float gamma);
/**
 * @brief Create a RTree classifier.
 *
 * @param V specifies the number of features used per node (ActiveVarCount).
 * @param T specifies the number of trees used.
 * @param E specifies the required OOB error.
 * @return the created classifier.
 */
cv::Ptr<cv::ml::StatModel> fsiv_create_rtrees_classifier(int V,
                                                         int T,
                                                         float E);

/**
 * @brief Train a classifier.
 *
 * @param clf is the classifier to be trained.
 * @param X are the samples.
 * @param y are the labels.
 * @pre X has a row per sample.
 * @pre y.type()==CV_32SC1
 */
void fsiv_train_classifier(cv::Ptr<cv::ml::StatModel> &clf,
                           cv::Mat const &X, cv::Mat const &y);

/**
 * @brief Predict labels using a trained classifier.
 *
 * @param clf is the classifier.
 * @param X are the new samples whose labels we want to predict.
 * @pre clf is trained.
 * @post ret_v.rows == X.rows
 * @post ret_v.type()==CV_32SC1
 */
cv::Mat fsiv_predict_labels(cv::Ptr<cv::ml::StatModel> &clf, cv::Mat const &X);

/**
 * @brief Save the model of a trained classifier to file.
 *
 * @param clf the classifier.
 * @param model_fname the filename where saving the model.
 */
void fsiv_save_classifier_model(cv::Ptr<cv::ml::StatModel> &clf,
                                const std::string &model_fname);

/**
 * @brief Load a knn classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
cv::Ptr<cv::ml::StatModel> fsiv_load_knn_classifier_model(
    const std::string &model_fname);

/**
 * @brief Load a svm classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
cv::Ptr<cv::ml::StatModel> fsiv_load_svm_classifier_model(
    const std::string &model_fname);

/**
 * @brief Load a rtrees classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
cv::Ptr<cv::ml::StatModel> fsiv_load_rtrees_classifier_model(
    const std::string &model_fname);

/**
 * @brief Load a classifier model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 */
cv::Ptr<cv::ml::StatModel> fsiv_load_classifier_model(
    const std::string &model_fname);