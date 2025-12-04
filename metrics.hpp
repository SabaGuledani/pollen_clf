#pragma once
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Computes a confusion matrix.
 *
 * @param true_labels are annotated true labels.
 * @param predicted_labels are the predicted labels.
 * @param n_categories specified the number of categories in the problem.
 * @return the confusion matrix.
 * @pre true_labels.rows==predicted_labels.rows
 * @pre true_labels.type()==CV_32SC1
 * @pre predicted_labels.type()==CV_32SC1
 * @post ret_v.type()==CV_32FC1
 * @post sum(ret_v)[0]==true_labels.rows
 */
cv::Mat fsiv_compute_confusion_matrix(const cv::Mat &true_labels,
                                      const cv::Mat &predicted_labels,
                                      int n_categories);

/**
 * @brief Compute the accuracy metric.
 *
 * @param cmat is the confusion matrix.
 * @return the metric.
 * @pre cmat.type()==CV_32FC1
 * @pre cmat.rows == cmat.cols && cmat.rows>1
 * @post 0.0<=ret_val<=1.0
 */
float fsiv_compute_accuracy(const cv::Mat &cmat);

/**
 * @brief Compute the Recognition Rate metric for each category.
 *
 * @param cmat is the confusion matrix.
 * @return a row vector with the Recognition Rates.
 * @post ret_v.rows == cmat.rows
 * @post ret_v.cols == 1
 * @post ret_v.type()==CV_32FC1
 */
cv::Mat fsiv_compute_recognition_rates(const cv::Mat &cmat);

/**
 * @brief Compute the Mean Recognition Rate metric.
 *
 * @param RRs are the Recognition Rates per category.
 * @return the mean Recognition Rate.
 */
float fsiv_compute_mean_recognition_rate(const cv::Mat &RRs);
