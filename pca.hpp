#pragma once

#include <opencv2/core.hpp>

/**
 * @brief Apply PCA (Principal Component Analysis) to reduce dimensionality.
 * 
 * This function fits PCA on training data and transforms both training and validation data.
 * It uses variance threshold to determine the number of components (e.g., 0.85 = 85% variance).
 * 
 * @param X_train Training features (will be modified in-place)
 * @param X_valid Validation features (will be modified in-place, can be empty)
 * @param variance_threshold Variance threshold (0.0-1.0), e.g., 0.85 means keep 85% of variance
 * @param max_components Maximum number of components (0 = no limit, use variance threshold)
 * @return cv::PCA object that can be saved/loaded for later use
 * @pre X_train.type() == CV_32FC1
 * @pre X_train.rows > 0 && X_train.cols > 0
 * @post X_train.cols <= original X_train.cols
 * @post X_valid.cols == X_train.cols (if X_valid is not empty)
 */
cv::PCA fsiv_apply_pca(cv::Mat &X_train, cv::Mat &X_valid, 
                        double variance_threshold = 0.85, 
                        int max_components = 0);


