#include "pca.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

cv::PCA fsiv_apply_pca(cv::Mat &X_train, cv::Mat &X_valid, 
                        double variance_threshold, 
                        int max_components)
{
    CV_Assert(X_train.type() == CV_32FC1);
    CV_Assert(X_train.rows > 0 && X_train.cols > 0);
    CV_Assert(variance_threshold > 0.0 && variance_threshold <= 1.0);
    
    int original_dim = X_train.cols;
    std::cout << "Applying PCA for dimensionality reduction..." << std::endl;
    
    // Create PCA object
    cv::PCA pca;
    
    // Compute PCA on training data with all components first to determine n_components
    cv::PCA pca_full = cv::PCA(X_train, cv::Mat(), cv::PCA::DATA_AS_ROW, 0);
    
    // Calculate cumulative explained variance
    cv::Mat eigenvalues = pca_full.eigenvalues.clone();
    double total_variance = cv::sum(eigenvalues)[0];
    
    // Find number of components needed to retain variance_threshold
    int n_components = 0;
    double cumulative_variance = 0.0;
    
    if (max_components > 0 && max_components < eigenvalues.rows)
    {
        // Use fixed number of components
        n_components = max_components;
        std::cout << "  Using " << n_components << " PCA components" << std::endl;
    }
    else
    {
        // Use variance threshold
        for (int i = 0; i < eigenvalues.rows; ++i)
        {
            cumulative_variance += eigenvalues.at<float>(i, 0);
            if (cumulative_variance / total_variance >= variance_threshold)
            {
                n_components = i + 1;
                break;
            }
        }
        
        // If we didn't reach threshold, use all components
        if (n_components == 0)
            n_components = eigenvalues.rows;
        
        double actual_variance = cumulative_variance / total_variance;
        std::cout << "  Using PCA to retain " << (variance_threshold * 100.0) 
                  << "% of variance (actual: " << (actual_variance * 100.0) << "%)" << std::endl;
        std::cout << "  Selected " << n_components << " components out of " << eigenvalues.rows << std::endl;
    }
    
    // Compute PCA with the selected number of components
    pca = cv::PCA(X_train, cv::Mat(), cv::PCA::DATA_AS_ROW, n_components);
    
    // Calculate explained variance ratio from original eigenvalues
    // The explained variance ratio is the sum of selected eigenvalues / sum of all eigenvalues
    double selected_variance = 0.0;
    for (int i = 0; i < n_components && i < eigenvalues.rows; ++i)
    {
        selected_variance += eigenvalues.at<float>(i, 0);
    }
    double explained_variance_ratio = 0.0;
    if (total_variance > 0)
    {
        explained_variance_ratio = selected_variance / total_variance;
    }
    
    // Transform training data
    cv::Mat X_train_transformed = pca.project(X_train);
    X_train = X_train_transformed;
    
    std::cout << "  Reduced features from " << original_dim << " to " << X_train.cols << " dimensions" << std::endl;
    std::cout << "  Explained variance ratio: " << std::fixed << std::setprecision(4) 
              << explained_variance_ratio << std::endl;
    
    // Transform validation data if provided
    if (!X_valid.empty())
    {
        cv::Mat X_valid_transformed = pca.project(X_valid);
        X_valid = X_valid_transformed;
        std::cout << "  Transformed validation features to " << X_valid.cols << " dimensions" << std::endl;
    }
    
    std::cout << std::endl;
    
    return pca;
}


