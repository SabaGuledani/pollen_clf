#include "pca.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
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

void fsiv_transform_pca(const cv::PCA &pca, cv::Mat &data)
{
    CV_Assert(!pca.eigenvectors.empty());
    CV_Assert(data.type() == CV_32FC1);
    
    cv::Mat transformed = pca.project(data);
    data = transformed;
}

bool fsiv_save_pca_model(const cv::PCA &pca, const std::string &model_fname)
{
    if (pca.eigenvectors.empty())
    {
        std::cerr << "Warning: PCA model is empty, cannot save." << std::endl;
        return false;
    }
    
    cv::FileStorage fs(model_fname, cv::FileStorage::APPEND);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open file for appending PCA model: " << model_fname << std::endl;
        return false;
    }
    
    fs << "fsiv_pca_eigenvectors" << pca.eigenvectors;
    fs << "fsiv_pca_eigenvalues" << pca.eigenvalues;
    fs << "fsiv_pca_mean" << pca.mean;
    fs << "fsiv_pca_use_pca" << true;
    
    fs.release();
    return true;
}

cv::PCA fsiv_load_pca_model(const std::string &model_fname)
{
    cv::PCA pca;
    cv::FileStorage fs(model_fname, cv::FileStorage::READ);
    
    if (!fs.isOpened())
    {
        std::cerr << "Error: Could not open file for reading PCA model: " << model_fname << std::endl;
        return pca;
    }
    
    cv::FileNode node = fs["fsiv_pca_use_pca"];
    if (node.empty())
    {
        // PCA model not found in file (PCA was not used)
        fs.release();
        return pca;
    }
    
    // Check if PCA was actually used (should be true or 1)
    bool use_pca = false;
    if (node.isInt())
    {
        use_pca = (node.operator int() != 0);
    }
    else if (node.isReal())
    {
        use_pca = (node.operator double() != 0.0);
    }
    
    if (!use_pca)
    {
        // PCA was not used during training
        fs.release();
        return pca;
    }
    
    node = fs["fsiv_pca_eigenvectors"];
    if (node.empty())
    {
        std::cerr << "Error: Could not load 'fsiv_pca_eigenvectors' from file." << std::endl;
        fs.release();
        return pca;
    }
    node >> pca.eigenvectors;
    
    node = fs["fsiv_pca_eigenvalues"];
    if (node.empty())
    {
        std::cerr << "Error: Could not load 'fsiv_pca_eigenvalues' from file." << std::endl;
        fs.release();
        return pca;
    }
    node >> pca.eigenvalues;
    
    node = fs["fsiv_pca_mean"];
    if (node.empty())
    {
        std::cerr << "Error: Could not load 'fsiv_pca_mean' from file." << std::endl;
        fs.release();
        return pca;
    }
    node >> pca.mean;
    
    fs.release();
    return pca;
}

