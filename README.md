# Pollen Classification System

A computer vision and machine learning project for classifying 15 different types of pollen grains using feature extraction and traditional ML classifiers. This project was developed as part of a course at the University of Córdoba (2024/2025 academic year).

## Overview

This project implements a complete image classification pipeline for pollen grain recognition. The system extracts visual features from pollen images using techniques like Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG), then classifies them using machine learning algorithms including K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Trees.

## Features

- **Multiple Feature Extractors**: 
  - Normalized gray levels
  - Local Binary Patterns (LBP)
  - Histogram of Oriented Gradients (HOG)
  - Combined LBP+HOG features

- **Machine Learning Classifiers**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM) with multiple kernels
  - Random Trees (RTrees)

- **Dimensionality Reduction**: Principal Component Analysis (PCA) support for feature reduction

- **Complete Pipeline**: Training, validation, testing, and model evaluation tools

- **Dual Implementation**: Both C++ (OpenCV) and Python implementations available

## Dataset

The project works with a pollen dataset containing 15 different pollen types:
- alnus, betula, carpinus, corylus, cupressaceae, fagus, fraxinus, picea, pinus, poaceae, populus, quercus, salix, tilia, urticaceae

Images are available in two resolutions: 128x128 (original) and 64x64 (cropped version).

## Project Structure

```
pollen_clf/
├── classifiers.cpp/hpp      # ML classifier implementations (KNN, SVM, RTrees)
├── dataset.cpp/hpp            # Dataset loading and management
├── features.cpp/hpp         # Feature extraction base classes
├── lbp_extractor.cpp/hpp   # LBP feature extractor
├── hog_extractor.cpp/hpp    # HOG feature extractor
├── lbp_hog_extractor.cpp/hpp # Combined LBP+HOG extractor
├── pca.cpp/hpp              # PCA dimensionality reduction
├── metrics.cpp/hpp           # Evaluation metrics (accuracy, confusion matrix)
├── train_clf.cpp            # Training program
├── test_clf.cpp             # Testing program
├── show_BAA500.cpp          # Visualization tool
├── python_helper.ipynb      # Python analysis and experiments
└── CMakeLists.txt           # Build configuration
```

## Requirements

- **C++17** or higher
- **CMake** 3.10 or higher
- **OpenCV** (with ML module)
- **OpenMP** (optional, for parallel processing)

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Training a Classifier

```bash
./train_clf --f_id=3 --f_params="1.5 6 5 5 64 64 32 16 16 9" \
            --use_pca --pca_variance=0.88 \
            --clf=1 --svm_K=2 --svm_C=2.0 --svm_G=3.5 \
            --rseed=4 ../data/train model_lbp_hog_pca.yml
```

### Testing a Classifier

```bash
./test_clf -t data/test model_lbp_hog_pca.yml submission.csv
```

### Listing Available Feature Extractors

```bash
./train_clf --f_list
```

## Key Technologies

- **C++17**: Core implementation
- **OpenCV**: Computer vision and machine learning library
- **CMake**: Build system
- **Python**: Data analysis and experimentation (Jupyter notebooks)

## Results

The project includes comprehensive experimentation with different feature extractors and classifier configurations. Results are tracked in `experiment_results.csv` and include metrics such as:
- Training accuracy
- Validation accuracy
- Model size
- Training time

## Course Information

This project was developed for a computer vision and machine learning course at the **University of Córdoba** during the **2024/2025 academic year**. The project demonstrates practical application of:
- Image feature extraction techniques
- Traditional machine learning classifiers
- Model evaluation and hyperparameter tuning
- Software engineering practices in C++

## Portfolio Summary

This project showcases my ability to:
- Implement computer vision algorithms from scratch (LBP, HOG)
- Work with OpenCV for image processing and ML
- Build end-to-end ML pipelines (data loading → feature extraction → training → evaluation)
- Optimize models through hyperparameter tuning and feature engineering
- Write clean, modular C++ code following best practices
- Conduct systematic experiments and analyze results

## License

This project was developed as part of an academic course. Please refer to the course guidelines for usage and distribution terms.
