"""
LBP+HOG Feature Extractor with SVM Classifier Training

This script replicates the C++ training code (train_clf.cpp) for training an SVM classifier 
using LBP+HOG feature extractor. The code is optimized for GPU usage where possible and 
allows easy hyperparameter tuning.

Optimized with vectorized LBP computation and multiprocessing for faster feature extraction.
"""

import numpy as np
import cv2
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
import pickle
import json
from multiprocessing import Pool, cpu_count
from functools import partial

# Machine Learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For GPU acceleration (optional - requires cuML)
CUML_AVAILABLE = False
cuSVC = None
try:
    import cuml
    from cuml.svm import SVC as cuSVC
    CUML_AVAILABLE = True
    print("cuML available - GPU acceleration enabled")
except ImportError:
    CUML_AVAILABLE = False
    print("cuML not available - using CPU sklearn")
    print("To enable GPU acceleration, install cuML. See instructions in the script comments.")

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available - progress bars disabled")

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")


class Dataset:
    """Dataset class to load and manage pollen classification dataset."""
    
    # Class names matching C++ code (15 pollen types + unknown)
    CLASS_NAMES = [
        "alnus", "betula", "carpinus", "corylus", "cupressaceae", "fagus",
        "fraxinus", "picea", "pinus", "poaceae", "populus", "quercus",
        "salix", "tilia", "urticaceae", "unknown"
    ]
    
    def __init__(self):
        self.sample_images = []
        self.sample_labels = []
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
    
    def load(self, folder: str, set_name: str) -> bool:
        """Load dataset from CSV file.
        
        Args:
            folder: Dataset folder path
            set_name: Set name (e.g., 'train', 'valid') - expects {set_name}.csv file
        
        Returns:
            True if successful
        """
        set_filename = os.path.join(folder, f"{set_name}.csv")
        
        if not os.path.exists(set_filename):
            print(f"Error: CSV file not found: {set_filename}")
            return False
        
        self.sample_images = []
        self.sample_labels = []
        
        with open(set_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip header
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                # Parse CSV: find last comma to separate filename and label
                comma_pos = line.rfind(',')
                if comma_pos == -1:
                    continue
                
                # Get label (after last comma)
                label = line[comma_pos + 1:].strip()
                
                # Get filename (before last comma, handle quotes)
                image_filename = line[:comma_pos].strip()
                if image_filename.startswith('"'):
                    image_filename = image_filename[1:]
                if image_filename.endswith('"'):
                    image_filename = image_filename[:-1]
                
                if image_filename and label:
                    # Build full path
                    image_path = os.path.join(folder, image_filename)
                    
                    if label in self.class_name_to_id:
                        self.sample_images.append(image_path)
                        self.sample_labels.append(self.class_name_to_id[label])
        
        return len(self.sample_images) > 0
    
    def get_sample(self, index: int) -> np.ndarray:
        """Get sample image at index.
        
        Args:
            index: Sample index
        
        Returns:
            Grayscale image resized to 64x64 (matching C++ code)
        """
        if index >= len(self.sample_images):
            raise IndexError(f"Index {index} out of range")
        
        img = cv2.imread(self.sample_images[index], cv2.IMREAD_GRAYSCALE)
        
        if img is None or img.size == 0:
            print(f"Warning: Could not load image: {self.sample_images[index]}")
            return np.zeros((64, 64), dtype=np.uint8)
        
        # Resize to 64x64 to match C++ code
        if img.shape[0] != 64 or img.shape[1] != 64:
            img = cv2.resize(img, (64, 64))
        
        return img
    
    def get_label(self, index: int) -> int:
        """Get label at index."""
        return self.sample_labels[index]
    
    def get_sample_filename(self, index: int) -> str:
        """Get sample filename at index."""
        return self.sample_images[index]
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.sample_images)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Get sample as tuple (image, label)."""
        return self.get_sample(index), self.get_label(index)
    
    @staticmethod
    def get_class_names() -> List[str]:
        """Get class names."""
        return Dataset.CLASS_NAMES.copy()


class LbpExtractor:
    """Local Binary Pattern (LBP) feature extractor - OPTIMIZED VERSION."""
    
    def __init__(self, radius: float = 1.0, neighbors: int = 8, 
                 grid_rows: int = 1, grid_cols: int = 1):
        self.radius = radius
        self.neighbors = neighbors
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Pre-compute neighbor offsets for vectorization
        angles = 2.0 * np.pi * np.arange(neighbors) / neighbors
        self.offset_x = np.round(radius * np.cos(angles)).astype(int)
        self.offset_y = np.round(radius * np.sin(angles)).astype(int)
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract LBP features from image - VECTORIZED for speed.
        
        Args:
            img: Grayscale image (single channel)
        
        Returns:
            Feature vector (1D array)
        """
        if img.ndim != 2:
            raise ValueError("Image must be grayscale (2D)")
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        h, w = img.shape
        border = int(np.ceil(self.radius))
        
        # Create coordinate grids for vectorized computation
        y_coords, x_coords = np.mgrid[border:h-border, border:w-border]
        
        # Initialize LBP image
        lbp_img = np.zeros((h - 2*border, w - 2*border), dtype=np.uint8)
        
        # Get center values
        center_values = img[y_coords, x_coords].astype(np.float32)
        
        # Compute LBP using vectorized operations
        for i in range(self.neighbors):
            # Calculate neighbor coordinates
            ny = y_coords + self.offset_y[i]
            nx = x_coords + self.offset_x[i]
            
            # Handle boundary conditions
            valid_mask = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            
            # Get neighbor values (use center value for out-of-bounds)
            neighbor_values = np.where(valid_mask, 
                                       img[ny, nx].astype(np.float32), 
                                       center_values)
            
            # Compare and set bit
            comparison = (neighbor_values >= center_values).astype(np.uint8)
            lbp_img |= (comparison << i)
        
        # Compute histogram(s)
        hist_size = 1 << self.neighbors  # 2^neighbors bins
        
        if self.grid_rows == 1 and self.grid_cols == 1:
            # Single histogram for entire image
            hist, _ = np.histogram(lbp_img.flatten(), bins=hist_size, range=(0, hist_size))
            # Normalize (L1 normalization)
            hist = hist.astype(np.float32)
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            features = hist
        else:
            # Spatial histogram: divide image into grid
            cell_height = lbp_img.shape[0] // self.grid_rows
            cell_width = lbp_img.shape[1] // self.grid_cols
            
            histograms = []
            
            for gr in range(self.grid_rows):
                for gc in range(self.grid_cols):
                    y_start = gr * cell_height
                    y_end = lbp_img.shape[0] if gr == self.grid_rows - 1 else (gr + 1) * cell_height
                    x_start = gc * cell_width
                    x_end = lbp_img.shape[1] if gc == self.grid_cols - 1 else (gc + 1) * cell_width
                    
                    cell = lbp_img[y_start:y_end, x_start:x_end]
                    hist, _ = np.histogram(cell.flatten(), bins=hist_size, range=(0, hist_size))
                    # Normalize (L1 normalization)
                    hist = hist.astype(np.float32)
                    if np.sum(hist) > 0:
                        hist = hist / np.sum(hist)
                    histograms.append(hist)
            
            # Concatenate all histograms
            features = np.concatenate(histograms)
        
        return features.astype(np.float32)


class HogExtractor:
    """Histogram of Oriented Gradients (HOG) feature extractor."""
    
    def __init__(self, win_width: int = 64, win_height: int = 64,
                 block_size: int = 16, block_stride: int = 8,
                 cell_size: int = 8, nbins: int = 9):
        self.win_width = win_width
        self.win_height = win_height
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        
        # Create HOG descriptor - use positional arguments for OpenCV 4.10.0
        # HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, ...)
        self.hog = cv2.HOGDescriptor(
            (win_width, win_height),           # winSize
            (block_size, block_size),            # blockSize
            (block_stride, block_stride),        # blockStride
            (cell_size, cell_size),              # cellSize
            nbins                                # nbins
        )
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract HOG features from image.
        
        Args:
            img: Grayscale image (single channel)
        
        Returns:
            Feature vector (1D array)
        """
        if img.ndim != 2:
            raise ValueError("Image must be grayscale (2D)")
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Resize image to window size if needed
        if img.shape[1] != self.win_width or img.shape[0] != self.win_height:
            img = cv2.resize(img, (self.win_width, self.win_height))
        
        # Compute HOG features
        descriptors = self.hog.compute(img)
        
        # Convert to 1D array
        if descriptors is not None:
            return descriptors.flatten().astype(np.float32)
        else:
            return np.array([], dtype=np.float32)


class LbpHogExtractor:
    """Combined LBP+HOG feature extractor."""
    
    def __init__(self, 
                 # LBP parameters (first 4)
                 lbp_radius: float = 1.0, lbp_neighbors: int = 8,
                 lbp_grid_rows: int = 1, lbp_grid_cols: int = 1,
                 # HOG parameters (next 6)
                 hog_win_width: int = 64, hog_win_height: int = 64,
                 hog_block_size: int = 16, hog_block_stride: int = 8,
                 hog_cell_size: int = 8, hog_nbins: int = 9):
        
        self.lbp_extractor = LbpExtractor(lbp_radius, lbp_neighbors, 
                                         lbp_grid_rows, lbp_grid_cols)
        self.hog_extractor = HogExtractor(hog_win_width, hog_win_height,
                                        hog_block_size, hog_block_stride,
                                        hog_cell_size, hog_nbins)
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract combined LBP+HOG features from image.
        
        Args:
            img: Grayscale image (single channel)
        
        Returns:
            Combined feature vector (1D array)
        """
        if img.ndim != 2:
            raise ValueError("Image must be grayscale (2D)")
        
        # Preprocess: Apply histogram equalization (matching C++ code)
        img_preprocessed = cv2.equalizeHist(img.astype(np.uint8))
        
        # Extract LBP features
        lbp_features = self.lbp_extractor.extract_features(img_preprocessed)
        
        # Extract HOG features
        hog_features = self.hog_extractor.extract_features(img_preprocessed)
        
        # Normalize each feature type independently (L2 normalization)
        # This ensures both feature types contribute equally
        lbp_norm = lbp_features / (np.linalg.norm(lbp_features) + 1e-10)
        hog_norm = hog_features / (np.linalg.norm(hog_features) + 1e-10)
        
        # Concatenate LBP and HOG features
        combined_features = np.concatenate([lbp_norm, hog_norm])
        
        return combined_features.astype(np.float32)


def _extract_single_feature(args):
    """Helper function for multiprocessing - extracts features from a single sample."""
    i, image_path, label, extractor_params = args
    try:
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return i, None, label, f"Empty sample: {image_path}"
        
        # Resize to 64x64 if needed
        if img.shape[0] != 64 or img.shape[1] != 64:
            img = cv2.resize(img, (64, 64))
        
        # Recreate extractor (can't pickle OpenCV objects easily)
        extractor = LbpHogExtractor(**extractor_params)
        features = extractor.extract_features(img)
        return i, features, label, None
    except Exception as e:
        return i, None, label, str(e)


def extract_features_from_dataset(dataset: Dataset, extractor: LbpHogExtractor, 
                                   use_tqdm: bool = True, n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from all samples in dataset - OPTIMIZED with multiprocessing.
    
    Args:
        dataset: Dataset object
        extractor: Feature extractor
        use_tqdm: Whether to show progress bar
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
    
    Returns:
        Tuple of (features matrix, labels array)
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Process first sample to get feature dimension
    first_sample = dataset.get_sample(0)
    if first_sample is None or first_sample.size == 0:
        raise ValueError(f"First sample is empty: {dataset.get_sample_filename(0)}")
    
    first_feature = extractor.extract_features(first_sample)
    feature_dim = first_feature.shape[0]
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = min(n_jobs, len(dataset), cpu_count())
    
    # Extract extractor parameters for multiprocessing
    extractor_params = {
        'lbp_radius': extractor.lbp_extractor.radius,
        'lbp_neighbors': extractor.lbp_extractor.neighbors,
        'lbp_grid_rows': extractor.lbp_extractor.grid_rows,
        'lbp_grid_cols': extractor.lbp_extractor.grid_cols,
        'hog_win_width': extractor.hog_extractor.win_width,
        'hog_win_height': extractor.hog_extractor.win_height,
        'hog_block_size': extractor.hog_extractor.block_size,
        'hog_block_stride': extractor.hog_extractor.block_stride,
        'hog_cell_size': extractor.hog_extractor.cell_size,
        'hog_nbins': extractor.hog_extractor.nbins
    }
    
    # Allocate memory
    X = np.zeros((len(dataset), feature_dim), dtype=np.float32)
    y = np.zeros(len(dataset), dtype=np.int32)
    
    # Copy first sample
    X[0] = first_feature
    y[0] = dataset.get_label(0)
    
    # Use multiprocessing for the rest if we have multiple samples
    if len(dataset) > 1 and n_jobs > 1:
        print(f"Extracting features using {n_jobs} parallel workers...")
        
        # Prepare arguments for multiprocessing (use file paths instead of dataset object)
        args_list = [(i, dataset.get_sample_filename(i), dataset.get_label(i), extractor_params) 
                     for i in range(1, len(dataset))]
        
        # Process in parallel
        with Pool(n_jobs) as pool:
            if use_tqdm and TQDM_AVAILABLE:
                results = list(tqdm(pool.imap(_extract_single_feature, args_list), 
                                   total=len(args_list), desc="Extracting features"))
            else:
                results = pool.map(_extract_single_feature, args_list)
        
        # Process results
        for i, features, label, error in results:
            if error:
                if "Empty sample" not in error:
                    print(f"Warning: sample {i} - {error}")
                X[i] = np.zeros(feature_dim, dtype=np.float32)
            else:
                X[i] = features
            y[i] = label
    else:
        # Fallback to sequential processing
        iterator = range(1, len(dataset))
        if use_tqdm and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Extracting features")
        
        for i in iterator:
            try:
                sample, label = dataset[i]
                if sample is None or sample.size == 0:
                    print(f"Warning: sample {i} is empty. File: {dataset.get_sample_filename(i)}")
                    X[i] = np.zeros(feature_dim, dtype=np.float32)
                    y[i] = label
                    continue
                
                features = extractor.extract_features(sample)
                X[i] = features
                y[i] = label
                
            except Exception as e:
                print(f"Error processing sample {i}: {dataset.get_sample_filename(i)}")
                print(f"Error: {e}")
                X[i] = np.zeros(feature_dim, dtype=np.float32)
                y[i] = dataset.get_label(i)
    
    return X, y


def compute_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, 
                            n_categories: int = 15) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        true_labels: True labels (1D array)
        predicted_labels: Predicted labels (1D array)
        n_categories: Number of categories
    
    Returns:
        Confusion matrix (n_categories x n_categories)
    """
    cmat = np.zeros((n_categories, n_categories), dtype=np.float32)
    
    for i in range(len(true_labels)):
        true_label = int(true_labels[i])
        pred_label = int(predicted_labels[i])
        if 0 <= true_label < n_categories and 0 <= pred_label < n_categories:
            cmat[true_label, pred_label] += 1.0
    
    return cmat


def compute_accuracy(cmat: np.ndarray) -> float:
    """Compute accuracy from confusion matrix.
    
    Args:
        cmat: Confusion matrix
    
    Returns:
        Accuracy (0.0 to 1.0)
    """
    diagonal_sum = np.trace(cmat)
    total_sum = np.sum(cmat)
    
    if total_sum > 0:
        return float(diagonal_sum / total_sum)
    else:
        return 0.0


def create_rtrees_classifier(V: int = 0, T: int = 50, E: float = 0.1) -> RandomForestClassifier:
    """Create Random Trees (Random Forest) classifier.
    
    Args:
        V: Number of random features sampled per node (0 = sqrt of total features)
        T: Max number of trees in the forest
        E: OOB error to stop adding more trees (used for monitoring, sklearn doesn't support direct epsilon stopping)
    
    Returns:
        Random Forest classifier
    """
    # Calculate max_features (V parameter)
    # If V is 0, use 'sqrt' (default), otherwise use V
    if V == 0:
        max_features = 'sqrt'
    else:
        max_features = V
    
    # Create Random Forest classifier
    # Note: sklearn doesn't support epsilon-based early stopping like OpenCV,
    # so we use n_estimators (T) and can monitor oob_score
    clf = RandomForestClassifier(
        n_estimators=T,           # T: Max number of trees
        max_features=max_features, # V: Features per node (0 = sqrt, matching C++ code)
        max_depth=40,              # Maximum depth of trees (matching C++ code)
        min_samples_leaf=5,        # Minimum samples per leaf node (matching C++ setMinSampleCount)
        max_leaf_nodes=None,       # No limit on leaf nodes
        n_jobs=-1,                # Use all CPUs
        random_state=42,          # For reproducibility
        oob_score=True,           # Calculate out-of-bag score (related to E parameter)
        class_weight=None,         # Balanced classes
        verbose=0
    )
    
    return clf


def create_svm_classifier(kernel: int = 0, C: float = 1.0, 
                         degree: float = 3.0, gamma: float = 1.0,
                         use_gpu: bool = False) -> SVC:
    """Create SVM classifier.
    
    Args:
        kernel: Kernel type (0:Linear, 1:Polynomial, 2:RBF, 3:Sigmoid, 4:CHI2, 5:INTER)
        C: Regularization parameter
        degree: Degree for polynomial kernel
        gamma: Gamma parameter for RBF, Polynomial, Sigmoid, CHI2 kernels
        use_gpu: Whether to use GPU (requires cuML)
    
    Returns:
        SVM classifier
    """
    kernel_map = {
        0: 'linear',
        1: 'poly',
        2: 'rbf',
        3: 'sigmoid'
    }
    
    if kernel == 4:  # CHI2 kernel - not directly supported in sklearn
        # Use RBF as approximation or custom kernel
        # For now, use RBF with adjusted gamma
        print("Warning: CHI2 kernel not directly supported in sklearn, using RBF approximation")
        kernel_type = 'rbf'
    elif kernel == 5:  # INTER (intersection) kernel
        # Use linear as approximation
        print("Warning: INTER kernel not directly supported in sklearn, using linear approximation")
        kernel_type = 'linear'
    else:
        kernel_type = kernel_map.get(kernel, 'rbf')
    
    if use_gpu and CUML_AVAILABLE:
        # cuML SVM (GPU accelerated)
        if kernel == 0:  # Linear
            clf = cuSVC(kernel='linear', C=C)
        elif kernel == 1:  # Polynomial
            clf = cuSVC(kernel='poly', C=C, degree=int(degree), gamma=gamma)
        elif kernel == 2:  # RBF
            clf = cuSVC(kernel='rbf', C=C, gamma=gamma)
        elif kernel == 3:  # Sigmoid
            clf = cuSVC(kernel='sigmoid', C=C, gamma=gamma)
        else:
            clf = cuSVC(kernel='rbf', C=C, gamma=gamma)
    else:
        # sklearn SVM (CPU)
        if kernel == 0:  # Linear
            clf = SVC(kernel='linear', C=C, random_state=42)
        elif kernel == 1:  # Polynomial
            clf = SVC(kernel='poly', C=C, degree=int(degree), gamma=gamma, random_state=42)
        elif kernel == 2:  # RBF
            clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        elif kernel == 3:  # Sigmoid
            clf = SVC(kernel='sigmoid', C=C, gamma=gamma, random_state=42)
        elif kernel == 4:  # CHI2 - use RBF approximation
            clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        elif kernel == 5:  # INTER - use linear
            clf = SVC(kernel='linear', C=C, random_state=42)
        else:
            clf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    
    return clf


def train_classifier(dataset_path: str, train_set: str = "train", valid_set: str = "valid",
                    # LBP parameters
                    lbp_radius: float = 1.5, lbp_neighbors: int = 6,
                    lbp_grid_rows: int = 2, lbp_grid_cols: int = 2,
                    # HOG parameters
                    hog_win_width: int = 64, hog_win_height: int = 64,
                    hog_block_size: int = 32, hog_block_stride: int = 16,
                    hog_cell_size: int = 16, hog_nbins: int = 9,
                    # Classifier selection
                    classifier: int = 1,  # 0: KNN (not implemented), 1: SVM, 2: RTrees
                    # SVM parameters
                    svm_kernel: int = 2, svm_C: float = 1.0,
                    svm_degree: float = 3.0, svm_gamma: float = 0.1,
                    # RTrees parameters
                    rtrees_V: int = 0, rtrees_T: int = 50, rtrees_E: float = 0.1,
                    # Other parameters
                    random_seed: int = 0, use_gpu: bool = False,
                    model_fname: Optional[str] = None,
                    results_csv: str = "experiment_results.csv",
                    n_jobs: int = -1,
                    # PCA parameters (to reduce overfitting)
                    use_pca: bool = False,
                    pca_components: Optional[int] = None,
                    pca_variance: float = 0.95) -> dict:
    """Train classifier (SVM or RTrees) with LBP+HOG features.
    
    Args:
        dataset_path: Path to dataset folder
        train_set: Training set name (expects {train_set}.csv)
        valid_set: Validation set name (expects {valid_set}.csv)
        lbp_radius: LBP radius
        lbp_neighbors: LBP number of neighbors
        lbp_grid_rows: LBP grid rows
        lbp_grid_cols: LBP grid cols
        hog_win_width: HOG window width
        hog_win_height: HOG window height
        hog_block_size: HOG block size
        hog_block_stride: HOG block stride
        hog_cell_size: HOG cell size
        hog_nbins: HOG number of bins
        classifier: Classifier type (0: KNN not implemented, 1: SVM, 2: RTrees)
        svm_kernel: SVM kernel type (0:Linear, 1:Poly, 2:RBF, 3:Sigmoid, 4:CHI2, 5:INTER)
        svm_C: SVM C parameter
        svm_degree: SVM degree (for polynomial kernel)
        svm_gamma: SVM gamma parameter
        rtrees_V: RTrees number of features per node (0 = sqrt of total features)
        rtrees_T: RTrees max number of trees
        rtrees_E: RTrees OOB error threshold (for monitoring)
        random_seed: Random seed (0 means use current time)
        use_gpu: Whether to use GPU acceleration (SVM only)
        model_fname: Model filename to save (None = don't save)
        results_csv: CSV file to save experiment results
        n_jobs: Number of parallel jobs for feature extraction (-1 = use all CPUs)
        use_pca: Whether to apply PCA for dimensionality reduction (helps with overfitting)
        pca_components: Number of PCA components (None = use variance threshold)
        pca_variance: Variance threshold for PCA (0.95 = keep 95% of variance)
    
    Returns:
        Dictionary with training results
    """
    # Set random seed
    if random_seed == 0:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    print(f"Set the random seed to: {random_seed}")
    
    # Load datasets
    print("Loading train dataset ...", end=" ")
    train_dataset = Dataset()
    if not train_dataset.load(dataset_path, train_set):
        raise RuntimeError(f"Error: could not open dataset path [{dataset_path}] or load train set [{train_set}]")
    print("done.")
    print(f"Train partition with {len(train_dataset)} samples.")
    
    print("Loading validation dataset ...", end=" ")
    valid_dataset = Dataset()
    if not valid_dataset.load(dataset_path, valid_set):
        print("Warning: could not load validation set. Continuing without validation.")
        valid_dataset = None
    else:
        print("done.")
        print(f"Validation partition with {len(valid_dataset)} samples.")
    print()
    
    # Create feature extractor
    extractor = LbpHogExtractor(
        lbp_radius=lbp_radius, lbp_neighbors=lbp_neighbors,
        lbp_grid_rows=lbp_grid_rows, lbp_grid_cols=lbp_grid_cols,
        hog_win_width=hog_win_width, hog_win_height=hog_win_height,
        hog_block_size=hog_block_size, hog_block_stride=hog_block_stride,
        hog_cell_size=hog_cell_size, hog_nbins=hog_nbins
    )
    
    extractor_name = "LBP+HOG Feature Extractor"
    extractor_params = f"{lbp_radius} {lbp_neighbors} {lbp_grid_rows} {lbp_grid_cols} {hog_win_width} {hog_win_height} {hog_block_size} {hog_block_stride} {hog_cell_size} {hog_nbins}"
    print(f"Feature extractor: {extractor_name}")
    print(f"Feature extractor params: {extractor_params}")
    
    # Extract features from training set
    print("Extracting features in train partition ...", end=" ")
    X_train, y_train = extract_features_from_dataset(train_dataset, extractor, n_jobs=n_jobs)
    print("done.")
    print(f"Extracted features vector dimension: 1x{X_train.shape[1]}")
    
    # Extract features from validation set
    X_valid, y_valid = None, None
    if valid_dataset is not None:
        print("Extracting features in validation partition ...", end=" ")
        X_valid, y_valid = extract_features_from_dataset(valid_dataset, extractor, n_jobs=n_jobs)
        print("done.")
    
    # Compute memory usage
    train_memory_mb = (X_train.nbytes) / (1024 * 1024)
    valid_memory_mb = (X_valid.nbytes if X_valid is not None else 0) / (1024 * 1024)
    total_memory_mb = train_memory_mb + valid_memory_mb
    print(f"Extracted features use {total_memory_mb:.2f} Mb of memory.")
    print()
    
    # Apply PCA if requested (helps reduce overfitting)
    pca = None
    original_feature_dim = X_train.shape[1]  # Store original dimension before PCA
    if use_pca:
        print("Applying PCA for dimensionality reduction...")
        if pca_components is not None:
            # Use fixed number of components
            pca = PCA(n_components=pca_components, random_state=random_seed)
            print(f"  Using {pca_components} PCA components")
        else:
            # Use variance threshold
            pca = PCA(n_components=pca_variance, random_state=random_seed)
            print(f"  Using PCA to retain {pca_variance*100:.1f}% of variance")
        
        # Fit PCA on training data only
        X_train = pca.fit_transform(X_train)
        print(f"  Reduced features from {original_feature_dim} to {X_train.shape[1]} dimensions")
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Transform validation data using the same PCA
        if X_valid is not None:
            X_valid = pca.transform(X_valid)
            print(f"  Transformed validation features to {X_valid.shape[1]} dimensions")
        print()
    
    # Create classifier based on type
    if classifier == 1:  # SVM
        kernel_names = {0: "Linear", 1: "Polynomial", 2: "RBF", 3: "Sigmoid", 4: "CHI2", 5: "INTER"}
        kernel_name = kernel_names.get(svm_kernel, f"Unknown({svm_kernel})")
        print(f"Using a SVM classifier with K={svm_kernel} ({kernel_name}) C={svm_C} D={svm_degree} G={svm_gamma}")
        
        clf = create_svm_classifier(
            kernel=svm_kernel, C=svm_C, degree=svm_degree, gamma=svm_gamma,
            use_gpu=use_gpu
        )
        classifier_name = "SVM"
        classifier_params = f"K={svm_kernel} C={svm_C}"
        if svm_kernel == 1:  # Polynomial
            classifier_params += f" D={svm_degree} G={svm_gamma}"
        elif svm_kernel in [2, 3, 4]:  # RBF, Sigmoid, CHI2
            classifier_params += f" G={svm_gamma}"
    elif classifier == 2:  # RTrees
        print(f"Using a RTrees classifier with V={rtrees_V} T={rtrees_T} E={rtrees_E}")
        
        clf = create_rtrees_classifier(V=rtrees_V, T=rtrees_T, E=rtrees_E)
        classifier_name = "RTrees"
        classifier_params = f"V={rtrees_V} T={rtrees_T} E={rtrees_E}"
    else:
        raise ValueError(f"Unknown classifier type: {classifier}. Use 1 for SVM or 2 for RTrees.")
    
    print()
    
    # Train classifier
    print("Training ...", end=" ")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"done. (took {train_time:.2f} seconds)")
    
    # Compute training accuracy
    print("Computing training accuracy ...", end=" ")
    y_train_pred = clf.predict(X_train)
    cmat_train = compute_confusion_matrix(y_train, y_train_pred, n_categories=15)
    train_acc = compute_accuracy(cmat_train)
    print("done.")
    print(f"Training accuracy: {train_acc:.6f}")
    
    # Print OOB score for RTrees if available
    if classifier == 2 and hasattr(clf, 'oob_score_'):
        print(f"Out-of-bag score: {clf.oob_score_:.6f}")
    print()
    
    # Compute validation accuracy
    valid_acc = -1.0
    if X_valid is not None and y_valid is not None:
        print("Validating ...", end=" ")
        y_valid_pred = clf.predict(X_valid)
        cmat_valid = compute_confusion_matrix(y_valid, y_valid_pred, n_categories=15)
        valid_acc = compute_accuracy(cmat_valid)
        print("done.")
        print(f"Validation accuracy: {valid_acc:.6f}")
        print()
    
    # Save model if requested
    model_size_mb = 0.0
    if model_fname:
        print(f"Saving the model to '{model_fname}'.")
        
        # Save classifier
        with open(model_fname, 'wb') as f:
            pickle.dump(clf, f)
        
        # Save PCA if used
        if pca is not None:
            pca_fname = model_fname.replace('.pkl', '_pca.pkl')
            with open(pca_fname, 'wb') as f:
                pickle.dump(pca, f)
            print(f"Saved PCA model to '{pca_fname}'.")
        
        # Save feature extractor parameters
        model_info = {
            'feature_extractor': 'LBP+HOG',
            'feature_params': extractor_params,
            'classifier': classifier_name,
            'classifier_type': classifier,
            'random_seed': random_seed,
            'use_pca': use_pca,
            'original_feature_dim': original_feature_dim if use_pca else None,
            'pca_feature_dim': X_train.shape[1] if use_pca else None,
            'pca_explained_variance': float(pca.explained_variance_ratio_.sum()) if use_pca and pca is not None else None
        }
        
        # Add classifier-specific parameters
        if classifier == 1:  # SVM
            model_info.update({
                'svm_kernel': svm_kernel,
                'svm_C': svm_C,
                'svm_degree': svm_degree,
                'svm_gamma': svm_gamma
            })
        elif classifier == 2:  # RTrees
            model_info.update({
                'rtrees_V': rtrees_V,
                'rtrees_T': rtrees_T,
                'rtrees_E': rtrees_E
            })
        
        info_fname = model_fname + '.json'
        with open(info_fname, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Compute model size
        if os.path.exists(model_fname):
            model_size_mb = os.path.getsize(model_fname) / (1024.0 * 1024.0)
            print(f"Model size: {model_size_mb:.6f} Mb.")
            
            # Compute size score (matching C++ code)
            dataset_size_mb = 4.0 * 45.06  # From C++ code
            size_score = max(0.0, 1.0 - (model_size_mb / dataset_size_mb))
            print(f"Size score max(0.0, 1.0-(model_size_mb/dataset_size_mb)) = {size_score:.6f}")
            
            # Predicted final score
            predicted_final_score = (2.0 * train_acc * size_score) / (train_acc + size_score + 1e-10)
            print(f"Predicted final score 2*(acc*size_score)/(acc+size_score) = {predicted_final_score:.6f}")
        else:
            size_score = 0.0
            predicted_final_score = 0.0
    else:
        size_score = 0.0
        predicted_final_score = 0.0
    
    # Write experiment results to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create results dictionary
    results = {
        'Timestamp': timestamp,
        'Classifier': classifier_name,
        'Classifier_Params': classifier_params,
        'Extractor': extractor_name,
        'Extractor_Params': extractor_params,
        'Train_Accuracy': train_acc,
        'Valid_Accuracy': valid_acc if valid_acc >= 0 else None,
        'Test_Accuracy': None,
        'Model_Size_MB': model_size_mb,
        'Size_Score': size_score if model_fname else None,
        'Predicted_Final_Score': predicted_final_score if model_fname else None,
        'Model_Filename': model_fname if model_fname else '',
        'Train_Time_Seconds': train_time
    }
    
    # Append to CSV
    file_exists = os.path.exists(results_csv)
    df_results = pd.DataFrame([results])
    
    if file_exists:
        df_results.to_csv(results_csv, mode='a', header=False, index=False)
    else:
        df_results.to_csv(results_csv, mode='w', header=True, index=False)
    
    print()
    print(f"Saving experiment results to {results_csv} ... done.")
    
    return results


if __name__ == '__main__':
    # Example usage - uncomment and modify as needed
    
    # Example 1: Train with SVM
    results = train_classifier(
        dataset_path="./data/data/train",  # Adjust path as needed
        train_set="train",
        valid_set="valid",
        # LBP parameters
        lbp_radius=2.0,
        lbp_neighbors=8,
        lbp_grid_rows=5,
        lbp_grid_cols=5,
        # HOG parameters
        hog_win_width=64,
        hog_win_height=64,
        hog_block_size=32,
        hog_block_stride=16,
        hog_cell_size=32,
        hog_nbins=18,
        # Classifier selection
        classifier=1,  # 1: SVM, 2: RTrees
        # SVM parameters
        svm_kernel=2,  # RBF
        svm_C=2.0,
        svm_gamma=2.0,
        random_seed=42,
        use_gpu=False,  # Use GPU if available, otherwise use CPU
        # PCA parameters (to reduce overfitting)
        use_pca=True,  # Enable PCA to reduce overfitting
        pca_components=None,  # None = use variance threshold
        pca_variance=0.85,  # Keep 95% of variance
        model_fname="model_lbp_hog_svm.pkl",
        results_csv="experiment_results.csv",
        n_jobs=-1  # Use all CPUs for multiprocessing
    )
    # V_values = [0, 10, 20, 30, 50, 75, 100]
    # T_values = [100, 200, 300, 400, 500]
    # for V in V_values:
    #     # for T in T_values:
    #     results = train_classifier(
    #     dataset_path="./data/data/train",  # Adjust path as needed
    #     train_set="train",
    #     valid_set="valid",
    #     # LBP parameters
    #     lbp_radius=2.0,
    #     lbp_neighbors=8,
    #     lbp_grid_rows=4,
    #     lbp_grid_cols=4,
    #     # HOG parameters
    #     hog_win_width=64,
    #     hog_win_height=64,
    #     hog_block_size=16,
    #     hog_block_stride=16,
    #     hog_cell_size=16,
    #     hog_nbins=9,
    #     # Classifier selection
    #     classifier=2,  # 1: SVM, 2: RTrees
    #     # RTrees parameters
    #     rtrees_V=V,    # 0 = sqrt of total features
    #     rtrees_T=100,   # Number of trees
    #     rtrees_E=0,  # OOB error threshold (for monitoring)
    #     random_seed=42,
    #     model_fname="model_lbp_hog_rtrees.pkl",
    #     results_csv="experiment_results.csv",
    #     n_jobs=-1  # Use all CPUs for multiprocessing
    # )
    # Example 2: Train with RTrees
   
    
    print("\nTraining completed!")
    print(f"Training accuracy: {results['Train_Accuracy']:.6f}")
    if results['Valid_Accuracy'] is not None:
        print(f"Validation accuracy: {results['Valid_Accuracy']:.6f}")
    else:
        print("No validation set")

