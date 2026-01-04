"""
Prediction Script for LBP+HOG SVM Model

This script loads a trained model, makes predictions on the validation set,
and displays a confusion matrix using matplotlib/seaborn.
Uses multiprocessing for fast feature extraction.
"""

import numpy as np
import cv2
import os
import pickle
import json
from pathlib import Path
from typing import Tuple, List, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

# Machine Learning
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Visualization
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    SEABORN_AVAILABLE = False
    print("seaborn not available - using matplotlib only")

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


def load_model(model_path: str) -> Tuple[object, Optional[object], dict]:
    """Load trained model, PCA (if used), and metadata.
    
    Args:
        model_path: Path to .pkl model file
    
    Returns:
        Tuple of (classifier, pca, metadata_dict)
    """
    # Load model
    print(f"Loading model from '{model_path}'...")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    print("Model loaded.")
    
    # Load PCA if exists
    pca_path = model_path.replace('.pkl', '_pca.pkl')
    pca = None
    if os.path.exists(pca_path):
        print(f"Loading PCA from '{pca_path}'...")
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        print("PCA loaded.")
    
    # Load metadata
    metadata_path = model_path + '.json'
    metadata = {}
    if os.path.exists(metadata_path):
        print(f"Loading metadata from '{metadata_path}'...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("Metadata loaded.")
    else:
        print(f"Warning: Metadata file not found: {metadata_path}")
    
    return classifier, pca, metadata


def parse_feature_params(feature_params_str: str) -> dict:
    """Parse feature parameters string into dictionary.
    
    Args:
        feature_params_str: Space-separated string "radius neighbors grid_rows grid_cols win_w win_h block_size block_stride cell_size nbins"
    
    Returns:
        Dictionary with parsed parameters
    """
    params = feature_params_str.split()
    if len(params) != 10:
        raise ValueError(f"Expected 10 parameters, got {len(params)}")
    
    return {
        'lbp_radius': float(params[0]),
        'lbp_neighbors': int(params[1]),
        'lbp_grid_rows': int(params[2]),
        'lbp_grid_cols': int(params[3]),
        'hog_win_width': int(params[4]),
        'hog_win_height': int(params[5]),
        'hog_block_size': int(params[6]),
        'hog_block_stride': int(params[7]),
        'hog_cell_size': int(params[8]),
        'hog_nbins': int(params[9])
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 10)):
    """Plot confusion matrix using matplotlib/seaborn.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix (percentages)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    
    # Plot 1: Raw counts
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
    else:
        im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0].figure.colorbar(im1, ax=axes[0])
        axes[0].set(xticks=np.arange(len(class_names)),
                   yticks=np.arange(len(class_names)),
                   xticklabels=class_names, yticklabels=class_names,
                   xlabel='Predicted', ylabel='True',
                   title='Confusion Matrix (Counts)')
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot 2: Normalized percentages
    if SEABORN_AVAILABLE:
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    else:
        im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        axes[1].figure.colorbar(im2, ax=axes[1], format='%.1f%%')
        axes[1].set(xticks=np.arange(len(class_names)),
                   yticks=np.arange(len(class_names)),
                   xticklabels=class_names, yticklabels=class_names,
                   xlabel='Predicted', ylabel='True',
                   title='Confusion Matrix (Percentages)')
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                axes[1].text(j, i, format(cm_normalized[i, j], '.1f'),
                           ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black")
    
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to '{save_path}'")
    
    plt.show()


def predict_validation_set(model_path: str, dataset_path: str, 
                          valid_set: str = "valid",
                          n_jobs: int = -1,
                          save_confusion_matrix: Optional[str] = None):
    """Load model and make predictions on validation set.
    
    Args:
        model_path: Path to .pkl model file
        dataset_path: Path to dataset folder
        valid_set: Validation set name (expects {valid_set}.csv)
        n_jobs: Number of parallel jobs for feature extraction (-1 = use all CPUs)
        save_confusion_matrix: Optional path to save confusion matrix figure
    """
    # Load model, PCA, and metadata
    classifier, pca, metadata = load_model(model_path)
    
    # Parse feature parameters from metadata
    if 'feature_params' not in metadata:
        raise ValueError("Metadata does not contain 'feature_params'")
    
    feature_params = parse_feature_params(metadata['feature_params'])
    print(f"\nFeature extractor parameters:")
    print(f"  LBP: radius={feature_params['lbp_radius']}, neighbors={feature_params['lbp_neighbors']}, "
          f"grid={feature_params['lbp_grid_rows']}x{feature_params['lbp_grid_cols']}")
    print(f"  HOG: win={feature_params['hog_win_width']}x{feature_params['hog_win_height']}, "
          f"block={feature_params['hog_block_size']}, stride={feature_params['hog_block_stride']}, "
          f"cell={feature_params['hog_cell_size']}, bins={feature_params['hog_nbins']}")
    
    # Create feature extractor
    extractor = LbpHogExtractor(**feature_params)
    
    # Load validation dataset
    print(f"\nLoading validation dataset from '{dataset_path}'...")
    valid_dataset = Dataset()
    if not valid_dataset.load(dataset_path, valid_set):
        raise RuntimeError(f"Error: could not load validation set '{valid_set}' from '{dataset_path}'")
    print(f"Validation partition with {len(valid_dataset)} samples.")
    
    # Extract features
    print("\nExtracting features from validation set...")
    X_valid, y_valid = extract_features_from_dataset(valid_dataset, extractor, n_jobs=n_jobs)
    print(f"Extracted features vector dimension: 1x{X_valid.shape[1]}")
    
    # Apply PCA if used during training
    if metadata.get('use_pca', False) and pca is not None:
        print("\nApplying PCA transformation...")
        X_valid = pca.transform(X_valid)
        print(f"Transformed features to {X_valid.shape[1]} dimensions")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = classifier.predict(X_valid)
    print("Predictions completed.")
    
    # Compute accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"\nValidation Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    
    # Get class names (only first 15 classes, excluding 'unknown')
    class_names = Dataset.get_class_names()[:15]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred, target_names=class_names, digits=4))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_valid, y_pred, class_names, save_path=save_confusion_matrix)
    
    return y_valid, y_pred, accuracy


if __name__ == '__main__':
    # Example usage
    model_path = "model_lbp_hog_svm.pkl"
    dataset_path = "./data/data/train"  # Adjust path as needed
    valid_set = "valid"
    
    # Make predictions
    y_true, y_pred, accuracy = predict_validation_set(
        model_path=model_path,
        dataset_path=dataset_path,
        valid_set=valid_set,
        n_jobs=-1,  # Use all CPUs
        save_confusion_matrix="confusion_matrix_validation.png"  # Optional: save figure
    )
    
    print(f"\n{'='*60}")
    print(f"Prediction completed!")
    print(f"Validation Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

