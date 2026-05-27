"""
🖼️ Image Data Processor
Specialized processor for image data with computer vision features
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
from skimage import feature, filters, measure, exposure
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
import warnings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image Data Processor
    
    Specialized processor for image data with intelligent:
    - Feature extraction (texture, shape, color)
    - Image preprocessing and enhancement
    - Computer vision features
    - Image statistics and metadata
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_extractors = {}
        self.processing_history = []
        
    def process(self, X: pd.DataFrame, y: pd.Series = None, 
                config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process image data with intelligent computer vision features
        
        Args:
            X: Feature data (should include image paths or image arrays)
            y: Target data (optional)
            config: Processing configuration
            
        Returns:
            Processed data and processing metadata
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if X is None:
            raise ValueError("X cannot be None")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        if len(X.shape) < 2:
            raise ValueError("X must have at least 2 columns")
        
        if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")
        
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary or None")
        
        logger.info("🖼️ Processing image data")
        start_time = time.time()
        
        config = config or self._get_default_config()
        metadata = {
            "original_shape": X.shape,
            "processing_steps": [],
            "image_columns": []
        }
        
        X_processed = X.copy()
        
        # Step 1: Detect image columns
        image_columns = self._detect_image_columns(X_processed)
        metadata["image_columns"] = image_columns
        
        if not image_columns:
            logger.warning("No image columns detected")
            return X_processed, metadata
        
        # Step 2: Load and validate images
        if config.get("load_images", True):
            X_processed, loading_metadata = self._load_and_validate_images(
                X_processed, image_columns, config
            )
            metadata["image_loading"] = loading_metadata
            metadata["processing_steps"].append("image_loading")
        
        # Step 3: Extract basic image statistics
        if config.get("extract_basic_stats", True):
            X_processed, stats_metadata = self._extract_basic_image_stats(
                X_processed, image_columns, config
            )
            metadata["basic_stats"] = stats_metadata
            metadata["processing_steps"].append("basic_stats")
        
        # Step 4: Extract texture features
        if config.get("extract_texture", True):
            X_processed, texture_metadata = self._extract_texture_features(
                X_processed, image_columns, config
            )
            metadata["texture_features"] = texture_metadata
            metadata["processing_steps"].append("texture_features")
        
        # Step 5: Extract shape features
        if config.get("extract_shape", True):
            X_processed, shape_metadata = self._extract_shape_features(
                X_processed, image_columns, config
            )
            metadata["shape_features"] = shape_metadata
            metadata["processing_steps"].append("shape_features")
        
        # Step 6: Extract color features
        if config.get("extract_color", True):
            X_processed, color_metadata = self._extract_color_features(
                X_processed, image_columns, config
            )
            metadata["color_features"] = color_metadata
            metadata["processing_steps"].append("color_features")
        
        # Step 7: Extract edge features
        if config.get("extract_edges", True):
            X_processed, edge_metadata = self._extract_edge_features(
                X_processed, image_columns, config
            )
            metadata["edge_features"] = edge_metadata
            metadata["processing_steps"].append("edge_features")
        
        # Step 8: Extract histogram features
        if config.get("extract_histograms", True):
            X_processed, hist_metadata = self._extract_histogram_features(
                X_processed, image_columns, config
            )
            metadata["histogram_features"] = hist_metadata
            metadata["processing_steps"].append("histogram_features")
        
        # Step 9: Handle missing values
        if config.get("handle_missing", True):
            X_processed, missing_metadata = self._handle_image_missing(
                X_processed, config
            )
            metadata["missing_handling"] = missing_metadata
            metadata["processing_steps"].append("missing_values")
        
        # Step 10: Scale features
        if config.get("scale_features", True):
            X_processed, scaling_metadata = self._scale_image_features(
                X_processed, config
            )
            metadata["feature_scaling"] = scaling_metadata
            metadata["processing_steps"].append("feature_scaling")
        
        # Step 11: Drop original image columns (optional)
        if config.get("drop_image_columns", True):
            X_processed = X_processed.drop(columns=image_columns)
        
        processing_time = time.time() - start_time
        
        metadata.update({
            "final_shape": X_processed.shape,
            "processing_time": processing_time,
            "features_created": X_processed.shape[1] - X.shape[1]
        })
        
        # Store processing history
        self.processing_history.append(metadata)
        
        logger.info(f"✅ Image processing complete: {X.shape} → {X_processed.shape} in {processing_time:.2f}s")
        return X_processed, metadata
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default image processing configuration"""
        return {
            "load_images": True,
            "extract_basic_stats": True,
            "extract_texture": True,
            "extract_shape": True,
            "extract_color": True,
            "extract_edges": True,
            "extract_histograms": True,
            "handle_missing": True,
            "scale_features": True,
            "drop_image_columns": True,
            
            # Image loading parameters
            "target_size": (224, 224),
            "color_mode": "rgb",  # rgb, grayscale, hsv
            "normalize": True,
            
            # Texture parameters
            "lbp_radius": 3,
            "lbp_points": 24,
            "gabor_frequencies": [0.1, 0.3, 0.5],
            "gabor_orientations": 8,
            
            # Shape parameters
            "contour_approximation": 0.01,
            "min_contour_area": 100,
            
            # Color parameters
            "color_bins": 256,
            "color_space": "hsv",  # rgb, hsv, lab
            
            # Edge parameters
            "edge_threshold1": 50,
            "edge_threshold2": 150,
            
            # Scaling parameters
            "scaling_method": "standard"  # standard, minmax, robust
        }
    
    def _detect_image_columns(self, X: pd.DataFrame) -> List[str]:
        """Detect image columns in the dataset"""
        image_columns = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                # Check if it contains file paths or image arrays
                sample_values = X[col].dropna().head(5)
                
                if len(sample_values) > 0:
                    # Check for file extensions
                    is_file_path = any(
                        str(val).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))
                        for val in sample_values
                    )
                    
                    # Check for numpy arrays (already loaded)
                    is_array = any(
                        isinstance(val, np.ndarray) and len(val.shape) >= 2
                        for val in sample_values
                    )
                    
                    if is_file_path or is_array:
                        image_columns.append(col)
        
        return image_columns
    
    def _load_and_validate_images(self, X: pd.DataFrame, image_columns: List[str],
                                config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and validate images"""
        logger.info("📁 Loading and validating images")
        
        target_size = config.get("target_size", (224, 224))
        color_mode = config.get("color_mode", "rgb")
        normalize = config.get("normalize", True)
        
        loading_stats = {}
        
        for col in image_columns:
            loaded_images = []
            failed_loads = 0
            
            for idx, img_data in enumerate(X[col]):
                try:
                    if img_data is None or pd.isna(img_data):
                        loaded_images.append(None)
                        failed_loads += 1
                        continue
                    
                    # Check if it's a file path or already loaded array
                    if isinstance(img_data, str):
                        # Load from file path
                        image = cv2.imread(img_data)
                        if image is None:
                            loaded_images.append(None)
                            failed_loads += 1
                            continue
                    elif isinstance(img_data, np.ndarray):
                        # Use provided array
                        image = img_data.copy()
                    else:
                        loaded_images.append(None)
                        failed_loads += 1
                        continue
                    
                    # Resize image
                    image = cv2.resize(image, target_size)
                    
                    # Convert color mode
                    if color_mode == "grayscale":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = np.expand_dims(image, axis=-1)  # Add channel dimension
                    elif color_mode == "hsv":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    elif color_mode == "rgb":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Normalize
                    if normalize:
                        image = image.astype(np.float32) / 255.0
                    
                    loaded_images.append(image)
                    
                except Exception as e:
                    logger.warning(f"Failed to load image at index {idx}: {e}")
                    loaded_images.append(None)
                    failed_loads += 1
            
            # Update column with loaded images
            X[col] = loaded_images
            
            loading_stats[col] = {
                "total_images": len(X[col]),
                "successful_loads": len(loaded_images) - failed_loads,
                "failed_loads": failed_loads,
                "success_rate": (len(loaded_images) - failed_loads) / len(loaded_images),
                "target_size": target_size,
                "color_mode": color_mode
            }
        
        metadata = {
            "loading_stats": loading_stats,
            "columns_processed": image_columns
        }
        
        return X, metadata
    
    def _extract_basic_image_stats(self, X: pd.DataFrame, image_columns: List[str],
                                 config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract basic image statistics"""
        logger.info("📊 Extracting basic image statistics")
        
        stats_features = {}
        
        for col in image_columns:
            images = X[col]
            
            # Basic statistics
            stats_features[f'{col}_width'] = [img.shape[1] if img is not None else 0 for img in images]
            stats_features[f'{col}_height'] = [img.shape[0] if img is not None else 0 for img in images]
            stats_features[f'{col}_channels'] = [img.shape[2] if img is not None and len(img.shape) > 2 else 1 for img in images]
            stats_features[f'{col}_aspect_ratio'] = [
                (img.shape[1] / img.shape[0]) if img is not None and img.shape[0] > 0 else 0
                for img in images
            ]
            
            # Pixel statistics
            stats_features[f'{col}_mean_intensity'] = [
                np.mean(img) if img is not None else 0 for img in images
            ]
            stats_features[f'{col}_std_intensity'] = [
                np.std(img) if img is not None else 0 for img in images
            ]
            stats_features[f'{col}_min_intensity'] = [
                np.min(img) if img is not None else 0 for img in images
            ]
            stats_features[f'{col}_max_intensity'] = [
                np.max(img) if img is not None else 0 for img in images
            ]
            
            # Image quality metrics
            stats_features[f'{col}_is_valid'] = [1 if img is not None else 0 for img in images]
        
        # Add features to dataframe
        for feature_name, feature_data in stats_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(stats_features.keys()),
            "statistics_calculated": [
                "dimensions", "aspect_ratio", "intensity_stats", "quality_metrics"
            ]
        }
        
        return X, metadata
    
    def _extract_texture_features(self, X: pd.DataFrame, image_columns: List[str],
                                config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract texture features"""
        logger.info("🔍 Extracting texture features")
        
        lbp_radius = config.get("lbp_radius", 3)
        lbp_points = config.get("lbp_points", 24)
        gabor_frequencies = config.get("gabor_frequencies", [0.1, 0.3, 0.5])
        gabor_orientations = config.get("gabor_orientations", 8)
        
        texture_features = {}
        
        for col in image_columns:
            images = X[col]
            
            # Local Binary Pattern features
            lbp_mean = []
            lbp_std = []
            
            # Gabor features
            gabor_mean = []
            gabor_std = []
            
            for img in images:
                if img is None:
                    lbp_mean.append(0)
                    lbp_std.append(0)
                    gabor_mean.append([0] * len(gabor_frequencies) * len(gabor_orientations))
                    gabor_std.append([0] * len(gabor_frequencies) * len(gabor_orientations))
                    continue
                
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img.squeeze()
                
                # Local Binary Pattern
                lbp = local_binary_pattern(gray_img, P=lbp_points, R=lbp_radius, method='uniform')
                lbp_mean.append(np.mean(lbp))
                lbp_std.append(np.std(lbp))
                
                # Gabor filters
                gabor_responses = []
                for freq in gabor_frequencies:
                    for theta in range(gabor_orientations):
                        theta_rad = theta * np.pi / gabor_orientations
                        real, _ = gabor(gray_img, frequency=freq, theta=theta_rad)
                        gabor_responses.append(np.mean(real))
                
                gabor_mean.append(gabor_responses)
                gabor_std.append([
                    np.std(gabor(gray_img, frequency=freq, theta=theta * np.pi / gabor_orientations)[0])
                    for freq in gabor_frequencies
                    for theta in range(gabor_orientations)
                ])
            
            # Add LBP features
            texture_features[f'{col}_lbp_mean'] = lbp_mean
            texture_features[f'{col}_lbp_std'] = lbp_std
            
            # Add Gabor features
            for i, (freq, theta) in enumerate([
                (freq, theta) for freq in gabor_frequencies 
                for theta in range(gabor_orientations)
            ]):
                gabor_mean_values = [g[i] for g in gabor_mean]
                gabor_std_values = [g[i] for g in gabor_std]
                
                texture_features[f'{col}_gabor_mean_{freq}_{theta}'] = gabor_mean_values
                texture_features[f'{col}_gabor_std_{freq}_{theta}'] = gabor_std_values
        
        # Add features to dataframe
        for feature_name, feature_data in texture_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(texture_features.keys()),
            "texture_methods": ["lbp", "gabor"],
            "parameters": {
                "lbp_radius": lbp_radius,
                "lbp_points": lbp_points,
                "gabor_frequencies": gabor_frequencies,
                "gabor_orientations": gabor_orientations
            }
        }
        
        return X, metadata
    
    def _extract_shape_features(self, X: pd.DataFrame, image_columns: List[str],
                              config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract shape features"""
        logger.info("🔷 Extracting shape features")
        
        contour_approx = config.get("contour_approximation", 0.01)
        min_contour_area = config.get("min_contour_area", 100)
        
        shape_features = {}
        
        for col in image_columns:
            images = X[col]
            
            # Shape statistics
            num_contours = []
            largest_contour_area = []
            contour_perimeter = []
            contour_circularity = []
            
            for img in images:
                if img is None:
                    num_contours.append(0)
                    largest_contour_area.append(0)
                    contour_perimeter.append(0)
                    contour_circularity.append(0)
                    continue
                
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img.squeeze()
                
                # Threshold to get binary image
                _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
                
                num_contours.append(len(valid_contours))
                
                if valid_contours:
                    # Find largest contour
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    largest_contour_area.append(cv2.contourArea(largest_contour))
                    
                    # Calculate perimeter
                    perimeter = cv2.arcLength(largest_contour, True)
                    contour_perimeter.append(perimeter)
                    
                    # Calculate circularity
                    area = cv2.contourArea(largest_contour)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    contour_circularity.append(circularity)
                else:
                    largest_contour_area.append(0)
                    contour_perimeter.append(0)
                    contour_circularity.append(0)
        
            # Add shape features
            shape_features[f'{col}_num_contours'] = num_contours
            shape_features[f'{col}_largest_contour_area'] = largest_contour_area
            shape_features[f'{col}_contour_perimeter'] = contour_perimeter
            shape_features[f'{col}_contour_circularity'] = contour_circularity
        
        # Add features to dataframe
        for feature_name, feature_data in shape_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(shape_features.keys()),
            "shape_metrics": [
                "num_contours", "largest_contour_area", "contour_perimeter", "contour_circularity"
            ]
        }
        
        return X, metadata
    
    def _extract_color_features(self, X: pd.DataFrame, image_columns: List[str],
                             config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract color features"""
        logger.info("🎨 Extracting color features")
        
        color_bins = config.get("color_bins", 256)
        color_space = config.get("color_space", "hsv")
        
        color_features = {}
        
        for col in image_columns:
            images = X[col]
            
            # Color statistics
            mean_colors = []
            std_colors = []
            
            # Color histograms (simplified - just means per channel)
            for img in images:
                if img is None:
                    mean_colors.append([0, 0, 0])
                    std_colors.append([0, 0, 0])
                    continue
                
                # Convert color space if needed
                if color_space == "hsv" and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                elif color_space == "lab" and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                
                # Calculate channel statistics
                if len(img.shape) == 3:
                    # Multi-channel image
                    channel_means = [np.mean(img[:, :, i]) for i in range(img.shape[2])]
                    channel_stds = [np.std(img[:, :, i]) for i in range(img.shape[2])]
                else:
                    # Grayscale image
                    channel_means = [np.mean(img)]
                    channel_stds = [np.std(img)]
                
                mean_colors.append(channel_means)
                std_colors.append(channel_stds)
            
            # Convert to separate columns
            for i in range(3):  # Assume max 3 channels
                channel_mean_values = [m[i] if i < len(m) else 0 for m in mean_colors]
                channel_std_values = [s[i] if i < len(s) else 0 for s in std_colors]
                
                color_features[f'{col}_channel_{i}_mean'] = channel_mean_values
                color_features[f'{col}_channel_{i}_std'] = channel_std_values
        
        # Add features to dataframe
        for feature_name, feature_data in color_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(color_features.keys()),
            "color_space": color_space,
            "color_metrics": ["channel_means", "channel_stds"]
        }
        
        return X, metadata
    
    def _extract_edge_features(self, X: pd.DataFrame, image_columns: List[str],
                             config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract edge features"""
        logger.info("📐 Extracting edge features")
        
        threshold1 = config.get("edge_threshold1", 50)
        threshold2 = config.get("edge_threshold2", 150)
        
        edge_features = {}
        
        for col in image_columns:
            images = X[col]
            
            # Edge statistics
            edge_density = []
            edge_mean = []
            edge_std = []
            
            for img in images:
                if img is None:
                    edge_density.append(0)
                    edge_mean.append(0)
                    edge_std.append(0)
                    continue
                
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img.squeeze()
                
                # Canny edge detection
                edges = cv2.Canny(gray_img, threshold1, threshold2)
                
                # Calculate edge statistics
                edge_density.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
                edge_mean.append(np.mean(edges))
                edge_std.append(np.std(edges))
            
            # Add edge features
            edge_features[f'{col}_edge_density'] = edge_density
            edge_features[f'{col}_edge_mean'] = edge_mean
            edge_features[f'{col}_edge_std'] = edge_std
        
        # Add features to dataframe
        for feature_name, feature_data in edge_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(edge_features.keys()),
            "edge_method": "canny",
            "parameters": {
                "threshold1": threshold1,
                "threshold2": threshold2
            }
        }
        
        return X, metadata
    
    def _extract_histogram_features(self, X: pd.DataFrame, image_columns: List[str],
                                  config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract histogram features"""
        logger.info("📊 Extracting histogram features")
        
        bins = config.get("color_bins", 256)
        
        hist_features = {}
        
        for col in image_columns:
            images = X[col]
            
            # Histogram statistics
            hist_means = []
            hist_stds = []
            hist_skewness = []
            hist_kurtosis = []
            
            for img in images:
                if img is None:
                    hist_means.append([0, 0, 0])
                    hist_stds.append([0, 0, 0])
                    hist_skewness.append([0, 0, 0])
                    hist_kurtosis.append([0, 0, 0])
                    continue
                
                # Calculate histograms for each channel
                channel_stats = []
                
                if len(img.shape) == 3:
                    # Multi-channel image
                    for i in range(min(3, img.shape[2])):
                        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
                        hist_flat = hist.flatten()
                        
                        channel_stats.append({
                            "mean": np.mean(hist_flat),
                            "std": np.std(hist_flat),
                            "skewness": self._calculate_skewness(hist_flat),
                            "kurtosis": self._calculate_kurtosis(hist_flat)
                        })
                else:
                    # Grayscale image
                    hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
                    hist_flat = hist.flatten()
                    
                    channel_stats.append({
                        "mean": np.mean(hist_flat),
                        "std": np.std(hist_flat),
                        "skewness": self._calculate_skewness(hist_flat),
                        "kurtosis": self._calculate_kurtosis(hist_flat)
                    })
                
                # Pad to 3 channels if needed
                while len(channel_stats) < 3:
                    channel_stats.append({"mean": 0, "std": 0, "skewness": 0, "kurtosis": 0})
                
                # Extract statistics
                hist_means.append([s["mean"] for s in channel_stats])
                hist_stds.append([s["std"] for s in channel_stats])
                hist_skewness.append([s["skewness"] for s in channel_stats])
                hist_kurtosis.append([s["kurtosis"] for s in channel_stats])
            
            # Convert to separate columns
            for i in range(3):
                hist_mean_values = [h[i] for h in hist_means]
                hist_std_values = [h[i] for h in hist_stds]
                hist_skew_values = [h[i] for h in hist_skewness]
                hist_kurt_values = [h[i] for h in hist_kurtosis]
                
                hist_features[f'{col}_hist_{i}_mean'] = hist_mean_values
                hist_features[f'{col}_hist_{i}_std'] = hist_std_values
                hist_features[f'{col}_hist_{i}_skewness'] = hist_skew_values
                hist_features[f'{col}_hist_{i}_kurtosis'] = hist_kurt_values
        
        # Add features to dataframe
        for feature_name, feature_data in hist_features.items():
            X[feature_name] = feature_data
        
        metadata = {
            "features_created": list(hist_features.keys()),
            "histogram_bins": bins,
            "histogram_metrics": ["mean", "std", "skewness", "kurtosis"]
        }
        
        return X, metadata
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _handle_image_missing(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in image features"""
        logger.info("🔧 Handling missing values in image features")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        missing_info = {}
        
        for col in numeric_cols:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    "missing_count": missing_count,
                    "missing_ratio": missing_count / len(X)
                }
                
                # Fill with 0 for image features
                X[col] = X[col].fillna(0)
        
        metadata = {
            "missing_info": missing_info,
            "total_missing_before": sum(X.isnull().sum()),
            "total_missing_after": sum(X.isnull().sum())
        }
        
        return X, metadata
    
    def _scale_image_features(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale image features"""
        logger.info("⚖️ Scaling image features")
        
        method = config.get("scaling_method", "standard")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            return X, {"method": "none"}
        
        # Fit and transform
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers["image_global"] = scaler
        
        metadata = {
            "method": method,
            "features_scaled": list(numeric_cols),
            "scaler_fitted": True
        }
        
        return X, metadata
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of image processing operations"""
        if not self.processing_history:
            return {"error": "No processing history available"}
        
        latest = self.processing_history[-1]
        
        return {
            "total_processing_sessions": len(self.processing_history),
            "latest_session": {
                "original_shape": latest["original_shape"],
                "final_shape": latest["final_shape"],
                "processing_time": latest["processing_time"],
                "steps_applied": latest["processing_steps"],
                "features_created": latest.get("features_created", 0)
            },
            "all_steps_applied": list(set(
                step for session in self.processing_history 
                for step in session["processing_steps"]
            ))
        }
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new image data using fitted processors"""
        X_processed = X.copy()
        
        # Detect image columns
        image_columns = self._detect_image_columns(X_processed)
        
        if not image_columns:
            return X_processed
        
        # Apply image loading and processing
        config = self._get_default_config()
        X_processed, _ = self._load_and_validate_images(X_processed, image_columns, config)
        
        # Apply scaling if fitted
        if "image_global" in self.scalers:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_cols] = self.scalers["image_global"].transform(X_processed[numeric_cols])
        
        return X_processed
    
    def reset(self):
        """Reset all fitted processors"""
        self.scalers.clear()
        self.feature_extractors.clear()
        self.processing_history.clear()
        logger.info("🔄 Image processor reset")
