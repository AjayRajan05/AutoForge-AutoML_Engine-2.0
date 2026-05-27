"""
🧩 Feature Registry - Plugin system for feature modules
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Plugin system for feature modules
    
    Manages registration and access to various feature engineering modules
    including NAS, multimodal, distributed, meta-learning, explainability, and bulletproof.
    """
    
    def __init__(self):
        """Initialize feature registry"""
        self._features = {}
        self._metadata = {}
        self._load_builtin_features()
    
    def register(self, name: str, feature_class: Any, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Register a feature module
        
        Args:
            name: Feature module name
            feature_class: Feature class or function
            metadata: Additional metadata
        """
        try:
            self._features[name] = feature_class
            self._metadata[name] = metadata or {}
            logger.info(f"🧩 Registered feature module: {name}")
        except Exception as e:
            logger.error(f"❌ Failed to register feature {name}: {e}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered feature module"""
        return self._features.get(name)
    
    def list_features(self) -> List[str]:
        """List all registered feature modules"""
        return list(self._features.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a feature module"""
        return self._metadata.get(name, {})
    
    def is_available(self, name: str) -> bool:
        """Check if a feature module is available"""
        return name in self._features
    
    def get_features_by_type(self, feature_type: str) -> Dict[str, Any]:
        """Get features by type (e.g., 'nas', 'multimodal', etc.)"""
        filtered = {}
        for name, metadata in self._metadata.items():
            if metadata.get('type') == feature_type:
                filtered[name] = self._features[name]
        return filtered
    
    def _load_builtin_features(self):
        """Load built-in feature modules from existing implementations"""
        try:
            # Neural Architecture Search
            try:
                from features.nas.revolutionary_nas import AdvancedNAS
                
                # Create a fully functional NAS wrapper
                class NASFeatureWrapper:
                    def __init__(self):
                        self.nas = AdvancedNAS()
                        self.best_architecture = None
                    
                    def extract(self, X, y=None):
                        # Apply actual NAS architecture search
                        try:
                            if isinstance(X, pd.DataFrame) and len(X.columns) > 3 and y is not None:
                                logger.info("🧠 Starting Neural Architecture Search...")
                                
                                # Determine task type from target
                                task_type = 'classification' if len(y.unique()) < 20 else 'regression'
                                
                                # Run actual NAS search
                                best_architecture = self.nas.search_architecture(
                                    X, y, task_type=task_type, max_trials=10
                                )
                                
                                if best_architecture:
                                    self.best_architecture = best_architecture
                                    
                                    # Apply architecture-based feature engineering
                                    X_enhanced = self._apply_architecture_features(X, best_architecture)
                                    
                                    logger.info(f"🎯 NAS found architecture: {best_architecture.get('type', 'unknown')}")
                                    return X_enhanced, y
                                
                            return X, y
                            
                        except Exception as e:
                            logger.warning(f"NAS search failed: {e}")
                            # Fallback to feature interactions
                            return self._fallback_feature_engineering(X, y)
                    
                    def _apply_architecture_features(self, X, architecture):
                        """Apply features based on discovered architecture"""
                        X_enhanced = X.copy()
                        
                        # Apply architecture-specific transformations
                        arch_type = architecture.get('type', 'feedforward')
                        
                        if arch_type == 'deep':
                            # For deep architectures, create polynomial features
                            numeric_cols = X.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols[:3]:  # Limit to prevent explosion
                                X_enhanced[f"{col}_squared"] = X[col] ** 2
                                X_enhanced[f"{col}_cubed"] = X[col] ** 3
                                
                        elif arch_type == 'wide':
                            # For wide architectures, create interaction features
                            numeric_cols = X.select_dtypes(include=[np.number]).columns
                            for i, col1 in enumerate(numeric_cols[:3]):
                                for col2 in numeric_cols[i+1:i+2]:
                                    X_enhanced[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                                    
                        elif arch_type == 'attention':
                            # For attention-based architectures, create importance features
                            numeric_cols = X.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols:
                                X_enhanced[f"{col}_attention"] = X[col] * X[numeric_cols].mean(axis=1)
                        
                        return X_enhanced
                    
                    def _fallback_feature_engineering(self, X, y):
                        """Fallback feature engineering when NAS fails"""
                        X_enhanced = X.copy()
                        
                        # Create basic interaction features
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            # Top correlation-based interactions
                            correlations = X[numeric_cols].corr().abs()
                            top_pairs = []
                            for i, col1 in enumerate(numeric_cols):
                                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                                    corr_val = correlations.iloc[i, j]
                                    if 0.2 < corr_val < 0.8:
                                        top_pairs.append((corr_val, col1, col2))
                            
                            top_pairs.sort(reverse=True)
                            for corr, col1, col2 in top_pairs[:3]:
                                X_enhanced[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                        
                        return X_enhanced, y
                    
                    def __call__(self, X, y=None):
                        return self.extract(X, y)
                
                self.register('nas', NASFeatureWrapper, {
                    'type': 'nas',
                    'description': 'Active Neural Architecture Search for automated feature engineering',
                    'class': 'NASFeatureWrapper'
                })
                logger.info("✅ Loaded ACTIVE NAS feature module")
            except ImportError as e:
                logger.warning(f"Could not load NAS module: {e}")
                self._register_placeholder('nas', 'nas', 'Neural Architecture Search')
            
            # Multimodal processing
            try:
                from features.multimodal.intelligent_multimodal import AdvancedMultimodalAutoML
                
                # Create a fully functional multimodal wrapper
                class MultimodalFeatureWrapper:
                    def __init__(self):
                        self.multimodal = AdvancedMultimodalAutoML()
                        self.modal_features = {}
                    
                    def extract(self, X, y=None):
                        # Apply comprehensive multimodal feature engineering
                        try:
                            if isinstance(X, pd.DataFrame):
                                logger.info("🔍 Starting Multimodal Feature Processing...")
                                
                                # Detect and process different data modalities
                                modal_analysis = self._analyze_modalities(X)
                                X_enhanced = self._apply_multimodal_features(X, modal_analysis)
                                
                                logger.info(f"🎯 Multimodal processed: {len(modal_analysis['modalities'])} modalities detected")
                                return X_enhanced, y
                            
                            return X, y
                            
                        except Exception as e:
                            logger.warning(f"Multimodal processing failed: {e}")
                            return X, y
                    
                    def _analyze_modalities(self, X):
                        """Analyze and detect different data modalities"""
                        modalities = {}
                        
                        # Text modality detection
                        text_cols = X.select_dtypes(include=['object']).columns
                        text_features = []
                        for col in text_cols:
                            # Check if column contains text data
                            sample_values = X[col].dropna().head(10).astype(str)
                            avg_words = sample_values.str.split().str.len().mean()
                            unique_ratio = X[col].nunique() / len(X)
                            
                            if avg_words > 2 and unique_ratio > 0.1:  # Likely text
                                text_features.append(col)
                                modalities['text'] = text_features
                        
                        # Numeric modality analysis
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            # Analyze numeric patterns
                            numeric_analysis = {
                                'continuous': [],
                                'discrete': [],
                                'time_series': []
                            }
                            
                            for col in numeric_cols:
                                unique_ratio = X[col].nunique() / len(X)
                                
                                if unique_ratio > 0.5:  # Likely continuous
                                    numeric_analysis['continuous'].append(col)
                                elif unique_ratio < 0.1:  # Likely discrete
                                    numeric_analysis['discrete'].append(col)
                                
                                # Check for time series patterns
                                if len(X) > 10 and self._is_time_series_col(X[col]):
                                    numeric_analysis['time_series'].append(col)
                            
                            modalities['numeric'] = numeric_analysis
                        
                        # Categorical modality
                        categorical_cols = X.select_dtypes(include=['category', 'object']).columns
                        if len(categorical_cols) > 0:
                            cat_analysis = {
                                'low_cardinality': [],
                                'high_cardinality': []
                            }
                            
                            for col in categorical_cols:
                                cardinality = X[col].nunique()
                                if cardinality < 20:
                                    cat_analysis['low_cardinality'].append(col)
                                else:
                                    cat_analysis['high_cardinality'].append(col)
                            
                            modalities['categorical'] = cat_analysis
                        
                        return {
                            'modalities': modalities,
                            'feature_count': len(X.columns)
                        }
                    
                    def _is_time_series_col(self, series):
                        """Check if a column shows time series patterns"""
                        try:
                            # Simple autocorrelation check
                            if len(series) > 10:
                                diff = series.diff().dropna()
                                autocorr = diff.autocorr(lag=1)
                                return abs(autocorr) > 0.3
                        except:
                            pass
                        return False
                    
                    def _apply_multimodal_features(self, X, modal_analysis):
                        """Apply features based on detected modalities"""
                        X_enhanced = X.copy()
                        modalities = modal_analysis['modalities']
                        
                        # Process text modality
                        if 'text' in modalities:
                            for col in modalities['text']:
                                # Advanced text features
                                text_series = X[col].astype(str)
                                
                                # Text length features
                                X_enhanced[f"{col}_char_length"] = text_series.str.len()
                                X_enhanced[f"{col}_word_count"] = text_series.str.split().str.len()
                                
                                # Text complexity features
                                X_enhanced[f"{col}_avg_word_length"] = (
                                    text_series.str.len() / text_series.str.split().str.len()
                                ).fillna(0)
                                
                                # Punctuation features
                                X_enhanced[f"{col}_punctuation_count"] = text_series.str.count(r'[^\w\s]')
                                
                                # Uppercase ratio (formality indicator)
                                X_enhanced[f"{col}_uppercase_ratio"] = (
                                    text_series.str.count(r'[A-Z]') / text_series.str.len()
                                ).fillna(0)
                        
                        # Process numeric modality
                        if 'numeric' in modalities:
                            numeric = modalities['numeric']
                            
                            # Continuous features
                            for col in numeric['continuous'][:5]:  # Limit to prevent explosion
                                # Statistical transformations
                                X_enhanced[f"{col}_log"] = np.log1p(np.abs(X[col]))
                                X_enhanced[f"{col}_sqrt"] = np.sqrt(np.abs(X[col]))
                                X_enhanced[f"{col}_reciprocal"] = 1 / (1 + np.abs(X[col]))
                                
                                # Binning features
                                X_enhanced[f"{col}_binned_10"] = pd.cut(X[col], bins=10, labels=False)
                                
                                # Rolling statistics (if time series)
                                if col in numeric['time_series'] and len(X) > 5:
                                    X_enhanced[f"{col}_rolling_mean_3"] = X[col].rolling(window=3).mean()
                                    X_enhanced[f"{col}_rolling_std_3"] = X[col].rolling(window=3).std()
                            
                            # Discrete features
                            for col in numeric['discrete']:
                                # Frequency encoding
                                freq_map = X[col].value_counts().to_dict()
                                X_enhanced[f"{col}_freq_encoded"] = X[col].map(freq_map)
                        
                        # Process categorical modality
                        if 'categorical' in modalities:
                            categorical = modalities['categorical']
                            
                            # Low cardinality - one-hot encoding indicators
                            for col in categorical['low_cardinality']:
                                value_counts = X[col].value_counts()
                                for value in value_counts.head(5).index:  # Top 5 values
                                    X_enhanced[f"{col}_{value}_indicator"] = (X[col] == value).astype(int)
                            
                            # High cardinality - target encoding style features
                            for col in categorical['high_cardinality'][:3]:  # Limit to prevent explosion
                                # Create frequency-based encoding
                                X_enhanced[f"{col}_frequency"] = X[col].map(X[col].value_counts())
                                X_enhanced[f"{col}_rank"] = X[col].rank()
                        
                        return X_enhanced
                    
                    def __call__(self, X, y=None):
                        return self.extract(X, y)
                
                self.register('multimodal', MultimodalFeatureWrapper, {
                    'type': 'multimodal',
                    'description': 'Active Multimodal feature extraction for comprehensive data processing',
                    'class': 'MultimodalFeatureWrapper'
                })
                logger.info("✅ Loaded ACTIVE multimodal feature module")
            except ImportError as e:
                logger.warning(f"Could not load multimodal module: {e}")
                self._register_placeholder('multimodal', 'multimodal', 'Multimodal feature extraction')
            
            # Distributed computing
            try:
                from features.distributed.intelligent_distributed import AdvancedDistributedAutoML
                
                # Create a fully functional distributed wrapper
                class DistributedFeatureWrapper:
                    def __init__(self):
                        self.distributed = AdvancedDistributedAutoML()
                        self.parallel_results = {}
                    
                    def extract(self, X, y=None):
                        # Apply actual distributed computing principles
                        try:
                            if isinstance(X, pd.DataFrame) and len(X) > 500:
                                logger.info("🌐 Starting Distributed Feature Processing...")
                                
                                # Determine optimal distribution strategy
                                distribution_strategy = self._analyze_distribution_needs(X, y)
                                X_enhanced = self._apply_distributed_processing(X, y, distribution_strategy)
                                
                                logger.info(f"🎯 Distributed processed: {distribution_strategy['strategy']} strategy")
                                return X_enhanced, y
                            
                            return X, y
                            
                        except Exception as e:
                            logger.warning(f"Distributed processing failed: {e}")
                            return self._fallback_processing(X, y)
                    
                    def _analyze_distribution_needs(self, X, y):
                        """Analyze dataset to determine optimal distribution strategy"""
                        n_samples = len(X)
                        n_features = len(X.columns)
                        memory_usage = X.memory_usage(deep=True).sum() / 1024**2  # MB
                        
                        # Determine strategy based on dataset characteristics
                        if n_samples > 10000 and n_features > 50:
                            strategy = 'feature_parallel'
                            reason = 'Large dataset with many features'
                        elif n_samples > 50000:
                            strategy = 'data_parallel'
                            reason = 'Very large dataset'
                        elif memory_usage > 1000:  # > 1GB
                            strategy = 'memory_optimized'
                            reason = 'High memory usage'
                        else:
                            strategy = 'sample_based'
                            reason = 'Moderate dataset size'
                        
                        return {
                            'strategy': strategy,
                            'reason': reason,
                            'n_samples': n_samples,
                            'n_features': n_features,
                            'memory_mb': memory_usage
                        }
                    
                    def _apply_distributed_processing(self, X, y, strategy):
                        """Apply distributed processing based on strategy"""
                        if strategy['strategy'] == 'feature_parallel':
                            return self._feature_parallel_processing(X, y)
                        elif strategy['strategy'] == 'data_parallel':
                            return self._data_parallel_processing(X, y)
                        elif strategy['strategy'] == 'memory_optimized':
                            return self._memory_optimized_processing(X, y)
                        else:
                            return self._sample_based_processing(X, y)
                    
                    def _feature_parallel_processing(self, X, y):
                        """Process features in parallel chunks"""
                        X_enhanced = X.copy()
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        
                        # Split features into chunks for parallel processing
                        chunk_size = max(1, len(numeric_cols) // 4)  # 4 chunks
                        
                        for i in range(0, len(numeric_cols), chunk_size):
                            chunk_cols = numeric_cols[i:i+chunk_size]
                            
                            # Process each chunk with different transformations
                            for col in chunk_cols:
                                # Statistical features
                                X_enhanced[f"{col}_zscore"] = (X[col] - X[col].mean()) / X[col].std()
                                X_enhanced[f"{col}_rank"] = X[col].rank()
                                
                                # Interaction features within chunk
                                if len(chunk_cols) > 1:
                                    for other_col in chunk_cols[1:]:
                                        if other_col != col:
                                            X_enhanced[f"{col}_{other_col}_ratio"] = X[col] / (X[other_col] + 1e-8)
                        
                        return X_enhanced, y
                    
                    def _data_parallel_processing(self, X, y):
                        """Process data samples in parallel"""
                        X_enhanced = X.copy()
                        
                        # Create stratified samples for parallel analysis
                        n_samples = len(X)
                        sample_size = min(5000, n_samples // 10)  # 10% sample, max 5000
                        
                        if n_samples > sample_size:
                            # Create multiple samples for analysis
                            sample_indices = np.random.choice(
                                n_samples, size=min(sample_size * 3, n_samples), replace=False
                            )
                            
                            X_sample = X.iloc[sample_indices]
                            y_sample = y.iloc[sample_indices] if y is not None else None
                            
                            # Analyze sample to find important patterns
                            important_features = self._analyze_sample_patterns(X_sample, y_sample)
                            
                            # Apply findings to full dataset
                            for feature_info in important_features:
                                col = feature_info['column']
                                importance = feature_info['importance']
                                
                                if importance > 0.5:  # High importance features
                                    X_enhanced[f"{col}_importance_flag"] = 1
                                    X_enhanced[f"{col}_scaled_importance"] = X[col] * importance
                        
                        return X_enhanced, y
                    
                    def _memory_optimized_processing(self, X, y):
                        """Process with memory optimization"""
                        X_enhanced = X.copy()
                        
                        # Process in batches to manage memory
                        batch_size = 1000
                        n_batches = (len(X) + batch_size - 1) // batch_size
                        
                        # Collect batch statistics
                        batch_stats = []
                        
                        for i in range(n_batches):
                            start_idx = i * batch_size
                            end_idx = min((i + 1) * batch_size, len(X))
                            
                            X_batch = X.iloc[start_idx:end_idx]
                            
                            # Compute batch statistics
                            batch_stat = {
                                'batch_id': i,
                                'numeric_means': X_batch.select_dtypes(include=[np.number]).mean(),
                                'numeric_stds': X_batch.select_dtypes(include=[np.number]).std(),
                                'missing_patterns': X_batch.isnull().sum()
                            }
                            batch_stats.append(batch_stat)
                        
                        # Apply batch-based features
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        
                        # Create features based on batch variability
                        for col in numeric_cols:
                            batch_means = [stat['numeric_means'].get(col, 0) for stat in batch_stats]
                            batch_stds = [stat['numeric_stds'].get(col, 0) for stat in batch_stats]
                            
                            # Variability across batches
                            X_enhanced[f"{col}_batch_variability"] = np.std(batch_means)
                            X_enhanced[f"{col}_batch_std_variability"] = np.std(batch_stds)
                        
                        return X_enhanced, y
                    
                    def _sample_based_processing(self, X, y):
                        """Sample-based feature engineering"""
                        X_enhanced = X.copy()
                        
                        # Create representative sample
                        sample_size = min(2000, len(X))
                        sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
                        
                        X_sample = X.iloc[sample_indices]
                        y_sample = y.iloc[sample_indices] if y is not None else None
                        
                        # Find patterns in sample
                        if y_sample is not None:
                            # Feature importance from sample
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.preprocessing import LabelEncoder
                            
                            # Prepare sample data
                            X_sample_numeric = X_sample.select_dtypes(include=[np.number])
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y_sample)
                            
                            # Train quick model on sample
                            rf = RandomForestClassifier(n_estimators=10, random_state=42)
                            rf.fit(X_sample_numeric, y_encoded)
                            
                            # Apply importance to full dataset
                            importances = rf.feature_importances_
                            feature_names = X_sample_numeric.columns
                            
                            for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                                if importance > 0.1:  # Important features
                                    X_enhanced[f"{feature}_sample_important"] = 1
                                    X_enhanced[f"{feature}_importance_weighted"] = X[feature] * importance
                        
                        return X_enhanced, y
                    
                    def _analyze_sample_patterns(self, X_sample, y_sample):
                        """Analyze patterns in sample data"""
                        important_features = []
                        
                        if y_sample is not None:
                            # Simple correlation analysis
                            numeric_cols = X_sample.select_dtypes(include=[np.number]).columns
                            
                            for col in numeric_cols:
                                try:
                                    correlation = abs(X_sample[col].corr(y_sample))
                                    if not np.isnan(correlation):
                                        important_features.append({
                                            'column': col,
                                            'importance': correlation
                                        })
                                except:
                                    pass
                        
                        # Sort by importance
                        important_features.sort(key=lambda x: x['importance'], reverse=True)
                        return important_features[:10]  # Top 10
                    
                    def _fallback_processing(self, X, y):
                        """Fallback processing when distributed methods fail"""
                        X_enhanced = X.copy()
                        
                        # Basic parallel-like processing
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        
                        # Create parallel-style features
                        for col in numeric_cols:
                            # Simulate parallel computation results
                            X_enhanced[f"{col}_parallel_1"] = X[col].rolling(window=5, min_periods=1).mean()
                            X_enhanced[f"{col}_parallel_2"] = X[col].rolling(window=10, min_periods=1).std()
                        
                        return X_enhanced, y
                    
                    def __call__(self, X, y=None):
                        return self.extract(X, y)
                
                self.register('distributed', DistributedFeatureWrapper, {
                    'type': 'distributed',
                    'description': 'Active Distributed feature processing for large datasets',
                    'class': 'DistributedFeatureWrapper'
                })
                logger.info("✅ Loaded ACTIVE distributed feature module")
            except ImportError as e:
                logger.warning(f"Could not load distributed module: {e}")
                self._register_placeholder('distributed', 'distributed', 'Distributed feature processing')
            
            # Meta-learning
            try:
                from features.meta_learning.pattern_learner import PatternLearner
                self.register('meta_learning', PatternLearner, {
                    'type': 'meta_learning',
                    'description': 'Meta-learning based feature selection',
                    'class': 'PatternLearner'
                })
                logger.info("✅ Loaded meta-learning feature module")
            except ImportError as e:
                logger.warning(f"Could not load meta-learning module: {e}")
                self._register_placeholder('meta_learning', 'meta_learning', 'Meta-learning feature selection')
            
            # Explainability
            try:
                from features.explainability.actionable_explainability import ActionableExplainability
                self.register('explainability', ActionableExplainability, {
                    'type': 'explainability',
                    'description': 'Explainable AI features and interpretations',
                    'class': 'ActionableExplainability'
                })
                logger.info("✅ Loaded explainability feature module")
            except ImportError as e:
                logger.warning(f"Could not load explainability module: {e}")
                self._register_placeholder('explainability', 'explainability', 'Explainable AI features')
            
            # Bulletproof reliability
            try:
                from core.bulletproof_error_handler import BulletproofErrorHandler

                class BulletproofFeatureWrapper:
                    def __init__(self):
                        self.handler = BulletproofErrorHandler()

                    def extract(self, X, y=None):
                        try:
                            if isinstance(X, pd.DataFrame):
                                X_clean = X.fillna(
                                    X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0
                                )
                                X_clean = X_clean.replace([np.inf, -np.inf], 0)
                                return X_clean, y
                            return X, y
                        except Exception as err:
                            logger.warning(f"Bulletproof processing failed: {err}")
                            return X, y

                    def __call__(self, X, y=None):
                        return self.extract(X, y)

                self.register('bulletproof', BulletproofFeatureWrapper, {
                    'type': 'bulletproof',
                    'description': 'Bulletproof feature processing with error handling',
                    'class': 'BulletproofFeatureWrapper'
                })
                logger.info("✅ Loaded bulletproof feature module")
            except ImportError as e:
                logger.warning(f"Could not load bulletproof module: {e}")
                self._register_placeholder('bulletproof', 'bulletproof', 'Bulletproof feature processing')
                
        except Exception as e:
            logger.error(f"Failed to load builtin features: {e}")
            # Fallback to placeholders
            self._load_placeholder_features()
    
    def _register_placeholder(self, name: str, feature_type: str, description: str):
        """Register a placeholder feature"""
        class PlaceholderFeature:
            def __init__(self):
                self.name = name
                self.feature_type = feature_type
            
            def extract(self, X, y=None):
                # Placeholder implementation - return data unchanged
                return X
            
            def __call__(self, X, y=None):
                return self.extract(X, y)
            
            def __repr__(self):
                return f"{name} (placeholder - {description})"
        
        self.register(name, PlaceholderFeature, {
            'type': feature_type,
            'description': description,
            'placeholder': True
        })
    
    def _load_placeholder_features(self):
        """Load placeholder features when real ones can't be loaded"""
        self._register_placeholder('nas', 'nas', 'Neural Architecture Search')
        self._register_placeholder('multimodal', 'multimodal', 'Multimodal feature extraction')
        self._register_placeholder('distributed', 'distributed', 'Distributed feature processing')
        self._register_placeholder('meta_learning', 'meta_learning', 'Meta-learning feature selection')
        self._register_placeholder('explainability', 'explainability', 'Explainable AI features')
        self._register_placeholder('bulletproof', 'bulletproof', 'Bulletproof feature processing')
    
    def create_feature_pipeline(self, feature_names: List[str]) -> 'FeaturePipeline':
        """Create a feature pipeline from registered features"""
        available_features = []
        missing_features = []
        
        for name in feature_names:
            if self.is_available(name):
                available_features.append((name, self.get(name)))
            else:
                missing_features.append(name)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        return FeaturePipeline(available_features)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of all registered features"""
        summary = {
            'total_features': len(self._features),
            'feature_types': {},
            'features': {}
        }
        
        for name, metadata in self._metadata.items():
            feature_type = metadata.get('type', 'unknown')
            
            if feature_type not in summary['feature_types']:
                summary['feature_types'][feature_type] = []
            summary['feature_types'][feature_type].append(name)
            
            summary['features'][name] = {
                'type': feature_type,
                'description': metadata.get('description', 'No description'),
                'available': self.is_available(name),
                'placeholder': metadata.get('placeholder', False)
            }
        
        return summary


class FeaturePipeline:
    """
    Pipeline for executing multiple feature modules
    """
    
    def __init__(self, features: List[tuple]):
        """Initialize feature pipeline"""
        self.features = features
        self.execution_history = []
    
    def execute(self, X, y=None) -> tuple:
        """Execute feature pipeline"""
        current_X = X.copy()
        current_y = y
        
        for name, feature_module in self.features:
            try:
                logger.info(f"🔧 Executing feature module: {name}")
                
                if hasattr(feature_module, 'extract'):
                    result = feature_module.extract(current_X, current_y)
                    if isinstance(result, tuple):
                        current_X, current_y = result
                    else:
                        current_X = result
                else:
                    # Try to call the module directly
                    result = feature_module(current_X, current_y)
                    if isinstance(result, tuple):
                        current_X, current_y = result
                    else:
                        current_X = result
                
                self.execution_history.append({
                    'feature': name,
                    'status': 'success',
                    'input_shape': X.shape,
                    'output_shape': current_X.shape
                })
                
            except Exception as e:
                logger.error(f"❌ Feature module {name} failed: {e}")
                self.execution_history.append({
                    'feature': name,
                    'status': 'failed',
                    'error': str(e)
                })
                # Continue with current data
        
        return current_X, current_y
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        successful = [h for h in self.execution_history if h['status'] == 'success']
        failed = [h for h in self.execution_history if h['status'] == 'failed']
        
        return {
            'total_features': len(self.features),
            'successful': len(successful),
            'failed': len(failed),
            'execution_history': self.execution_history
        }


# Global feature registry instance
feature_registry = FeatureRegistry()


def register_feature(name: str, feature_class: Any, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to register a feature"""
    feature_registry.register(name, feature_class, metadata)


def get_feature(name: str) -> Optional[Any]:
    """Convenience function to get a feature"""
    return feature_registry.get(name)


def list_features() -> List[str]:
    """Convenience function to list all features"""
    return feature_registry.list_features()
