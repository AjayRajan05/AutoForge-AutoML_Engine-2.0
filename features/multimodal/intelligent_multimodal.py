"""
Advanced Multimodal AutoML with Meta-Learning Intelligence
Intelligent processing of multiple data types
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

# Setup logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class AdvancedMultimodalAutoML:
    """
    Advanced Multimodal AutoML with Meta-Learning
    
    Current AutoML Multimodal Limitations:
    - Separate processing for each modality
    - No cross-modal learning
    - Fixed preprocessing pipelines
    - No intelligence in modality selection
    
    Our Approach:
    - Cross-modal meta-learning
    - Intelligent modality fusion
    - Adaptive preprocessing
    - Smart modality selection
    """
    
    def __init__(self):
        self.modality_patterns = {}  # Learned modality patterns
        self.cross_modal_knowledge = {}  # Cross-modal learning
        self.modality_performance = {}  # Modality performance history
        
    def analyze_multimodal_data(self, X, y=None):
        """
        Advanced multimodal data analysis with intelligence
        
        Args:
            X: Input data (could be mixed modalities)
            y: Target variable (optional)
            
        Returns:
            Multimodal analysis and recommendations
        """
        logger.info("🌐 Starting Advanced Multimodal Analysis...")
        
        # Step 1: Detect data modalities
        modalities = self._detect_modalities(X)
        
        # Step 2: Analyze each modality
        modality_analysis = {}
        for modality, data in modalities.items():
            modality_analysis[modality] = self._analyze_modality(data, modality)
        
        # Step 3: Cross-modal intelligence
        cross_modal_insights = self._cross_modal_analysis(modality_analysis)
        
        # Step 4: Generate intelligent recommendations
        recommendations = self._generate_multimodal_recommendations(
            modality_analysis, cross_modal_insights
        )
        
        logger.info(f"🎯 Detected {len(modalities)} modalities with intelligent analysis")
        
        return {
            'modalities': modalities,
            'analysis': modality_analysis,
            'cross_modal_insights': cross_modal_insights,
            'recommendations': recommendations
        }
    
    def _detect_modalities(self, X):
        """Intelligent modality detection"""
        modalities = {}
        
        if isinstance(X, pd.DataFrame):
            # Analyze each column
            for col in X.columns:
                modality = self._detect_column_modality(X[col])
                if modality not in modalities:
                    modalities[modality] = {}
                modalities[modality][col] = X[col]
        elif isinstance(X, np.ndarray):
            # Analyze array structure
            if X.dtype == object:
                # Likely text or categorical
                modalities['text'] = X
            else:
                # Likely numerical
                modalities['tabular'] = X
        else:
            # Single modality
            modalities['tabular'] = X
        
        return modalities
    
    def _detect_column_modality(self, series):
        """Detect modality of a single column"""
        # Check for text data
        if series.dtype == object:
            sample_values = series.dropna().head(100)
            
            # Text detection
            if self._is_text_data(sample_values):
                return 'text'
            
            # Categorical detection
            elif self._is_categorical_data(sample_values):
                return 'categorical'
            
            # Mixed data
            else:
                return 'mixed'
        
        # Check for numerical data
        elif pd.api.types.is_numeric_dtype(series):
            if self._is_time_series_data(series):
                return 'time_series'
            else:
                return 'numerical'
        
        # Default to tabular
        return 'tabular'
    
    def _is_text_data(self, sample_values):
        """Detect if data is text"""
        if len(sample_values) == 0:
            return False
        
        # Check for textual characteristics
        avg_length = sample_values.astype(str).str.len().mean()
        unique_words = len(set(' '.join(sample_values.astype(str)).split()))
        
        # Heuristics for text data
        return (avg_length > 20 and unique_words > 50)
    
    def _is_categorical_data(self, sample_values):
        """Detect if data is categorical"""
        if len(sample_values) == 0:
            return False
        
        unique_ratio = sample_values.nunique() / len(sample_values)
        return unique_ratio < 0.5 and len(sample_values.unique()) < 50
    
    def _is_time_series_data(self, series):
        """Detect if data is time series"""
        # Simple heuristic: check for temporal patterns
        if len(series) < 10:
            return False
        
        # Check for autocorrelation (simple version)
        try:
            autocorr = series.autocorr(lag=1)
            return abs(autocorr) > 0.5
        except:
            return False
    
    def _analyze_modality(self, data, modality_type):
        """Intelligent analysis of specific modality"""
        analysis = {
            'type': modality_type,
            'shape': data.shape if hasattr(data, 'shape') else len(data),
            'quality_score': 0.0,
            'complexity': 'low',
            'recommendations': []
        }
        
        if modality_type == 'text':
            analysis.update(self._analyze_text_modality(data))
        elif modality_type == 'numerical':
            analysis.update(self._analyze_numerical_modality(data))
        elif modality_type == 'categorical':
            analysis.update(self._analyze_categorical_modality(data))
        elif modality_type == 'time_series':
            analysis.update(self._analyze_time_series_modality(data))
        else:
            analysis.update(self._analyze_tabular_modality(data))
        
        return analysis
    
    def _analyze_text_modality(self, data):
        """Analyze text modality with intelligence"""
        if isinstance(data, pd.Series):
            text_data = data.dropna().astype(str)
        else:
            text_data = pd.Series(data).dropna().astype(str)
        
        # Text characteristics
        avg_length = text_data.str.len().mean()
        unique_words = len(set(' '.join(text_data).split()))
        vocab_size = len(set(' '.join(text_data).lower().split()))
        
        # Quality assessment
        quality_score = min(1.0, avg_length / 100) * min(1.0, vocab_size / 1000)
        
        # Complexity assessment
        if vocab_size > 5000:
            complexity = 'high'
        elif vocab_size > 1000:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        recommendations = []
        if avg_length < 10:
            recommendations.append("Consider using n-grams for short text")
        if vocab_size > 10000:
            recommendations.append("Consider dimensionality reduction for large vocabulary")
        
        return {
            'avg_length': avg_length,
            'vocab_size': vocab_size,
            'unique_words': unique_words,
            'quality_score': quality_score,
            'complexity': complexity,
            'recommendations': recommendations
        }
    
    def _analyze_numerical_modality(self, data):
        """Analyze numerical modality with intelligence"""
        if isinstance(data, pd.Series):
            num_data = data.dropna()
        else:
            num_data = pd.Series(data).dropna()
        
        # Numerical characteristics
        mean_val = num_data.mean()
        std_val = num_data.std()
        skewness = num_data.skew()
        kurtosis = num_data.kurtosis()
        
        # Quality assessment
        missing_ratio = num_data.isna().sum() / len(num_data) if hasattr(num_data, 'isna') else 0
        quality_score = 1.0 - missing_ratio
        
        # Complexity assessment
        if abs(skewness) > 2 or abs(kurtosis) > 5:
            complexity = 'high'
        elif abs(skewness) > 1 or abs(kurtosis) > 2:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        recommendations = []
        if abs(skewness) > 2:
            recommendations.append("Consider log transformation for skewed data")
        if std_val / abs(mean_val) > 1:
            recommendations.append("Consider scaling for high variance data")
        
        return {
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'quality_score': quality_score,
            'complexity': complexity,
            'recommendations': recommendations
        }
    
    def _analyze_categorical_modality(self, data):
        """Analyze categorical modality with intelligence"""
        if isinstance(data, pd.Series):
            cat_data = data.dropna()
        else:
            cat_data = pd.Series(data).dropna()
        
        # Categorical characteristics
        n_categories = cat_data.nunique()
        category_counts = cat_data.value_counts()
        entropy = -((category_counts / len(cat_data)) * np.log2(category_counts / len(cat_data))).sum()
        
        # Quality assessment
        balance_score = 1.0 - (category_counts.std() / category_counts.mean())
        quality_score = balance_score
        
        # Complexity assessment
        if n_categories > 100:
            complexity = 'high'
        elif n_categories > 20:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        recommendations = []
        if n_categories > 100:
            recommendations.append("Consider grouping rare categories")
        if balance_score < 0.5:
            recommendations.append("Consider balanced sampling for imbalanced categories")
        
        return {
            'n_categories': n_categories,
            'entropy': entropy,
            'balance_score': balance_score,
            'quality_score': quality_score,
            'complexity': complexity,
            'recommendations': recommendations
        }
    
    def _analyze_time_series_modality(self, data):
        """Analyze time series modality with intelligence"""
        if isinstance(data, pd.Series):
            ts_data = data.dropna()
        else:
            ts_data = pd.Series(data).dropna()
        
        # Time series characteristics
        trend_strength = abs(ts_data.autocorr(lag=len(ts_data)//4))
        seasonality_strength = abs(ts_data.autocorr(lag=1))
        
        # Quality assessment
        quality_score = max(trend_strength, seasonality_strength)
        
        # Complexity assessment
        if trend_strength > 0.8 or seasonality_strength > 0.8:
            complexity = 'high'
        elif trend_strength > 0.5 or seasonality_strength > 0.5:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        recommendations = []
        if trend_strength > 0.7:
            recommendations.append("Consider detrending the time series")
        if seasonality_strength > 0.7:
            recommendations.append("Consider seasonal decomposition")
        
        return {
            'trend_strength': trend_strength,
            'seasonality_strength': seasonality_strength,
            'quality_score': quality_score,
            'complexity': complexity,
            'recommendations': recommendations
        }
    
    def _analyze_tabular_modality(self, data):
        """Analyze tabular modality with intelligence"""
        if hasattr(data, 'shape'):
            n_samples, n_features = data.shape
        else:
            n_samples = len(data)
            n_features = 1
        
        # Tabular characteristics
        complexity = 'high' if n_features > 100 else 'medium' if n_features > 20 else 'low'
        quality_score = min(1.0, n_samples / 1000)  # More samples = better quality
        
        recommendations = []
        if n_features > 100:
            recommendations.append("Consider feature selection for high-dimensional data")
        if n_samples < 100:
            recommendations.append("Consider data augmentation for small datasets")
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'quality_score': quality_score,
            'complexity': complexity,
            'recommendations': recommendations
        }
    
    def _cross_modal_analysis(self, modality_analysis):
        """Intelligent cross-modal analysis"""
        insights = {
            'modal synergies': [],
            'conflicts': [],
            'fusion_recommendations': []
        }
        
        # Analyze modality synergies
        modalities = list(modality_analysis.keys())
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                synergy = self._analyze_modality_synergy(
                    modality_analysis[mod1], modality_analysis[mod2]
                )
                if synergy > 0.5:
                    insights['modal synergies'].append((mod1, mod2, synergy))
                elif synergy < -0.5:
                    insights['conflicts'].append((mod1, mod2, synergy))
        
        # Generate fusion recommendations
        if insights['modal synergies']:
            insights['fusion_recommendations'].append(
                "Consider early fusion for synergistic modalities"
            )
        if insights['conflicts']:
            insights['fusion_recommendations'].append(
                "Consider late fusion for conflicting modalities"
            )
        
        return insights
    
    def _analyze_modality_synergy(self, analysis1, analysis2):
        """Analyze synergy between two modalities"""
        # Simple heuristic based on quality scores
        quality1 = analysis1.get('quality_score', 0.5)
        quality2 = analysis2.get('quality_score', 0.5)
        
        # Complementarity bonus
        if analysis1['type'] != analysis2['type']:
            complementarity = 0.2
        else:
            complementarity = 0.0
        
        # Calculate synergy
        synergy = (quality1 + quality2) / 2 + complementarity - 0.5
        
        return np.clip(synergy, -1.0, 1.0)
    
    def _generate_multimodal_recommendations(self, modality_analysis, cross_modal_insights):
        """Generate intelligent multimodal recommendations"""
        recommendations = {
            'primary_modality': None,
            'fusion_strategy': None,
            'preprocessing_pipeline': [],
            'model_recommendations': []
        }
        
        # Find best modality
        best_modality = max(modality_analysis.keys(), 
                          key=lambda x: modality_analysis[x].get('quality_score', 0))
        recommendations['primary_modality'] = best_modality
        
        # Fusion strategy
        if cross_modal_insights['modal synergies']:
            recommendations['fusion_strategy'] = 'early_fusion'
        elif len(modality_analysis) > 1:
            recommendations['fusion_strategy'] = 'late_fusion'
        else:
            recommendations['fusion_strategy'] = 'single_modality'
        
        # Preprocessing pipeline
        for modality, analysis in modality_analysis.items():
            recommendations['preprocessing_pipeline'].extend(
                analysis.get('recommendations', [])
            )
        
        # Model recommendations
        if 'text' in modality_analysis:
            recommendations['model_recommendations'].append('text_neural_network')
        if 'numerical' in modality_analysis:
            recommendations['model_recommendations'].append('gradient_boosting')
        if 'categorical' in modality_analysis:
            recommendations['model_recommendations'].append('random_forest')
        if 'time_series' in modality_analysis:
            recommendations['model_recommendations'].append('lstm')
        
        return recommendations
