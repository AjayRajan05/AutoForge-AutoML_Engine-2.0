"""
📥 Input/Output Types - Standardized data structures for AutoForge
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class AutoMLInput:
    """
    Standardized input format for AutoForge AutoML
    
    Supports both production and testing inputs with comprehensive metadata
    """
    data: pd.DataFrame
    target_column: str
    task_type: Optional[str] = None
    data_type: Optional[str] = None
    model_family: str = "ml"
    problem_description: Optional[str] = None
    user_preference: str = "auto"
    search_depth: str = "balanced"
    scoring: Optional[str] = None
    ensemble_method: str = "none"
    ensemble_top_n: int = 3
    enable_refinement: bool = False
    max_time: Optional[float] = None
    max_trials: Optional[int] = None
    features_to_exclude: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    validation_split: float = 0.2
    random_state: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate input after initialization"""
        self._validate_basic()
    
    def _validate_basic(self):
        """Basic validation of input data"""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        if self.data.empty:
            raise ValueError("Data cannot be empty")
        
        if len(self.data) < 10:
            raise ValueError("Dataset must have at least 10 samples")
        
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
    
    def get_features(self) -> pd.DataFrame:
        """Get feature data (exclude target and excluded columns)"""
        exclude_cols = [self.target_column] + self.features_to_exclude
        return self.data.drop(columns=exclude_cols, errors='ignore')
    
    def get_target(self) -> pd.Series:
        """Get target data"""
        return self.data[self.target_column]
    
    def get_shape(self) -> tuple:
        """Get dataset shape"""
        return self.data.shape
    
    def get_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "shape": self.data.shape,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "data_type": self.data_type,
            "user_preference": self.user_preference,
            "max_time": self.max_time,
            "max_trials": self.max_trials,
            "features_to_exclude": self.features_to_exclude,
            "categorical_features": self.categorical_features,
            "validation_split": self.validation_split,
            "random_state": self.random_state
        }


@dataclass
class AutoMLOutput:
    """
    Standardized output format for AutoForge AutoML
    
    Contains all results from AutoML execution
    """
    best_model: Any
    best_score: float
    metrics: Dict[str, float]
    selected_features: List[str]
    pipeline_summary: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    explainability_report: Optional[Dict[str, Any]]
    training_time: float
    logs: List[str]
    model_path: Optional[str]
    predictions: Optional[np.ndarray] = None
    predictions_proba: Optional[np.ndarray] = None
    validation_scores: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    data_profile: Optional[Dict[str, Any]] = None
    strategy_used: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        return {
            "best_score": self.best_score,
            "training_time": self.training_time,
            "n_features_used": len(self.selected_features),
            "model_type": type(self.best_model).__name__ if self.best_model else "Unknown",
            "task_type": self.metadata.get("task_type", "unknown"),
            "data_type": self.metadata.get("data_type", "unknown"),
            "strategy": self.strategy_used.get("primary_strategy", "unknown") if self.strategy_used else "unknown",
            "has_predictions": self.predictions is not None,
            "has_explainability": self.explainability_report is not None,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings)
        }
    
    def get_detailed_report(self) -> str:
        """Get detailed human-readable report"""
        lines = []
        
        # Header
        lines.append("🚀 AutoForge AutoML Execution Report")
        lines.append("=" * 50)
        
        # Performance
        lines.append(f"📊 Best Score: {self.best_score:.4f}")
        lines.append(f"⏱️  Training Time: {self.training_time:.2f} seconds")
        lines.append(f"🔧 Features Used: {len(self.selected_features)}")
        
        # Model info
        if self.best_model:
            lines.append(f"🤖 Model Type: {type(self.best_model).__name__}")
        
        # Strategy
        if self.strategy_used:
            strategy = self.strategy_used.get("primary_strategy", "unknown")
            confidence = self.strategy_used.get("metadata", {}).get("confidence", 0)
            lines.append(f"🎯 Strategy: {strategy} (confidence: {confidence:.1%})")
        
        # Metrics
        if self.metrics:
            lines.append("\n📈 Performance Metrics:")
            for metric, value in self.metrics.items():
                lines.append(f"  • {metric}: {value:.4f}")
        
        # Feature importance
        if self.feature_importance:
            lines.append("\n🔍 Top Feature Importance:")
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                lines.append(f"  • {feature}: {importance:.4f}")
        
        # Warnings and errors
        if self.warnings:
            lines.append(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:3]:  # Show first 3
                lines.append(f"  • {warning}")
        
        if self.errors:
            lines.append(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors[:3]:  # Show first 3
                lines.append(f"  • {error}")
        
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    corrected_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_test_input() -> AutoMLInput:
    """
    Generate test input for development and testing
    
    Returns:
        Sample AutoMLInput with synthetic data
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features
    data = {}
    for i in range(n_features):
        if i < 5:  # Numeric features
            data[f"numeric_{i}"] = np.random.randn(n_samples)
        else:  # Categorical features
            data[f"cat_{i}"] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # Create target (classification)
    data['target'] = np.random.choice([0, 1], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'numeric_0'] = np.nan
    
    return AutoMLInput(
        data=df,
        target_column='target',
        task_type='classification',
        data_type='tabular',
        problem_description='Test dataset for AutoForge development',
        user_preference='auto',
        max_time=300,
        max_trials=50,
        categorical_features=[f'cat_{i}' for i in range(5, 10)],
        validation_split=0.2,
        random_state=42,
        metadata={'source': 'synthetic', 'purpose': 'testing'}
    )


def get_small_test_input() -> AutoMLInput:
    """Generate small test input for quick testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    return AutoMLInput(
        data=df,
        target_column='target',
        task_type='classification',
        user_preference='fast',
        max_time=60,
        max_trials=10
    )


def get_large_test_input() -> AutoMLInput:
    """Generate large test input for performance testing"""
    np.random.seed(42)
    n_samples = 50000
    n_features = 50
    
    data = {}
    for i in range(n_features):
        if i < 30:  # Numeric features
            data[f"numeric_{i}"] = np.random.randn(n_samples)
        else:  # Categorical features
            data[f"cat_{i}"] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    
    # Create target (regression for large dataset)
    data['target'] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    
    return AutoMLInput(
        data=df,
        target_column='target',
        task_type='regression',
        data_type='tabular',
        problem_description='Large test dataset for performance testing',
        user_preference='fast',
        max_time=600,
        max_trials=30,
        categorical_features=[f'cat_{i}' for i in range(30, 50)],
        validation_split=0.1
    )


def get_text_test_input() -> AutoMLInput:
    """Generate test input with text data"""
    np.random.seed(42)
    n_samples = 500
    
    # Text features
    texts = [
        "This is a positive review",
        "Great product, highly recommend",
        "Not worth the money",
        "Average quality, nothing special",
        "Excellent service and fast delivery"
    ]
    
    data = {
        'review_text': np.random.choice(texts, n_samples),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'price_category': np.random.choice(['low', 'medium', 'high'], n_samples),
        'sentiment': np.random.choice([0, 1], n_samples)  # 0: negative, 1: positive
    }
    
    df = pd.DataFrame(data)
    
    return AutoMLInput(
        data=df,
        target_column='sentiment',
        task_type='classification',
        data_type='text',
        problem_description='Text sentiment analysis dataset',
        user_preference='accurate',
        max_time=300,
        max_trials=40,
        categorical_features=['price_category']
    )


def get_time_series_test_input() -> AutoMLInput:
    """Generate test input with time series data"""
    np.random.seed(42)
    n_samples = 200
    
    # Create time series with trend and seasonality
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    trend = np.linspace(100, 200, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.randn(n_samples) * 5
    
    data = {
        'date': dates,
        'value': trend + seasonal + noise,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': dates.dayofweek >= 5
    }
    
    df = pd.DataFrame(data)
    
    return AutoMLInput(
        data=df,
        target_column='value',
        task_type='regression',
        data_type='time_series',
        problem_description='Time series forecasting dataset',
        user_preference='accurate',
        max_time=300,
        max_trials=30,
        categorical_features=['day_of_week', 'month', 'is_weekend']
    )
