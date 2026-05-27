"""
Model definitions and HPO search spaces.

Runtime access (get_model, recommend_models) lives in registry/model_registry.py.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Import advanced models
try:
    from .advanced_models import LightGBMWrapper, SimpleNeuralNetwork
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    LightGBMWrapper = None
    SimpleNeuralNetwork = None

# PRODUCTION-GRADE: Model-Specific Search Spaces
MODEL_SEARCH_SPACE = {
    "logistic_regression": {
        "solver": ["lbfgs", "liblinear", "saga"],
        "C": (1e-4, 10.0),
        "penalty": ["l1", "l2"],
        "max_iter": [100, 200, 500]
    },
    "neural_network": {
        "optimizer": ["adam", "sgd"],
        "learning_rate": (1e-4, 1e-1),
        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "activation": ["relu", "tanh", "logistic"],
        "max_iter": [200, 500, 1000]
    },
    "svm": {
        "kernel": ["linear", "rbf", "poly"],
        "C": (0.1, 100.0),
        "gamma": ["scale", "auto"]
    },
    "random_forest": {
        "n_estimators": (50, 300),
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10)
    },
    "extra_trees": {
        "n_estimators": (50, 300),
        "max_depth": (3, 20),
        "min_samples_leaf": (1, 10),
    },
    "elastic_net": {
        "alpha": (1e-4, 10.0),
        "l1_ratio": (0.1, 0.9),
    },
    "xgboost": {
        "n_estimators": (50, 300),
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0)
    }
}

# PRODUCTION-GRADE: Default Parameters for Fallback
try:
    from config.settings import get_config_value
    DEFAULT_PARAMS = {
        "logistic_regression": {
            "solver": get_config_value('models', 'logistic_regression_solver', "lbfgs"),
            #"C": get_config_value('models', 'logistic_regression_C', 1.0),
            "C": get_config_value('models', 'logistic_regression_C', 1.0),
            #"penalty": get_config_value('models', 'logistic_regression_penalty', "l2"),
            "penalty": get_config_value('models', 'logistic_regression_penalty', "l2"),
            #"max_iter": get_config_value('models', 'logistic_regression_max_iter', 100)
            "max_iter": get_config_value('models', 'logistic_regression_max_iter', 100)
        },
        "neural_network": {
            "optimizer": get_config_value('models', 'neural_network_optimizer', "adam"),
            #"learning_rate": get_config_value('models', 'neural_network_learning_rate', 0.001),
            "learning_rate": get_config_value('models', 'neural_network_learning_rate', 0.001),
            #"hidden_layer_sizes": get_config_value('models', 'neural_network_hidden_layers', (64,)),
            "hidden_layer_sizes": get_config_value('models', 'neural_network_hidden_layers', (64,)),
            "activation": get_config_value('models', 'neural_network_activation', "relu"),
            #"max_iter": get_config_value('models', 'neural_network_max_iter', 500)
            "max_iter": get_config_value('models', 'neural_network_max_iter', 500)
        },
        "svm": {
            "kernel": get_config_value('models', 'svm_kernel', "rbf"),
            "C": get_config_value('models', 'svm_C', 1.0),
            "gamma": get_config_value('models', 'svm_gamma', "scale")
        },
        "random_forest": {
            "n_estimators": get_config_value('models', 'random_forest_n_estimators', 100),
            "max_depth": get_config_value('models', 'random_forest_max_depth', 10),
            "min_samples_split": get_config_value('models', 'random_forest_min_samples_split', 2),
            "min_samples_leaf": get_config_value('models', 'random_forest_min_samples_leaf', 1)
        },
        "xgboost": {
            "n_estimators": get_config_value('models', 'xgboost_n_estimators', 100),
            "max_depth": get_config_value('models', 'xgboost_max_depth', 6),
            "learning_rate": get_config_value('models', 'xgboost_learning_rate', 0.1),
            "subsample": get_config_value('models', 'xgboost_subsample', 0.8)
        }
    }
except ImportError:
    # Fallback to hardcoded values if config not available
    DEFAULT_PARAMS = {
        "logistic_regression": {"solver": "lbfgs", "C": 1.0, "penalty": "l2", "max_iter": 100},
        "neural_network": {"optimizer": "adam", "learning_rate": 0.001, "hidden_layer_sizes": (64,), "activation": "relu", "max_iter": 500},
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "random_forest": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
        "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}
    }

# PRODUCTION-GRADE: Parameter Validation
def validate_params(model_name: str, params: Dict[str, Any]) -> bool:
    """Validate parameters for specific model"""
    if model_name not in MODEL_SEARCH_SPACE:
        logger.warning(f"Unknown model: {model_name}")
        return False
    
    valid_space = MODEL_SEARCH_SPACE[model_name]
    
    for param_name, param_value in params.items():
        if param_name not in valid_space:
            logger.warning(f"Invalid param {param_name} for model {model_name}")
            return False
        
        if isinstance(valid_space[param_name], list):
            if param_value not in valid_space[param_name]:
                logger.warning(f"Invalid value {param_value} for param {param_name} in model {model_name}")
                return False
    
    return True

# PRODUCTION-GRADE: Safe Parameter Correction
def safe_params(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-correct invalid parameters"""
    try:
        if validate_params(model_name, params):
            return params
        else:
            logger.warning(f"Invalid params for {model_name}, using defaults")
            return DEFAULT_PARAMS.get(model_name, {})
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        return DEFAULT_PARAMS.get(model_name, {})

# Enhanced MODEL_REGISTRY with advanced models
CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "naive_bayes": GaussianNB,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
}

REGRESSION_MODELS = {
    "random_forest_regressor": RandomForestRegressor,
    "extra_trees_regressor": ExtraTreesRegressor,
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "svr": SVR,
    "xgboost_regressor": XGBRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
}

# Classical tabular canon (used for recommendations; neural_network excluded by default)
CLASSICAL_CLASSIFICATION = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "xgboost",
    "gradient_boosting",
    "svm",
    "knn",
    "naive_bayes",
]
CLASSICAL_REGRESSION = [
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest_regressor",
    "extra_trees_regressor",
    "xgboost_regressor",
    "gradient_boosting_regressor",
    "svr",
]

# Add advanced models if available
if ADVANCED_MODELS_AVAILABLE:
    CLASSIFICATION_MODELS.update({
        "lightgbm": lambda **kwargs: LightGBMWrapper(task_type="classification", **kwargs),
        "neural_network": lambda **kwargs: SimpleNeuralNetwork(task_type="classification", **kwargs),
        "classification_lightgbm": lambda **kwargs: LightGBMWrapper(task_type="classification", **kwargs),
        "classification_neural_network": lambda **kwargs: SimpleNeuralNetwork(task_type="classification", **kwargs),
    })
    
    REGRESSION_MODELS.update({
        "lightgbm_regressor": lambda **kwargs: LightGBMWrapper(task_type="regression", **kwargs),
        "neural_network_regressor": lambda **kwargs: SimpleNeuralNetwork(task_type="regression", **kwargs),
        "regression_lightgbm_regressor": lambda **kwargs: LightGBMWrapper(task_type="regression", **kwargs),
        "regression_neural_network_regressor": lambda **kwargs: SimpleNeuralNetwork(task_type="regression", **kwargs),
    })

# Unified MODEL_REGISTRY with prefixed names for search space compatibility
MODEL_REGISTRY = {
    **CLASSIFICATION_MODELS,
    **REGRESSION_MODELS,
    # Add prefixed names for search space compatibility
    **{f"classification_{k}": v for k, v in CLASSIFICATION_MODELS.items()},
    **{f"regression_{k}": v for k, v in REGRESSION_MODELS.items()}
}

# Task type mapping
TASK_TYPES = {
    **{f"classification_{k}": "classification" for k in CLASSIFICATION_MODELS.keys()},
    **{f"regression_{k}": "regression" for k in REGRESSION_MODELS.keys()},
}

# Add advanced model task types
if ADVANCED_MODELS_AVAILABLE:
    TASK_TYPES.update({
        "classification_lightgbm": "classification",
        "classification_neural_network": "classification",
        "regression_lightgbm_regressor": "regression",
        "regression_neural_network_regressor": "regression",
    })

def get_model_class(model_name: str):
    """
    Get model class by name with proper instantiation for advanced models
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class or factory function
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models(task_type: str = "all") -> Dict[str, Any]:
    """
    Get available models for a specific task type
    
    Args:
        task_type: "classification", "regression", or "all"
        
    Returns:
        Dictionary of available models
    """
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    elif task_type == "regression":
        return REGRESSION_MODELS
    elif task_type == "all":
        return MODEL_REGISTRY
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def is_advanced_model(model_name: str) -> bool:
    """
    Check if a model is an advanced model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Whether the model is advanced
    """
    advanced_models = ["lightgbm", "neural_network", "lightgbm_regressor", "neural_network_regressor"]
    return any(adv in model_name for adv in advanced_models)


def create_model_instance(model_name: str, task_type: str = "classification", **kwargs):
    """
    Create model instance with proper handling for advanced models
    
    Args:
        model_name: Name of the model
        task_type: "classification" or "regression"
        **kwargs: Model parameters
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_name is invalid
        TypeError: If parameters are invalid
    """
    # Input validation
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")

    # Resolve task-specific model when names overlap (e.g. random_forest, xgboost)
    regression_aliases = {
        "random_forest": "random_forest_regressor",
        "xgboost": "xgboost_regressor",
        "gradient_boosting": "gradient_boosting_regressor",
        "logistic_regression": "linear_regression",
        "svm": "svr",
        "knn": "random_forest_regressor",
        "decision_tree": "random_forest_regressor",
        "naive_bayes": "linear_regression",
    }
    if task_type == "regression" and model_name in regression_aliases:
        model_name = regression_aliases[model_name]

    if model_name in REGRESSION_MODELS and task_type == "regression":
        model_class = REGRESSION_MODELS[model_name]
    elif model_name in CLASSIFICATION_MODELS and task_type == "classification":
        model_class = CLASSIFICATION_MODELS[model_name]
    else:
        model_class = get_model_class(model_name)

    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    if is_advanced_model(model_name):
        # Advanced models need special handling
        if "lightgbm" in model_name:
            resolved_task = "regression" if task_type == "regression" or "regression" in model_name else "classification"
            return LightGBMWrapper(task_type=resolved_task, **kwargs)
        elif "neural_network" in model_name:
            resolved_task = "regression" if task_type == "regression" or "regressor" in model_name else "classification"
            return SimpleNeuralNetwork(task_type=resolved_task, **kwargs)
    else:
        # Standard sklearn models
        return model_class(**kwargs)


SKLEARN_DL_MODELS = {
    "neural_network",
    "neural_network_regressor",
    "classification_neural_network",
    "regression_neural_network_regressor",
}

KERAS_DL_MODELS = {"dnn", "lstm", "cnn"}


def is_sklearn_dl_model(model_name: str) -> bool:
    return any(token in model_name for token in SKLEARN_DL_MODELS)


def filter_models_by_family(model_names: List[str], model_family: str = "ml") -> List[str]:
    if model_family == "dl":
        dl_models = [m for m in model_names if is_sklearn_dl_model(m)]
        return dl_models or ["neural_network", "neural_network_regressor"]
    if model_family == "ml":
        return [m for m in model_names if not is_sklearn_dl_model(m)]
    return model_names