"""
Pipeline Registry
Registry for ML pipeline components and builders
"""

import logging
from typing import Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# Pipeline registry
PIPELINE_REGISTRY = {
    'standard': {
        'imputer': SimpleImputer(strategy='mean'),
        'scaler': StandardScaler()
    },
    'robust': {
        'imputer': SimpleImputer(strategy='median'),
        'scaler': RobustScaler()
    },
    'minmax': {
        'imputer': SimpleImputer(strategy='mean'),
        'scaler': MinMaxScaler()
    }
}

def build_pipeline(params: Dict[str, Any], model: Any) -> Pipeline:
    """
    Build ML pipeline from parameters and model
    
    Args:
        params: Pipeline configuration parameters
        model: ML model to include in pipeline
        
    Returns:
        sklearn Pipeline instance
    """
    try:
        steps = []
        
        # Add imputation step
        if params.get('imputer', True):
            strategy = params.get('imputation_strategy', 'mean')
            steps.append(('imputer', SimpleImputer(strategy=strategy)))
        
        # Add scaling step
        if params.get('scaler', True):
            scaler_type = params.get('scaler_type', 'standard')
            if scaler_type == 'standard':
                steps.append(('scaler', StandardScaler()))
            elif scaler_type == 'robust':
                steps.append(('scaler', RobustScaler()))
            elif scaler_type == 'minmax':
                steps.append(('scaler', MinMaxScaler()))
        
        # Add model step
        steps.append(('model', model))
        
        pipeline = Pipeline(steps)
        logger.info(f"✅ Built pipeline with {len(steps)} steps")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to build pipeline: {e}")
        # Return minimal pipeline
        return Pipeline([('model', model)])

def get_pipeline_templates() -> Dict[str, Dict[str, Any]]:
    """
    Get available pipeline templates
    
    Returns:
        Dictionary of pipeline templates
    """
    return PIPELINE_REGISTRY.copy()

def register_pipeline_template(name: str, template: Dict[str, Any]) -> None:
    """
    Register a new pipeline template
    
    Args:
        name: Template name
        template: Template configuration
    """
    PIPELINE_REGISTRY[name] = template
    logger.info(f"📝 Registered pipeline template: {name}")
