"""
🚀 AutoForge Unified AutoML - Core System
Intelligent automated machine learning with comprehensive capabilities
"""

import logging
import time
import warnings
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.dtype_helpers import categorical_columns, numeric_columns

# Ensure project root is on path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Intelligence components with fallback handling
try:
    from intelligence.intelligence_engine import IntelligenceEngine, DatasetProfile
except ImportError:
    IntelligenceEngine = None
    DatasetProfile = None

# Input/Output components with fallback handling
try:
    from input_output.input_types import AutoMLInput, AutoMLOutput
except ImportError:
    AutoMLInput = None
    AutoMLOutput = None

try:
    from input_output.input_validator import InputValidator, ValidationResult
except ImportError:
    InputValidator = None
    ValidationResult = None

# Persistence components with fallback handling
try:
    from persistence.model_saver import ModelSaver
except ImportError:
    ModelSaver = None

# Core components
from .bulletproof_error_handler import BulletproofErrorHandler, bulletproof_method
# Feature components with fallback handling
try:
    from features.feature_selector import FeatureSelector
except ImportError:
    FeatureSelector = None

try:
    from features.explainability.actionable_explainability import ActionableExplainability
except ImportError:
    ActionableExplainability = None

try:
    from features.explainability.decision_explainer import DecisionExplainer
except ImportError:
    DecisionExplainer = None

# Execution components with fallback handling
try:
    from execution.execution_engine import ExecutionEngine
except ImportError:
    ExecutionEngine = None

try:
    from execution.pipeline_builder import PipelineBuilder
except ImportError:
    PipelineBuilder = None

try:
    from execution.optimizer import Optimizer
except ImportError:
    Optimizer = None

try:
    from execution.evaluation import ModelEvaluator
except ImportError:
    ModelEvaluator = None
# Registry components with fallback handling
try:
    from registry.feature_registry import feature_registry
except ImportError:
    feature_registry = None

try:
    from registry.model_registry import model_registry
except ImportError:
    model_registry = None

# Utility components with fallback handling
try:
    from utils.helpers import timer_decorator, ProgressTracker
except ImportError:
    timer_decorator = None
    ProgressTracker = None

try:
    from utils.performance_optimizer import performance_optimizer, performance_monitor
except ImportError:
    performance_optimizer = None
    performance_monitor = None

try:
    from utils.monitoring import autoforge_monitor
except ImportError:
    autoforge_monitor = None

# Integration modules with fallback handling
try:
    from api.api_integration import api_integrator, integrate_with_autoforge
except ImportError:
    api_integrator = None
    integrate_with_autoforge = None

try:
    from benchmarking.benchmark_integration import benchmarking_integrator, benchmark_autoforge
except ImportError:
    benchmarking_integrator = None
    benchmark_autoforge = None

try:
    from systemization.systemization_integration import systemization_integrator, setup_autoforge_monitoring, run_autoforge_ab_test
except ImportError:
    systemization_integrator = None
    setup_autoforge_monitoring = None
    run_autoforge_ab_test = None

try:
    from tracking.tracking_integration import tracking_integrator, start_autoforge_experiment, log_autoforge_training, end_autoforge_experiment
except ImportError:
    tracking_integrator = None
    start_autoforge_experiment = None
    log_autoforge_training = None
    end_autoforge_experiment = None

try:
    from optimizer.optimizer_integration import optimizer_integrator, optimize_autoforge_model
except ImportError:
    optimizer_integrator = None
    optimize_autoforge_model = None

try:
    from processors.processor_integration import processor_integrator
except ImportError:
    processor_integrator = None

try:
    from ensemble.ensemble_integration import ensemble_integrator
except ImportError:
    ensemble_integrator = None

logger = logging.getLogger(__name__)


class UnifiedAutoML:
    """
    AutoForge Unified AutoML System
    
    The core AutoML system that combines intelligence, execution, and persistence
    to provide comprehensive automated machine learning capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AutoForge AutoML
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        logger.info("🚀 Initializing AutoForge Unified AutoML...")
        
        # Initialize components with fallback handling
        self.intelligence_engine = IntelligenceEngine() if IntelligenceEngine is not None else None
        self.input_validator = InputValidator() if InputValidator is not None else None
        self.model_saver = ModelSaver(self.config.get('model_save_path', './models')) if ModelSaver is not None else None
        self.error_handler = BulletproofErrorHandler()
        self.feature_selector = FeatureSelector() if FeatureSelector is not None else None
        self.execution_engine = ExecutionEngine() if ExecutionEngine is not None else None
        self.pipeline_builder = PipelineBuilder() if PipelineBuilder is not None else None
        self.optimizer = Optimizer() if Optimizer is not None else None
        self.evaluator = ModelEvaluator() if ModelEvaluator is not None else None
        self.feature_registry = feature_registry
        self.model_registry = model_registry
        
        # Integration components
        self.api_integrator = api_integrator
        self.benchmarking_integrator = benchmarking_integrator
        self.systemization_integrator = systemization_integrator
        self.tracking_integrator = tracking_integrator
        self.optimizer_integrator = optimizer_integrator
        self.processor_integrator = processor_integrator
        self.ensemble_integrator = ensemble_integrator
        
        # Explainability components
        self.actionable_explainer = ActionableExplainability() if ActionableExplainability is not None else None
        self.decision_explainer = DecisionExplainer() if DecisionExplainer is not None else None
        
        # State tracking
        self.is_fitted = False
        self.current_profile = None
        self.execution_history = []
        self.best_model = None
        self.best_score = None
        self.training_metadata = {}
        self.task_type_: Optional[str] = None
        self.data_type_: Optional[str] = None
        
        logger.info("✅ AutoForge Unified AutoML initialized")
    
    def fit(self, automl_input: Union[AutoMLInput, pd.DataFrame], 
            target_column: Optional[str] = None,
            enable_tracking: bool = True,
            enable_monitoring: bool = True,
            enable_optimization: bool = True) -> 'UnifiedAutoML':
        """
        Fit AutoML to training data
        
        Args:
            automl_input: Training data (AutoMLInput or DataFrame)
            target_column: Target column name (if automl_input is DataFrame)
            enable_tracking: Enable experiment tracking
            enable_monitoring: Enable performance monitoring
            enable_optimization: Enable hyperparameter optimization
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
        """
        # Input validation
        if automl_input is None:
            raise ValueError("automl_input cannot be None")
        
        if isinstance(automl_input, pd.DataFrame):
            if automl_input.empty:
                raise ValueError("automl_input DataFrame cannot be empty")
            if len(automl_input.shape) < 2:
                raise ValueError("automl_input DataFrame must have at least 2 columns")
        elif not self._looks_like_automl_input(automl_input):
            raise TypeError("automl_input must be AutoMLInput or pandas DataFrame")
        
        logger.info("🚀 Starting AutoForge AutoML training...")
        start_time = time.time()
        execution_id = f"autoforge_{int(time.time())}"

        self._ensure_core_components()
        
        # Start monitoring
        if enable_monitoring and autoforge_monitor:
            autoforge_monitor.log_system_health()
        
        # Start experiment tracking
        experiment_id = None
        if enable_tracking and start_autoforge_experiment:
            experiment_id = start_autoforge_experiment(
                self, 
                f"autoforge_training_{int(time.time())}",
                "AutoForge unified training",
                tags=['autoforge', 'automl']
            )
        
        try:
            # Step 1: Input preparation and validation
            logger.info("📋 Preparing and validating input...")
            automl_input = self._prepare_input(automl_input, target_column)
            self._current_user_preference = getattr(automl_input, 'user_preference', 'auto')
            
            validation_result = self.input_validator.validate(automl_input)
            if not validation_result.is_valid:
                if validation_result.corrected_data is not None:
                    logger.warning("⚠️  Input validation failed, using corrected data")
                    automl_input.data = validation_result.corrected_data
                else:
                    raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            # Step 2: Intelligence analysis
            logger.info("🧠 Performing intelligence analysis...")
            X = automl_input.get_features()
            y = automl_input.get_target()
            self._raw_feature_columns_ = list(X.columns)

            # Step 2b: Data processing via modality-specific processors
            detected_data_type = getattr(automl_input, 'data_type', None)
            preprocessing_artifacts = None
            if self.processor_integrator and self.config.get('use_processors', True):
                logger.info("🔧 Running data processors...")
                if not detected_data_type:
                    detected_data_type = self.processor_integrator.detect_data_type(X, y)
                data_type = detected_data_type
                proc_config = {
                    'task_type': getattr(automl_input, 'task_type', None),
                    'scale_features': self.config.get('scale_features', True),
                    'return_artifacts': True,
                }
                proc_result = self.processor_integrator.process_data(
                    X, y, data_type, proc_config
                )
                if len(proc_result) == 3:
                    X, y, preprocessing_artifacts = proc_result
                else:
                    X, y = proc_result
                automl_input.data = pd.concat([X, y.rename(automl_input.target_column)], axis=1)
            else:
                data_type = detected_data_type or 'tabular'
            self.data_type_ = data_type
            self.preprocessing_artifacts_ = preprocessing_artifacts
            automl_input.data_type = data_type
            
            self.current_profile = self.intelligence_engine.analyze(
                X, y,
                user_preference=automl_input.user_preference,
                max_time=automl_input.max_time,
                max_trials=automl_input.max_trials
            )
            
            # Step 3: Holdout split before feature selection (prevents target leakage)
            logger.info("📂 Creating train/holdout split...")
            X_features = automl_input.get_features()
            y_full = automl_input.get_target()
            if hasattr(y_full, 'loc') and hasattr(X_features, 'index'):
                y_full = y_full.loc[X_features.index]

            task_type = self._resolve_task_type(automl_input, self.current_profile)
            self.task_type_ = task_type

            stratify = None
            if task_type == 'classification' and y_full.nunique() > 1:
                stratify = y_full
            X_train_raw, X_holdout, y_train, y_holdout = train_test_split(
                X_features,
                y_full,
                test_size=0.2,
                random_state=automl_input.random_state or self.config.get('random_state', 42),
                stratify=stratify,
            )
            self._holdout_split_ = {
                'X_holdout': X_holdout,
                'y_holdout': y_holdout,
                'train_size': len(X_train_raw),
                'holdout_size': len(X_holdout),
            }

            # Step 4: Feature selection on train split only
            logger.info("🔧 Performing feature selection (train split only)...")
            feature_selection_config = self.current_profile.recommendations.copy()
            n_cols = X_train_raw.shape[1]
            feature_selection_config.update({
                'max_features': self.config.get('max_features') or n_cols,
                'min_features': self.config.get('min_features', max(1, n_cols // 10)),
                'selection_pressure': self.config.get('selection_pressure', 'medium'),
            })

            feature_results = self.feature_selector.select_features(
                X_train_raw,
                y_train,
                self.current_profile.__dict__,
                feature_selection_config,
            )
            selected_features = feature_results['selected_features']
            feature_results['feature_selection_on_train_only'] = True

            X_train = X_train_raw[selected_features]
            X_holdout = X_holdout[selected_features]
            X_selected = X_train

            # Step 5: Model training execution (CV on train split only)
            logger.info("⚡ Executing model training...")
            if self.current_profile and hasattr(self.current_profile, 'complexity_profile'):
                if self.current_profile.complexity_profile is None:
                    self.current_profile.complexity_profile = {}
                self.current_profile.complexity_profile['task_type'] = task_type

            model_family = getattr(automl_input, 'model_family', None) or self.config.get('model_family', 'ml')
            if model_family not in ('ml', 'dl', 'both'):
                model_family = 'ml'

            if automl_input.task_type in ('regression', 'classification'):
                models = model_registry.recommend_models(
                    len(X_selected), X_selected.shape[1], task_type, model_family=model_family
                )
            else:
                models = self.current_profile.recommendations.get(
                    'models',
                    model_registry.recommend_models(
                        len(X_selected), X_selected.shape[1], task_type, model_family=model_family
                    )
                )

            advanced_features = self._build_advanced_features_config(
                X_selected, automl_input, model_family
            )

            search_depth = getattr(automl_input, 'search_depth', None) or self.config.get(
                'search_depth', 'balanced'
            )
            if search_depth not in ('fast', 'balanced', 'deep'):
                search_depth = 'balanced'

            scoring = (
                getattr(automl_input, 'scoring', None)
                or self.config.get('scoring')
            )
            if scoring is None:
                from execution.preprocessing_search import default_scoring
                scoring = default_scoring(task_type)

            from core.adaptive_resource_manager import adaptive_resource_manager
            search_budget = adaptive_resource_manager.get_search_budget(
                search_depth,
                max_trials=automl_input.max_trials,
                max_time=automl_input.max_time,
            )

            preprocessing_recipe = None
            preprocessing_search_results = None
            prep_search_meta: Dict[str, Any] = {}

            run_prep_search = (
                search_depth in ('balanced', 'deep')
                and (self.data_type_ or 'tabular') == 'tabular'
                and preprocessing_artifacts is None
            )
            if run_prep_search:
                logger.info("🔍 Preprocessing recipe search...")
                from execution.preprocessing_search import PreprocessingSearchSpace
                prep_searcher = PreprocessingSearchSpace(
                    random_state=automl_input.random_state or self.config.get('random_state', 42)
                )
                prep_result = prep_searcher.search(
                    X_train,
                    y_train,
                    task_type=task_type,
                    cv_folds=min(3, search_budget.get('cv_folds', 3)),
                    scoring=scoring,
                )
                preprocessing_recipe = prep_result['preprocessing_recipe']
                preprocessing_search_results = prep_result['preprocessing_search_results']
                preprocessing_artifacts = prep_result['preprocessing_artifacts']
                self.preprocessing_artifacts_ = preprocessing_artifacts
                prep_search_meta = {
                    'preprocessing_recipe': preprocessing_recipe,
                    'preprocessing_recipe_config': prep_result.get('preprocessing_recipe_config'),
                    'preprocessing_search_results': preprocessing_search_results,
                    'preprocessing_cv_mean': prep_result.get('preprocessing_cv_mean'),
                    'preprocessing_cv_std': prep_result.get('preprocessing_cv_std'),
                    'preprocessing_scoring': prep_result.get('preprocessing_scoring'),
                }
                X_train = prep_result['X_processed']
                y_train = prep_result['y_processed']
                if X_holdout is not None:
                    from execution.preprocessing_pipeline import PreprocessingPipeline
                    holdout_pipe = PreprocessingPipeline()
                    holdout_pipe.artifacts = preprocessing_artifacts
                    holdout_pipe.is_fitted = True
                    holdout_pipe.column_transformer = preprocessing_artifacts.get('column_transformer')
                    X_holdout = holdout_pipe.transform(X_holdout)
                self._prep_search_meta_ = prep_search_meta

            execution_config = {
                'task_type': task_type,
                'model_family': model_family,
                'models': models,
                'advanced_features': advanced_features,
                'cv_folds': self.current_profile.recommendations.get('validation', {}).get('cv_folds', 5),
                'max_trials': automl_input.max_trials or 50,
                'random_state': automl_input.random_state,
                'n_jobs': self.config.get('n_jobs', -1),
                'scale_features': self.config.get('scale_features', True),
                'dl_epochs': self.config.get('dl_epochs', 10),
                'dl_max_trials': self.config.get('dl_max_trials', 5),
                'search_depth': search_depth,
                'search_budget': search_budget,
                'scoring': scoring,
                'early_stop_models': search_depth in ('balanced', 'deep'),
                'evaluate_preprocessing_variants': False,
                'skip_preprocessing': preprocessing_artifacts is not None or run_prep_search,
                'preprocessing_artifacts': preprocessing_artifacts,
                'preprocessing_recipe': preprocessing_recipe,
                'preprocessing_recipe_config': prep_search_meta.get('preprocessing_recipe_config'),
                'X_holdout': X_holdout,
                'y_holdout': y_holdout,
                'holdout_fraction': 0.2,
            }
            self.search_depth_ = search_depth
            
            # Optimize execution configuration for performance
            if performance_optimizer:
                execution_config = performance_optimizer.optimize_execution_speed(execution_config)
            
            training_results = self.execution_engine.run(
                X_train, y_train, execution_config
            )

            # Step 4b: Ensemble top models when beneficial
            if self.ensemble_integrator and self.config.get('enable_ensemble', True):
                training_results = self._maybe_build_ensemble(
                    training_results, execution_config['task_type']
                )

            # Step 4c: Holdout evaluation via ModelEvaluator (post-selection)
            if self.evaluator and X_holdout is not None:
                eval_report = self.evaluator.evaluate_model(
                    training_results['best_model'],
                    X_holdout,
                    y_holdout,
                    execution_config['task_type'],
                )
                training_results['evaluation_results'] = {
                    **training_results.get('evaluation_results', {}),
                    **eval_report,
                }
                training_results['holdout_metrics'] = {
                    k: v for k, v in eval_report.items()
                    if isinstance(v, (int, float)) and not k.startswith('_')
                }
            
            # Step 5: Hyperparameter optimization (refinement only when needed)
            run_refinement = enable_optimization and (
                search_depth == 'fast'
                or getattr(automl_input, 'enable_refinement', False)
            )
            if (
                run_refinement
                and training_results['best_model'] is not None
                and self.optimizer_integrator
            ):
                logger.info("⚡ Performing hyperparameter optimization...")
                
                search_space = self.optimizer_integrator.create_search_space(
                    type(training_results['best_model']).__name__,
                    execution_config['task_type']
                )
                
                optimization_config = {
                    'max_trials': min(20, automl_input.max_trials or 20),
                    'timeout': min(60, automl_input.max_time or 60)
                }

                X_opt = training_results.get('X_train', X_selected)
                y_opt = training_results.get('y_train', automl_input.get_target())
                
                optimization_results = optimize_autoforge_model(
                    training_results['best_model'],
                    X_opt, 
                    y_opt,
                    'adaptive',
                    **optimization_config
                )
                
                if optimization_results.get('best_score', 0) > training_results['best_score']:
                    training_results['best_score'] = optimization_results['best_score']
                    training_results['optimization_results'] = optimization_results
                    if optimization_results.get('best_model') is not None:
                        training_results['best_model'] = optimization_results['best_model']
                    logger.info(f"✅ Optimization improved score to {optimization_results['best_score']:.4f}")
            
            # Step 6: Post-processing and learning
            logger.info("📚 Post-processing and learning...")
            self._post_process_training(training_results, feature_results, start_time)
            
            self.is_fitted = True

            if self.config.get('auto_report'):
                from pathlib import Path as _Path
                from core.training_report import save_report_bundle
                report_dir = self.config.get('report_dir') or (
                    f"./runs/autoforge_{int(time.time())}"
                )
                save_report_bundle(_Path(report_dir), self)
                self.training_metadata['report_dir'] = str(report_dir)
                logger.info("📄 Training report saved to %s", report_dir)

            # Step 7: Save model if configured (after is_fitted is set)
            if self.config.get('auto_save_model', False):
                model_name = self.config.get('model_name') or f"autoforge_model_{int(time.time())}"
                self.save_model(model_name)
            
            # Step 8: Log training completion
            if enable_tracking and log_autoforge_training and end_autoforge_experiment:
                log_autoforge_training(self, X, y)
                end_autoforge_experiment('completed', {'best_score': self.best_score})
            
            # Step 9: Log execution monitoring
            if enable_monitoring and autoforge_monitor:
                end_time = time.time()
                dataset_info = {
                    'n_samples': len(X),
                    'n_features': len(X.columns),
                    'task_type': execution_config.get('task_type', 'unknown')
                }
                autoforge_monitor.log_execution(
                    execution_id, dataset_info, execution_config, 
                    start_time, end_time, True
                )
                
                # Log model performance
                if training_results.get('best_model') and training_results.get('best_score'):
                    autoforge_monitor.log_model_performance(
                        execution_id, 
                        type(training_results['best_model']).__name__,
                        training_results['best_score'],
                        training_results.get('training_time', 0)
                    )
            
            if self.config.get('verbose', True):
                self.print_model_comparison()

            self._register_training_with_monitor(X, y, execution_config, training_results)
            
            logger.info(f"✅ AutoML training completed successfully! "
                       f"Best score: {self.best_score:.4f}, Time: {training_results.get('training_time', 0):.2f}s")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ AutoML training failed: {e}")
            if enable_tracking and end_autoforge_experiment:
                end_autoforge_experiment('failed', {'error': str(e)})
            raise
    
    @bulletproof_method(max_retries=3)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model
        
        Args:
            X: Feature data
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            logger.info("🔮 Making predictions...")
            
            # Convert to DataFrame if needed
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            self._validate_predict_input(X)
            
            # Apply same preprocessing as training
            X_processed = self._apply_preprocessing(X)
            
            # Make predictions
            if hasattr(self.best_model, 'backend') and self.best_model.backend == 'keras':
                predictions = self.best_model.predict(X_processed)
            elif hasattr(self.best_model, 'predict'):
                predictions = self.best_model.predict(X_processed)
            else:
                predictions = self.best_model.predict(X_processed)
            
            logger.info(f"✅ Predictions completed for {len(X)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    @bulletproof_method(max_retries=3)
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions (classification only)
        
        Args:
            X: Feature data
            
        Returns:
            Probability predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            logger.info("🔮 Making probability predictions...")
            
            # Convert to DataFrame if needed
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # Apply same preprocessing as training
            X_processed = self._apply_preprocessing(X)
            
            # Check if model supports predict_proba
            if not hasattr(self.best_model, 'predict_proba'):
                raise ValueError("Model does not support probability predictions")
            
            # Make probability predictions
            predictions_proba = self.best_model.predict_proba(X_processed)
            
            logger.info(f"✅ Probability predictions completed for {len(X)} samples")
            return predictions_proba
            
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            raise
    
    @bulletproof_method(max_retries=3)
    def explain(self, X: Union[pd.DataFrame, np.ndarray], 
                 explanation_type: str = 'both') -> Dict[str, Any]:
        """
        Generate explanations for model predictions
        
        Args:
            X: Feature data to explain
            explanation_type: 'actionable', 'decision', or 'both'
            
        Returns:
            Explanation dictionary
        """
        if not self.is_fitted:
            return {"error": "Model must be fitted before generating explanations"}
        
        try:
            logger.info(f"💡 Generating {explanation_type} explanations...")
            
            explanations = {}
            
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                feature_names = self.training_metadata.get('feature_names', 
                    [f'feature_{i}' for i in range(X.shape[1])])
                X = pd.DataFrame(X, columns=feature_names)
            
            # Get predictions for explanations
            predictions = self.predict(X)
            
            # Generate actionable explanations
            if explanation_type in ['actionable', 'both']:
                try:
                    logger.info("💡 Generating actionable explanations...")
                    
                    actionable_explanations = self.actionable_explainer.generate_explanations(
                        model=self.best_model,
                        X=X,
                        predictions=predictions,
                        feature_names=X.columns.tolist()
                    )
                    
                    explanations['actionable'] = actionable_explanations
                    
                except Exception as e:
                    logger.warning(f"⚠️ Actionable explanations failed: {e}")
                    explanations['actionable'] = {"error": str(e)}
            
            # Generate decision explanations
            if explanation_type in ['decision', 'both']:
                try:
                    logger.info("💡 Generating decision explanations...")
                    
                    decision_explanations = self.decision_explainer.explain_decisions(
                        model=self.best_model,
                        X=X,
                        predictions=predictions,
                        feature_names=X.columns.tolist()
                    )
                    
                    explanations['decision'] = decision_explanations
                    
                except Exception as e:
                    logger.warning(f"⚠️ Decision explanations failed: {e}")
                    explanations['decision'] = {"error": str(e)}
            
            # Add metadata
            explanations['metadata'] = {
                'explanation_type': explanation_type,
                'model_type': type(self.best_model).__name__,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info("✅ Explanations generated successfully")
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate explanations: {e}")
            return {"error": f"Explanation generation failed: {str(e)}"}
    
    def save_model(self, model_name: str, overwrite: bool = False) -> str:
        """
        Save the trained model
        
        Args:
            model_name: Name for the saved model
            overwrite: Whether to overwrite existing model
            
        Returns:
            Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            logger.info(f"💾 Saving model: {model_name}")
            
            # Prepare metadata
            metadata = {
                'autoforge_version': '1.0.0',
                'training_time': self.training_metadata.get('training_time', 0),
                'best_score': self.best_score,
                'model_type': type(self.best_model).__name__,
                'profile': self.current_profile.__dict__ if self.current_profile else None,
                'config': self.config,
                'training_metadata': self.training_metadata
            }
            
            # Save pipeline
            prep_data = (
                self.training_metadata.get('preprocessing_artifacts')
                or self.training_metadata.get('preprocessing_steps', {})
            )
            from core.training_report import save_report_bundle

            model_path = self.model_saver.save_pipeline(
                model=self.best_model,
                preprocessing_steps=prep_data,
                feature_info=self.training_metadata.get('feature_info', {}),
                training_info=self.training_metadata,
                pipeline_name=model_name,
                report_bundle=True,
            )
            report_paths = save_report_bundle(Path(model_path), self)
            self.training_metadata['report_paths'] = report_paths
            
            logger.info(f"✅ Model saved successfully: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def load_model(self, model_name: str) -> 'UnifiedAutoML':
        """
        Load a saved model
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info(f"📂 Loading model: {model_name}")
            
            # Load pipeline
            pipeline_components = self.model_saver.load_pipeline(model_name)
            
            # Restore model and metadata
            self.best_model = pipeline_components['model']
            self.training_metadata = pipeline_components.get('training', {})
            self.task_type_ = self.training_metadata.get('task_type')
            self.data_type_ = self.training_metadata.get('data_type')
            self.best_score = self.training_metadata.get('best_score')
            preprocessing = pipeline_components.get('preprocessing', {})
            if preprocessing:
                self.training_metadata['preprocessing_artifacts'] = preprocessing
                steps = dict(self.training_metadata.get('preprocessing_steps', {}))
                for key, value in preprocessing.items():
                    if key not in steps or isinstance(steps.get(key), str):
                        steps[key] = value
                self.training_metadata['preprocessing_steps'] = steps
                self.preprocessing_artifacts_ = preprocessing
            
            # Restore profile if available
            if 'profile' in pipeline_components.get('training', {}):
                from ..intelligence.intelligence_engine import DatasetProfile
                profile_data = pipeline_components['training']['profile']
                self.current_profile = DatasetProfile(**profile_data)
            
            self.is_fitted = True

            ensemble_run_dir = self.training_metadata.get('ensemble_run_dir')
            if ensemble_run_dir and self.ensemble_integrator:
                try:
                    disk_ensemble = self.ensemble_integrator.load_disk_ensemble(
                        ensemble_run_dir
                    )
                    if disk_ensemble is not None:
                        self.best_model = disk_ensemble
                except Exception as exc:
                    logger.warning("Disk ensemble load skipped: %s", exc)
            
            logger.info(f"✅ Model loaded successfully: {model_name}")
            return self
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_profile(self) -> Optional[DatasetProfile]:
        """Get the current dataset profile"""
        return self.current_profile
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model"""
        if not self.is_fitted:
            return None
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                feature_names = self.training_metadata.get('feature_names', [f'feature_{i}' for i in range(len(self.best_model.feature_importances_))])
                return dict(zip(feature_names, self.best_model.feature_importances_))
            elif hasattr(self.best_model, 'coef_'):
                feature_names = self.training_metadata.get('feature_names', [f'feature_{i}' for i in range(len(self.best_model.coef_[0]))])
                return dict(zip(feature_names, np.abs(self.best_model.coef_[0])))
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return None
    
    def explain(self) -> Dict[str, Any]:
        """Get explanations for the model and decisions"""
        if not self.is_fitted:
            return {"error": "Model must be fitted before getting explanations"}
        
        try:
            explanations = {
                "model_summary": {
                    "model_type": type(self.best_model).__name__,
                    "best_score": self.best_score,
                    "training_time": self.training_metadata.get('training_time', 0),
                    "n_features_used": len(self.training_metadata.get('selected_features', []))
                },
                "dataset_profile": self.current_profile.__dict__ if self.current_profile else None,
                "strategy_used": self.current_profile.recommendations if self.current_profile else None,
                "feature_importance": self.get_feature_importance(),
                "training_metadata": self.training_metadata
            }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate explanations: {e}")
            return {"error": f"Explanation generation failed: {str(e)}"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.is_fitted:
            return {"error": "Model must be fitted before getting performance stats"}

        task_type = (
            self.task_type_
            or self.training_metadata.get('task_type')
            or (
                self.current_profile.complexity_profile.get('task_type')
                if self.current_profile and self.current_profile.complexity_profile
                else None
            )
            or 'unknown'
        )
        data_type = (
            self.data_type_
            or self.training_metadata.get('data_type')
            or (self.current_profile.data_type if self.current_profile else None)
            or 'unknown'
        )

        return {
            'model_type': type(self.best_model).__name__,
            'best_score': self.best_score,
            'training_time': self.training_metadata.get('training_time', 0),
            'n_features': len(self.training_metadata.get('selected_features', [])),
            'data_type': data_type,
            'task_type': task_type,
            'model_family': self.training_metadata.get('model_family', self.config.get('model_family', 'ml')),
            'strategy': self.current_profile.recommendations.get('primary_strategy', 'unknown') if self.current_profile else "unknown"
        }

    def get_model_comparison(self) -> list:
        """Return sorted model comparison rows from the last training run."""
        if not self.is_fitted:
            return []
        rows = []
        model_results = self.training_metadata.get('training_results', {})
        eval_scores = self.training_metadata.get('evaluation_results', {}).get('all_model_scores', {})
        holdout_primary = (self.training_metadata.get('holdout_metrics') or {}).get('primary_score')
        for name, result in model_results.items():
            if isinstance(result, dict) and 'error' not in result:
                rows.append({
                    'model': name,
                    'cv_mean': result.get('cv_mean', 0.0),
                    'cv_std': result.get('cv_std', 0.0),
                    'scoring': result.get('scoring', ''),
                    'backend': result.get('backend', 'sklearn'),
                    'best_params': result.get('best_params', {}),
                    'trials_run': result.get('trials_run', 0),
                    'early_stopped': result.get('early_stopped', False),
                    'skip_reason': result.get('skip_reason'),
                    'train_time': result.get('train_time'),
                    'holdout': result.get('holdout_score'),
                })
        for name, scores in eval_scores.items():
            if not any(row['model'] == name for row in rows):
                rows.append({
                    'model': name,
                    'cv_mean': scores.get('mean', 0.0),
                    'cv_std': scores.get('std', 0.0),
                    'scoring': '',
                    'backend': 'sklearn',
                })
        task_type = self.task_type_ or 'classification'
        reverse = task_type == 'classification'
        rows.sort(key=lambda row: row['cv_mean'], reverse=reverse)
        if self.best_model is not None:
            rows.insert(0, {
                'model': f"* {type(self.best_model).__name__} (selected)",
                'cv_mean': self.best_score,
                'cv_std': 0.0,
                'scoring': 'best',
                'backend': getattr(self.best_model, 'backend', 'sklearn'),
                'holdout': holdout_primary,
            })
        return rows

    def print_model_comparison(self, n_space: int = 30) -> str:
        """Print a formatted comparison table for all trained models."""
        rows = self.get_model_comparison()
        if not rows:
            return "No model comparison data available."
        fmt = '{{0:{0}}}|{{1:{0}}}|{{2:{0}}}|{{3:{0}}}|{{4:{0}}}'.format(n_space)
        header = fmt.format('Model', 'Score', 'Std', 'Backend', 'BestParams')
        lines = [header, '-' * len(header)]
        for row in rows:
            params = row.get('best_params') or {}
            params_str = str(params)[:n_space] if params else '{}'
            lines.append(fmt.format(
                str(row['model'])[:n_space],
                f"{row['cv_mean']:.4f}",
                f"{row['cv_std']:.4f}",
                str(row.get('backend', 'sklearn')),
                params_str,
            ))
        output = '\n'.join(lines)
        logger.info("\n%s", output)
        print(output)
        return output

    def get_selection_report(self) -> Dict[str, Any]:
        """Structured selection report (winner, leaderboard, holdout)."""
        from core.training_report import get_selection_report as _build_report
        return _build_report(self)

    def selection_summary(self) -> str:
        """Short console summary of model selection."""
        from core.training_report import selection_summary as _summary
        return _summary(self)

    def generate_training_report(self) -> str:
        """Human-readable markdown training report."""
        from core.training_report import generate_training_report as _gen
        return _gen(self)

    def get_system_status(self) -> Dict[str, Any]:
        """Return integration and registry status for diagnostics."""
        status = {
            'is_fitted': self.is_fitted,
            'best_score': self.best_score,
            'task_type': self.task_type_,
            'data_type': self.data_type_,
        }
        if self.api_integrator:
            status['api'] = self.api_integrator.get_status()
        if self.model_registry:
            status['models'] = self.model_registry.get_registry_summary()
        if self.feature_registry:
            status['features'] = self.feature_registry.list_features()
        return status
    
    def run_full_integration(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Run full integration with all existing systems
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Integration results
        """
        logger.info(" Running full AutoForge integration...")
        
        results = {
            'autoforge_performance': {},
            'api_comparison': {},
            'benchmarking_results': {},
            'systemization_results': {},
            'tracking_summary': {},
            'optimization_comparison': {}
        }
        
        try:
            # 1. Train AutoForge
            logger.info(" Training AutoForge...")
            automl_input = AutoMLInput(
                data=pd.concat([X, y.rename('target')], axis=1),
                target_column='target'
            )
            self.fit(automl_input)
            
            results['autoforge_performance'] = self.get_performance_stats()
            
            # 2. API Integration
            logger.info(" Running API integration...")
            api_results = integrate_with_autoforge(self)
            results['api_comparison'] = api_results.get_status() if hasattr(api_results, 'get_status') else {}
            
            # 3. Benchmarking
            logger.info(" Running benchmarking...")
            test_datasets = [{'name': 'current_dataset', 'X': X, 'y': y}]
            benchmark_results = benchmark_autoforge(self, test_datasets)
            results['benchmarking_results'] = benchmark_results
            
            # 4. Systemization
            logger.info(" Running systemization...")
            monitoring = setup_autoforge_monitoring(self)
            if monitoring:
                monitoring_results = self.systemization_integrator.monitor_training(self, X, y)
                results['systemization_results'] = monitoring_results
            
            # 5. Tracking Summary
            results['tracking_summary'] = self.tracking_integrator.get_systemization_summary()
            
            # 6. Optimization Comparison
            if self.best_model:
                logger.info(" Running optimization comparison...")
                opt_results = self.optimizer_integrator.compare_optimizers(
                    self.best_model, X, y, 
                    self.optimizer_integrator.create_search_space(
                        type(self.best_model).__name__,
                        'classification' if len(np.unique(y)) < 20 else 'regression'
                    ),
                    {'max_trials': 10}
                )
                results['optimization_comparison'] = opt_results
            
            logger.info(" Full integration complete!")
            return results
            
        except Exception as e:
            logger.error(f" Full integration failed: {e}")
            results['error'] = str(e)
            return results
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of all integration components"""
        summary = {
            'model_registry': self.model_registry.get_registry_summary() if self.model_registry else {},
            'feature_registry': self.feature_registry.list_features() if self.feature_registry else [],
            'processors': bool(self.processor_integrator),
            'ensemble': bool(self.ensemble_integrator),
            'optimizer': bool(self.optimizer_integrator),
        }
        if self.api_integrator:
            summary['api'] = self.api_integrator.get_status()
        return summary
    
    def _prepare_input(self, input_data: Union[AutoMLInput, pd.DataFrame], 
                       target_column: Optional[str]) -> AutoMLInput:
        """Prepare and standardize input data"""
        if self._looks_like_automl_input(input_data):
            return input_data
        elif isinstance(input_data, pd.DataFrame):
            if target_column is None:
                raise ValueError("target_column must be specified when input_data is DataFrame")
            return AutoMLInput(data=input_data, target_column=target_column)
        else:
            raise ValueError("input_data must be AutoMLInput or pandas DataFrame")

    @staticmethod
    def _looks_like_automl_input(obj) -> bool:
        return all(hasattr(obj, attr) for attr in ('data', 'target_column', 'get_features', 'get_target'))

    def _ensure_core_components(self):
        """Lazily initialize components if optional imports failed at module load."""
        if self.execution_engine is None:
            from execution.execution_engine import ExecutionEngine
            self.execution_engine = ExecutionEngine()
        if self.feature_selector is None:
            from features.feature_selector import FeatureSelector
            self.feature_selector = FeatureSelector()
        if self.intelligence_engine is None:
            from intelligence.intelligence_engine import IntelligenceEngine
            self.intelligence_engine = IntelligenceEngine()
        if self.input_validator is None:
            from input_output.input_validator import InputValidator
            self.input_validator = InputValidator()
        if self.model_saver is None:
            from persistence.model_saver import ModelSaver
            self.model_saver = ModelSaver(self.config.get('model_save_path', './models'))
        if self.model_registry is None:
            from registry.model_registry import model_registry
            self.model_registry = model_registry
    
        
    def _maybe_build_ensemble(self, training_results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Build voting or disk-backed stacking ensemble from top CV models."""
        model_results = training_results.get('model_results', {})
        valid = [
            (name, result) for name, result in model_results.items()
            if 'model' in result and 'error' not in result
        ]
        if len(valid) < 2:
            training_results['ensemble_info'] = {
                'tried': False,
                'skip_reason': 'Fewer than two successful models.',
            }
            return training_results

        valid.sort(key=lambda item: item[1]['cv_mean'], reverse=True)
        top_n = self.config.get('ensemble_top_n', 3)
        top_items = valid[:min(top_n, len(valid))]
        winner_score = top_items[0][1]['cv_mean']

        search_depth = getattr(self, 'search_depth_', self.config.get('search_depth', 'balanced'))
        epsilon = self.config.get('ensemble_epsilon', 0.01)
        ensemble_delta = self.config.get('ensemble_delta', 0.001)

        close_items = [
            item for item in top_items
            if (winner_score - item[1]['cv_mean']) <= epsilon
        ]
        if search_depth == 'balanced' and len(close_items) < 2:
            training_results['ensemble_info'] = {
                'tried': False,
                'skip_reason': (
                    f'Top-{top_n} models not within epsilon={epsilon} of winner '
                    f'(CV={winner_score:.4f}).'
                ),
            }
            return training_results

        top_models = [result['model'] for _, result in (close_items if close_items else top_items)]
        top_items = close_items if close_items else top_items
        X_train = training_results.get('X_train')
        y_train = training_results.get('y_train')
        if X_train is None or y_train is None:
            training_results['ensemble_info'] = {
                'tried': False,
                'skip_reason': 'Training data unavailable for ensemble.',
            }
            return training_results

        ensemble_method = self.config.get('ensemble_method', 'none')
        user_pref = getattr(self, '_current_user_preference', 'auto')
        if ensemble_method == 'none' and user_pref == 'accurate':
            ensemble_method = 'stacking'
        if ensemble_method == 'none' and search_depth == 'balanced':
            ensemble_method = 'voting'

        try:
            if ensemble_method == 'stacking' and self.ensemble_integrator:
                save_dir = self.config.get('model_save_path', './models')
                run_id = f"ensemble_{int(time.time())}"
                ensemble = self.ensemble_integrator.build_disk_stacking_ensemble(
                    top_items, X_train, y_train, task_type, save_dir, run_id
                )
                training_results['ensemble_run_id'] = run_id
                training_results['ensemble_run_dir'] = str(Path(save_dir) / run_id)
                method_label = 'stacking'
            else:
                ensemble = self.ensemble_integrator.create_voting_ensemble(
                    top_models, task_type, voting='hard'
                )
                ensemble.fit(X_train, y_train)
                method_label = 'voting'

            ensemble_eval = self.ensemble_integrator.evaluate_ensemble(
                ensemble, X_train, y_train, top_models
            )
            ensemble_eval['tried'] = True
            ensemble_score = ensemble_eval.get('ensemble_score', training_results['best_score'])
            single_model_score = training_results['best_score']
            if ensemble_score >= single_model_score + ensemble_delta:
                training_results['best_model'] = ensemble
                training_results['best_score'] = ensemble_score
                training_results['model_type'] = type(ensemble).__name__
                training_results['ensemble_info'] = {
                    **ensemble_eval,
                    'selected': True,
                    'tried': True,
                    'close_models': [name for name, _ in top_items],
                    'member_scores': {
                        name: res['cv_mean'] for name, res in top_items
                    },
                    'ensemble_score': ensemble_score,
                    'single_model_score': single_model_score,
                }
                training_results['ensemble_method'] = method_label
                logger.info(
                    "✅ Ensemble (%s) selected as best model (score=%.4f)",
                    method_label,
                    ensemble_score,
                )
            else:
                training_results['ensemble_info'] = {
                    **ensemble_eval,
                    'tried': True,
                    'selected': False,
                    'close_models': [name for name, _ in top_items],
                    'member_scores': {
                        name: res['cv_mean'] for name, res in top_items
                    },
                    'ensemble_score': ensemble_score,
                    'single_model_score': training_results['best_score'],
                    'skip_reason': (
                        f'Ensemble score {ensemble_score:.4f} did not beat single model '
                        f'{training_results["best_score"]:.4f} by delta={ensemble_delta}.'
                    ),
                }
                training_results['ensemble_method'] = method_label
        except Exception as exc:
            logger.warning("Ensemble step skipped: %s", exc)
            training_results['ensemble_info'] = {
                'tried': True,
                'selected': False,
                'skip_reason': str(exc),
            }

        return training_results

    def _register_training_with_monitor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        execution_config: Dict[str, Any],
        training_results: Dict[str, Any],
    ) -> None:
        """Register training completion with LightweightMonitor."""
        if not systemization_integrator:
            return
        try:
            monitor = getattr(self, '_production_monitor', None)
            if monitor is None:
                monitor = systemization_integrator.setup_monitoring(self)
                self._production_monitor = monitor
            if monitor is None:
                return
            monitor.log_data_profile(X, dataset_name="training")
            record = {
                "timestamp": time.time(),
                "model_name": type(self.best_model).__name__,
                "metrics": {
                    "best_score": float(self.best_score),
                    "training_time": training_results.get("training_time", 0),
                },
                "sample_size": len(X),
                "task_type": execution_config.get("task_type"),
                "search_depth": execution_config.get("search_depth"),
                "event": "training_complete",
            }
            if hasattr(monitor, "accuracy_history"):
                monitor.accuracy_history.append(record)
            if systemization_integrator.available_components.get("model_versioning"):
                if not self.config.get("disable_auto_version_save", True):
                    systemization_integrator.save_model_version(
                        self,
                        f"autoforge_{int(time.time())}",
                        description=f"score={self.best_score:.4f}",
                    )
        except Exception as exc:
            logger.warning("Monitor registration skipped: %s", exc)

    def _post_process_training(self, training_results: Dict[str, Any],
                           feature_results: Dict[str, Any], start_time: float):
        """Post-process training results"""
        training_time = time.time() - start_time
        
        # Store results
        self.best_model = training_results['best_model']
        self.best_score = training_results['best_score']
        
        # Update metadata
        task_type = self.task_type_ or training_results.get('task_type')
        holdout_metrics = (
            training_results.get('holdout_metrics')
            or training_results.get('evaluation_results', {}).get('holdout_metrics')
            or {
                k: v for k, v in training_results.get('evaluation_results', {}).items()
                if isinstance(v, (int, float)) and k in (
                    'accuracy', 'f1_score', 'mse', 'rmse', 'mae', 'r2_score',
                    'primary_score', 'precision', 'recall',
                )
            }
        )
        regression_metrics = {}
        if task_type == 'regression':
            regression_metrics = {
                'cv_neg_mse': float(self.best_score),
                'cv_scoring': self.training_metadata.get('scoring', 'neg_mean_squared_error'),
            }
            regression_metrics.update({
                k: float(v) for k, v in holdout_metrics.items()
                if k in ('mse', 'rmse', 'mae', 'r2_score', 'primary_score')
            })

        prep_meta = getattr(self, '_prep_search_meta_', {}) or {}

        dataset_profile = None
        if self.current_profile:
            qp = getattr(self.current_profile, 'quality_profile', None) or {}
            missing_ratio = qp.get('missing_ratio', 0.0) if isinstance(qp, dict) else 0.0
            dataset_profile = {
                'missing_pct': round(float(missing_ratio) * 100, 2),
                'n_samples': getattr(self.current_profile, 'size_profile', {}).get('n_samples'),
                'n_features': getattr(self.current_profile, 'size_profile', {}).get('n_features'),
            }

        self.training_metadata = {
            'training_time': training_results['training_time'],
            'best_score': self.best_score,
            'task_type': task_type,
            'data_type': self.data_type_ or self.config.get('data_type', 'tabular'),
            'model_family': training_results.get('metadata', {}).get('model_family', self.config.get('model_family', 'ml')),
            'dataset_profile': dataset_profile,
            'selected_features': feature_results['selected_features'],
            'raw_feature_columns': getattr(self, '_raw_feature_columns_', feature_results['selected_features']),
            'n_features_selected': len(feature_results['selected_features']),
            'feature_selection_ratio': feature_results['selection_ratio'],
            'preprocessing_steps': training_results['preprocessing_info'],
            'preprocessing_artifacts': getattr(self, 'preprocessing_artifacts_', None)
                or training_results.get('preprocessing_info'),
            'preprocessing_recipe': prep_meta.get('preprocessing_recipe')
                or training_results.get('preprocessing_info', {}).get('recipe_name'),
            'preprocessing_search_results': prep_meta.get('preprocessing_search_results'),
            'preprocessing_cv_mean': prep_meta.get('preprocessing_cv_mean'),
            'preprocessing_cv_std': prep_meta.get('preprocessing_cv_std'),
            'preprocessing_scoring': prep_meta.get('preprocessing_scoring'),
            'scoring': training_results.get('metadata', {}).get('scoring')
                or prep_meta.get('preprocessing_scoring'),
            'feature_names': training_results.get('feature_names', feature_results['selected_features']),
            'training_results': training_results['model_results'],
            'evaluation_results': training_results['evaluation_results'],
            'holdout_metrics': holdout_metrics,
            'regression_metrics': regression_metrics,
            'holdout_split': getattr(self, '_holdout_split_', {}),
            'ensemble_method': training_results.get('ensemble_method'),
            'ensemble_info': training_results.get('ensemble_info'),
            'model_type': training_results['model_type'],
            'feature_scores': feature_results['feature_scores'],
            'feature_selection_on_train_only': feature_results.get(
                'feature_selection_on_train_only', True
            ),
            'ensemble_run_dir': training_results.get('ensemble_run_dir'),
            'ensemble_run_id': training_results.get('ensemble_run_id'),
        }

        from core.training_report import build_selection_decision
        self.training_metadata['selection_decision'] = (
            build_selection_decision(self) if self.best_model else {}
        )
        
        # Learn from execution
        if self.current_profile:
            self.intelligence_engine.learn_from_execution(self.current_profile, self.best_score)
        
        # Add to execution history
        self.execution_history.append({
            'timestamp': time.time(),
            'training_time': training_time,
            'best_score': self.best_score,
            'model_type': training_results['model_type'],
            'n_features': len(feature_results['selected_features'])
        })
    
    def _validate_predict_input(self, X: pd.DataFrame) -> None:
        """Ensure required raw input feature columns are present."""
        meta = self.training_metadata or {}
        candidates = [
            meta.get('raw_feature_columns') or [],
            meta.get('selected_features') or [],
        ]
        for expected in candidates:
            if not expected:
                continue
            missing = [c for c in expected if c not in X.columns]
            if not missing:
                return
        for expected in candidates:
            if not expected:
                continue
            missing = [c for c in expected if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns for prediction: {missing[:5]}"
                    + ("..." if len(missing) > 5 else "")
                )
            return

    def _apply_preprocessing(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Apply the same preprocessing used during training."""
        feature_names = (
            self.training_metadata.get('feature_names')
            or self.training_metadata.get('selected_features', [])
        )

        artifacts = self.training_metadata.get('preprocessing_artifacts')
        if artifacts:
            try:
                from execution.preprocessing_pipeline import PreprocessingPipeline
                pipe = PreprocessingPipeline()
                pipe.artifacts = artifacts
                pipe.is_fitted = True
                X_processed = pipe.transform(X)
                if feature_names and isinstance(X_processed, pd.DataFrame):
                    for col in feature_names:
                        if col not in X_processed.columns:
                            X_processed[col] = 0
                    X_processed = X_processed[feature_names]
                return X_processed
            except Exception as exc:
                logger.warning("Pipeline transform failed, falling back: %s", exc)

        if 'preprocessing_steps' not in self.training_metadata:
            if feature_names and isinstance(X, pd.DataFrame):
                cols = [c for c in feature_names if c in X.columns]
                return X[cols] if cols else X
            return X.values if isinstance(X, pd.DataFrame) else X

        steps = self.training_metadata['preprocessing_steps']
        X_processed = X.copy()

        if steps.get('missing_values_handled'):
            numeric_cols = steps.get('numeric_features', numeric_columns(X_processed))
            for col in numeric_cols:
                if col in X_processed.columns and X_processed[col].dtype in [np.number, 'float64', 'int64']:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mean())
            for col in steps.get('categorical_features', []):
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].fillna('unknown')

        encoders = steps.get('categorical_encoding', {})
        object_cols = categorical_columns(X_processed)
        for col in object_cols:
            if col in encoders and hasattr(encoders[col], 'transform'):
                values = X_processed[col].astype(str)
                known = set(encoders[col].classes_)
                values = values.where(values.isin(known), encoders[col].classes_[0])
                X_processed[col] = encoders[col].transform(values)
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        numeric_features = steps.get('numeric_features', [])
        scaler = steps.get('scaler')
        if scaler is not None and hasattr(scaler, 'transform') and numeric_features:
            cols = [
                col for col in numeric_features
                if col in X_processed.columns and pd.api.types.is_numeric_dtype(X_processed[col])
            ]
            if cols:
                X_processed[cols] = scaler.transform(X_processed[cols])

        expected = self.training_metadata.get('feature_names')
        if expected and isinstance(X_processed, pd.DataFrame):
            for col in expected:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[expected]

        return X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed

    @staticmethod
    def _infer_task_type_from_target(y: pd.Series) -> str:
        if len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64', 'int32', 'bool']:
            return 'classification'
        if pd.api.types.is_numeric_dtype(y) and len(y.unique()) >= 20:
            return 'regression'
        if len(y.unique()) < 20:
            return 'classification'
        return 'regression'

    def _resolve_task_type(self, automl_input, profile) -> str:
        user_task = getattr(automl_input, 'task_type', None)
        if user_task in ('classification', 'regression'):
            return user_task
        if profile and getattr(profile, 'complexity_profile', None):
            profile_task = profile.complexity_profile.get('task_type')
            if profile_task in ('classification', 'regression'):
                return profile_task
        return self._infer_task_type_from_target(automl_input.get_target())

    @staticmethod
    def _has_mixed_modalities(X: pd.DataFrame) -> bool:
        numeric_cols = numeric_columns(X)
        text_like = []
        for col in categorical_columns(X):
            sample = X[col].dropna()
            if len(sample) == 0:
                continue
            value = sample.iloc[0]
            if isinstance(value, str) and len(str(value).split()) > 2:
                text_like.append(col)
        return len(numeric_cols) > 0 and len(text_like) > 0

    def _build_advanced_features_config(
        self, X: pd.DataFrame, automl_input, model_family: str
    ) -> Dict[str, Any]:
        use_multimodal_cfg = self.config.get('use_multimodal', 'auto')
        use_nas_cfg = self.config.get('use_nas', False)
        use_multimodal = False
        if use_multimodal_cfg is True or use_multimodal_cfg == 'true':
            use_multimodal = True
        elif use_multimodal_cfg == 'auto':
            use_multimodal = self._has_mixed_modalities(X)
        use_nas = bool(use_nas_cfg) or model_family in ('dl', 'both')
        use_meta = self.config.get('use_meta_learning', False)
        return {
            'use_nas': use_nas,
            'use_multimodal': use_multimodal,
            'use_meta_learning': use_meta,
            'use_distributed': self.config.get('use_distributed', False),
            'use_explainability': self.config.get('use_explainability', False),
            'nas_min_features': self.config.get('nas_min_features', 10),
            'nas_min_samples': self.config.get('nas_min_samples', 100),
        }
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'model_save_path': './models',
            'auto_save_model': False,
            'model_name': None,
            'max_execution_time': 3600,
            'random_state': 42,
            'verbose': True,
            'n_jobs': -1,
            'use_processors': True,
            'enable_ensemble': True,
            'scale_features': True,
            'max_features': None,
            'min_features': 1,
            'model_family': 'ml',
            'data_type': None,
            'use_multimodal': 'auto',
            'use_nas': False,
            'use_meta_learning': False,
            'search_depth': 'balanced',
            'ensemble_method': 'none',
            'ensemble_top_n': 3,
            'ensemble_epsilon': 0.01,
            'ensemble_delta': 0.001,
            'scoring': None,
            'dl_epochs': 10,
            'dl_max_trials': 5,
            'nas_min_features': 10,
            'nas_min_samples': 100,
        }
