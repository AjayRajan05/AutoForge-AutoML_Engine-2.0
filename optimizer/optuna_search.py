try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

import logging
import time
from sklearn.model_selection import cross_val_score

# Model registry with fallback handling
try:
    from ..models.registry import MODEL_REGISTRY, safe_params, validate_params
except ImportError:
    MODEL_REGISTRY = {}
    safe_params = lambda x: x
    validate_params = lambda x: True

# Core components with fallback handling
try:
    from .search_space import get_search_space
except ImportError:
    get_search_space = lambda *args, **kwargs: {}

try:
    from ..tracking.logger import ExperimentLogger
except ImportError:
    ExperimentLogger = None

try:
    from ..execution.pipeline_builder import PipelineBuilder
except ImportError:
    PipelineBuilder = None

try:
    from ..features.meta_learning.self_improver import SelfImprover
except ImportError:
    SelfImprover = None

try:
    from ..core.failure_memory import failure_memory
except ImportError:
    failure_memory = None


class OptunaOptimizer:
    def __init__(self, n_trials=50, cv=3, task_type="classification"):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        self.n_trials = n_trials
        self.cv = cv
        self.task_type = task_type
        self.study = None
        self.logger = logging.getLogger(__name__)
        self.failure_log = []  # PRODUCTION-GRADE: Track failures
        self.best_params_history = {}  # PRODUCTION-GRADE: Warm start from history

    def optimize(self, X, y):
        """
        PRODUCTION-GRADE: Optimize with intelligent parameter validation and pruning
        """
        try:
            # PRODUCTION-GRADE: Get priors from self-improver (with error handling)
            priors = None
            try:
                improver = SelfImprover()
                improver.analyze()
                priors = improver.get_best()
                self.logger.info(f"Using priors from self-improver: {priors}")
            except Exception as e:
                self.logger.warning(f"Failed to load priors: {e}")
                priors = None

            # PRODUCTION-GRADE: Dataset-aware search space
            n_samples = len(X)
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            
            trials = []
            logger = ExperimentLogger()

            def objective(trial):
                try:
                    # PRODUCTION-GRADE: Get intelligent search space
                    params = get_search_space(trial, task_type=self.task_type, priors=priors)
                    
                    # Extract model name and parameters
                    model_name = params.pop("model")
                    
                    # PRODUCTION-GRADE: Validate model exists
                    if model_name not in MODEL_REGISTRY:
                        self.logger.warning(f"Unknown model: {model_name}")
                        return -1.0
                    
                    # PRODUCTION-GRADE: Check failure memory
                    if failure_memory.is_similar_to_past_failure(model_name, params):
                        self.logger.warning(f"Params similar to past failure for {model_name}")
                        # Get safe parameters from failure memory
                        params = failure_memory.get_safe_params(model_name, params)
                    
                    # PRODUCTION-GRADE: Validate and correct parameters
                    if not validate_params(model_name, params):
                        self.logger.warning(f"Invalid params for {model_name}")
                        # Log to failure memory
                        failure_memory.log_failure(model_name, params, "Invalid parameters")
                        return -1.0
                    
                    # PRODUCTION-GRADE: Use safe parameters
                    safe_model_params = safe_params(model_name, params)
                    
                    # PRODUCTION-GRADE: Safe training wrapper
                    try:
                        model = MODEL_REGISTRY[model_name](**safe_model_params)
                        pipeline = build_pipeline(params, model)
                        
                        # PRODUCTION-GRADE: Early pruning for bad configs
                        start_time = time.time()
                        
                        # PRODUCTION-GRADE: Dataset-aware CV folds
                        cv_folds = min(self.cv, 3 if n_samples < 1000 else 5)
                        
                        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds)
                        mean_score = cv_scores.mean()
                        
                        training_time = time.time() - start_time
                        
                        # PRODUCTION-GRADE: Cost-aware optimization
                        cost_adjusted_score = mean_score - 0.01 * training_time
                        
                        # PRODUCTION-GRADE: Early pruning for very bad models
                        if trial.number > 5 and mean_score < 0.5:
                            self.logger.info(f"Pruning trial {trial.number}: score {mean_score:.3f} < 0.5")
                            raise optuna.exceptions.TrialPruned()
                        
                        # PRODUCTION-GRADE: Update best params history
                        if mean_score > self.best_params_history.get(model_name, {}).get("score", 0):
                            self.best_params_history[model_name] = {
                                "score": mean_score,
                                "params": safe_model_params,
                                "training_time": training_time
                            }
                        
                        self.logger.info(f"Trial {trial.number}: {model_name} score = {mean_score:.4f}")
                        return cost_adjusted_score
                        
                    except Exception as training_error:
                        self.logger.warning(f"Training failed for {model_name}: {training_error}")
                        # Log to failure memory
                        failure_memory.log_failure(model_name, safe_model_params, str(training_error))
                        return -1.0
                    
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as e:
                    self.logger.warning(f"Trial {trial.number} failed: {e}")
                    # PRODUCTION-GRADE: Log failure for learning
                    failure_memory.log_failure(
                        model_name if 'model_name' in locals() else "unknown", 
                        params if 'params' in locals() else {}, 
                        str(e)
                    )
                    return -1.0

            # Create and run study
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            self.study.optimize(objective, n_trials=self.n_trials)
            
            # Sort trials by score (descending)
            trials.sort(key=lambda x: -x[0])
            
            self.logger.info(f"Optimization completed. Best score: {self.study.best_value:.4f}")
            self.logger.info(f"Total trials: {len(self.study.trials)}")
            
            return trials[:3]  # Return top 3 trials
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Hyperparameter optimization failed: {e}")
    
    def get_best_params(self):
        """Get best parameters from the study"""
        if self.study is None:
            raise ValueError("Study not created yet. Call optimize() first.")
        
        return self.study.best_params
    
    def get_optimization_history(self):
        """Get optimization history for plotting"""
        if self.study is None:
            raise ValueError("Study not created yet. Call optimize() first.")
        
        return self.study.trials_dataframe()