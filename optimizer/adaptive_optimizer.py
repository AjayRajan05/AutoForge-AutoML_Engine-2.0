"""
Adaptive Hyperparameter Optimization with Optuna Pruning and Dynamic Trial Budget
Smart search, not brute force
"""

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    MedianPruner = None

import numpy as np
import pandas as pd
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

logger = logging.getLogger(__name__)


class AdaptiveOptimizer:
    """
    Adaptive hyperparameter optimization with intelligent pruning and dynamic budgeting
    """
    
    def __init__(self,
                 initial_trials: int = 20,
                 max_trials: int = 100,
                 pruning_patience: int = 5,
                 improvement_threshold: float = 0.001,
                 time_budget: Optional[float] = None,
                 cv_folds: int = 5,
                 large_dataset_threshold: int = 10000,
                 sample_ratio: float = 0.3):
        """
        Initialize adaptive optimizer
        
        Args:
            initial_trials: Starting number of trials
            max_trials: Maximum number of trials allowed
            pruning_patience: Trials to wait before pruning
            improvement_threshold: Minimum improvement to continue
            time_budget: Maximum time in seconds
            cv_folds: Cross-validation folds
            large_dataset_threshold: Dataset size to trigger sampling
            sample_ratio: Fraction of data to sample for large datasets
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        self.initial_trials = initial_trials
        self.max_trials = max_trials
        self.pruning_patience = pruning_patience
        self.improvement_threshold = improvement_threshold
        self.time_budget = time_budget
        self.cv_folds = cv_folds
        self.large_dataset_threshold = large_dataset_threshold
        self.sample_ratio = sample_ratio
        
        self.study = None
        self.trial_history = []
        self.best_scores = []
        self.start_time = None
        
    def create_study(self, study_name: str = "adaptive_optimization") -> optuna.Study:
        """
        Create Optuna study with adaptive configuration
        
        Args:
            study_name: Name of the study
            
        Returns:
            Optuna study object
        """
        # Adaptive sampler based on trial count
        sampler = TPESampler(
            n_startup_trials=max(10, self.initial_trials // 2),
            n_ei_candidates=24,
            seed=42
        )
        
        # Adaptive pruner
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=max(1, self.cv_folds // 2),
            interval_steps=1
        )
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        self.start_time = time.time()
        return self.study
    
    def optimize_adaptive(self, 
                         objective_func: Callable,
                         X, 
                         y,
                         task_type: str = "classification",
                         progress_callback: Optional[Callable] = None) -> Tuple[List[Tuple], Dict[str, Any]]:
        """
        Adaptive optimization with dynamic trial budget
        
        Args:
            objective_func: Objective function to optimize
            X: Features
            y: Target
            task_type: Type of ML task
            progress_callback: Progress reporting function
            
        Returns:
            List of (score, model_name, params) and optimization metadata
        """
        # Smart sampling for large datasets
        if len(X) > self.large_dataset_threshold:
            import numpy as np
            sample_size = max(1000, int(len(X) * self.sample_ratio))
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sampled = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_sampled = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            logger.info(f"Large dataset detected ({len(X)} samples). Using sample of {sample_size} for optimization")
            X, y = X_sampled, y_sampled
        
        # Reduce CV folds for large datasets to speed up optimization
        if len(X) > self.large_dataset_threshold:
            original_cv_folds = self.cv_folds
            self.cv_folds = min(3, self.cv_folds)
            logger.info(f"Reduced CV folds from {original_cv_folds} to {self.cv_folds} for large dataset")
        
        if self.study is None:
            self.create_study()
        
        # Wrap objective with adaptive logic
        def wrapped_objective(trial):
            return self._adaptive_objective(trial, objective_func, X, y, task_type)
        
        # Adaptive optimization loop
        current_trials = min(self.initial_trials, 15)  # Reduce initial trials for speed
        iteration = 0
        optimization_metadata = {
            "total_trials": 0,
            "pruned_trials": 0,
            "optimization_time": 0,
            "convergence_achieved": False,
            "dynamic_budget_used": False,
            "dataset_sampled": len(X) != (len(X) if not hasattr(X, '__len__') else len(X)),
            "original_cv_folds": original_cv_folds if 'original_cv_folds' in locals() else self.cv_folds
        }
        
        while current_trials <= self.max_trials:
            logger.info(f"Starting optimization iteration {iteration + 1} with {current_trials} trials")
            
            # Check time budget
            if self.time_budget and (time.time() - self.start_time) > self.time_budget:
                logger.info("Time budget reached, stopping optimization")
                break
            
            # Run trials for this iteration
            iteration_start = time.time()
            
            # Optimize with current trial budget
            self.study.optimize(
                wrapped_objective,
                n_trials=current_trials,
                callbacks=[self._trial_callback] if progress_callback else None
            )
            
            iteration_time = time.time() - iteration_start
            optimization_metadata["optimization_time"] += iteration_time
            
            # Analyze progress and decide whether to continue
            should_continue, next_trials = self._analyze_progress()
            
            optimization_metadata["total_trials"] = len(self.study.trials)
            optimization_metadata["pruned_trials"] = sum(1 for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED)
            
            if not should_continue:
                optimization_metadata["convergence_achieved"] = True
                logger.info("Convergence achieved, stopping optimization")
                break
            
            if current_trials < self.max_trials:
                optimization_metadata["dynamic_budget_used"] = True
                current_trials = min(next_trials, self.max_trials)
                iteration += 1
        
        # Extract best trials
        best_trials = self._extract_best_trials()
        
        optimization_metadata["final_trials"] = len(self.study.trials)
        optimization_metadata["best_score"] = self.study.best_value if self.study.trials else None
        optimization_metadata["best_params"] = self.study.best_params if self.study.trials else None
        
        return best_trials, optimization_metadata
    
    def _adaptive_objective(self, 
                           trial: optuna.Trial,
                           objective_func: Callable,
                           X, 
                           y,
                           task_type: str) -> float:
        """
        Adaptive objective function with pruning support
        
        Args:
            trial: Optuna trial object
            objective_func: Original objective function
            X: Features
            y: Target
            task_type: Type of ML task
            
        Returns:
            Objective value
        """
        try:
            # Get model and parameters from objective function
            score, model_name, params = objective_func(trial, X, y, task_type)
            
            # Report intermediate scores for pruning (simulated CV steps)
            if hasattr(trial, 'report') and self.cv_folds > 1:
                # Simulate cross-validation progress
                intermediate_scores = np.linspace(score * 0.7, score, self.cv_folds)
                for step, intermediate_score in enumerate(intermediate_scores):
                    trial.report(intermediate_score, step)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.debug(f"Trial {trial.number} pruned at step {step}")
                        raise optuna.TrialPruned()
            
            return score
            
        except optuna.TrialPruned:
            raise
        except optuna.exceptions.OptunaError as e:
            # Handle Optuna-specific errors (like CategoricalDistribution issues)
            logger.warning(f"Trial {trial.number} failed with Optuna error: {e}")
            # Return a very bad score instead of failing
            return -1.0 if task_type == "classification" else -1e6
        except Exception as e:
            # Handle other errors (model fitting, data issues, etc.)
            logger.warning(f"Trial {trial.number} failed: {e}")
            # Return a very bad score instead of failing
            return -1.0 if task_type == "classification" else -1e6
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """
        Callback for trial progress tracking
        
        Args:
            study: Optuna study
            trial: Current trial
        """
        self.trial_history.append({
            "trial_number": trial.number,
            "value": trial.value if trial.value is not None else None,
            "state": trial.state.name,
            "params": trial.params.copy()
        })
        
        # Track best scores
        if trial.value is not None and (not self.best_scores or trial.value > max(self.best_scores)):
            self.best_scores.append(trial.value)
        elif self.best_scores:
            self.best_scores.append(max(self.best_scores))
    
    def _analyze_progress(self) -> Tuple[bool, int]:
        """
        Analyze optimization progress and decide next steps
        
        Returns:
            Tuple of (should_continue, next_trial_budget)
        """
        if len(self.study.trials) < self.initial_trials:
            return True, self.initial_trials
        
        # Check recent improvements
        recent_trials = self.study.trials[-10:]  # Last 10 trials
        recent_scores = [t.value for t in recent_trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(recent_scores) < 5:
            # Need more data to decide
            return True, min(len(self.study.trials) + 10, self.max_trials)
        
        # Check if all scores are infinite (regression problem)
        if all(score == float('inf') for score in recent_scores):
            logger.warning("All recent trials have infinite scores - stopping optimization")
            return False, 0
        
        # Filter out infinite scores for improvement calculation
        finite_scores = [score for score in recent_scores if score != float('inf')]
        if not finite_scores:
            logger.warning("No finite scores available - stopping optimization")
            return False, 0
        
        # Calculate improvement
        best_recent = max(finite_scores)
        best_overall = self.study.best_value if self.study.best_value != float('inf') else max(finite_scores)
        
        improvement = best_recent - best_overall if best_overall is not None else 0
        
        # Decision logic with more aggressive convergence for large datasets
        # Note: We can't access X here, so we use a simpler convergence strategy
        if improvement < self.improvement_threshold:
            logger.info(f"Small improvement ({improvement:.6f}), considering convergence")
            
            # Check if we've had enough trials
            if len(self.study.trials) >= self.initial_trials * 2:
                return False, 0  # Stop optimization
        
        # Check time budget
        if self.time_budget and (time.time() - self.start_time) > self.time_budget * 0.8:
            logger.info("Approaching time budget limit")
            return False, 0
        
        # Continue with adaptive trial budget
        if improvement > self.improvement_threshold * 2:
            # Good improvement, increase trials
            next_trials = min(len(self.study.trials) + 20, self.max_trials)
            logger.info(f"Good improvement detected, increasing trials to {next_trials}")
        else:
            # Moderate improvement, maintain current pace
            next_trials = min(len(self.study.trials) + 10, self.max_trials)
            logger.info(f"Moderate improvement, continuing with {next_trials} trials")
        
        return True, next_trials
    
    def _extract_best_trials(self, top_k: int = 10) -> List[Tuple]:
        """
        Extract top trials from optimization
        
        Args:
            top_k: Number of top trials to extract
            
        Returns:
            List of (score, model_name, params) tuples
        """
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        # Sort by score (descending)
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        
        best_trials = []
        for trial in completed_trials[:top_k]:
            model_name = trial.params.get("model", "unknown")
            params = trial.params.copy()
            score = trial.value
            
            best_trials.append((score, model_name, params))
        
        return best_trials
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report
        
        Returns:
            Optimization report dictionary
        """
        if not self.study or not self.study.trials:
            return {"error": "No optimization performed"}
        
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        report = {
            "study_name": self.study.study_name,
            "total_trials": len(self.study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_score": self.study.best_value,
            "best_params": self.study.best_params,
            "optimization_time": time.time() - self.start_time if self.start_time else None,
            "pruning_efficiency": len(pruned_trials) / len(self.study.trials) if self.study.trials else 0,
            "convergence_rate": self._calculate_convergence_rate(),
            "model_performance": self._analyze_model_performance()
        }
        
        return report
    
    def _calculate_convergence_rate(self) -> float:
        """
        Calculate convergence rate based on score improvements
        
        Returns:
            Convergence rate (0-1)
        """
        if len(self.best_scores) < 2:
            return 0.0
        
        # Calculate rate of improvement
        improvements = []
        for i in range(1, len(self.best_scores)):
            improvement = (self.best_scores[i] - self.best_scores[i-1]) / max(abs(self.best_scores[i-1]), 1e-8)
            improvements.append(improvement)
        
        # Average improvement rate (negative for decreasing improvements)
        avg_improvement = np.mean(improvements)
        
        # Convert to convergence rate (higher = more converged)
        convergence_rate = max(0, 1 - abs(avg_improvement) * 10)
        
        return convergence_rate
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """
        Analyze performance by model type
        
        Returns:
            Model performance analysis
        """
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        model_performance = {}
        for trial in completed_trials:
            model_name = trial.params.get("model", "unknown")
            score = trial.value
            
            if model_name not in model_performance:
                model_performance[model_name] = {
                    "scores": [],
                    "count": 0,
                    "best_score": -float('inf'),
                    "avg_score": 0.0
                }
            
            model_performance[model_name]["scores"].append(score)
            model_performance[model_name]["count"] += 1
            model_performance[model_name]["best_score"] = max(model_performance[model_name]["best_score"], score)
        
        # Calculate averages
        for model_name, stats in model_performance.items():
            if stats["scores"]:
                stats["avg_score"] = np.mean(stats["scores"])
                stats["std_score"] = np.std(stats["scores"])
        
        return model_performance
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history
        
        Args:
            save_path: Path to save the plot
        """
        if not self.study or not self.study.trials:
            logger.warning("No trials to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig = plot_optimization_history(self.study)
            plt.title("Optimization History")
            plt.xlabel("Trial")
            plt.ylabel("Objective Value")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")


    def optimize_with_search_space(
        self,
        model,
        X,
        y,
        search_space: Dict[str, Any],
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """Optimize a single model using a parameter search space."""
        from copy import deepcopy

        model_template = deepcopy(model)

        def objective(trial, X_data, y_data, task_type):
            params = {}
            for name, spec in search_space.items():
                if isinstance(spec, list):
                    params[name] = trial.suggest_categorical(name, spec)
                elif isinstance(spec, tuple) and len(spec) >= 3:
                    low, high, kind = spec[:3]
                    if kind == "int":
                        params[name] = trial.suggest_int(name, int(low), int(high))
                    else:
                        params[name] = trial.suggest_float(name, float(low), float(high))
                else:
                    params[name] = spec

            trial_model = deepcopy(model_template)
            trial_model.set_params(**params)
            scores = cross_val_score(trial_model, X_data, y_data, cv=self.cv_folds, scoring=scoring, n_jobs=-1)
            return float(scores.mean()), type(model).__name__, params

        task_type = "classification" if scoring == "accuracy" else "regression"
        _, metadata = self.optimize_adaptive(objective, X, y, task_type=task_type)
        return {
            "best_params": metadata.get("best_params", {}),
            "best_score": metadata.get("best_score", -np.inf),
            "trials_completed": metadata.get("total_trials", 0),
        }


def create_adaptive_optimizer(**kwargs) -> AdaptiveOptimizer:
    """
    Convenience function to create adaptive optimizer
    
    Args:
        **kwargs: Arguments for AdaptiveOptimizer
        
    Returns:
        AdaptiveOptimizer instance
    """
    return AdaptiveOptimizer(**kwargs)
