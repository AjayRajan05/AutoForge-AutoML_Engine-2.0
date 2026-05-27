import json
import os
import logging
from collections import defaultdict
from typing import Dict, List, Any


class SelfImprover:
    def __init__(self, path="automl/experiments/logs.json"):
        self.path = path
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "model_scores": defaultdict(list),
            "scaler_scores": defaultdict(list),
            "imputer_scores": defaultdict(list),
        }

    def load_logs(self) -> List[Dict[str, Any]]:
        """
        Load experiment logs with error handling
        """
        try:
            if not os.path.exists(self.path):
                self.logger.warning(f"Log file not found: {self.path}")
                return []
            
            with open(self.path, "r") as f:
                logs = json.load(f)
            
            if not isinstance(logs, list):
                self.logger.warning(f"Invalid log format, expected list got {type(logs)}")
                return []
            
            self.logger.info(f"Loaded {len(logs)} experiment logs")
            return logs
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in logs: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to load logs: {e}")
            return []

    def analyze(self) -> bool:
        """
        Analyze experiment logs to extract performance patterns
        """
        try:
            logs = self.load_logs()
            
            if not logs:
                self.logger.info("No logs to analyze")
                return False
            
            # Reset stats
            self.stats = {
                "model_scores": defaultdict(list),
                "scaler_scores": defaultdict(list),
                "imputer_scores": defaultdict(list),
            }
            
            successful_analyses = 0
            
            for run in logs:
                try:
                    params = run.get("params", {})
                    metrics = run.get("metrics", {})
                    score = metrics.get("cv_score")
                    
                    if score is None:
                        continue
                    
                    # Extract model performance
                    model = run.get("model")
                    if model:
                        self.stats["model_scores"][model].append(score)
                    
                    # Extract preprocessing performance
                    scaler = params.get("scaler")
                    if scaler:
                        self.stats["scaler_scores"][scaler].append(score)
                    
                    imputer = params.get("imputer")
                    if imputer:
                        self.stats["imputer_scores"][imputer].append(score)
                    
                    successful_analyses += 1
                    
                except Exception as e:
                    self.logger.debug(f"Failed to analyze run: {e}")
                    continue
            
            # Use configurable threshold for successful analyses
            try:
                from config.settings import get_config_value
                min_analyses = get_config_value('meta_learning', 'min_successful_analyses', 5)
            except ImportError:
                min_analyses = 5
            
            if successful_analyses < min_analyses:
                self.logger.warning(f"Only {successful_analyses} successful analyses, minimum {min_analyses} required")
                return False
            
            self.logger.info(f"Successfully analyzed {successful_analyses} runs")
            return successful_analyses > 0
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return False

    def get_best(self) -> Dict[str, List[str]]:
        """
        Get best performing components based on historical data
        """
        try:
            def avg(scores: List[float]) -> float:
                return sum(scores) / len(scores) if scores else 0.0
            
            def get_top_items(scores_dict: Dict[str, List[float]], top_k: int = 5) -> List[str]:
                """Get top-k items by average score"""
                sorted_items = sorted(
                    scores_dict.items(),
                    key=lambda x: avg(x[1]),
                    reverse=True
                )
                return [item[0] for item in sorted_items[:top_k]]
            
            best = {
                "model_priority": get_top_items(self.stats["model_scores"]),
                "scaler_priority": get_top_items(self.stats["scaler_scores"]),
                "imputer_priority": get_top_items(self.stats["imputer_scores"]),
            }
            
            # Log the findings
            for category, priority_list in best.items():
                if priority_list:
                    top_score = avg(self.stats[category.replace("_priority", "_scores")][priority_list[0]])
                    self.logger.info(f"{category}: {priority_list[:3]} (avg score: {top_score:.4f})")
            
            return best
            
        except Exception as e:
            self.logger.error(f"Failed to get best components: {e}")
            # Return empty priorities as fallback
            return {
                "model_priority": [],
                "scaler_priority": [],
                "imputer_priority": [],
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get detailed performance summary for analysis
        """
        try:
            summary = {}
            
            for category, scores_dict in self.stats.items():
                category_summary = {}
                
                for component, scores in scores_dict.items():
                    if scores:
                        category_summary[component] = {
                            "avg_score": sum(scores) / len(scores),
                            "min_score": min(scores),
                            "max_score": max(scores),
                            "count": len(scores),
                            "std_score": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5
                        }
                
                summary[category] = category_summary
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance summary: {e}")
            return {}

    def should_avoid_component(self, component: str, category: str, threshold: float = 0.3) -> bool:
        """
        Determine if a component should be avoided based on poor historical performance
        """
        try:
            scores_key = f"{category}_scores"
            if scores_key not in self.stats or component not in self.stats[scores_key]:
                return False  # No data, don't avoid
            
            scores = self.stats[scores_key][component]
            avg_score = sum(scores) / len(scores)
            
            should_avoid = avg_score < threshold
            if should_avoid:
                self.logger.debug(f"Avoiding {component} ({category}) due to low avg score: {avg_score:.4f}")
            
            return should_avoid
            
        except Exception as e:
            self.logger.error(f"Failed to check component avoidance: {e}")
            return False

    def update_with_new_run(self, run_data: Dict[str, Any]) -> bool:
        """
        Update stats with a new experiment run
        """
        try:
            params = run_data.get("params", {})
            metrics = run_data.get("metrics", {})
            score = metrics.get("cv_score")
            
            if score is None:
                return False
            
            # Update model scores
            model = run_data.get("model")
            if model:
                self.stats["model_scores"][model].append(score)
            
            # Update preprocessing scores
            scaler = params.get("scaler")
            if scaler:
                self.stats["scaler_scores"][scaler].append(score)
            
            imputer = params.get("imputer")
            if imputer:
                self.stats["imputer_scores"][imputer].append(score)
            
            self.logger.debug(f"Updated stats with new run: {model} = {score:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update with new run: {e}")
            return False