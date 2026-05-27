import json
import os
import logging
from typing import List, Dict, Any


class KnowledgeBase:
    def __init__(self, path="automl/experiments/logs.json"):
        self.path = path
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        # Initialize empty logs file if it doesn't exist
        if not os.path.exists(self.path):
            self._create_empty_logs()

    def _create_empty_logs(self):
        """Create an empty logs file"""
        try:
            with open(self.path, "w") as f:
                json.dump([], f)
            self.logger.info(f"Created empty logs file at {self.path}")
        except Exception as e:
            self.logger.error(f"Failed to create logs file: {e}")
            raise

    def load(self) -> List[Dict[str, Any]]:
        """
        Load all experiments from knowledge base
        """
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                self.logger.warning(f"Invalid knowledge base format, expected list got {type(data)}")
                return []
            
            # Filter valid records
            valid_records = []
            for record in data:
                if self._is_valid_record(record):
                    valid_records.append(record)
                else:
                    self.logger.debug(f"Skipping invalid record: {record}")
            
            self.logger.info(f"Loaded {len(valid_records)} valid records from knowledge base")
            return valid_records
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in knowledge base: {e}")
            # Backup corrupted file and create new one
            self._backup_corrupted_file()
            self._create_empty_logs()
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return []

    def get_similar_experiments(self, n_features, n_samples, task_type, top_k=None):
        """
        Get experiments similar to current dataset characteristics
        
        Args:
            n_features: Number of features in current dataset
            n_samples: Number of samples in current dataset
            task_type: Type of task (classification/regression)
            top_k: Number of similar experiments to return (configurable)
        """
        # Use configurable default for top_k
        if top_k is None:
            try:
                from config.settings import get_config_value
                top_k = get_config_value('meta_learning', 'similar_experiments_top_k', 5)
            except ImportError:
                top_k = 5
        
        experiments = self.load()
        similar_experiments = []
        
        for exp in experiments:
            # Check if experiment has relevant characteristics
            exp_n_features = exp.get('dataset_profile', {}).get('num_cols', 0)
            exp_n_samples = exp.get('dataset_profile', {}).get('num_rows', 0)
            exp_task_type = exp.get('task_type')
            
            # Calculate similarity (simple heuristic)
            if exp_task_type == task_type:
                feature_similarity = 1.0 / (1.0 + abs(exp_n_features - n_features))
                sample_similarity = 1.0 / (1.0 + abs(exp_n_samples - n_samples))
                overall_similarity = (feature_similarity + sample_similarity) / 2.0
                
                similar_experiments.append({
                    'experiment': exp,
                    'similarity': overall_similarity
                })
        
        # Sort by similarity and return top_k
        similar_experiments.sort(key=lambda x: x['similarity'], reverse=True)
        return [item['experiment'] for item in similar_experiments[:top_k]]
    
    def add_experiment(self, experiment_data):
        """
        Add a new experiment to the knowledge base
        """
        experiments = self.load()
        experiments.append(experiment_data)
        
        try:
            with open(self.path, "w") as f:
                json.dump(experiments, f, indent=2)
            self.logger.info(f"Added experiment to knowledge base")
        except Exception as e:
            self.logger.error(f"Failed to add experiment: {e}")

    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        """Check if a record has the required structure"""
        required_fields = ["run_id", "timestamp"]
        
        if not isinstance(record, dict):
            return False
        
        for field in required_fields:
            if field not in record:
                return False
        
        # At least one of params, metrics, or model should be present
        if not any(field in record for field in ["params", "metrics", "model"]):
            return False
        
        return True

    def _backup_corrupted_file(self):
        """Backup corrupted knowledge base file"""
        try:
            backup_path = f"{self.path}.backup"
            if os.path.exists(self.path):
                os.rename(self.path, backup_path)
                self.logger.info(f"Backed up corrupted file to {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup corrupted file: {e}")

    def save_record(self, record: Dict[str, Any]) -> bool:
        """
        Save a single record to the knowledge base
        """
        try:
            # Load existing data
            existing_data = self.load()
            
            # Add new record
            existing_data.append(record)
            
            # Save back to file
            with open(self.path, "w") as f:
                json.dump(existing_data, f, indent=2)
            
            self.logger.debug(f"Saved record: {record.get('run_id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save record: {e}")
            return False

    def get_records_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all records for a specific model
        """
        all_records = self.load()
        return [record for record in all_records if record.get("model") == model_name]

    def get_records_by_score_range(self, min_score: float, max_score: float) -> List[Dict[str, Any]]:
        """
        Get records within a score range
        """
        all_records = self.load()
        filtered_records = []
        
        for record in all_records:
            metrics = record.get("metrics", {})
            score = metrics.get("cv_score")
            
            if score is not None and min_score <= score <= max_score:
                filtered_records.append(record)
        
        return filtered_records

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge base
        """
        records = self.load()
        
        if not records:
            return {
                "total_records": 0,
                "unique_models": 0,
                "avg_score": 0.0,
                "score_range": (0.0, 0.0)
            }
        
        # Extract scores
        scores = []
        models = set()
        
        for record in records:
            metrics = record.get("metrics", {})
            score = metrics.get("cv_score")
            
            if score is not None:
                scores.append(score)
            
            model = record.get("model")
            if model:
                models.add(model)
        
        return {
            "total_records": len(records),
            "unique_models": len(models),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "score_range": (min(scores), max(scores)) if scores else (0.0, 0.0),
            "models": list(models)
        }