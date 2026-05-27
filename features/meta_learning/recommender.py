import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


class MetaRecommender:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base.load()
        self.failure_patterns = self._analyze_failures()

    def _analyze_failures(self):
        """Analyze patterns of failed experiments"""
        failures = defaultdict(list)
        for record in self.kb:
            if record.get("metrics", {}).get("cv_score", 0) < 0.5:  # Poor performance
                params = record.get("params", {})
                model = record.get("model")
                profile = record.get("dataset_profile", {})
                
                # Store failure patterns
                failures[model].append({
                    "profile": profile,
                    "params": params,
                    "score": record.get("metrics", {}).get("cv_score", 0)
                })
        
        return failures

    def similarity(self, a, b):
        """Enhanced similarity metric with multiple features"""
        keys = ["num_rows", "num_cols", "num_numeric", "num_categorical", "missing_ratio"]
        
        # Normalize features
        features_a = []
        features_b = []
        
        for k in keys:
            val_a = a.get(k, 0)
            val_b = b.get(k, 0)
            
            # Log transform for scale invariance
            if k in ["num_rows", "num_cols"] and val_a > 0 and val_b > 0:
                val_a = np.log1p(val_a)
                val_b = np.log1p(val_b)
            
            features_a.append(val_a)
            features_b.append(val_b)
        
        # Cosine similarity
        features_a = np.array(features_a).reshape(1, -1)
        features_b = np.array(features_b).reshape(1, -1)
        
        return cosine_similarity(features_a, features_b)[0][0]

    def _avoid_failures(self, model_name, dataset_profile):
        """Check if model is likely to fail on this dataset"""
        if model_name not in self.failure_patterns:
            return True  # No failure data, allow
        
        failures = self.failure_patterns[model_name]
        if not failures:
            return True
        
        # Check similarity to failed cases
        for failure in failures:
            sim = self.similarity(dataset_profile, failure["profile"])
            if sim > 0.8:  # Very similar to a failed case
                return False
        
        return True

    def recommend(self, dataset_profile, top_k=5):
        """Enhanced recommendations with failure avoidance"""
        try:
            scored = []

            for record in self.kb:
                past_profile = record.get("dataset_profile", {})
                score = record.get("metrics", {}).get("cv_score", 0)
                model = record.get("model")

                # Skip models that are likely to fail
                if not self._avoid_failures(model, dataset_profile):
                    continue

                sim = self.similarity(dataset_profile, past_profile)
                
                # Boost score for high similarity
                adjusted_score = score * (1 + sim)
                
                scored.append((sim, adjusted_score, record))

            # Sort by similarity (high) + performance (high)
            scored.sort(key=lambda x: (-x[0], -x[1]))

            # Add diversity - ensure different model types
            diverse_recommendations = []
            seen_models = set()
            
            for sim, score, record in scored:
                model = record.get("model")
                if model not in seen_models and len(diverse_recommendations) < top_k:
                    diverse_recommendations.append(record)
                    seen_models.add(model)

            return diverse_recommendations
            
        except Exception as e:
            print(f"Meta-learning recommend failed: {e}")
            return []  # Return empty list instead of None

    def get_preprocessing_hints(self, dataset_profile):
        """Get preprocessing recommendations based on dataset characteristics"""
        try:
            hints = {}
            
            # Missing data handling
            missing_ratio = dataset_profile.get("missing_ratio", 0)
            if missing_ratio > 0.2:
                hints["imputer"] = ["knn", "most_frequent"]
            elif missing_ratio > 0.05:
                hints["imputer"] = ["mean", "median"]
            else:
                hints["imputer"] = ["mean"]
            
            # Scaling recommendations
            num_numeric = dataset_profile.get("num_numeric", 0)
            num_categorical = dataset_profile.get("num_categorical", 0)
            
            if num_numeric > 10:
                hints["scaler"] = ["standard", "robust"]
            elif num_numeric > 0:
                hints["scaler"] = ["standard"]
            
            # Feature selection
            total_features = num_numeric + num_categorical
            if total_features > 50:
                hints["feature_selection"] = [True]
                hints["k_features"] = min(20, total_features // 2)
            
            return hints
            
        except Exception as e:
            print(f"get_preprocessing_hints failed: {e}")
            return {'imputer': ['mean'], 'scaler': ['standard']}  # Return defaults instead of None