"""
Advanced Neural Architecture Search with Meta-Learning
Intelligent architecture search for neural networks
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.neural_network import MLPClassifier, MLPRegressor
from features.meta_learning.knowledge_base import KnowledgeBase
from features.meta_learning.pattern_learner import PatternLearner

# Setup logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class AdvancedNAS:
    """
    Advanced Neural Architecture Search with Meta-Learning Intelligence
    
    Current AutoML NAS Limitations:
    - Fixed search patterns
    - No learning from previous architectures
    - Brute force exploration
    
    Our Approach:
    - Meta-learning guided search
    - Architecture pattern recognition
    - Intelligent convergence
    - Knowledge transfer between tasks
    """
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.architecture_patterns = {}  # Learned architecture patterns
        self.performance_history = {}    # Architecture performance memory
        
    def search_architecture(self, X, y, task_type="classification", max_trials=50):
        """
        Advanced NAS with meta-learning guidance
        
        Args:
            X: Input features
            y: Target variable  
            task_type: Classification or regression
            max_trials: Maximum architecture trials
            
        Returns:
            Best neural network architecture
        """
        logger.info("🧠 Starting Advanced Neural Architecture Search...")
        
        # Step 1: Analyze data characteristics for architecture guidance
        data_profile = self._analyze_data_characteristics(X, y, task_type)
        
        # Step 2: Get meta-learning guidance from similar datasets
        architecture_priors = self._get_architecture_priors(data_profile)
        
        # Step 3: Intelligent architecture generation
        best_architecture = None
        best_score = -np.inf if task_type == "classification" else np.inf
        
        for trial in range(max_trials):
            # Generate architecture with meta-learning guidance
            architecture = self._generate_intelligent_architecture(
                data_profile, architecture_priors, trial
            )
            
            # Evaluate architecture
            score = self._evaluate_architecture(architecture, X, y, task_type)
            
            # Update best
            is_better = (score > best_score) if task_type == "classification" else (score < best_score)
            if is_better:
                best_score = score
                best_architecture = architecture
                logger.info(f"🎯 New best architecture found: Score {score:.4f}")
        
        # Step 4: Store learned pattern
        self._store_architecture_pattern(data_profile, best_architecture, best_score)
        
        logger.info(f"🏆 Advanced NAS completed: Best score {best_score:.4f}")
        return best_architecture
    
    def _analyze_data_characteristics(self, X, y, task_type):
        """Analyze data to guide architecture search"""
        n_samples, n_features = X.shape
        
        # Analyze complexity
        if task_type == "classification":
            n_classes = len(np.unique(y))
            complexity = n_samples * n_features * n_classes
        else:
            complexity = n_samples * n_features
            
        # Analyze linearity (simple heuristic)
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        if task_type == "regression":
            lr = LinearRegression()
            lr.fit(X, y)
            pred = lr.predict(X)
            linearity = r2_score(y, pred)
        else:
            # For classification, use simple logistic regression
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X, y)
            pred = lr.predict(X)
            linearity = np.mean(pred == y)
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'complexity': complexity,
            'linearity': linearity,
            'task_type': task_type
        }
    
    def _get_architecture_priors(self, data_profile):
        """Get architecture guidance from similar past experiments"""
        # Query knowledge base for similar datasets
        similar_experiments = self.knowledge_base.get_similar_experiments(
            data_profile['n_features'],
            data_profile['n_samples'],
            data_profile['task_type']
        )
        
        if similar_experiments:
            # Extract successful architecture patterns
            successful_patterns = []
            for exp in similar_experiments[:5]:  # Top 5 similar
                if exp.get('model_type') == 'neural_network':
                    pattern = exp.get('architecture_pattern')
                    if pattern and exp.get('cv_score', 0) > 0.8:
                        successful_patterns.append(pattern)
            
            return successful_patterns
        return []
    
    def _generate_intelligent_architecture(self, data_profile, priors, trial):
        """Generate architecture with meta-learning guidance"""
        n_features = data_profile['n_features']
        complexity = data_profile['complexity']
        linearity = data_profile['linearity']
        
        # Base architecture on data characteristics
        if priors and trial < len(priors):
            # Use meta-learning guidance for early trials
            base_pattern = priors[trial % len(priors)]
            architecture = self._adapt_pattern(base_pattern, data_profile)
        else:
            # Intelligent exploration for later trials
            architecture = self._intelligent_exploration(data_profile, trial)
        
        return architecture
    
    def _adapt_pattern(self, pattern, data_profile):
        """Adapt learned pattern to current data"""
        n_features = data_profile['n_features']
        
        # Scale architecture based on input size
        if n_features < 10:
            # Small input: simpler architecture
            layers = pattern.get('layers', 2)
            neurons = [max(16, n_features * 2)] * layers
        elif n_features < 50:
            # Medium input: balanced architecture
            layers = pattern.get('layers', 3)
            neurons = [max(32, n_features)] + [max(16, n_features // 2)] * (layers - 1)
        else:
            # Large input: deep architecture
            layers = pattern.get('layers', 4)
            neurons = [max(64, n_features // 2)] + [max(32, n_features // 4)] * (layers - 1)
        
        return {
            'layers': layers,
            'neurons': neurons,
            'activation': pattern.get('activation', 'relu'),
            'solver': pattern.get('solver', 'adam'),
            'alpha': pattern.get('alpha', 0.0001),
            'learning_rate_init': pattern.get('learning_rate_init', 0.001)
        }
    
    def _intelligent_exploration(self, data_profile, trial):
        """Intelligent architecture exploration"""
        n_features = data_profile['n_features']
        linearity = data_profile['linearity']
        complexity = data_profile['complexity']
        
        # Adjust depth based on complexity and linearity
        if linearity > 0.8:  # Very linear - simple architecture
            layers = np.random.choice([1, 2])
        elif linearity > 0.6:  # Moderately linear
            layers = np.random.choice([2, 3])
        else:  # Non-linear - deeper architecture
            layers = np.random.choice([3, 4, 5])
        
        # Adjust width based on input size
        if n_features < 10:
            base_neurons = np.random.choice([16, 32, 64])
        elif n_features < 50:
            base_neurons = np.random.choice([32, 64, 128])
        else:
            base_neurons = np.random.choice([64, 128, 256])
        
        # Create decreasing neuron pattern
        neurons = [base_neurons]
        for i in range(1, layers):
            neurons.append(max(8, base_neurons // (2**i)))
        
        return {
            'layers': layers,
            'neurons': neurons,
            'activation': np.random.choice(['relu', 'tanh']),
            'solver': np.random.choice(['adam', 'sgd']),
            'alpha': 10 ** np.random.uniform(-4, -2),
            'learning_rate_init': 10 ** np.random.uniform(-3, -1)
        }
    
    def _evaluate_architecture(self, architecture, X, y, task_type):
        """Evaluate neural network architecture"""
        try:
            # Create neural network with this architecture
            if task_type == "classification":
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(architecture['neurons']),
                    activation=architecture['activation'],
                    solver=architecture['solver'],
                    alpha=architecture['alpha'],
                    learning_rate_init=architecture['learning_rate_init'],
                    max_iter=500,  # Quick evaluation
                    random_state=42,
                    verbose=False
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(architecture['neurons']),
                    activation=architecture['activation'],
                    solver=architecture['solver'],
                    alpha=architecture['alpha'],
                    learning_rate_init=architecture['learning_rate_init'],
                    max_iter=500,
                    random_state=42,
                    verbose=False
                )
            
            # Quick cross-validation
            from sklearn.model_selection import cross_val_score
            
            if task_type == "classification":
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                return np.mean(scores)
            else:
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                return np.mean(scores)
                
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return -1.0 if task_type == "classification" else 1e6
    
    def _store_architecture_pattern(self, data_profile, architecture, score):
        """Store successful architecture pattern for future learning"""
        pattern_key = f"{data_profile['task_type']}_{data_profile['n_features']}_{data_profile['n_samples']}"
        
        # Store in performance history
        self.performance_history[pattern_key] = {
            'data_profile': data_profile,
            'architecture': architecture,
            'score': score,
            'timestamp': np.datetime64('now')
        }
        
        # Add to knowledge base
        self.knowledge_base.add_experiment({
            'model_type': 'neural_network',
            'architecture_pattern': architecture,
            'data_characteristics': data_profile,
            'cv_score': score,
            'task_type': data_profile['task_type']
        })
        
        logger.info(f"🧠 Stored architecture pattern for future learning")
