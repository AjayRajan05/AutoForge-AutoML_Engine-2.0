import random
from sklearn.neural_network import MLPClassifier


def random_nn():
    layers = random.choice([
        (64,),
        (128, 64),
        (256, 128, 64)
    ])

    return MLPClassifier(hidden_layer_sizes=layers, max_iter=200)


class NeuralArchitectureSearch:
    """Neural Architecture Search for AutoML"""
    
    def __init__(self, max_trials=10):
        self.max_trials = max_trials
        self.best_architecture = None
        
    def search(self, X, y, task_type="classification"):
        """Search for best neural architecture"""
        best_score = 0
        best_model = None
        
        for _ in range(self.max_trials):
            model = random_nn()
            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=3)
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            except:
                continue
                
        self.best_architecture = best_model
        return best_model