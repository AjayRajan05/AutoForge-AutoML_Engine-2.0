"""
AutoForge Meta-Learning Module
"""

try:
    from .dataset_profiler import profile_dataset
    from .knowledge_base import KnowledgeBase
    from .recommender import MetaRecommender
    from .pattern_learner import PatternLearner
    __all__ = [
        'profile_dataset', 'KnowledgeBase', 'MetaRecommender', 'PatternLearner'
    ]
except ImportError as e:
    # Fallback if modules have import issues
    __all__ = []
    profile_dataset = None
    KnowledgeBase = None
    MetaRecommender = None
    PatternLearner = None
