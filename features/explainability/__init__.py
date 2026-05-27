"""
AutoForge Explainability Module
"""

try:
    from .model_explainability import ModelExplainability, explain_model
    from .actionable_explainability import ActionableExplainability, generate_actionable_insights, get_actionable_summary
    from .decision_explainer import DecisionExplainer, DecisionExplanation
    from .actionable_generator import ActionableInsights, ActionableInsight
    __all__ = [
        'ModelExplainability', 'explain_model', 'ActionableExplainability',
        'generate_actionable_insights', 'get_actionable_summary',
        'DecisionExplainer', 'DecisionExplanation',
        'ActionableInsights', 'ActionableInsight'
    ]
except ImportError as e:
    # Fallback if modules have import issues
    __all__ = []
    ModelExplainability = None
    explain_model = None
    ActionableExplainability = None
    generate_actionable_insights = None
    get_actionable_summary = None
    DecisionExplainer = None
    DecisionExplanation = None
    ActionableInsights = None
    ActionableInsight = None
