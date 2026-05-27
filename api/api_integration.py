"""
API Integration — lightweight wrapper around UnifiedAutoML.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.unified_automl import UnifiedAutoML

logger = logging.getLogger(__name__)


class APIIntegrator:
    """Expose UnifiedAutoML through a simple API integration layer."""

    def __init__(self):
        self.automl = None
        self.integration_history = []

    def _get_automl(self):
        if self.automl is None:
            from core.unified_automl import UnifiedAutoML
            self.automl = UnifiedAutoML()
        return self.automl

    def create_automl(self, config: Optional[Dict[str, Any]] = None):
        from core.unified_automl import UnifiedAutoML
        return UnifiedAutoML(config)

    def get_status(self) -> Dict[str, Any]:
        automl = self._get_automl()
        return {
            "is_fitted": automl.is_fitted,
            "best_score": automl.best_score,
            "model_type": getattr(automl, "training_metadata", {}).get("model_type"),
        }


api_integrator = APIIntegrator()


def integrate_with_autoforge(automl) -> APIIntegrator:
    integrator = APIIntegrator()
    integrator.automl = automl
    return integrator
