"""ForgeCast Model Lifecycle page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgecast.ui.views import configure_page, render_model_lifecycle

configure_page("Model Lifecycle")
render_model_lifecycle()
