"""ForgeCast Forecast Demo page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgecast.ui.views import configure_page, render_forecast_demo

configure_page("Forecast Demo")
render_forecast_demo()
