"""ForgeCast Data & Reliability page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgecast.ui.views import configure_page, render_data_reliability

configure_page("Data & Reliability")
render_data_reliability()
