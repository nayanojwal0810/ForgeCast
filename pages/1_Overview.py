"""ForgeCast Overview page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgecast.ui.views import configure_page, render_overview

configure_page("Overview")
render_overview()
