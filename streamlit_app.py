"""ForgeCast - Streamlit Community Cloud Application Entrypoint.

This thin entrypoint delegates to src/forgecast/ui/app.py while ensuring
repository-relative paths and sys.path configuration are portable across
local and hosted Linux environments.
"""

from pathlib import Path
import sys

# Ensure src/ is on sys.path for direct module resolution
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forgecast.ui.app import main

if __name__ == "__main__":
    main()
