import sys
import runpy
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Execute the main dashboard
app_path = BASE_DIR / "dashboard" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
