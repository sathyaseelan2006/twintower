import sys
from pathlib import Path

# Ensure backend root is on sys.path in Vercel Serverless environment
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import app
