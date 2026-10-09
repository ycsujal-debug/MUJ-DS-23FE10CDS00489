"""
Pytest configuration for ReviewRadar.
Ensures project root is added to sys.path during test discovery and execution.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
