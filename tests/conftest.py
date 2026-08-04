"""Make the package importable when running pytest from the repo root without
installing it (`pip install -e .` also works; this just removes the need)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
