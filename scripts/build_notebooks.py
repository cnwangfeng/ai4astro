"""Compatibility entry: build current v3 notebooks, not the old draft."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_notebooks_v3.py")),run_name="__main__")
