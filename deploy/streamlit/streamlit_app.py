"""Streamlit Community Cloud entry point: runs the real app at the repo root.

It lives here so that Community Cloud uses deploy/streamlit/requirements.txt
(it looks next to the entry point first), a slim list without the CLI,
eval and test dependencies.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "streamlit_app.py"), run_name="__main__")
