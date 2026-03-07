"""DARDE/DARDO document validation pipeline — entry point.

Launches the Streamlit web interface where ONG workers can
upload PDF or JPG documents for automated validation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless=true"],
        check=True,
    )


if __name__ == "__main__":
    main()
