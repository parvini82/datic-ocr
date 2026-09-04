"""Convenience script to run the solver on all sample questions and export deliverables."""

import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from scripts.export_deliverables import export_deliverables


def main() -> None:
    export_deliverables(
        samples_dir=root_dir / "samples",
        output_json_path=root_dir / "sample_outputs.json",
        output_html_path=root_dir / "sample_outputs.html",
    )


if __name__ == "__main__":
    main()


