"""Executable script to run all sample question blocks and format output as Phase 4 JSON."""

import json
import logging
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from ocr_solver.formatter import format_single_result, generate_html_report, to_json_string
from ocr_solver.pipeline import SolverPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    samples_dir = root_dir / "samples"
    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}", file=sys.stderr)
        sys.exit(1)

    pipeline = SolverPipeline()

    sample_images = sorted(list(samples_dir.glob("*.png")))
    print(f"=== Running OCR-Aware Solver on {len(sample_images)} Sample Questions ===")

    clean_results = []
    for img in sample_images:
        print(f"\n---> Evaluating: {img.name}")
        res = pipeline.process_image(img, inject_synthetic_noise=False)
        clean_results.append(res)
        print(f"Solved: {img.name} -> Answer: {res.output.answer} (Changed: {res.output.changed})")

    # 1. Save strict JSON output
    output_json_path = root_dir / "outputs" / "sample_results.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(to_json_string(clean_results), encoding="utf-8")
    print(f"\n[✓] Results JSON saved to: {output_json_path}")

    # 2. Generate and save styled RTL HTML Report
    output_html_path = root_dir / "outputs" / "report.html"
    generate_html_report(clean_results, str(output_html_path))
    print(f"[✓] Beautiful RTL HTML Report generated at: {output_html_path}")
    print(f"    👉 Open in browser: file://{output_html_path.resolve()}")


if __name__ == "__main__":
    main()

