"""Export Project Deliverables (sample_outputs.json and sample_outputs.html).

Processes all sample exam questions through the OCR-aware solver pipeline
and exports the final results in both strict JSON and a styled RTL HTML viewer.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from ocr_solver.formatter import generate_html_report, to_json_string
from ocr_solver.pipeline import SolverPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_deliverables")


def export_deliverables(
    samples_dir: Path = root_dir / "samples",
    output_json_path: Path = root_dir / "sample_outputs.json",
    output_html_path: Path = root_dir / "sample_outputs.html",
    inject_synthetic_noise: bool = False,
) -> None:
    """Run pipeline on all sample images and write sample_outputs.json and sample_outputs.html."""
    if not samples_dir.exists():
        logger.error("Samples directory does not exist: %s", samples_dir)
        sys.exit(1)

    sample_images = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    if not sample_images:
        logger.error("No sample images found in %s", samples_dir)
        sys.exit(1)

    logger.info("Found %d sample images in %s", len(sample_images), samples_dir)
    pipeline = SolverPipeline()

    results = []
    for img in sample_images:
        logger.info("--> Evaluating sample: %s", img.name)
        res = pipeline.process_image(img, inject_synthetic_noise=inject_synthetic_noise)
        results.append(res)
        logger.info(
            "Solved %s => Option %s (Changed: %s, Resolved: %s in %d attempts)",
            img.name,
            res.output.answer,
            res.output.changed,
            res.is_resolved,
            res.total_attempts,
        )

    # 1. Export strict JSON deliverable
    json_content = to_json_string(results, indent=2, ensure_ascii=False)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json_content, encoding="utf-8")
    logger.info("✓ Saved JSON deliverable to: %s", output_json_path)

    # 2. Export styled Persian RTL HTML viewer
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(results, str(output_html_path))
    logger.info("✓ Saved HTML viewer deliverable to: %s", output_html_path)

    print("\n" + "=" * 60)
    print("🎯 DELIVERABLES EXPORTED SUCCESSFULLY")
    print(f"📄 JSON Output: {output_json_path.resolve()}")
    print(f"🌐 HTML Viewer: {output_html_path.resolve()}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deliverables for OCR-Aware Question Solver.")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=root_dir / "samples",
        help="Path to directory containing sample question crops",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root_dir / "sample_outputs.json",
        help="Destination path for sample_outputs.json",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=root_dir / "sample_outputs.html",
        help="Destination path for sample_outputs.html",
    )
    parser.add_argument(
        "--inject-noise",
        action="store_true",
        help="Inject synthetic OCR noise into sample texts before solving",
    )
    args = parser.parse_args()

    export_deliverables(
        samples_dir=args.samples_dir,
        output_json_path=args.output_json,
        output_html_path=args.output_html,
        inject_synthetic_noise=args.inject_noise,
    )


if __name__ == "__main__":
    main()
