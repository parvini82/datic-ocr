"""Command-line Interface for OCR-Aware Question Solver."""

import argparse
import json
import logging
import sys
from pathlib import Path

from ocr_solver.config import settings
from ocr_solver.formatter import format_single_result, to_json_string
from ocr_solver.pipeline import SolverPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr_solver.cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR-Aware Question Solving Agent with Visual Refinement"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single question block image (e.g., samples/q113.png)",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="samples",
        help="Directory containing question block images (default: samples)",
    )
    parser.add_argument(
        "--inject-noise",
        action="store_true",
        help="Inject realistic Persian OCR noise before solving to stress-test refinement",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.08,
        help="Character perturbation rate for noise injection (default: 0.08)",
    )
    parser.add_argument(
        "--max-perturbations",
        type=int,
        default=3,
        help="Max perturbations injected per question block (default: 3)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry cap for OCR correction loop (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save output JSON",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include full audit trail and attempts in output",
    )

    args = parser.parse_args()

    pipeline = SolverPipeline()
    pipeline.agent.max_retries = args.max_retries

    if args.image:
        result = pipeline.process_image(
            image_path=args.image,
            inject_synthetic_noise=args.inject_noise,
            noise_rate=args.noise_rate,
            max_noise_perturbations=args.max_perturbations,
        )
        if args.detailed:
            out_str = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        else:
            out_str = to_json_string(result)

        print(out_str)

        if args.output:
            Path(args.output).write_text(out_str, encoding="utf-8")
            logger.info("Saved result to %s", args.output)

    else:
        results = pipeline.process_directory(
            dir_path=args.samples_dir,
            inject_synthetic_noise=args.inject_noise,
        )
        if args.detailed:
            out_str = json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False)
        else:
            out_str = to_json_string(results)

        print(out_str)

        if args.output:
            Path(args.output).write_text(out_str, encoding="utf-8")
            logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
