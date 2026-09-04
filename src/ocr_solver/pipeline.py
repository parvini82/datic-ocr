"""End-to-end OCR-Aware Solver Pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from ocr_solver.agent.loop import OCRAwareSolverAgent
from ocr_solver.formatter import format_single_result
from ocr_solver.models import DetailedSolverResult, OCRResult
from ocr_solver.noise.persian_noise import PersianOCRNoiseInjector
from ocr_solver.ocr.datalab import DatalabOCRClient

logger = logging.getLogger(__name__)


class SolverPipeline:
    """Orchestrates OCR extraction, optional noise injection, and agent solving."""

    def __init__(
        self,
        ocr_client: Optional[DatalabOCRClient] = None,
        agent: Optional[OCRAwareSolverAgent] = None,
        noise_injector: Optional[PersianOCRNoiseInjector] = None,
    ):
        self.ocr_client = ocr_client or DatalabOCRClient()
        self.agent = agent or OCRAwareSolverAgent()
        self.noise_injector = noise_injector or PersianOCRNoiseInjector()

    def process_image(
        self,
        image_path: Union[str, Path],
        inject_synthetic_noise: bool = False,
        noise_rate: float = 0.08,
        max_noise_perturbations: int = 3,
    ) -> DetailedSolverResult:
        """
        Process a single question crop image through the full pipeline.

        Args:
            image_path: Path to the image file.
            inject_synthetic_noise: Whether to corrupt the initial OCR text with synthetic Persian noise.
            noise_rate: Character corruption rate for noise injector.
            max_noise_perturbations: Max character perturbations.

        Returns:
            DetailedSolverResult
        """
        img = Path(image_path)
        logger.info("Processing image: %s", img.name)

        # 1. OCR Extraction (Datalab OCR)
        ocr_res: OCRResult = self.ocr_client.extract_text(img)
        initial_text = ocr_res.text

        # 2. Optional Synthetic Noise Injection (Phase 3)
        if inject_synthetic_noise:
            corrupted_text, perts = self.noise_injector.perturb_text(
                initial_text,
                char_rate=noise_rate,
                max_perturbations=max_noise_perturbations,
            )
            logger.info(
                "Injected %d synthetic OCR perturbations into %s",
                len(perts),
                img.name,
            )
            for p in perts:
                logger.debug("  - Mutated '%s' -> '%s' (category: %s)", p.original_char, p.perturbed_char, p.category)
            initial_text = corrupted_text

        # 3. Agent Solve and Refinement Loop (Phase 2)
        result = self.agent.solve(image_path=img, ocr_text=initial_text)
        return result

    def process_directory(
        self,
        dir_path: Union[str, Path],
        pattern: str = "*.png",
        inject_synthetic_noise: bool = False,
    ) -> List[DetailedSolverResult]:
        """Process all matching images in a directory."""
        folder = Path(dir_path)
        images = sorted(list(folder.glob(pattern)))
        logger.info("Found %d images in %s", len(images), folder)

        results: List[DetailedSolverResult] = []
        for img in images:
            res = self.process_image(
                image_path=img,
                inject_synthetic_noise=inject_synthetic_noise,
            )
            results.append(res)
        return results
