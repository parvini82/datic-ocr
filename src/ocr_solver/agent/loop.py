"""Core Refinement and Retry Agent Loop for OCR-Aware Question Solving."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ocr_solver.agent.options import (
    clean_text,
    extract_options_from_text,
    match_answer_against_options,
    normalize_digits_to_ascii,
)
from ocr_solver.agent.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    FALLBACK_USER_PROMPT,
    INITIAL_SOLVE_USER_PROMPT,
    OCR_REFINEMENT_SYSTEM_PROMPT,
    OCR_REFINEMENT_USER_PROMPT,
    RESOLVE_USER_PROMPT,
    SOLVER_SYSTEM_PROMPT,
)
from ocr_solver.agent.solver import VisionLLMClient
from ocr_solver.config import settings
from ocr_solver.models import DetailedSolverResult, QuestionOutput, SolveAttempt

logger = logging.getLogger(__name__)


class OCRAwareSolverAgent:
    """
    Agent implementing the OCR-aware retry and visual refinement loop:
    1. Solve using Image + OCR text.
    2. Check result against multiple-choice options.
    3. Return if matched.
    4. If no match, re-inspect image crop to find OCR errors and produce corrected text.
    5. Re-solve with corrected text.
    6. Terminate on match or when retry cap is reached, with clean best-guess fallback.
    """

    def __init__(
        self,
        llm_client: Optional[VisionLLMClient] = None,
        max_retries: Optional[int] = None,
    ):
        self.llm = llm_client or VisionLLMClient()
        self.max_retries = (
            max_retries if max_retries is not None else settings.MAX_RETRIES
        )

    def solve(
        self,
        image_path: Union[str, Path],
        ocr_text: str,
    ) -> DetailedSolverResult:
        """
        Execute the OCR-aware solving and refinement loop on a question block.

        Args:
            image_path: Path to the scanned question crop image.
            ocr_text: Initial OCR text string.

        Returns:
            DetailedSolverResult containing strict QuestionOutput and audit history.
        """
        img_path = Path(image_path)
        original_ocr = ocr_text
        current_text = ocr_text
        attempts: List[SolveAttempt] = []
        is_changed = False
        options = extract_options_from_text(current_text)

        logger.info("Starting OCR-aware solver on %s (Max retries: %d)", img_path.name, self.max_retries)

        # Step 1: Initial Attempt
        attempt_num = 1
        attempt_result = self._execute_solve_attempt(
            image_path=img_path,
            question_text=current_text,
            attempt_number=attempt_num,
            is_correction=False,
        )
        attempts.append(attempt_result)

        # Step 2 & 3: Check match against options
        if attempt_result.matched_option:
            logger.info(
                "Initial attempt succeeded. Matched option: %s",
                attempt_result.matched_option,
            )
            return DetailedSolverResult(
                output=QuestionOutput(
                    answer=attempt_result.matched_option,
                    question_text=current_text,
                    changed=False,
                    original_ocr_text=original_ocr,
                ),
                is_resolved=True,
                total_attempts=1,
                attempts=attempts,
                image_path=str(img_path),
            )

        # Step 4, 5, 6: Retry & Refinement Loop
        while attempt_num <= self.max_retries:
            logger.warning(
                "Attempt %d failed to match options. Triggering visual OCR refinement.",
                attempt_num,
            )

            # Step 4: Re-read image, identify misreads, and produce corrected text
            correction_res = self._refine_ocr_with_vision(
                image_path=img_path,
                current_text=current_text,
                previous_attempt=attempts[-1],
            )
            corrected_text = correction_res.get("corrected_question_text") or current_text
            notes = correction_res.get("correction_notes", "")

            # Check if text actually changed
            if clean_text(corrected_text) != clean_text(current_text):
                is_changed = True
                current_text = corrected_text
                options = extract_options_from_text(current_text)

            attempt_num += 1

            # Step 5: Re-solve with corrected text
            logger.info("Re-solving with corrected OCR text (Attempt %d)", attempt_num)
            attempt_result = self._execute_solve_attempt(
                image_path=img_path,
                question_text=current_text,
                attempt_number=attempt_num,
                is_correction=True,
                correction_notes=notes,
            )
            attempts.append(attempt_result)

            # Check if answer matched an option
            if attempt_result.matched_option:
                logger.info(
                    "Refinement attempt %d succeeded. Matched option: %s",
                    attempt_num,
                    attempt_result.matched_option,
                )
                return DetailedSolverResult(
                    output=QuestionOutput(
                        answer=attempt_result.matched_option,
                        question_text=current_text,
                        changed=is_changed,
                        original_ocr_text=original_ocr,
                    ),
                    is_resolved=True,
                    total_attempts=attempt_num,
                    attempts=attempts,
                    image_path=str(img_path),
                )

        # Step 6 Fallback: Retry cap reached without clean match
        logger.warning(
            "Retry cap of %d reached without match. Engaging best-guess fallback.",
            self.max_retries,
        )
        best_guess, fallback_reason = self._resolve_fallback(
            image_path=img_path,
            question_text=current_text,
            options=options,
            attempts=attempts,
        )

        return DetailedSolverResult(
            output=QuestionOutput(
                answer=best_guess,
                question_text=current_text,
                changed=is_changed,
                original_ocr_text=original_ocr,
            ),
            is_resolved=False,
            total_attempts=len(attempts),
            attempts=attempts,
            image_path=str(img_path),
            unresolved_reason=fallback_reason,
        )

    def _execute_solve_attempt(
        self,
        image_path: Path,
        question_text: str,
        attempt_number: int,
        is_correction: bool,
        correction_notes: Optional[str] = None,
    ) -> SolveAttempt:
        """Run a single solve attempt using the Vision LLM."""
        if not is_correction:
            user_prompt = INITIAL_SOLVE_USER_PROMPT.format(ocr_text=question_text)
        else:
            user_prompt = RESOLVE_USER_PROMPT.format(
                corrected_text=question_text,
                correction_notes=correction_notes or "Revised via visual inspection",
            )

        try:
            res = self.llm.chat_completion_with_image(
                system_prompt=SOLVER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path=image_path,
            )
        except Exception as e:
            logger.error("LLM solve call failed: %s", e)
            res = {
                "reasoning": f"Execution error: {e}",
                "computed_value": None,
                "matches_option": False,
                "chosen_option_label": None,
                "confidence": 0.0,
            }

        reasoning = res.get("reasoning", "")
        computed_val = res.get("computed_value")
        chosen_label = res.get("chosen_option_label")
        raw_matches = res.get("matches_option", False)

        options = extract_options_from_text(question_text)

        # Match verification
        matched_option = None
        match_conf = 0.0
        if raw_matches and chosen_label:
            matched_option, match_conf = match_answer_against_options(
                computed_value=computed_val,
                options=options,
                raw_answer_label=str(chosen_label),
            )
        elif computed_val:
            matched_option, match_conf = match_answer_against_options(
                computed_value=computed_val,
                options=options,
                raw_answer_label=None,
            )

        return SolveAttempt(
            attempt_number=attempt_number,
            question_text=question_text,
            reasoning=reasoning,
            computed_value=str(computed_val) if computed_val is not None else None,
            matched_option=matched_option,
            match_confidence=match_conf,
            is_correction_attempt=is_correction,
            correction_notes=correction_notes,
        )

    def _refine_ocr_with_vision(
        self,
        image_path: Path,
        current_text: str,
        previous_attempt: SolveAttempt,
    ) -> Dict[str, Any]:
        """Prompt the Vision model to visually inspect image and correct OCR mistakes."""
        mismatch_reason = (
            f"Derived value '{previous_attempt.computed_value}' did not match available options."
        )
        user_prompt = OCR_REFINEMENT_USER_PROMPT.format(
            ocr_text=current_text,
            computed_value=previous_attempt.computed_value or "None",
            mismatch_reason=mismatch_reason,
        )

        try:
            return self.llm.chat_completion_with_image(
                system_prompt=OCR_REFINEMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path=image_path,
                temperature=0.0,
            )
        except Exception as e:
            logger.error("Vision OCR refinement call failed: %s", e)
            return {
                "identified_errors": [str(e)],
                "corrected_question_text": current_text,
                "correction_notes": f"Refinement failed: {e}",
            }

    def _resolve_fallback(
        self,
        image_path: Path,
        question_text: str,
        options: Dict[str, str],
        attempts: List[SolveAttempt],
    ) -> tuple[str, str]:
        """
        Heuristic and LLM fallback strategy when retry cap is reached.
        Picks the closest candidate option based on distance/likelihood.
        """
        attempts_summary = "\n".join(
            f"Attempt {a.attempt_number}: Computed='{a.computed_value}', Reasoning Summary={a.reasoning[:120]}..."
            for a in attempts
        )

        try:
            res = self.llm.chat_completion_text(
                system_prompt=FALLBACK_SYSTEM_PROMPT,
                user_prompt=FALLBACK_USER_PROMPT.format(
                    question_text=question_text,
                    attempts_summary=attempts_summary,
                ),
                temperature=0.0,
            )
            best_guess = str(res.get("best_guess_option", "1"))
            norm_guess = normalize_digits_to_ascii(best_guess)
            if norm_guess in ["1", "2", "3", "4"]:
                best_guess = norm_guess
            else:
                best_guess = "1"
            justification = res.get(
                "fallback_justification",
                "Selected highest likelihood option via fallback heuristics.",
            )
            return best_guess, justification
        except Exception as e:
            logger.error("Fallback heuristic call failed: %s. Using default option 1.", e)
            return "1", f"Fallback default to option 1 due to error: {e}"
