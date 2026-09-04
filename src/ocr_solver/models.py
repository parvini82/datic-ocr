"""Data Models and Pydantic Schemas for OCR-Aware Solver."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Raw and processed output from an OCR engine."""

    text: str = Field(description="Extracted OCR text")
    provider: str = Field(default="datalab", description="OCR provider name")
    success: bool = Field(default=True, description="Whether the OCR request succeeded")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw API response data")
    error: Optional[str] = Field(default=None, description="Error message if OCR failed")


class Option(BaseModel):
    """Single multiple-choice option."""

    label: str = Field(description="Option label, e.g., '1', '2', '3', '4' or 'A', 'B', 'C', 'D'")
    value: str = Field(description="Text or formula content of the option")


class SolveAttempt(BaseModel):
    """Record of a single solving attempt in the refinement loop."""

    attempt_number: int = Field(description="1-indexed attempt number")
    question_text: str = Field(description="The question text used for this solve attempt")
    reasoning: str = Field(description="Step-by-step mathematical reasoning")
    computed_value: Optional[str] = Field(default=None, description="The derived result/value from reasoning")
    matched_option: Optional[str] = Field(default=None, description="Matched option label, or None if no match")
    match_confidence: float = Field(default=0.0, description="Confidence score of the option match (0.0 - 1.0)")
    is_correction_attempt: bool = Field(default=False, description="True if this attempt used revised/corrected OCR text")
    correction_notes: Optional[str] = Field(default=None, description="Notes on why and how OCR text was revised")


class QuestionOutput(BaseModel):
    """Strict final JSON output schema required by Phase 4."""

    answer: str = Field(description="Selected option label (e.g., '1', '2', '3', '4' or 'A', 'B', 'C', 'D')")
    question_text: str = Field(description="Final (possibly corrected) question text")
    changed: bool = Field(description="True if the question text was corrected/modified from the original OCR")
    original_ocr_text: str = Field(description="The original OCR text before any refinement")


class DetailedSolverResult(BaseModel):
    """Detailed solver result containing full audit trail, attempts, and strict output."""

    output: QuestionOutput = Field(description="The strict final JSON output")
    is_resolved: bool = Field(description="True if an option matched cleanly; False if resolved by fallback")
    total_attempts: int = Field(description="Total number of attempts executed")
    attempts: List[SolveAttempt] = Field(default_factory=list, description="Audit log of all solver attempts")
    image_path: Optional[str] = Field(default=None, description="Path to the question crop image")
    unresolved_reason: Optional[str] = Field(default=None, description="Explanation if retry cap was reached without clean match")
