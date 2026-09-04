"""OCR Module initialization."""

from ocr_solver.ocr.base import BaseOCRClient
from ocr_solver.ocr.datalab import (
    DatalabOCRClient,
    DatalabAPIError,
    DatalabAuthError,
    DatalabRateLimitError,
    SAMPLE_MOCK_OCR,
)

__all__ = [
    "BaseOCRClient",
    "DatalabOCRClient",
    "DatalabAPIError",
    "DatalabAuthError",
    "DatalabRateLimitError",
    "SAMPLE_MOCK_OCR",
]
