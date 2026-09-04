"""Abstract Base Class for OCR Clients."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from ocr_solver.models import OCRResult


class BaseOCRClient(ABC):
    """Abstract interface for OCR providers."""

    @abstractmethod
    def extract_text(self, image_path: Union[str, Path]) -> OCRResult:
        """Extract text from an image synchronously."""
        pass

    @abstractmethod
    async def extract_text_async(self, image_path: Union[str, Path]) -> OCRResult:
        """Extract text from an image asynchronously."""
        pass
