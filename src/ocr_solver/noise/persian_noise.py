"""Synthetic Persian OCR Noise Injection Engine.

Simulates realistic Persian OCR degradation by perturbing look-alike Persian letters,
confusable Persian digits, diacritics/dots, and common mathematical notation artifacts.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class NoiseCategory(str, Enum):
    DIGIT = "digit"
    LETTER_DOTS = "letter_dots"
    LETTER_SHAPE = "letter_shape"
    MATH_SYMBOL = "math_symbol"


@dataclass
class NoisePerturbation:
    """Record of an applied synthetic perturbation."""
    index: int
    original_char: str
    perturbed_char: str
    category: NoiseCategory


# Character confusion maps reflecting Persian OCR misidentifications
DIGIT_CONFUSIONS: Dict[str, List[str]] = {
    "۲": ["۳", "۲"],       # 2 vs 3 (similar teeth)
    "۳": ["۲", "۳"],
    "۴": ["۶", "۴"],       # 4 vs 6 (similar curve in handwritten/low-res)
    "۶": ["۴", "۶"],
    "۰": ["۵", "."],       # 0 (dot) vs 5 (circle) or period
    "۵": ["۰", "۵"],
    "۱": ["۹", "۱"],       # 1 vs 9
    "۹": ["۱", "۹"],
    "۷": ["۸", "۷"],       # 7 vs 8 (inverted chevron)
    "۸": ["۷", "۸"],
    "2": ["3"],
    "3": ["2"],
    "4": ["6"],
    "6": ["4"],
}

# Persian letters differing only by dot counts / positions
DOT_CONFUSIONS: Dict[str, List[str]] = {
    "ب": ["پ", "ت", "ث"],
    "پ": ["ب", "ت"],
    "ت": ["ب", "ث", "پ"],
    "ث": ["ت", "ب"],
    "ج": ["چ", "ح", "خ"],
    "چ": ["ج", "ح", "خ"],
    "ح": ["خ", "ج"],
    "خ": ["ح", "ج"],
    "د": ["ذ"],
    "ذ": ["د"],
    "ر": ["ز", "ژ"],
    "ز": ["ر", "ژ"],
    "ژ": ["ز", "ر"],
    "س": ["ش"],
    "ش": ["س"],
    "ص": ["ض"],
    "ض": ["ص"],
    "ط": ["ظ"],
    "ظ": ["ط"],
    "ع": ["غ"],
    "غ": ["ع"],
    "ف": ["ق"],
    "ق": ["ف"],
}

# Persian letters differing by minor stroke/shape features
SHAPE_CONFUSIONS: Dict[str, List[str]] = {
    "ک": ["گ"],
    "گ": ["ک"],
    "ی": ["ى", "ئ"],
    "ئ": ["ی"],
    "ه": ["ة", "ه"],
    "ا": ["آ", "۱", "l"],
    "آ": ["ا"],
}

# OCR corruptions on mathematical expressions
MATH_CONFUSIONS: Dict[str, List[str]] = {
    "+": ["-", "±"],
    "-": ["+", "_"],
    "=": ["-", "≈"],
    "√": ["r", "/"],
    "α": ["a", "∝"],
    "β": ["B", "8"],
    "^۲": ["۲", "^۳"],
    "^2": ["2", "^3"],
}


class PersianOCRNoiseInjector:
    """Injects realistic OCR degradation into Persian text."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def perturb_text(
        self,
        text: str,
        char_rate: float = 0.05,
        max_perturbations: int = 4,
        include_digits: bool = True,
        include_letters: bool = True,
        include_math: bool = True,
    ) -> Tuple[str, List[NoisePerturbation]]:
        """
        Perturb text by probabilistically replacing eligible characters with their OCR look-alikes.

        Args:
            text: Input Persian text string.
            char_rate: Probability of mutating an eligible character.
            max_perturbations: Strict upper bound on number of perturbations per block.
            include_digits: Whether to mutate Persian digits.
            include_letters: Whether to mutate Persian letters (dots & shapes).
            include_math: Whether to mutate math operators/symbols.

        Returns:
            Tuple of (perturbed_text, list_of_perturbations).
        """
        if not text or max_perturbations <= 0:
            return text, []

        chars = list(text)
        candidates: List[Tuple[int, str, NoiseCategory]] = []

        # Find all replaceable positions
        for i, ch in enumerate(chars):
            if include_digits and ch in DIGIT_CONFUSIONS:
                candidates.append((i, ch, NoiseCategory.DIGIT))
            elif include_letters and ch in DOT_CONFUSIONS:
                candidates.append((i, ch, NoiseCategory.LETTER_DOTS))
            elif include_letters and ch in SHAPE_CONFUSIONS:
                candidates.append((i, ch, NoiseCategory.LETTER_SHAPE))
            elif include_math and ch in MATH_CONFUSIONS:
                candidates.append((i, ch, NoiseCategory.MATH_SYMBOL))

        if not candidates:
            return text, []

        # Shuffle candidates and select up to max_perturbations based on rate
        self.rng.shuffle(candidates)
        perturbations: List[NoisePerturbation] = []

        for idx, orig_ch, category in candidates:
            if len(perturbations) >= max_perturbations:
                break

            if self.rng.random() > char_rate and len(perturbations) > 0:
                continue

            replacement_pool: List[str] = []
            if category == NoiseCategory.DIGIT:
                replacement_pool = [c for c in DIGIT_CONFUSIONS[orig_ch] if c != orig_ch] or DIGIT_CONFUSIONS[orig_ch]
            elif category == NoiseCategory.LETTER_DOTS:
                replacement_pool = [c for c in DOT_CONFUSIONS[orig_ch] if c != orig_ch]
            elif category == NoiseCategory.LETTER_SHAPE:
                replacement_pool = [c for c in SHAPE_CONFUSIONS[orig_ch] if c != orig_ch]
            elif category == NoiseCategory.MATH_SYMBOL:
                replacement_pool = [c for c in MATH_CONFUSIONS[orig_ch] if c != orig_ch]

            if not replacement_pool:
                continue

            chosen_replacement = self.rng.choice(replacement_pool)
            chars[idx] = chosen_replacement
            perturbations.append(
                NoisePerturbation(
                    index=idx,
                    original_char=orig_ch,
                    perturbed_char=chosen_replacement,
                    category=category,
                )
            )

        perturbed_text = "".join(chars)
        return perturbed_text, perturbations

    def perturb_digits_only(
        self, text: str, count: int = 1
    ) -> Tuple[str, List[NoisePerturbation]]:
        """Convenience method to corrupt exactly N digits (e.g. swap ۲ -> ۳)."""
        return self.perturb_text(
            text=text,
            char_rate=1.0,
            max_perturbations=count,
            include_digits=True,
            include_letters=False,
            include_math=False,
        )

    def perturb_letters_only(
        self, text: str, count: int = 1
    ) -> Tuple[str, List[NoisePerturbation]]:
        """Convenience method to corrupt exactly N letters (e.g. swap پ -> ب or ک -> گ)."""
        return self.perturb_text(
            text=text,
            char_rate=1.0,
            max_perturbations=count,
            include_digits=False,
            include_letters=True,
            include_math=False,
        )


def inject_noise(
    text: str,
    char_rate: float = 0.08,
    max_perturbations: int = 3,
    seed: Optional[int] = 42,
) -> Tuple[str, List[NoisePerturbation]]:
    """Functional interface for injecting synthetic Persian OCR noise."""
    injector = PersianOCRNoiseInjector(seed=seed)
    return injector.perturb_text(
        text=text,
        char_rate=char_rate,
        max_perturbations=max_perturbations,
    )
