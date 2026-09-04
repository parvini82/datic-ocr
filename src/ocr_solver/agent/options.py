"""Option Parsing, Normalization, and Verification Utilities."""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from ocr_solver.models import Option

# Mapping of Persian and Arabic digits to standard ASCII digits
PERSIAN_DIGITS_MAP = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

ASCII_TO_PERSIAN_MAP = {v: k for k, v in PERSIAN_DIGITS_MAP.items() if k in "۰۱۲۳۴۵۶۷۸۹"}


def normalize_digits_to_ascii(text: str) -> str:
    """Convert Persian/Arabic digits to ASCII digits."""
    return "".join(PERSIAN_DIGITS_MAP.get(ch, ch) for ch in text)


def normalize_digits_to_persian(text: str) -> str:
    """Convert ASCII digits to standard Persian digits."""
    return "".join(ASCII_TO_PERSIAN_MAP.get(ch, ch) for ch in text)


def clean_text(text: str) -> str:
    """Normalize Unicode and clean excessive whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", " ")  # Zero-width non-joiner normalization
    return re.sub(r"\s+", " ", text).strip()


def extract_options_from_text(text: str) -> Dict[str, str]:
    """
    Extract multiple choice options (1, 2, 3, 4) from Persian exam text.
    Handles formats such as:
      - '-۱ (۱    -√۵ (۲    ۱ (۳    √۵ (۴'
      - '(۱) -۱   (۲) -√۵   (۳) ۱   (۴) √۵'
      - '1) -1   2) -√5   3) 1   4) √5'
      - '1. -1   2. -√5   3. 1   4. √5'
    """
    options: Dict[str, str] = {}
    normalized = normalize_digits_to_ascii(text)

    # Strategy 1: Look for trailing option labels like "value (1" or "value (1)" or "value [1]"
    # Matches: non-parenthesis text followed by (1) or (1
    pattern_trailing = re.findall(r"([^\(\)\[\]\n]+?)\s*[\(\[]([1-4])[\)\]]?", normalized)
    if len(pattern_trailing) >= 2:
        for val, label in pattern_trailing:
            val_cleaned = clean_text(val)
            if val_cleaned and label not in options:
                options[label.strip()] = val_cleaned
        if len(options) >= 2:
            return options

    # Strategy 2: Look for leading option labels like "(1) value" or "1) value" or "1. value"
    pattern_leading = re.findall(
        r"(?:[\(\[]([1-4])[\)\]]|([1-4])[\)\.-])\s*([^\(\)\[\]1-4\n]+)",
        normalized,
    )
    if len(pattern_leading) >= 2:
        for match in pattern_leading:
            label = (match[0] or match[1]).strip()
            val = clean_text(match[2])
            if label and val:
                options[label] = val
        if len(options) >= 2:
            return options

    return options


def match_answer_against_options(
    computed_value: Optional[str],
    options: Dict[str, str],
    raw_answer_label: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """
    Check if a computed answer or raw label matches any available option.
    Uses exact matching first, followed by mathematical equivalence, then containment.

    Returns:
        Tuple of (matched_option_label, confidence). If no match, returns (None, 0.0).
    """
    # 1. Direct label match (e.g. model explicitly chose option "1", "2", "3", "4" or "A", "B", "C", "D")
    if raw_answer_label:
        norm_label = normalize_digits_to_ascii(str(raw_answer_label)).strip().upper()
        if norm_label in options or norm_label in ["1", "2", "3", "4"]:
            return norm_label, 0.95

        letter_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
        if norm_label in letter_map:
            return letter_map[norm_label], 0.95

    if not computed_value:
        return None, 0.0

    comp_clean = clean_text(normalize_digits_to_ascii(computed_value)).lower()

    # Pass 1: Exact string match
    for label, val in options.items():
        val_clean = clean_text(normalize_digits_to_ascii(val)).lower()
        if comp_clean == val_clean:
            return label, 1.0

    # Pass 2: Exact numerical / float match or zero equivalence
    for label, val in options.items():
        val_clean = clean_text(normalize_digits_to_ascii(val)).lower()

        if comp_clean in ["0", "zero"] and ("صفر" in val or val_clean == "0"):
            return label, 1.0

        try:
            f_comp = float(comp_clean)
            f_val = float(val_clean)
            if abs(f_comp - f_val) < 1e-4:
                return label, 1.0
        except ValueError:
            pass

    # Pass 3: Substring / expression match (only when length is substantial)
    for label, val in options.items():
        val_clean = clean_text(normalize_digits_to_ascii(val)).lower()
        if len(comp_clean) > 2 and (comp_clean in val_clean or val_clean in comp_clean):
            return label, 0.85

    return None, 0.0
