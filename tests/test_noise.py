"""Unit tests for synthetic Persian OCR noise generator."""

import pytest
from ocr_solver.noise.persian_noise import (
    PersianOCRNoiseInjector,
    NoiseCategory,
    inject_noise,
    DIGIT_CONFUSIONS,
    DOT_CONFUSIONS,
    SHAPE_CONFUSIONS,
)


def test_digit_noise_perturbation():
    """Test that Persian digits are perturbed to lookalike digits."""
    injector = PersianOCRNoiseInjector(seed=42)
    original_text = "مقدار f(۲) برابر ۴ است."
    perturbed, perts = injector.perturb_digits_only(original_text, count=2)

    assert len(perts) > 0
    assert len(perts) <= 2
    for p in perts:
        assert p.category == NoiseCategory.DIGIT
        assert p.original_char in ["۲", "۴"]
        assert p.perturbed_char in DIGIT_CONFUSIONS[p.original_char]


def test_letter_noise_perturbation():
    """Test that Persian letters with dots or lookalike shapes are perturbed."""
    injector = PersianOCRNoiseInjector(seed=42)
    original_text = "پاسخ صحیح گزینه ب است."
    perturbed, perts = injector.perturb_letters_only(original_text, count=2)

    assert len(perts) > 0
    assert len(perts) <= 2
    for p in perts:
        assert p.category in (NoiseCategory.LETTER_DOTS, NoiseCategory.LETTER_SHAPE)
        assert p.original_char != p.perturbed_char


def test_noise_rate_and_bounds():
    """Test that max_perturbations bound is strictly respected."""
    injector = PersianOCRNoiseInjector(seed=123)
    text = "۱۱۵- α و β ریشه‌های معادله ax^۲ - ۸x + ۴ = ۰ است. ۱ (۱  ۲ (۲  ۳ (۳  ۴ (۴"

    perturbed, perts = injector.perturb_text(text, char_rate=1.0, max_perturbations=3)
    assert len(perts) <= 3
    assert perturbed != text


def test_noise_zero_perturbations():
    """Test zero perturbations when rate is 0 or max_perturbations is 0."""
    injector = PersianOCRNoiseInjector(seed=42)
    text = "متن بدون تغییر"
    perturbed, perts = injector.perturb_text(text, max_perturbations=0)
    assert perturbed == text
    assert len(perts) == 0


def test_deterministic_seed():
    """Test that identical seeds yield identical perturbations."""
    text = "تابع f(x) در نقطه ۲ و ۳ پیوسته است."
    p1, perts1 = inject_noise(text, char_rate=0.5, max_perturbations=3, seed=99)
    p2, perts2 = inject_noise(text, char_rate=0.5, max_perturbations=3, seed=99)

    assert p1 == p2
    assert len(perts1) == len(perts2)
    assert [p.index for p in perts1] == [p.index for p in perts2]
