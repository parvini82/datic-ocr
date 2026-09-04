"""Noise Injection Module."""

from ocr_solver.noise.persian_noise import (
    PersianOCRNoiseInjector,
    NoisePerturbation,
    NoiseCategory,
    inject_noise,
    DIGIT_CONFUSIONS,
    DOT_CONFUSIONS,
    SHAPE_CONFUSIONS,
    MATH_CONFUSIONS,
)

__all__ = [
    "PersianOCRNoiseInjector",
    "NoisePerturbation",
    "NoiseCategory",
    "inject_noise",
    "DIGIT_CONFUSIONS",
    "DOT_CONFUSIONS",
    "SHAPE_CONFUSIONS",
    "MATH_CONFUSIONS",
]
