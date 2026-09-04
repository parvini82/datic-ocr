"""Strict JSON schema validation tests for Phase 4 output."""

import json
from ocr_solver.formatter import format_single_result, to_json_string
from ocr_solver.models import DetailedSolverResult, QuestionOutput, SolveAttempt


def test_question_output_schema():
    """Verify QuestionOutput matches the exact 4 fields required by Phase 4."""
    out = QuestionOutput(
        answer="C",
        question_text="سوال نمونه تستی",
        changed=True,
        original_ocr_text="متن قدیمی OCR",
    )

    formatted = format_single_result(out)

    # Exactly 4 keys
    assert set(formatted.keys()) == {"answer", "question_text", "changed", "original_ocr_text"}
    assert isinstance(formatted["answer"], str)
    assert isinstance(formatted["question_text"], str)
    assert isinstance(formatted["changed"], bool)
    assert isinstance(formatted["original_ocr_text"], str)
    assert formatted["answer"] == "C"
    assert formatted["changed"] is True


def test_json_string_serialization():
    """Verify JSON string formatting preserves Persian unicode and exact structure."""
    out = QuestionOutput(
        answer="3",
        question_text="مقدار f(√۵) کدام است؟",
        changed=False,
        original_ocr_text="مقدار f(√۵) کدام است؟",
    )

    json_str = to_json_string(out)
    parsed = json.loads(json_str)

    assert parsed["answer"] == "3"
    assert "√۵" in parsed["question_text"]
    assert parsed["changed"] is False
    assert parsed["original_ocr_text"] == parsed["question_text"]


def test_detailed_result_formatting():
    """Verify DetailedSolverResult correctly delegates to format_single_result."""
    detailed = DetailedSolverResult(
        output=QuestionOutput(
            answer="1",
            question_text="متن اصلاح شده",
            changed=True,
            original_ocr_text="متن خراب",
        ),
        is_resolved=True,
        total_attempts=2,
    )

    formatted = format_single_result(detailed)
    assert set(formatted.keys()) == {"answer", "question_text", "changed", "original_ocr_text"}
    assert formatted["changed"] is True
