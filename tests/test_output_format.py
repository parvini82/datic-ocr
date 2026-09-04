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


def test_generate_html_report(tmp_path):
    """Verify HTML report generation includes RTL direction, Vazirmatn font, and required card fields."""
    from ocr_solver.formatter import generate_html_report
    from pathlib import Path

    sample_results = [
        {
            "image_name": "q113.png",
            "answer": "3",
            "question_text": "متن نهایی سوال ۱۱۳",
            "changed": False,
            "original_ocr_text": "متن اولیه OCR سوال ۱۱۳",
        },
        {
            "image_name": "q115.png",
            "answer": "2",
            "question_text": "متن اصلاح شده سوال ۱۱۵",
            "changed": True,
            "original_ocr_text": "متن دارای خطای OCR سوال ۱۱۵",
        },
    ]

    report_file = tmp_path / "report.html"
    html = generate_html_report(sample_results, str(report_file))

    assert report_file.exists()
    assert 'dir="rtl"' in html
    assert "Vazirmatn" in html
    assert "q113.png" in html
    assert "q115.png" in html
    assert "متن اولیه OCR" in html
    assert "متن نهایی سوال" in html
    assert "گزینه 3" in html
    assert "گزینه 2" in html
    assert "تغییر یافته" in html


def test_extract_json_with_latex_escapes():
    """Verify JSON extractor correctly recovers unescaped LaTeX backslashes from LLM."""
    from ocr_solver.agent.solver import extract_json_from_response

    # Simulated LLM output with unescaped LaTeX backslashes: \{, \implies, \sqrt, etc.
    llm_raw = (
        '{\n'
        '  "reasoning": "f(x) is constant. S = \\{(m, n-1), (0, k)\\}. -1 = k \\implies k = -1. f(\\sqrt{5}) = 1.",\n'
        '  "computed_value": "1",\n'
        '  "matches_option": true,\n'
        '  "chosen_option_label": "3",\n'
        '  "confidence": 1.0\n'
        '}'
    )

    parsed = extract_json_from_response(llm_raw)
    assert parsed["chosen_option_label"] == "3"
    assert parsed["computed_value"] == "1"
    assert parsed["matches_option"] is True
    assert "\\sqrt{5}" in parsed["reasoning"] or "sqrt{5}" in parsed["reasoning"]

