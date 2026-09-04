"""Unit and loop tests for OCR-Aware Solver Agent."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
import httpx
import openai

from ocr_solver.agent.options import (
    extract_options_from_text,
    match_answer_against_options,
    normalize_digits_to_ascii,
)
from ocr_solver.agent.loop import OCRAwareSolverAgent
from ocr_solver.agent.solver import VisionLLMClient
from ocr_solver.models import DetailedSolverResult, QuestionOutput


def test_extract_options_persian_standard():
    """Test extracting standard Persian options in (1), (2), (3), (4) format."""
    text = "سوال تستی ریاضی\n-۱ (۱    -√۵ (۲    ۱ (۳    √۵ (۴"
    options = extract_options_from_text(text)

    assert "1" in options or "۱" in options or "1" in normalize_digits_to_ascii(str(options.keys()))
    assert len(options) == 4


def test_match_answer_exact_and_label():
    """Test matching computed answer or label to options."""
    options = {"1": "-1", "2": "-sqrt(5)", "3": "1", "4": "sqrt(5)"}

    # Direct label match
    matched, conf = match_answer_against_options(None, options, raw_answer_label="2")
    assert matched == "2"
    assert conf > 0.9

    # Value match
    matched, conf = match_answer_against_options("1", options, raw_answer_label=None)
    assert matched == "3"
    assert conf == 1.0


def test_agent_loop_immediate_match(tmp_path: Path):
    """Test agent terminates on attempt 1 when answer matches an option (changed=False)."""
    dummy_img = tmp_path / "q113.png"
    dummy_img.write_bytes(b"dummy")

    mock_llm = MagicMock(spec=VisionLLMClient)
    # LLM returns matching option directly on attempt 1
    mock_llm.chat_completion_with_image.return_value = {
        "reasoning": "Since f(x) is constant, m=0, n=0. Calculating f(sqrt(5)) gives -1.",
        "computed_value": "-1",
        "matches_option": True,
        "chosen_option_label": "1",
        "confidence": 0.95,
    }

    agent = OCRAwareSolverAgent(llm_client=mock_llm, max_retries=3)
    ocr_text = "۱۱۳- تابع f(x) ... -۱ (۱    -√۵ (۲    ۱ (۳    √۵ (۴"

    result: DetailedSolverResult = agent.solve(dummy_img, ocr_text)

    assert result.is_resolved is True
    assert result.total_attempts == 1
    assert result.output.answer == "1"
    assert result.output.changed is False
    assert result.output.original_ocr_text == ocr_text


def test_agent_loop_refinement_recovery(tmp_path: Path):
    """Test agent triggers visual OCR correction and succeeds on attempt 2 (changed=True)."""
    dummy_img = tmp_path / "q115.png"
    dummy_img.write_bytes(b"dummy")

    mock_llm = MagicMock(spec=VisionLLMClient)

    # Attempt 1: Solves corrupted text, result = 7.5 (does not match options 1, 2, 3, 4)
    # Step 4 (OCR correction): Returns corrected text
    # Attempt 2: Re-solves corrected text, result = 3 (matches option 3)
    mock_llm.chat_completion_with_image.side_effect = [
        # Call 1: Attempt 1 Solve
        {
            "reasoning": "Derived value is 7.5 which matches no option.",
            "computed_value": "7.5",
            "matches_option": False,
            "chosen_option_label": None,
            "confidence": 0.3,
        },
        # Call 2: OCR Refinement
        {
            "identified_errors": ["OCR misread 8 as 9 in ax^2 - 8x + 4"],
            "corrected_question_text": "۱۱۵- α و β ریشه‌های ax^۲ - ۸x + ۴ = ۰ ... ۱ (۱  ۲ (۲  ۳ (۳  ۴ (۴",
            "correction_notes": "Fixed coefficient 8",
        },
        # Call 3: Attempt 2 Solve
        {
            "reasoning": "With corrected formula, log_sqrt(2) a = 3.",
            "computed_value": "3",
            "matches_option": True,
            "chosen_option_label": "3",
            "confidence": 0.98,
        },
    ]

    agent = OCRAwareSolverAgent(llm_client=mock_llm, max_retries=3)
    corrupted_ocr_text = "۱۱۵- α و β ریشه‌های ax^۲ - ۹x + ۴ = ۰ ... ۱ (۱  ۲ (۲  ۳ (۳  ۴ (۴"

    result = agent.solve(dummy_img, corrupted_ocr_text)

    assert result.is_resolved is True
    assert result.total_attempts == 2
    assert result.output.answer == "3"
    assert result.output.changed is True
    assert result.output.original_ocr_text == corrupted_ocr_text
    assert "۸x" in result.output.question_text


def test_agent_loop_retry_cap_and_fallback(tmp_path: Path):
    """Test agent hits retry cap and gracefully uses best-guess fallback."""
    dummy_img = tmp_path / "unsolvable.png"
    dummy_img.write_bytes(b"dummy")

    mock_llm = MagicMock(spec=VisionLLMClient)

    # Always return no match
    mock_llm.chat_completion_with_image.return_value = {
        "reasoning": "Inconclusive proof.",
        "computed_value": "999",
        "matches_option": False,
        "chosen_option_label": None,
        "confidence": 0.1,
    }
    # Fallback heuristic call
    mock_llm.chat_completion_text.return_value = {
        "best_guess_option": "2",
        "fallback_justification": "Option 2 is algebraically closest.",
    }

    agent = OCRAwareSolverAgent(llm_client=mock_llm, max_retries=2)
    ocr_text = "سوال مبهم ۱ (۱  ۲ (۲  ۳ (۳  ۴ (۴"

    result = agent.solve(dummy_img, ocr_text)

    assert result.is_resolved is False
    assert result.output.answer == "2"
    assert result.unresolved_reason is not None


def test_agent_aborts_on_auth_error(tmp_path: Path):
    """Test that authentication errors immediately bubble up and abort without retrying."""
    dummy_img = tmp_path / "q113.png"
    dummy_img.write_bytes(b"dummy")

    mock_llm = MagicMock(spec=VisionLLMClient)
    # Simulate a 401 AuthenticationError from OpenAI/OpenRouter
    mock_request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    mock_response = httpx.Response(401, request=mock_request, json={"error": {"message": "Invalid API Key"}})
    mock_llm.chat_completion_with_image.side_effect = openai.AuthenticationError(
        message="Invalid API Key", response=mock_response, body=None
    )

    agent = OCRAwareSolverAgent(llm_client=mock_llm, max_retries=3)
    ocr_text = "۱۱۳- تابع ... -۱ (۱    -√۵ (۲    ۱ (۳    √۵ (۴"

    with pytest.raises(openai.AuthenticationError):
        agent.solve(dummy_img, ocr_text)

    # Verify that it only called once and did NOT attempt to retry/refine
    assert mock_llm.chat_completion_with_image.call_count == 1
