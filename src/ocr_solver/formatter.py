"""Output Formatting and Schema Enforcement for Phase 4."""

import json
from typing import Any, Dict, List, Union
from ocr_solver.models import DetailedSolverResult, QuestionOutput


def format_single_result(result: Union[DetailedSolverResult, QuestionOutput]) -> Dict[str, Any]:
    """
    Format a solver result into the exact strict JSON dictionary specified in Phase 4:
    {
      "answer": "...",
      "question_text": "...",
      "changed": true/false,
      "original_ocr_text": "..."
    }
    """
    if isinstance(result, DetailedSolverResult):
        output = result.output
    else:
        output = result

    return {
        "answer": str(output.answer),
        "question_text": str(output.question_text),
        "changed": bool(output.changed),
        "original_ocr_text": str(output.original_ocr_text),
    }


def to_json_string(
    data: Union[DetailedSolverResult, QuestionOutput, List[Union[DetailedSolverResult, QuestionOutput]]],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> str:
    """Serialize result or list of results to pretty JSON string with Persian unicode preserved."""
    if isinstance(data, list):
        formatted = [format_single_result(item) for item in data]
    else:
        formatted = format_single_result(data)

    return json.dumps(formatted, indent=indent, ensure_ascii=ensure_ascii)
