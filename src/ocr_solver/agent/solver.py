"""LLM and Vision Model Client Wrapper."""

import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI

from ocr_solver.config import settings

logger = logging.getLogger(__name__)


def encode_image_to_data_url(image_path: Union[str, Path]) -> str:
    """Read an image file and convert it to a base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/png"

    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"


def extract_json_from_response(content: str) -> Dict[str, Any]:
    """Extract and parse JSON object from LLM text response."""
    content = content.strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try markdown json code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding outermost braces { ... }
    brace_match = re.search(r"\{[\s\S]*\}", content)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse valid JSON from LLM response:\n{content}")


class VisionLLMClient:
    """Client for invoking Vision and Reasoning models via OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        solver_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.solver_model = solver_model or settings.SOLVER_MODEL
        self.vision_model = vision_model or settings.VISION_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.TEMPERATURE
        )

        self._client: Optional[OpenAI] = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if self.api_key:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                raise ValueError("OPENAI_API_KEY is not configured in settings or environment.")
        return self._client

    def chat_completion_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a prompt accompanied by an image to the vision model and parse JSON output."""
        image_data_url = encode_image_to_data_url(image_path)
        chosen_model = model or self.vision_model
        chosen_temp = temperature if temperature is not None else self.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "high"},
                    },
                ],
            },
        ]

        logger.debug("Calling Vision Model %s with image %s", chosen_model, image_path)
        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=messages,  # type: ignore
            temperature=chosen_temp,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content or "{}"
        return extract_json_from_response(response_text)

    def chat_completion_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a text-only prompt and parse JSON output."""
        chosen_model = model or self.solver_model
        chosen_temp = temperature if temperature is not None else self.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=messages,  # type: ignore
            temperature=chosen_temp,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content or "{}"
        return extract_json_from_response(response_text)
