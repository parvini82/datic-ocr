"""Unit tests for OCR clients and Datalab integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ocr_solver.ocr.datalab import (
    DatalabOCRClient,
    DatalabAuthError,
    DatalabRateLimitError,
    SAMPLE_MOCK_OCR,
)
from ocr_solver.models import OCRResult


def test_datalab_mock_fallback(tmp_path: Path):
    """Test that mock fallback returns curated OCR text for known sample names."""
    sample_file = tmp_path / "q113.png"
    sample_file.write_bytes(b"dummy image data")

    client = DatalabOCRClient(api_key=None, enable_mock_fallback=True)
    result = client.extract_text(sample_file)

    assert result.success is True
    assert result.provider == "datalab-mock"
    assert "mx^۲" in result.text
    assert "-۱ (۱" in result.text


def test_datalab_missing_api_key_no_fallback(tmp_path: Path):
    """Test that DatalabAuthError is raised when no API key and fallback disabled."""
    sample_file = tmp_path / "custom.png"
    sample_file.write_bytes(b"dummy image data")

    client = DatalabOCRClient(api_key="", enable_mock_fallback=False)
    with pytest.raises(DatalabAuthError):
        client.extract_text(sample_file)


def test_datalab_file_not_found():
    """Test FileNotFoundError for non-existent file path."""
    client = DatalabOCRClient(api_key="mock_key")
    with pytest.raises(FileNotFoundError):
        client.extract_text("non_existent_image.png")


@patch("httpx.Client.post")
def test_datalab_direct_api_success(mock_post, tmp_path: Path):
    """Test successful direct OCR response from Datalab API."""
    sample_file = tmp_path / "test.png"
    sample_file.write_bytes(b"dummy image data")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "markdown": "Sample extracted Persian text: ۱۲۳",
        "status": "complete",
    }
    mock_post.return_value = mock_response

    client = DatalabOCRClient(api_key="valid_test_key", enable_mock_fallback=False)
    result = client.extract_text(sample_file)

    assert result.success is True
    assert result.provider == "datalab"
    assert result.text == "Sample extracted Persian text: ۱۲۳"


@patch("httpx.Client.post")
def test_datalab_rate_limit_error(mock_post, tmp_path: Path):
    """Test that HTTP 429 raises DatalabRateLimitError when fallback disabled."""
    sample_file = tmp_path / "test.png"
    sample_file.write_bytes(b"dummy image data")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_post.return_value = mock_response

    client = DatalabOCRClient(api_key="valid_test_key", enable_mock_fallback=False)
    with pytest.raises(DatalabRateLimitError):
        client.extract_text(sample_file)


@patch("httpx.Client.get")
@patch("httpx.Client.post")
def test_datalab_polling_flow(mock_post, mock_get, tmp_path: Path):
    """Test asynchronous check_url polling flow matching Datalab Convert API."""
    sample_file = tmp_path / "test.png"
    sample_file.write_bytes(b"dummy image data")

    # Initial POST returns request_check_url
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {
        "success": True,
        "request_id": "req_123",
        "request_check_url": "https://www.datalab.to/api/v1/convert/req_123",
    }
    mock_post.return_value = mock_post_resp

    # First GET returns processing, second returns complete
    mock_get_resp1 = MagicMock()
    mock_get_resp1.status_code = 200
    mock_get_resp1.json.return_value = {"status": "processing"}

    mock_get_resp2 = MagicMock()
    mock_get_resp2.status_code = 200
    mock_get_resp2.json.return_value = {
        "status": "complete",
        "success": True,
        "markdown": "Converted markdown content",
    }
    mock_get.side_effect = [mock_get_resp1, mock_get_resp2]

    client = DatalabOCRClient(api_key="valid_key", enable_mock_fallback=False)
    result = client.extract_text(sample_file)

    assert result.success is True
    assert result.text == "Converted markdown content"
    assert result.provider == "datalab"


def test_datalab_header_format():
    """Verify X-API-Key header formatting."""
    client = DatalabOCRClient(api_key="my_secret_key")
    headers = client._get_headers()
    assert headers == {"X-API-Key": "my_secret_key"}

