"""Tests for the multimodal helper module (image loading, provider conversion)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from checkllm import multimodal
from checkllm.models import JudgeResponse
from checkllm.multimodal import (
    ImagePayload,
    call_vision_judge,
    load_image,
    load_images,
    to_anthropic_content,
    to_gemini_part,
    to_openai_content,
)


PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture(scope="session")
def tiny_png_bytes() -> bytes:
    """Return bytes for a tiny in-memory PNG (red 8x8 square)."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_jpeg_bytes() -> bytes:
    """Return bytes for a tiny in-memory JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(0, 128, 255)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    """Return a temporary directory where per-session image files live."""
    return tmp_path_factory.mktemp("image_fixtures")


@pytest.fixture(scope="session")
def tiny_png_path(fixtures_dir: Path, tiny_png_bytes: bytes) -> Path:
    """Write tiny_png_bytes to a file and return its path."""
    path = fixtures_dir / "tiny.png"
    path.write_bytes(tiny_png_bytes)
    return path


class TestImagePayload:
    def test_requires_url_or_data(self):
        with pytest.raises(ValueError):
            ImagePayload()

    def test_rejects_both_url_and_data(self):
        with pytest.raises(ValueError):
            ImagePayload(url="http://x", data=b"abc", mime_type="image/png")

    def test_requires_mime_for_data(self):
        with pytest.raises(ValueError):
            ImagePayload(data=b"abc")

    def test_is_url_true_for_url(self):
        assert ImagePayload(url="https://example.com/a.png").is_url

    def test_is_url_false_for_data(self):
        assert not ImagePayload(data=b"abc", mime_type="image/png").is_url

    def test_to_base64(self):
        p = ImagePayload(data=b"hello", mime_type="image/png")
        assert base64.b64decode(p.to_base64()) == b"hello"

    def test_to_data_url_for_data(self):
        p = ImagePayload(data=b"hello", mime_type="image/png")
        assert p.to_data_url().startswith("data:image/png;base64,")

    def test_to_data_url_for_url(self):
        p = ImagePayload(url="https://example.com/a.png")
        assert p.to_data_url() == "https://example.com/a.png"

    def test_to_base64_fails_for_url(self):
        with pytest.raises(ValueError):
            ImagePayload(url="https://x").to_base64()


class TestLoadImage:
    def test_from_path(self, tiny_png_path: Path):
        payload = load_image(tiny_png_path)
        assert payload.data is not None
        assert payload.mime_type == "image/png"

    def test_from_path_string(self, tiny_png_path: Path):
        payload = load_image(str(tiny_png_path))
        assert payload.mime_type == "image/png"

    def test_missing_path_raises(self, fixtures_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_image(fixtures_dir / "does-not-exist.png")

    def test_from_bytes(self, tiny_png_bytes: bytes):
        payload = load_image(tiny_png_bytes)
        assert payload.data == tiny_png_bytes
        assert payload.mime_type == "image/png"

    def test_from_jpeg_bytes(self, tiny_jpeg_bytes: bytes):
        payload = load_image(tiny_jpeg_bytes)
        assert payload.mime_type == "image/jpeg"

    def test_from_data_url(self, tiny_png_bytes: bytes):
        b64 = base64.b64encode(tiny_png_bytes).decode("ascii")
        payload = load_image(f"data:image/png;base64,{b64}")
        assert payload.data == tiny_png_bytes
        assert payload.mime_type == "image/png"

    def test_from_https_url(self):
        payload = load_image("https://example.com/a.png")
        assert payload.is_url
        assert payload.url == "https://example.com/a.png"

    def test_from_base64_string(self, tiny_png_bytes: bytes):
        payload = load_image(base64.b64encode(tiny_png_bytes).decode("ascii"))
        assert payload.mime_type == "image/png"
        assert payload.data == tiny_png_bytes

    def test_from_file_like(self, tiny_png_bytes: bytes):
        payload = load_image(io.BytesIO(tiny_png_bytes))
        assert payload.data == tiny_png_bytes
        assert payload.mime_type == "image/png"

    def test_passes_through_payload(self):
        original = ImagePayload(url="https://example.com/x.png")
        assert load_image(original) is original

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            load_image(12345)  # type: ignore[arg-type]

    def test_load_images_list(self, tiny_png_bytes: bytes):
        payloads = load_images([tiny_png_bytes, tiny_png_bytes])
        assert len(payloads) == 2
        assert all(p.mime_type == "image/png" for p in payloads)


class TestProviderConverters:
    def test_openai_url(self):
        p = ImagePayload(url="https://example.com/a.png")
        block = to_openai_content(p)
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/a.png"

    def test_openai_data(self, tiny_png_bytes: bytes):
        p = ImagePayload(data=tiny_png_bytes, mime_type="image/png")
        block = to_openai_content(p)
        assert block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_anthropic_url(self):
        p = ImagePayload(url="https://example.com/a.png")
        block = to_anthropic_content(p)
        assert block["type"] == "image"
        assert block["source"]["type"] == "url"

    def test_anthropic_base64(self, tiny_png_bytes: bytes):
        p = ImagePayload(data=tiny_png_bytes, mime_type="image/png")
        block = to_anthropic_content(p)
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert base64.b64decode(block["source"]["data"]) == tiny_png_bytes

    def test_gemini_inline(self, tiny_png_bytes: bytes):
        p = ImagePayload(data=tiny_png_bytes, mime_type="image/png")
        part = to_gemini_part(p)
        assert part["inline_data"]["mime_type"] == "image/png"

    def test_gemini_url(self):
        p = ImagePayload(url="https://example.com/a.png")
        part = to_gemini_part(p)
        assert "file_data" in part


class TestCallVisionJudge:
    @pytest.mark.asyncio
    async def test_uses_evaluate_with_images_when_available(self, tiny_png_bytes: bytes):
        judge = AsyncMock()
        judge.evaluate_with_images = AsyncMock(
            return_value=JudgeResponse(score=0.9, reasoning="ok", raw_output="")
        )
        payload = ImagePayload(data=tiny_png_bytes, mime_type="image/png")
        response = await call_vision_judge(
            judge, prompt="hello", images=[payload], system_prompt="sys"
        )
        assert response.score == 0.9
        judge.evaluate_with_images.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_text_evaluate_without_images(self):
        judge = AsyncMock()
        judge.evaluate = AsyncMock(
            return_value=JudgeResponse(score=0.5, reasoning="ok", raw_output="")
        )
        response = await call_vision_judge(judge, prompt="hi", images=[])
        assert response.score == 0.5
        judge.evaluate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_judge_lacks_vision(self, tiny_png_bytes: bytes):
        class _Judge:
            async def evaluate(self, prompt, system_prompt=None):
                return JudgeResponse(score=0.5, reasoning="", raw_output="")

        payload = ImagePayload(data=tiny_png_bytes, mime_type="image/png")
        with pytest.raises(TypeError):
            await call_vision_judge(_Judge(), prompt="x", images=[payload])


class TestCaseImageFields:
    def test_case_accepts_image(self):
        from checkllm.datasets.case import Case

        case = Case(input="x", image="path/to.png")
        assert case.image == "path/to.png"
        assert case.image_sources == ["path/to.png"]

    def test_case_accepts_images(self):
        from checkllm.datasets.case import Case

        case = Case(input="x", images=["a.png", "b.png"])
        assert case.image_sources == ["a.png", "b.png"]

    def test_case_defaults_empty(self):
        from checkllm.datasets.case import Case

        case = Case(input="x")
        assert case.image_sources == []

    def test_images_takes_precedence(self):
        from checkllm.datasets.case import Case

        case = Case(input="x", image="single.png", images=["a.png", "b.png"])
        assert case.image_sources == ["a.png", "b.png"]


def test_module_exports():
    """Public surface sanity check."""
    for name in (
        "ImagePayload",
        "ImageSource",
        "load_image",
        "load_images",
        "to_anthropic_content",
        "to_gemini_part",
        "to_openai_content",
        "call_vision_judge",
    ):
        assert hasattr(multimodal, name), name
