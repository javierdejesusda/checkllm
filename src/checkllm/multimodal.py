"""Multimodal input helpers for vision-capable judges.

This module normalizes image inputs (file path, URL, base64 string, bytes, or
file-like object) into a provider-agnostic ``ImagePayload`` and exposes
helpers that convert the payload into the format each vision-capable provider
expects (OpenAI chat, Anthropic messages, Gemini inline_data).
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

ImageSource = Union[str, bytes, os.PathLike[str], "ImagePayload"]

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


@dataclass(frozen=True)
class ImagePayload:
    """Normalized representation of an image input.

    Exactly one of ``url`` or ``data`` is populated. When ``data`` is set,
    ``mime_type`` is also always set.

    Attributes:
        url: Remote HTTP(S) URL to the image, if the source was a URL.
        data: Raw image bytes, if the source was a path, file-like, bytes,
            base64 string, or data URL.
        mime_type: MIME type for the image (e.g., ``image/png``). Only
            meaningful when ``data`` is populated.
    """

    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.url is None and self.data is None:
            raise ValueError("ImagePayload requires either url or data")
        if self.url is not None and self.data is not None:
            raise ValueError("ImagePayload must have url or data, not both")
        if self.data is not None and not self.mime_type:
            raise ValueError("ImagePayload with inline data requires mime_type")

    @property
    def is_url(self) -> bool:
        """Return True if this payload references a remote URL."""
        return self.url is not None

    def to_base64(self) -> str:
        """Return the base64-encoded image bytes.

        Raises:
            ValueError: If this payload is URL-only (no inline data).
        """
        if self.data is None:
            raise ValueError("Cannot base64-encode a URL-only ImagePayload")
        return base64.b64encode(self.data).decode("ascii")

    def to_data_url(self) -> str:
        """Return a ``data:<mime>;base64,...`` URL for the inline bytes.

        URL payloads are returned unchanged.
        """
        if self.url is not None:
            return self.url
        return f"data:{self.mime_type};base64,{self.to_base64()}"


def _guess_mime_from_bytes(data: bytes) -> str:
    """Guess MIME type from the first few bytes of an image."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "application/octet-stream"


def _guess_mime_from_path(path: str | os.PathLike[str]) -> str:
    """Guess MIME type from a file path or URL suffix."""
    guess, _ = mimetypes.guess_type(str(path))
    return guess or "application/octet-stream"


def load_image(source: ImageSource | Any, mime_type: str | None = None) -> ImagePayload:
    """Normalize an image input into an ``ImagePayload``.

    Args:
        source: The image source. Accepted forms:

            * An ``ImagePayload`` (returned unchanged).
            * A path to a local file (``str`` or ``os.PathLike``).
            * An ``http://``/``https://`` URL.
            * A ``data:<mime>;base64,...`` data URL.
            * A plain base64-encoded string.
            * Raw ``bytes``.
        mime_type: Optional explicit MIME type override. Used when the MIME
            type cannot be inferred (e.g., bare bytes or base64 strings).

    Returns:
        The normalized ``ImagePayload``.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If the source cannot be interpreted as an image.
    """
    if isinstance(source, ImagePayload):
        return source

    if isinstance(source, bytes):
        resolved_mime = mime_type or _guess_mime_from_bytes(source)
        return ImagePayload(data=source, mime_type=resolved_mime)

    if isinstance(source, os.PathLike):
        return _load_from_path(source, mime_type)

    if isinstance(source, str):
        data_match = _DATA_URL_RE.match(source)
        if data_match:
            mime = mime_type or data_match.group("mime")
            raw = base64.b64decode(data_match.group("data"))
            return ImagePayload(data=raw, mime_type=mime)

        if _URL_RE.match(source):
            return ImagePayload(url=source)

        if os.sep in source or "/" in source or Path(source).suffix:
            path = Path(source)
            if path.exists():
                return _load_from_path(path, mime_type)

        if _BASE64_RE.match(source) and len(source) > 16:
            try:
                raw = base64.b64decode(source, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("Could not decode source as base64 string") from exc
            resolved_mime = mime_type or _guess_mime_from_bytes(raw)
            return ImagePayload(data=raw, mime_type=resolved_mime)

        raise ValueError(f"Could not interpret string source as image: {source[:40]!r}")

    read_fn = getattr(source, "read", None)
    if callable(read_fn):
        raw = read_fn()
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError("File-like source must return bytes from read()")
        resolved_mime = mime_type or _guess_mime_from_bytes(bytes(raw))
        return ImagePayload(data=bytes(raw), mime_type=resolved_mime)

    raise TypeError(f"Unsupported image source type: {type(source).__name__}")


def _load_from_path(path: str | os.PathLike[str], mime_type: str | None) -> ImagePayload:
    """Read image bytes from a filesystem path."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {resolved}")
    data = resolved.read_bytes()
    resolved_mime = mime_type or _guess_mime_from_path(resolved) or _guess_mime_from_bytes(data)
    return ImagePayload(data=data, mime_type=resolved_mime)


def load_images(sources: Iterable[ImageSource]) -> list[ImagePayload]:
    """Normalize a sequence of image sources into a list of payloads."""
    return [load_image(s) for s in sources]


def to_openai_content(payload: ImagePayload) -> dict[str, Any]:
    """Convert an ``ImagePayload`` to an OpenAI chat ``image_url`` block."""
    return {
        "type": "image_url",
        "image_url": {"url": payload.to_data_url()},
    }


def to_anthropic_content(payload: ImagePayload) -> dict[str, Any]:
    """Convert an ``ImagePayload`` to an Anthropic messages ``image`` block."""
    if payload.is_url:
        return {
            "type": "image",
            "source": {"type": "url", "url": payload.url},
        }
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": payload.mime_type,
            "data": payload.to_base64(),
        },
    }


def to_gemini_part(payload: ImagePayload) -> dict[str, Any]:
    """Convert an ``ImagePayload`` to a Gemini ``inline_data`` part.

    Gemini does not accept remote URLs directly. URL payloads are returned as
    a ``file_data`` part which callers may need to resolve via the files API.
    """
    if payload.is_url:
        return {"file_data": {"mime_type": "image/*", "file_uri": payload.url}}
    return {
        "inline_data": {
            "mime_type": payload.mime_type,
            "data": payload.to_base64(),
        }
    }


async def call_vision_judge(
    judge: Any,
    prompt: str,
    images: Iterable[ImagePayload],
    system_prompt: str | None = None,
) -> Any:
    """Invoke a judge backend with vision inputs if supported.

    Args:
        judge: A judge backend instance (text or vision-capable).
        prompt: The text portion of the prompt.
        images: Image payloads to include.
        system_prompt: Optional system prompt.

    Returns:
        The raw ``JudgeResponse`` returned by the judge.

    Raises:
        TypeError: If ``judge`` does not implement ``evaluate_with_images``
            and images are supplied.
    """
    image_list = list(images)
    method = getattr(judge, "evaluate_with_images", None)
    if image_list and callable(method):
        return await method(prompt=prompt, images=image_list, system_prompt=system_prompt)
    if image_list:
        raise TypeError(
            f"Judge {type(judge).__name__} does not support image inputs. "
            "Use a vision-capable model (OpenAI gpt-4o, Anthropic claude-3.5-sonnet, "
            "or Gemini) or pass image_description= for text-only evaluation."
        )
    return await judge.evaluate(prompt=prompt, system_prompt=system_prompt)


__all__ = [
    "ImagePayload",
    "ImageSource",
    "call_vision_judge",
    "load_image",
    "load_images",
    "to_anthropic_content",
    "to_gemini_part",
    "to_openai_content",
]
