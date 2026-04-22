"""Image-text alignment metric.

Scores whether a text accurately describes an accompanying image using a
vision-capable LLM judge. Useful for evaluating captioning, alt-text, and
image-grounded chat responses on GPT-4V / Claude Vision / Gemini Vision.
"""

from __future__ import annotations

import time
from typing import Iterable, cast

from checkllm.judge import JudgeBackend
from checkllm.models import CheckResult
from checkllm.multimodal import ImagePayload, ImageSource, call_vision_judge, load_image

IMAGE_TEXT_ALIGNMENT_SYSTEM_PROMPT = """You are an expert multimodal evaluator. You will be shown one or more images and a candidate text. Decide how faithfully the text describes what the image actually shows.

Score from 0.0 to 1.0:
- 1.0 = Every claim in the text is directly supported by the image; no hallucinated objects, attributes, counts, or relationships.
- 0.7 = Mostly accurate with small inaccuracies (minor color or count drift).
- 0.5 = Partially accurate; key elements are right but other claims are wrong or unsupported.
- 0.3 = Largely unsupported; most claims do not reflect the image.
- 0.0 = Text is unrelated to or contradicts the image.

Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}"""


class ImageTextAlignmentMetric:
    """Scores text-image alignment using a vision-capable judge."""

    def __init__(self, judge: JudgeBackend, threshold: float = 0.8) -> None:
        self.judge = judge
        self.threshold = threshold
        self.system_prompt: str = IMAGE_TEXT_ALIGNMENT_SYSTEM_PROMPT

    async def evaluate(
        self,
        image: ImageSource | Iterable[ImageSource],
        text: str,
    ) -> CheckResult:
        """Evaluate whether ``text`` faithfully describes ``image``.

        Args:
            image: A single image source or a sequence of sources. Accepts
                file paths, URLs, base64 strings, bytes, or ``ImagePayload``.
            text: The candidate text description.

        Returns:
            A ``CheckResult`` with the alignment score and reasoning.
        """
        payloads = _ensure_payloads(image)
        prompt = f"Candidate text:\n{text}\n\nScore how faithfully the text describes the image(s)."
        start = time.perf_counter_ns()
        response = await call_vision_judge(
            self.judge, prompt=prompt, images=payloads, system_prompt=self.system_prompt
        )
        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

        return CheckResult(
            passed=response.score >= self.threshold,
            score=response.score,
            reasoning=response.reasoning,
            cost=response.cost,
            latency_ms=int(elapsed_ms),
            metric_name="image_text_alignment",
        )


def _ensure_payloads(
    image: ImageSource | Iterable[ImageSource],
) -> list[ImagePayload]:
    """Coerce a single source or iterable of sources into ``ImagePayload``s."""
    if isinstance(image, ImagePayload):
        return [image]
    # Single scalar sources (path/URL/base64/bytes/file-like) go through load_image.
    if isinstance(image, (str, bytes)) or hasattr(image, "read"):
        return [load_image(cast(ImageSource, image))]
    try:
        iter(cast(Iterable[ImageSource], image))
    except TypeError:
        return [load_image(cast(ImageSource, image))]
    return [load_image(s) for s in cast(Iterable[ImageSource], image)]
