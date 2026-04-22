"""Tests for the new multimodal evaluation metrics."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest

from checkllm.metrics import (
    ChartValueExtractionMetric,
    DiagramComprehensionMetric,
    ImageCaptioningQualityMetric,
    ImageConsistencyMetric,
    ImageSafetyMetric,
    ImageTextAlignmentMetric,
    OCRAccuracyMetric,
    VisualFaithfulnessMetric,
    VisualHallucinationMetric,
    VisualReasoningMetric,
)
from checkllm.models import JudgeResponse

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture
def tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(0, 200, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def vision_judge() -> AsyncMock:
    """A judge mock that implements both evaluate and evaluate_with_images."""
    judge = AsyncMock()
    judge.evaluate_with_images = AsyncMock()
    judge.evaluate = AsyncMock()
    return judge


class TestImageTextAlignment:
    @pytest.mark.asyncio
    async def test_pass(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.95, reasoning="matches", raw_output=""
        )
        metric = ImageTextAlignmentMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(image=tiny_png_bytes, text="A small green square")
        assert result.passed is True
        assert result.metric_name == "image_text_alignment"
        vision_judge.evaluate_with_images.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_accepts_iterable(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.9, reasoning="ok", raw_output=""
        )
        metric = ImageTextAlignmentMetric(judge=vision_judge)
        await metric.evaluate(image=[tiny_png_bytes, tiny_png_bytes], text="two squares")
        args = vision_judge.evaluate_with_images.await_args
        assert len(args.kwargs["images"]) == 2


class TestImageCaptioningQuality:
    @pytest.mark.asyncio
    async def test_with_reference(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.8, reasoning="good", raw_output=""
        )
        metric = ImageCaptioningQualityMetric(judge=vision_judge, threshold=0.7)
        result = await metric.evaluate(
            image=tiny_png_bytes,
            caption="a small green square",
            reference_caption="a tiny green square on white",
        )
        assert result.passed is True
        prompt = vision_judge.evaluate_with_images.await_args.kwargs["prompt"]
        assert "reference" in prompt.lower()

    @pytest.mark.asyncio
    async def test_without_reference(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.4, reasoning="bad", raw_output=""
        )
        metric = ImageCaptioningQualityMetric(judge=vision_judge, threshold=0.7)
        result = await metric.evaluate(image=tiny_png_bytes, caption="a cat")
        assert result.passed is False


class TestOCRAccuracy:
    @pytest.mark.asyncio
    async def test_judge_path(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.9, reasoning="transcript matches", raw_output=""
        )
        metric = OCRAccuracyMetric(judge=vision_judge, threshold=0.85)
        result = await metric.evaluate(image=tiny_png_bytes, extracted_text="HELLO WORLD")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_ground_truth_fallback(self, vision_judge):
        metric = OCRAccuracyMetric(judge=vision_judge, threshold=0.85)
        result = await metric.evaluate(
            image=None,
            extracted_text="Hello World",
            ground_truth="Hello World",
        )
        assert result.score == pytest.approx(1.0)
        assert result.cost == 0.0

    @pytest.mark.asyncio
    async def test_ground_truth_partial(self, vision_judge):
        metric = OCRAccuracyMetric(judge=vision_judge, threshold=0.85)
        result = await metric.evaluate(
            image=None,
            extracted_text="Hello Wrld",
            ground_truth="Hello World",
        )
        assert 0.5 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_requires_image_or_truth(self, vision_judge):
        metric = OCRAccuracyMetric(judge=vision_judge)
        with pytest.raises(ValueError):
            await metric.evaluate(image=None, extracted_text="x")


class TestDiagramComprehension:
    @pytest.mark.asyncio
    async def test_correct(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.95, reasoning="matches expected", raw_output=""
        )
        metric = DiagramComprehensionMetric(judge=vision_judge, threshold=0.7)
        result = await metric.evaluate(
            image=tiny_png_bytes,
            question="How many nodes are in the graph?",
            expected_answer="3",
            candidate_answer="3",
        )
        assert result.passed is True


class TestVisualHallucination:
    @pytest.mark.asyncio
    async def test_grounded(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.95, reasoning="grounded", raw_output=""
        )
        metric = VisualHallucinationMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(image=tiny_png_bytes, response="A green square.")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_hallucinated(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.1, reasoning="hallucinated cat", raw_output=""
        )
        metric = VisualHallucinationMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(image=tiny_png_bytes, response="A cat playing the piano.")
        assert result.passed is False


class TestChartValueExtraction:
    @pytest.mark.asyncio
    async def test_deterministic_within_tolerance(self, vision_judge):
        metric = ChartValueExtractionMetric(judge=vision_judge, threshold=0.7, tolerance=0.05)
        result = await metric.evaluate(
            expected_values={"Q1": 100.0, "Q2": 200.0, "Q3": 150.0},
            extracted_values={"Q1": 101.0, "Q2": 204.0, "Q3": 149.0},
        )
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_deterministic_out_of_tolerance(self, vision_judge):
        metric = ChartValueExtractionMetric(judge=vision_judge, threshold=0.7, tolerance=0.05)
        result = await metric.evaluate(
            expected_values={"A": 100.0, "B": 200.0},
            extracted_values={"A": 120.0, "B": 150.0},
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_value(self, vision_judge):
        metric = ChartValueExtractionMetric(judge=vision_judge, threshold=0.5)
        result = await metric.evaluate(
            expected_values={"A": 10.0, "B": 20.0},
            extracted_values={"A": 10.0},
        )
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_judge_extraction(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.0,
            reasoning="read values",
            raw_output='{"values": {"A": 10.0, "B": 20.0}, "reasoning": "read values"}',
            cost=0.001,
        )
        metric = ChartValueExtractionMetric(judge=vision_judge, threshold=0.5, tolerance=0.1)
        result = await metric.evaluate(
            expected_values={"A": 10.0, "B": 20.0},
            image=tiny_png_bytes,
        )
        assert result.score == 1.0
        assert result.cost == 0.001

    @pytest.mark.asyncio
    async def test_requires_image_or_values(self, vision_judge):
        metric = ChartValueExtractionMetric(judge=vision_judge)
        with pytest.raises(ValueError):
            await metric.evaluate(expected_values={"A": 1.0})


class TestImageSafety:
    @pytest.mark.asyncio
    async def test_safe(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=1.0, reasoning="safe", raw_output=""
        )
        metric = ImageSafetyMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(image=tiny_png_bytes)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unsafe_with_custom_categories(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.0, reasoning="violates weapons category", raw_output=""
        )
        metric = ImageSafetyMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(image=tiny_png_bytes, categories=["weapons", "violence"])
        assert result.passed is False
        prompt = vision_judge.evaluate_with_images.await_args.kwargs["prompt"]
        assert "weapons" in prompt


class TestVisualFaithfulness:
    @pytest.mark.asyncio
    async def test_pass(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.9, reasoning="faithful", raw_output=""
        )
        metric = VisualFaithfulnessMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(
            image=tiny_png_bytes, output="a small green square", query="describe"
        )
        assert result.passed is True
        assert result.metric_name == "visual_faithfulness"


class TestImageConsistency:
    @pytest.mark.asyncio
    async def test_consistent(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.9, reasoning="consistent across both", raw_output=""
        )
        metric = ImageConsistencyMetric(judge=vision_judge, threshold=0.8)
        result = await metric.evaluate(
            images=[tiny_png_bytes, tiny_png_bytes],
            response="Both images show a green square.",
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_requires_two_images(self, vision_judge, tiny_png_bytes):
        metric = ImageConsistencyMetric(judge=vision_judge)
        with pytest.raises(ValueError):
            await metric.evaluate(images=[tiny_png_bytes], response="x")


class TestVisualReasoning:
    @pytest.mark.asyncio
    async def test_correct_answer(self, vision_judge, tiny_png_bytes):
        vision_judge.evaluate_with_images.return_value = JudgeResponse(
            score=0.95,
            reasoning="step 1... step 2... correct",
            raw_output="",
        )
        metric = VisualReasoningMetric(judge=vision_judge, threshold=0.7)
        result = await metric.evaluate(
            image=tiny_png_bytes,
            question="How many red objects are left of the blue one?",
            expected_answer="2",
            candidate_answer="2",
        )
        assert result.passed is True


class TestMetricExports:
    def test_all_metrics_importable_from_package(self):
        from checkllm.metrics import (
            ChartValueExtractionMetric,
            DiagramComprehensionMetric,
            ImageCaptioningQualityMetric,
            ImageConsistencyMetric,
            ImageSafetyMetric,
            ImageTextAlignmentMetric,
            OCRAccuracyMetric,
            VisualFaithfulnessMetric,
            VisualHallucinationMetric,
            VisualReasoningMetric,
        )

        for cls in [
            ChartValueExtractionMetric,
            DiagramComprehensionMetric,
            ImageCaptioningQualityMetric,
            ImageConsistencyMetric,
            ImageSafetyMetric,
            ImageTextAlignmentMetric,
            OCRAccuracyMetric,
            VisualFaithfulnessMetric,
            VisualHallucinationMetric,
            VisualReasoningMetric,
        ]:
            assert callable(cls)


class TestCheckFluentAPI:
    """Ensure the check fluent API exposes all new metrics."""

    def test_check_has_methods(self):
        from checkllm.check import CheckCollector
        from checkllm.config import CheckllmConfig

        collector = CheckCollector(config=CheckllmConfig(cache_enabled=False))
        for name in (
            "image_text_alignment",
            "image_captioning_quality",
            "ocr_accuracy",
            "diagram_comprehension",
            "visual_hallucination",
            "chart_value_extraction",
            "image_safety",
            "visual_faithfulness",
            "image_consistency",
            "visual_reasoning",
        ):
            assert callable(getattr(collector, name)), name
