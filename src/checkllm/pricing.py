"""Provider-specific LLM pricing tables and cost breakdown records.

Pricing snapshot date: 2026-04-08.

All prices are expressed in USD per 1M tokens (input / output).  Values are
sourced from each provider's public pricing page:

    * OpenAI           -- https://openai.com/api/pricing/
    * Anthropic        -- https://www.anthropic.com/pricing
    * Google Gemini    -- https://ai.google.dev/pricing
    * DeepSeek         -- https://api-docs.deepseek.com/quick_start/pricing
    * AWS Bedrock      -- https://aws.amazon.com/bedrock/pricing/

The :data:`PRICING` mapping uses lowercase, normalized model identifiers that
match the canonical ``model=`` argument passed to checkllm's judge backends.
Aliases (e.g. ``"claude-4.7-sonnet"`` vs ``"claude-sonnet-4-7"``) are also
registered so lookups succeed regardless of which form callers use.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------------------
# Pricing tables (USD per 1M tokens)
# ---------------------------------------------------------------------------

#: OpenAI chat-completion models (GPT-4o, GPT-4.1 family, o1/o3 reasoning).
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Reasoning (o-series) -- public snapshot Apr 2026.
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o1-preview": (15.00, 60.00),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
}

#: Anthropic Claude models -- Sonnet, Opus, Haiku across the 3.x -> 4.x series.
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    # Claude 3.5 generation
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    # Claude 3.7 generation
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    # Claude 4 family (Sonnet / Opus / Haiku)
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-haiku-4": (0.80, 4.00),
    # Claude 4.5 / 4.6 / 4.7
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-5": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-haiku-4-6": (0.80, 4.00),
    "claude-sonnet-4-7": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-haiku-4-7": (0.80, 4.00),
}

#: Google Gemini pricing (1.5 / 2.0 / 2.5 lines).
GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
}

#: DeepSeek (v3 chat and R1 reasoner).
DEEPSEEK_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v3": (0.27, 1.10),
    "deepseek-r1": (0.55, 2.19),
}

#: AWS Bedrock variants (prices tracked per model family, not per region).
BEDROCK_PRICING: dict[str, tuple[float, float]] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00),
    "anthropic.claude-3-opus-20240229-v1:0": (15.00, 75.00),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.25, 1.25),
    "anthropic.claude-sonnet-4-v1:0": (3.00, 15.00),
    "anthropic.claude-opus-4-v1:0": (15.00, 75.00),
    "amazon.nova-pro-v1:0": (0.80, 3.20),
    "amazon.nova-lite-v1:0": (0.06, 0.24),
    "amazon.nova-micro-v1:0": (0.035, 0.14),
    "meta.llama3-70b-instruct-v1:0": (2.65, 3.50),
    "meta.llama3-8b-instruct-v1:0": (0.30, 0.60),
    "mistral.mistral-large-2407-v1:0": (2.00, 6.00),
    "cohere.command-r-plus-v1:0": (2.50, 10.00),
}

#: Fallback price used when the model identifier is unknown.  Intentionally
#: pessimistic (matches GPT-4 tier) so unknown models don't undercount cost.
DEFAULT_PRICING: tuple[float, float] = (5.00, 15.00)


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "google"),
    ("deepseek-", "deepseek"),
    ("anthropic.", "bedrock"),
    ("amazon.", "bedrock"),
    ("meta.", "bedrock"),
    ("mistral.", "bedrock"),
    ("cohere.", "bedrock"),
)


def infer_provider(model: str) -> str:
    """Infer the provider for a model identifier.

    Args:
        model: Canonical model name (case-insensitive).

    Returns:
        One of ``"openai"``, ``"anthropic"``, ``"google"``, ``"deepseek"``,
        ``"bedrock"``, or ``"unknown"``.
    """
    key = (model or "").lower().strip()
    for prefix, provider in _PROVIDER_PREFIXES:
        if key.startswith(prefix):
            return provider
    return "unknown"


# ---------------------------------------------------------------------------
# Unified lookup
# ---------------------------------------------------------------------------

#: Merged mapping used by :func:`lookup_price`.
PRICING: dict[str, tuple[float, float]] = {
    **OPENAI_PRICING,
    **ANTHROPIC_PRICING,
    **GEMINI_PRICING,
    **DEEPSEEK_PRICING,
    **BEDROCK_PRICING,
}


def lookup_price(model: str) -> tuple[float, float]:
    """Return ``(input_price, output_price)`` per 1M tokens for ``model``.

    Falls back to :data:`DEFAULT_PRICING` if the model is unknown.
    Lookup is case-insensitive.

    Args:
        model: Canonical model identifier (e.g. ``"gpt-4o"``).

    Returns:
        ``(input_price_per_million, output_price_per_million)`` in USD.
    """
    key = (model or "").lower().strip()
    if key in PRICING:
        return PRICING[key]
    # Tolerate "claude-4.7-sonnet" style aliases by normalizing dots.
    canonical = key.replace(".", "-")
    return PRICING.get(canonical, DEFAULT_PRICING)


# ---------------------------------------------------------------------------
# Cost breakdown record
# ---------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    """Granular cost attribution for a single check.

    This is attached to :class:`checkllm.models.CheckResult` so the dashboard
    can roll up spend by provider, model, metric, test, or time bucket.

    Attributes:
        input_tokens: Prompt / input tokens billed by the provider.
        output_tokens: Completion / output tokens billed by the provider.
        input_price: Input price in USD per 1M tokens.
        output_price: Output price in USD per 1M tokens.
        provider: Provider label (``"openai"``, ``"anthropic"``, ...).
        model: Canonical model identifier used for the call.
        metric: Name of the check metric (e.g. ``"hallucination"``).
        test_id: Test identifier (pytest node id or caller-supplied).
        timestamp: Unix timestamp in seconds when the check completed.
        total_cost: Pre-computed USD cost; keeps the record self-contained
            after serialization.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    input_price: float = 0.0
    output_price: float = 0.0
    provider: str = "unknown"
    model: str = ""
    metric: str = ""
    test_id: str = ""
    timestamp: float = field(default_factory=time.time)
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


def build_cost_breakdown(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    metric: str = "",
    test_id: str = "",
    timestamp: float | None = None,
) -> CostBreakdown:
    """Construct a :class:`CostBreakdown` from raw token counts.

    Args:
        model: Canonical model identifier.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        metric: Optional metric name.
        test_id: Optional test identifier.
        timestamp: Optional explicit timestamp (seconds since epoch); defaults
            to ``time.time()`` when ``None``.

    Returns:
        A fully populated :class:`CostBreakdown`, including computed
        ``total_cost`` and inferred ``provider``.
    """
    input_price, output_price = lookup_price(model)
    total = (input_tokens / 1_000_000.0) * input_price + (
        output_tokens / 1_000_000.0
    ) * output_price
    return CostBreakdown(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        input_price=float(input_price),
        output_price=float(output_price),
        provider=infer_provider(model),
        model=model,
        metric=metric,
        test_id=test_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        total_cost=round(total, 10),
    )


def breakdown_from_dict(data: dict[str, object] | None) -> CostBreakdown | None:
    """Rehydrate a :class:`CostBreakdown` from its serialized dict form.

    Missing keys fall back to the dataclass defaults so older persisted runs
    keep deserializing cleanly.

    Args:
        data: Serialized breakdown (as produced by :meth:`CostBreakdown.to_dict`)
            or ``None``.

    Returns:
        A :class:`CostBreakdown` instance, or ``None`` when ``data`` is falsy.
    """
    if not data:
        return None
    known = {f for f in CostBreakdown.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in known}
    return CostBreakdown(**filtered)  # type: ignore[arg-type]
