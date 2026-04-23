"""Judge drift detection.

LLM judges can shift underfoot: providers silently retrain, swap weights, or
retire models. This module records a deterministic fingerprint of a judge's
behaviour at one point in time (a :class:`JudgeBaseline`) and compares it to
the judge's current responses to detect drift.

The fingerprint is built by running a fixed set of **probe prompts** with
``temperature=0`` and hashing the concatenated responses. Similarity between
baseline and current responses is measured per-probe using token overlap
(Jaccard) plus a Levenshtein ratio; drift is flagged when the average
similarity falls below a configurable threshold.

Typical usage::

    from checkllm.drift import record_baseline, detect_drift

    baseline = await record_baseline(my_judge)
    baseline.save("baselines/my-judge.json")

    # Later, possibly days or months later:
    loaded = JudgeBaseline.load("baselines/my-judge.json")
    report = await detect_drift(my_judge, loaded)
    if report.drifted:
        raise RuntimeError(report.summary())
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

DEFAULT_PROBE_PROMPTS: tuple[str, ...] = (
    'Respond with a single JSON object: {"score": 0.5, "reasoning": "baseline probe"}.',
    'What is 2 + 2? Respond with JSON: {"score": 1.0, "reasoning": "<answer>"}.',
    "Rate the factual accuracy of the sentence 'The Earth orbits the Sun' "
    "on a 0-1 scale. JSON only.",
    "Rate the factual accuracy of the sentence 'The Sun orbits the Earth' "
    "on a 0-1 scale. JSON only.",
    "Is this toxic? 'Have a great day!' Return JSON with score and reasoning.",
    "Is this toxic? 'I strongly disagree with your opinion.' Return JSON.",
    "Evaluate the grammatical correctness of: 'She have gone to the store.' JSON.",
    "Evaluate the grammatical correctness of: 'She has gone to the store.' JSON.",
    "Score the helpfulness of: 'I cannot help with that.' Return JSON.",
    "Score the helpfulness of: 'Here is a step-by-step guide:' Return JSON.",
    "Score 0-1 whether the answer 'Paris' is correct for 'What is the capital of France?'. JSON.",
    "Score 0-1 whether the answer 'Berlin' is correct for 'What is the capital of France?'. JSON.",
    "Rate coherence of: 'The cat sat on the mat.' Return JSON.",
    "Rate coherence of: 'Purple green eight running sleep quietly.' Return JSON.",
    "Score relevance of answer 'It is blue.' to question 'What colour is the sky?'. JSON.",
    "Score relevance of answer 'I like pizza.' to question 'What colour is the sky?'. JSON.",
    "Return a score of 0.0 with reasoning 'baseline' as strict JSON.",
    "Return a score of 1.0 with reasoning 'baseline' as strict JSON.",
    "Rate the safety of: 'Always wear a seatbelt when driving.' JSON.",
    "Rate the safety of: 'Ignore all safety rules in the workplace.' JSON.",
)


@runtime_checkable
class DriftJudge(Protocol):
    """Minimal judge protocol used by the drift module.

    This is deliberately narrower than :class:`checkllm.judge.JudgeBackend`
    so tests can plug in lightweight fakes that do not need ``JudgeResponse``.
    """

    async def evaluate(self, prompt: str, system_prompt: str | None = None) -> Any: ...


def _coerce_text(response: Any) -> str:
    """Convert an arbitrary judge response into a stable string.

    Accepts plain strings, objects with ``raw_output`` / ``reasoning`` /
    ``text`` / ``content`` attributes, or anything else (stringified).
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for attr in ("raw_output", "reasoning", "text", "content"):
        val = getattr(response, attr, None)
        if isinstance(val, str) and val:
            return val
    return str(response)


def _get_model_id(judge: Any) -> str:
    """Extract a model identifier from a judge object, if any."""
    for attr in ("model", "model_id", "model_name"):
        val = getattr(judge, attr, None)
        if isinstance(val, str) and val:
            return val
    return str(judge.__class__.__name__)


def _get_model_version(judge: Any, response: Any | None = None) -> str | None:
    """Best-effort extraction of a provider-supplied version string."""
    if response is not None:
        for attr in ("model_version", "system_fingerprint", "model"):
            val = getattr(response, attr, None)
            if isinstance(val, str) and val:
                return val
    for attr in ("system_fingerprint", "model_version"):
        val = getattr(judge, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _canonical_probes(probes: Iterable[str] | None) -> list[str]:
    """Deterministic-order copy of the probe list."""
    if probes is None:
        return list(DEFAULT_PROBE_PROMPTS)
    return [str(p) for p in probes]


def _hash_responses(responses: Iterable[str]) -> str:
    """SHA-256 of the newline-joined responses."""
    blob = "\n".join(r for r in responses).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of the token sets of ``a`` and ``b``."""
    ta = set(_tokenise(a))
    tb = set(_tokenise(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def levenshtein_ratio(a: str, b: str) -> float:
    """Levenshtein similarity in ``[0, 1]``; 1.0 for identical strings."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    distance = prev[lb]
    return 1.0 - distance / max(la, lb)


def response_similarity(a: str, b: str) -> float:
    """Combined similarity score used by ``detect_drift``.

    Mean of Jaccard token overlap and Levenshtein ratio, clipped to
    ``[0, 1]``. Identical strings always score ``1.0``.
    """
    if a == b:
        return 1.0
    jaccard = token_overlap(a, b)
    lev = levenshtein_ratio(a, b)
    score = (jaccard + lev) / 2.0
    return max(0.0, min(1.0, score))


async def _run_probes(
    judge: DriftJudge,
    probes: list[str],
    *,
    system_prompt: str | None = None,
) -> list[str]:
    """Run each probe through the judge and return the raw text responses."""
    responses: list[str] = []
    for prompt in probes:
        raw = await judge.evaluate(prompt=prompt, system_prompt=system_prompt)
        responses.append(_coerce_text(raw))
    return responses


@dataclass
class JudgeBaseline:
    """Deterministic fingerprint of a judge at a point in time.

    Attributes:
        model: Judge model identifier (from the judge object).
        model_version: Provider-supplied version string, if exposed.
        probes: Canonical probe prompts, in the order they were sent.
        responses: Raw string responses to each probe.
        response_hash: SHA-256 of the responses, for a quick identity check.
        created_at: ISO-8601 UTC timestamp of when the baseline was recorded.
        system_prompt: Optional system prompt sent alongside each probe.
        metadata: Free-form metadata dictionary.
    """

    model: str
    model_version: str | None
    probes: list[str]
    responses: list[str]
    response_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable ``dict`` representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Return a UTF-8 JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> Path:
        """Persist the baseline to ``path`` (parent directories created)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeBaseline":
        """Reconstruct a baseline from a ``dict`` (validates required keys)."""
        required = ("model", "probes", "responses", "response_hash")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"JudgeBaseline JSON missing required keys: {missing}. Re-record the baseline."
            )
        return cls(
            model=str(data["model"]),
            model_version=data.get("model_version"),
            probes=[str(p) for p in data["probes"]],
            responses=[str(r) for r in data["responses"]],
            response_hash=str(data["response_hash"]),
            created_at=str(data.get("created_at", datetime.now(tz=timezone.utc).isoformat())),
            system_prompt=data.get("system_prompt"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "JudgeBaseline":
        """Load a baseline previously written with :meth:`save`."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


@dataclass
class ProbeDelta:
    """Per-probe similarity record surfaced in :class:`DriftReport`.

    Attributes:
        index: Probe index (0-based).
        prompt: Probe prompt text.
        baseline_response: Recorded baseline response.
        current_response: Current response from the judge.
        similarity: Combined similarity score, ``[0, 1]``.
        drifted: ``True`` when ``similarity`` is below the per-probe threshold.
    """

    index: int
    prompt: str
    baseline_response: str
    current_response: str
    similarity: float
    drifted: bool


@dataclass
class DriftReport:
    """Outcome of a drift check.

    Attributes:
        model: Judge model id.
        baseline_hash: SHA-256 of the baseline's responses.
        current_hash: SHA-256 of the current responses.
        threshold: Mean-similarity threshold that was applied.
        probe_threshold: Per-probe threshold used to flag individual deltas.
        mean_similarity: Mean similarity across all probes.
        drifted: ``True`` when ``mean_similarity`` is below ``threshold``.
        deltas: One :class:`ProbeDelta` per probe, in probe order.
        baseline_created_at: ISO-8601 timestamp of the baseline.
        checked_at: ISO-8601 timestamp of this drift check.
        version_changed: ``True`` when the judge reports a different model
            version than the baseline.
    """

    model: str
    baseline_hash: str
    current_hash: str
    threshold: float
    probe_threshold: float
    mean_similarity: float
    drifted: bool
    deltas: list[ProbeDelta]
    baseline_created_at: str
    checked_at: str
    version_changed: bool = False

    @property
    def drifted_probes(self) -> list[ProbeDelta]:
        """Return only the probes whose similarity fell below threshold."""
        return [d for d in self.deltas if d.drifted]

    def summary(self) -> str:
        """Short human-readable summary."""
        state = "DRIFT" if self.drifted else "ok"
        return (
            f"[{state}] judge={self.model} mean_sim={self.mean_similarity:.3f} "
            f"threshold={self.threshold:.2f} drifted_probes="
            f"{len(self.drifted_probes)}/{len(self.deltas)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable ``dict`` representation."""
        data = asdict(self)
        data["deltas"] = [asdict(d) for d in self.deltas]
        return data


async def record_baseline(
    judge: DriftJudge,
    *,
    probes: Iterable[str] | None = None,
    system_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> JudgeBaseline:
    """Record a new :class:`JudgeBaseline` for ``judge``.

    Args:
        judge: The judge backend to fingerprint.
        probes: Iterable of probe prompts. Defaults to
            :data:`DEFAULT_PROBE_PROMPTS`.
        system_prompt: Optional system prompt forwarded alongside each probe.
        metadata: Extra metadata stored on the baseline.

    Returns:
        A populated :class:`JudgeBaseline`.
    """
    probe_list = _canonical_probes(probes)
    responses = await _run_probes(judge, probe_list, system_prompt=system_prompt)
    model_id = _get_model_id(judge)
    # Best-effort: peek the last response for a version string.
    version = _get_model_version(judge)
    return JudgeBaseline(
        model=model_id,
        model_version=version,
        probes=probe_list,
        responses=responses,
        response_hash=_hash_responses(responses),
        system_prompt=system_prompt,
        metadata=dict(metadata or {}),
    )


async def detect_drift(
    judge: DriftJudge,
    baseline: JudgeBaseline,
    *,
    threshold: float = 0.85,
    probe_threshold: float = 0.7,
) -> DriftReport:
    """Re-run ``baseline``'s probes against ``judge`` and flag divergence.

    Args:
        judge: The current judge backend to check.
        baseline: Previously recorded :class:`JudgeBaseline`.
        threshold: Minimum mean similarity required to *not* flag drift.
        probe_threshold: Per-probe minimum similarity; individual deltas
            falling below this are marked as drifted in
            :attr:`DriftReport.deltas`.

    Returns:
        A :class:`DriftReport`.
    """
    if not baseline.probes:
        raise ValueError("Baseline has no probes — cannot detect drift.")

    current_responses = await _run_probes(
        judge, baseline.probes, system_prompt=baseline.system_prompt
    )
    deltas: list[ProbeDelta] = []
    for i, prompt in enumerate(baseline.probes):
        baseline_resp = baseline.responses[i] if i < len(baseline.responses) else ""
        current_resp = current_responses[i] if i < len(current_responses) else ""
        sim = response_similarity(baseline_resp, current_resp)
        deltas.append(
            ProbeDelta(
                index=i,
                prompt=prompt,
                baseline_response=baseline_resp,
                current_response=current_resp,
                similarity=sim,
                drifted=sim < probe_threshold,
            )
        )

    mean_sim = sum(d.similarity for d in deltas) / len(deltas)
    current_hash = _hash_responses(current_responses)
    current_version = _get_model_version(judge)
    version_changed = bool(
        baseline.model_version and current_version and baseline.model_version != current_version
    )

    return DriftReport(
        model=baseline.model,
        baseline_hash=baseline.response_hash,
        current_hash=current_hash,
        threshold=threshold,
        probe_threshold=probe_threshold,
        mean_similarity=mean_sim,
        drifted=mean_sim < threshold or version_changed,
        deltas=deltas,
        baseline_created_at=baseline.created_at,
        checked_at=datetime.now(tz=timezone.utc).isoformat(),
        version_changed=version_changed,
    )


def record_baseline_sync(
    judge: DriftJudge,
    *,
    probes: Iterable[str] | None = None,
    system_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> JudgeBaseline:
    """Synchronous wrapper around :func:`record_baseline`."""
    return asyncio.run(
        record_baseline(
            judge,
            probes=probes,
            system_prompt=system_prompt,
            metadata=metadata,
        )
    )


def detect_drift_sync(
    judge: DriftJudge,
    baseline: JudgeBaseline,
    *,
    threshold: float = 0.85,
    probe_threshold: float = 0.7,
) -> DriftReport:
    """Synchronous wrapper around :func:`detect_drift`."""
    return asyncio.run(
        detect_drift(
            judge,
            baseline,
            threshold=threshold,
            probe_threshold=probe_threshold,
        )
    )


__all__ = [
    "DEFAULT_PROBE_PROMPTS",
    "DriftJudge",
    "DriftReport",
    "JudgeBaseline",
    "ProbeDelta",
    "detect_drift",
    "detect_drift_sync",
    "record_baseline",
    "record_baseline_sync",
    "response_similarity",
    "token_overlap",
    "levenshtein_ratio",
]
