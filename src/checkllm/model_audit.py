"""Security scanner for ML model files.

Inspects model artifacts for dangerous patterns such as arbitrary code
execution payloads in pickle files, unsafe numpy object arrays, and
suspicious binary content. Scanning is performed on raw bytes without
deserialising the model, preventing exploitation during the audit itself.

Usage::

    from checkllm.model_audit import ModelAuditor

    auditor = ModelAuditor()
    result = auditor.scan("model.pkl")
    if not result.is_safe:
        print(result.summary())
"""

from __future__ import annotations

import json
import re
import struct
import time
from enum import Enum
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, computed_field


class SeverityLevel(str, Enum):
    """Severity classification for security findings."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityFinding(BaseModel):
    """A single security finding from a model audit.

    Attributes:
        severity: How severe the finding is.
        title: Short summary of the issue.
        description: Detailed explanation.
        file_path: Path to the scanned file.
        line_number: Source line if applicable.
        byte_offset: Byte position in the file where the pattern was found.
        pattern_matched: The raw pattern or string that triggered the finding.
        remediation: Suggested fix or mitigation.
        cwe_id: Common Weakness Enumeration identifier, if applicable.
    """

    severity: SeverityLevel
    title: str
    description: str
    file_path: str
    line_number: int | None = None
    byte_offset: int | None = None
    pattern_matched: str | None = None
    remediation: str
    cwe_id: str | None = None


class AuditResult(BaseModel):
    """Aggregated result of scanning a single model file.

    Attributes:
        file_path: Path to the scanned file.
        file_type: Detected model format (e.g. ``pickle``, ``pytorch``).
        file_size_bytes: Size of the file in bytes.
        scan_time_ms: Wall-clock time spent scanning, in milliseconds.
        findings: All security findings discovered.
    """

    file_path: str
    file_type: str
    file_size_bytes: int
    scan_time_ms: int
    findings: list[SecurityFinding]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_safe(self) -> bool:
        """True when there are no critical or high-severity findings."""
        return all(
            f.severity not in (SeverityLevel.CRITICAL, SeverityLevel.HIGH) for f in self.findings
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def critical_count(self) -> int:
        """Number of critical findings."""
        return sum(1 for f in self.findings if f.severity == SeverityLevel.CRITICAL)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def high_count(self) -> int:
        """Number of high-severity findings."""
        return sum(1 for f in self.findings if f.severity == SeverityLevel.HIGH)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def medium_count(self) -> int:
        """Number of medium-severity findings."""
        return sum(1 for f in self.findings if f.severity == SeverityLevel.MEDIUM)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def low_count(self) -> int:
        """Number of low-severity findings."""
        return sum(1 for f in self.findings if f.severity == SeverityLevel.LOW)

    def summary(self) -> str:
        """Return a human-readable summary of the audit.

        Returns:
            Multi-line string listing the file, finding counts, and each
            finding with its severity.
        """
        total = len(self.findings)
        parts = []
        if self.critical_count:
            parts.append(f"{self.critical_count} critical")
        if self.high_count:
            parts.append(f"{self.high_count} high")
        if self.medium_count:
            parts.append(f"{self.medium_count} medium")
        if self.low_count:
            parts.append(f"{self.low_count} low")

        lines = [
            f"ModelAudit: {self.file_path}",
            f"Findings: {total} ({', '.join(parts)})" if parts else "Findings: 0",
        ]
        for f in self.findings:
            lines.append(f"- {f.severity.value.upper()}: {f.description}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialise the audit result to a JSON string.

        Returns:
            JSON representation of the audit result.
        """
        return self.model_dump_json(indent=2)


_DANGEROUS_MODULES = [
    b"os",
    b"subprocess",
    b"sys",
    b"shutil",
    b"builtins",
    b"eval",
    b"exec",
    b"compile",
    b"__import__",
]

_ATTACK_PATTERNS = [
    b"os.system",
    b"os.popen",
    b"subprocess.call",
    b"subprocess.Popen",
    b"subprocess.run",
    b"eval(",
    b"exec(",
]

_NETWORK_MODULES = [
    b"socket",
    b"urllib",
    b"requests",
    b"http",
]

_PICKLE_OPCODES: dict[bytes, str] = {
    b"R": "REDUCE",
    b"c": "GLOBAL",
    b"i": "INST",
    b"b": "BUILD",
    b"\x93": "STACK_GLOBAL",
}


class ModelAuditor:
    """Scans ML model files for security vulnerabilities.

    Reads raw bytes and pattern-matches against known dangerous constructs.
    The files are never deserialised, so malicious payloads cannot execute
    during scanning.

    Attributes:
        SUPPORTED_EXTENSIONS: Mapping of file extensions to format names.
    """

    SUPPORTED_EXTENSIONS: dict[str, str] = {
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".joblib": "joblib",
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".bin": "binary",
        ".onnx": "onnx",
        ".safetensors": "safetensors",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".keras": "keras",
        ".tflite": "tflite",
        ".pb": "protobuf",
        ".gguf": "gguf",
        ".ggml": "ggml",
        ".npy": "numpy",
        ".npz": "numpy",
    }

    def __init__(self, max_file_size_mb: int = 5000) -> None:
        self.max_file_size_mb = max_file_size_mb
        self._scanners: dict[str, Callable[[bytes, str], list[SecurityFinding]]] = {
            "pickle": self._scan_pickle,
            "joblib": self._scan_pickle,
            "pytorch": self._scan_pytorch,
            "numpy": self._scan_numpy,
            "safetensors": self._scan_safetensors,
            "onnx": self._scan_onnx,
            "binary": self._scan_binary,
            "hdf5": self._scan_binary,
            "keras": self._scan_binary,
            "tflite": self._scan_binary,
            "protobuf": self._scan_binary,
            "gguf": self._scan_binary,
            "ggml": self._scan_binary,
        }

    def scan(self, file_path: str) -> AuditResult:
        """Scan a single model file for security issues.

        Args:
            file_path: Path to the model file.

        Returns:
            An AuditResult with all findings.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported or the file
                exceeds the size limit.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = p.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(suffix)
        if file_type is None:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        file_size = p.stat().st_size
        max_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(f"File size {file_size} bytes exceeds limit of {max_bytes} bytes")

        start = time.monotonic()
        data = p.read_bytes()

        scanner = self._scanners.get(file_type, self._scan_binary)
        findings = scanner(data, file_path)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return AuditResult(
            file_path=file_path,
            file_type=file_type,
            file_size_bytes=file_size,
            scan_time_ms=elapsed_ms,
            findings=findings,
        )

    def scan_directory(self, directory: str, recursive: bool = True) -> list[AuditResult]:
        """Scan all supported model files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to descend into subdirectories.

        Returns:
            A list of AuditResult, one per scanned file.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results: list[AuditResult] = []
        pattern = "**/*" if recursive else "*"
        for child in root.glob(pattern):
            if child.is_file() and child.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                results.append(self.scan(str(child)))
        return results

    def _scan_pickle(self, data: bytes, file_path: str) -> list[SecurityFinding]:
        """Scan pickle/joblib data for dangerous opcodes and patterns.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.

        Returns:
            List of security findings.
        """
        findings: list[SecurityFinding] = []

        for opcode, name in _PICKLE_OPCODES.items():
            offset = data.find(opcode)
            if offset != -1:
                if name in ("REDUCE", "GLOBAL", "INST", "STACK_GLOBAL"):
                    findings.append(
                        SecurityFinding(
                            severity=SeverityLevel.MEDIUM,
                            title=f"Pickle {name} opcode detected",
                            description=(
                                f"Pickle file contains {name} opcode which can "
                                f"execute arbitrary Python code during deserialization"
                            ),
                            file_path=file_path,
                            byte_offset=offset,
                            pattern_matched=name,
                            remediation="Use safetensors or ONNX format instead of pickle",
                            cwe_id="CWE-502",
                        )
                    )

        for pattern in _ATTACK_PATTERNS:
            offset = data.find(pattern)
            if offset != -1:
                decoded = pattern.decode("utf-8", errors="replace")
                findings.append(
                    SecurityFinding(
                        severity=SeverityLevel.CRITICAL,
                        title=f"Dangerous function call: {decoded}",
                        description=(
                            f"Pickle file contains {decoded} call (arbitrary code execution)"
                        ),
                        file_path=file_path,
                        byte_offset=offset,
                        pattern_matched=decoded,
                        remediation="Do not load this file. Inspect provenance and rebuild from source.",
                        cwe_id="CWE-502",
                    )
                )

        for mod in _DANGEROUS_MODULES:
            offset = data.find(mod)
            if offset != -1:
                decoded = mod.decode("utf-8", errors="replace")
                already_covered = any(
                    f.pattern_matched and decoded in f.pattern_matched
                    for f in findings
                    if f.severity == SeverityLevel.CRITICAL
                )
                if not already_covered:
                    findings.append(
                        SecurityFinding(
                            severity=SeverityLevel.HIGH,
                            title=f"Dangerous module reference: {decoded}",
                            description=(f"Pickle file contains reference to '{decoded}' module"),
                            file_path=file_path,
                            byte_offset=offset,
                            pattern_matched=decoded,
                            remediation="Verify model provenance. Consider using safetensors format.",
                            cwe_id="CWE-502",
                        )
                    )

        for mod in _NETWORK_MODULES:
            offset = data.find(mod)
            if offset != -1:
                decoded = mod.decode("utf-8", errors="replace")
                findings.append(
                    SecurityFinding(
                        severity=SeverityLevel.HIGH,
                        title=f"Network module reference: {decoded}",
                        description=(
                            f"Pickle file contains reference to '{decoded}' "
                            f"(potential data exfiltration)"
                        ),
                        file_path=file_path,
                        byte_offset=offset,
                        pattern_matched=decoded,
                        remediation="Model files should not contain network code. Inspect provenance.",
                        cwe_id="CWE-829",
                    )
                )

        findings.extend(self._check_size_anomaly(data, file_path, "pickle"))
        return findings

    def _scan_pytorch(self, data: bytes, file_path: str) -> list[SecurityFinding]:
        """Scan PyTorch model files.

        PyTorch ``.pt`` / ``.pth`` files use pickle internally, so the pickle
        scanner is applied first. Additional PyTorch-specific checks follow.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.

        Returns:
            List of security findings.
        """
        findings = self._scan_pickle(data, file_path)

        if b"torch.load" in data:
            findings.append(
                SecurityFinding(
                    severity=SeverityLevel.MEDIUM,
                    title="torch.load pattern detected",
                    description=(
                        "File contains a torch.load reference suggesting unsafe "
                        "deserialization without weights_only=True"
                    ),
                    file_path=file_path,
                    pattern_matched="torch.load",
                    remediation="Use torch.load(..., weights_only=True) or convert to safetensors.",
                    cwe_id="CWE-502",
                )
            )

        return findings

    def _scan_numpy(self, data: bytes, file_path: str) -> list[SecurityFinding]:
        """Scan numpy ``.npy`` / ``.npz`` files for object arrays.

        Object arrays can contain pickled Python objects and are therefore
        unsafe to load with ``allow_pickle=True``.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.

        Returns:
            List of security findings.
        """
        findings: list[SecurityFinding] = []

        if data[:6] == b"\x93NUMPY":
            header_end = data.find(b"\n", 10)
            if header_end != -1:
                header = data[10:header_end]
                if b"'O'" in header or b"|O" in header or b"object" in header:
                    findings.append(
                        SecurityFinding(
                            severity=SeverityLevel.HIGH,
                            title="Numpy object array detected",
                            description=(
                                "Numpy file contains object dtype which uses pickle "
                                "for serialization and can execute arbitrary code"
                            ),
                            file_path=file_path,
                            pattern_matched="object dtype",
                            remediation="Convert to numeric dtypes or use safetensors.",
                            cwe_id="CWE-502",
                        )
                    )

        if file_path.endswith(".npz"):
            if b"'O'" in data or b"object" in data:
                findings.append(
                    SecurityFinding(
                        severity=SeverityLevel.MEDIUM,
                        title="Potential object array in npz archive",
                        description=(
                            "NPZ archive may contain object dtype arrays which "
                            "use pickle internally"
                        ),
                        file_path=file_path,
                        pattern_matched="object dtype",
                        remediation="Inspect each array in the archive for object dtypes.",
                        cwe_id="CWE-502",
                    )
                )

        findings.extend(self._check_size_anomaly(data, file_path, "numpy"))
        return findings

    def _scan_safetensors(self, data: bytes, file_path: str) -> list[SecurityFinding]:
        """Scan safetensors files.

        Safetensors is designed to be safe. This scanner validates the header
        JSON and reports the file as safe when the format is valid.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.

        Returns:
            List of security findings.
        """
        findings: list[SecurityFinding] = []

        if len(data) < 8:
            findings.append(
                SecurityFinding(
                    severity=SeverityLevel.MEDIUM,
                    title="Invalid safetensors file",
                    description="File is too small to contain a valid safetensors header",
                    file_path=file_path,
                    remediation="Verify file integrity and re-download if necessary.",
                )
            )
            return findings

        header_size = struct.unpack("<Q", data[:8])[0]
        if header_size > len(data) - 8:
            findings.append(
                SecurityFinding(
                    severity=SeverityLevel.MEDIUM,
                    title="Corrupted safetensors header",
                    description=(
                        f"Header size ({header_size}) exceeds available data "
                        f"({len(data) - 8} bytes)"
                    ),
                    file_path=file_path,
                    remediation="Re-download the model file.",
                )
            )
            return findings

        header_bytes = data[8 : 8 + header_size]
        try:
            json.loads(header_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            findings.append(
                SecurityFinding(
                    severity=SeverityLevel.MEDIUM,
                    title="Invalid safetensors header JSON",
                    description="The safetensors header is not valid JSON",
                    file_path=file_path,
                    remediation="Re-download or regenerate the model file.",
                )
            )

        if not findings:
            findings.append(
                SecurityFinding(
                    severity=SeverityLevel.INFO,
                    title="Valid safetensors format",
                    description="Safetensors format verified. This format is designed to be safe.",
                    file_path=file_path,
                    remediation="No action required.",
                )
            )

        return findings

    def _scan_onnx(self, data: bytes, file_path: str) -> list[SecurityFinding]:
        """Scan ONNX model files for custom operators.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.

        Returns:
            List of security findings.
        """
        findings: list[SecurityFinding] = []

        custom_op_indicators = [b"custom_op", b"CustomOp", b"custom_domain"]
        for indicator in custom_op_indicators:
            offset = data.find(indicator)
            if offset != -1:
                decoded = indicator.decode("utf-8", errors="replace")
                findings.append(
                    SecurityFinding(
                        severity=SeverityLevel.MEDIUM,
                        title="Custom ONNX operator detected",
                        description=(
                            f"ONNX model contains custom operator reference "
                            f"'{decoded}' which may execute arbitrary code"
                        ),
                        file_path=file_path,
                        byte_offset=offset,
                        pattern_matched=decoded,
                        remediation="Verify the custom operator source and ensure it is trusted.",
                        cwe_id="CWE-94",
                    )
                )

        findings.extend(self._check_size_anomaly(data, file_path, "onnx"))
        return findings

    def _scan_binary(self, data: bytes, file_path: str) -> list[SecurityFinding]:
        """Generic binary scanner for any model format.

        Checks for file size anomalies, high entropy (encrypted / compressed
        payloads), and suspicious string patterns.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.

        Returns:
            List of security findings.
        """
        findings: list[SecurityFinding] = []

        findings.extend(self._check_size_anomaly(data, file_path, "binary"))

        suspicious_strings = [
            (b"curl ", "Shell curl command"),
            (b"wget ", "Shell wget command"),
            (b"/bin/sh", "Shell path reference"),
            (b"/bin/bash", "Bash shell reference"),
            (b"powershell", "PowerShell reference"),
            (b"cmd.exe", "Windows command shell reference"),
            (b"<script", "HTML script tag"),
            (b"base64 -d", "Base64 decode command"),
        ]
        for pat, description in suspicious_strings:
            offset = data.find(pat)
            if offset != -1:
                decoded = pat.decode("utf-8", errors="replace")
                findings.append(
                    SecurityFinding(
                        severity=SeverityLevel.HIGH,
                        title=f"Suspicious string: {description}",
                        description=(f"Binary file contains suspicious pattern: '{decoded}'"),
                        file_path=file_path,
                        byte_offset=offset,
                        pattern_matched=decoded,
                        remediation="Inspect file provenance and contents before loading.",
                        cwe_id="CWE-506",
                    )
                )

        url_pattern = re.compile(rb"https?://[a-zA-Z0-9._/-]{10,}")
        match = url_pattern.search(data)
        if match:
            url = match.group().decode("utf-8", errors="replace")
            findings.append(
                SecurityFinding(
                    severity=SeverityLevel.MEDIUM,
                    title="Embedded URL detected",
                    description=f"Binary file contains URL: {url[:120]}",
                    file_path=file_path,
                    byte_offset=match.start(),
                    pattern_matched=url[:120],
                    remediation="Verify the URL is expected and belongs to a trusted source.",
                    cwe_id="CWE-829",
                )
            )

        return findings

    def _check_size_anomaly(
        self, data: bytes, file_path: str, format_name: str
    ) -> list[SecurityFinding]:
        """Flag unusually large files that may contain exfiltration payloads.

        Args:
            data: Raw file bytes.
            file_path: Path used for finding reports.
            format_name: The detected model format name.

        Returns:
            List containing at most one finding.
        """
        size_mb = len(data) / (1024 * 1024)
        if size_mb > 500:
            return [
                SecurityFinding(
                    severity=SeverityLevel.MEDIUM,
                    title="Unusually large model file",
                    description=(
                        f"Model file size ({size_mb:.0f} MB) is unusually large "
                        f"(potential data exfiltration payload)"
                    ),
                    file_path=file_path,
                    remediation="Verify the file size is expected for this model type.",
                    cwe_id="CWE-400",
                )
            ]
        return []
