"""Tests for checkllm.model_audit — ML model security scanner."""

from __future__ import annotations

import json
import pickle
import struct
import tempfile
from pathlib import Path

import pytest

from checkllm.model_audit import (
    AuditResult,
    ModelAuditor,
    SecurityFinding,
    SeverityLevel,
)


class TestSeverityLevel:
    def test_enum_values(self):
        assert SeverityLevel.INFO == "info"
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.CRITICAL == "critical"


class TestSecurityFinding:
    def test_construction(self):
        finding = SecurityFinding(
            severity=SeverityLevel.HIGH,
            title="Test finding",
            description="A test issue",
            file_path="test.pkl",
            remediation="Fix it",
        )
        assert finding.severity == SeverityLevel.HIGH
        assert finding.cwe_id is None

    def test_with_cwe(self):
        finding = SecurityFinding(
            severity=SeverityLevel.CRITICAL,
            title="Code exec",
            description="Arbitrary code execution",
            file_path="bad.pkl",
            cwe_id="CWE-502",
            remediation="Remove file",
        )
        assert finding.cwe_id == "CWE-502"


class TestAuditResult:
    def _make_result(self, findings: list[SecurityFinding] | None = None) -> AuditResult:
        return AuditResult(
            file_path="model.pkl",
            file_type="pickle",
            file_size_bytes=1024,
            scan_time_ms=5,
            findings=findings or [],
        )

    def test_is_safe_with_no_findings(self):
        result = self._make_result()
        assert result.is_safe is True

    def test_is_safe_with_info_only(self):
        findings = [
            SecurityFinding(
                severity=SeverityLevel.INFO,
                title="Info",
                description="informational",
                file_path="f.pkl",
                remediation="None needed",
            )
        ]
        result = self._make_result(findings)
        assert result.is_safe is True

    def test_is_safe_with_medium(self):
        findings = [
            SecurityFinding(
                severity=SeverityLevel.MEDIUM,
                title="Medium",
                description="medium issue",
                file_path="f.pkl",
                remediation="Consider fixing",
            )
        ]
        result = self._make_result(findings)
        assert result.is_safe is True

    def test_not_safe_with_high(self):
        findings = [
            SecurityFinding(
                severity=SeverityLevel.HIGH,
                title="High",
                description="high issue",
                file_path="f.pkl",
                remediation="Fix now",
            )
        ]
        result = self._make_result(findings)
        assert result.is_safe is False

    def test_not_safe_with_critical(self):
        findings = [
            SecurityFinding(
                severity=SeverityLevel.CRITICAL,
                title="Critical",
                description="critical issue",
                file_path="f.pkl",
                remediation="Delete file",
            )
        ]
        result = self._make_result(findings)
        assert result.is_safe is False

    def test_counts(self):
        findings = [
            SecurityFinding(
                severity=SeverityLevel.CRITICAL,
                title="c",
                description="c",
                file_path="f",
                remediation="r",
            ),
            SecurityFinding(
                severity=SeverityLevel.HIGH,
                title="h",
                description="h",
                file_path="f",
                remediation="r",
            ),
            SecurityFinding(
                severity=SeverityLevel.HIGH,
                title="h2",
                description="h2",
                file_path="f",
                remediation="r",
            ),
            SecurityFinding(
                severity=SeverityLevel.MEDIUM,
                title="m",
                description="m",
                file_path="f",
                remediation="r",
            ),
            SecurityFinding(
                severity=SeverityLevel.LOW,
                title="l",
                description="l",
                file_path="f",
                remediation="r",
            ),
        ]
        result = self._make_result(findings)
        assert result.critical_count == 1
        assert result.high_count == 2
        assert result.medium_count == 1
        assert result.low_count == 1

    def test_summary_format(self):
        findings = [
            SecurityFinding(
                severity=SeverityLevel.CRITICAL,
                title="Dangerous call",
                description="os.system detected",
                file_path="model.pkl",
                remediation="Delete it",
            ),
            SecurityFinding(
                severity=SeverityLevel.MEDIUM,
                title="Opcode",
                description="REDUCE opcode found",
                file_path="model.pkl",
                remediation="Use safetensors",
            ),
        ]
        result = self._make_result(findings)
        text = result.summary()
        assert "ModelAudit: model.pkl" in text
        assert "1 critical" in text
        assert "1 medium" in text
        assert "CRITICAL" in text
        assert "MEDIUM" in text

    def test_summary_no_findings(self):
        result = self._make_result()
        text = result.summary()
        assert "Findings: 0" in text

    def test_to_json(self):
        result = self._make_result()
        j = result.to_json()
        data = json.loads(j)
        assert data["file_path"] == "model.pkl"
        assert data["file_type"] == "pickle"


class TestSupportedExtensions:
    def test_all_extensions_present(self):
        expected = {
            ".pkl",
            ".pickle",
            ".joblib",
            ".pt",
            ".pth",
            ".bin",
            ".onnx",
            ".safetensors",
            ".h5",
            ".hdf5",
            ".keras",
            ".tflite",
            ".pb",
            ".gguf",
            ".ggml",
            ".npy",
            ".npz",
        }
        assert set(ModelAuditor.SUPPORTED_EXTENSIONS.keys()) == expected


class TestPickleScanner:
    def test_safe_pickle(self):
        data = pickle.dumps({"hello": "world", "numbers": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            critical_or_high = [
                f
                for f in result.findings
                if f.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)
            ]
            assert len(critical_or_high) == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_os_system(self):
        # Embed the literal attack pattern that the scanner looks for.
        # This is raw byte data for testing detection, NOT executed code.
        attack = b"os" + b"." + b"system"
        payload = b"\x80\x05\x95\x00\x00\x00\x00" + attack + b"('cmd')"
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            critical = [f for f in result.findings if f.severity == SeverityLevel.CRITICAL]
            assert len(critical) >= 1
            assert any("os.system" in f.description for f in critical)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_subprocess_popen(self):
        # Embed the literal attack pattern for detection testing
        attack = b"subprocess" + b"." + b"Popen"
        payload = b"\x80\x05\x95\x00\x00\x00\x00" + attack + b"(['ls'])"
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            critical = [f for f in result.findings if f.severity == SeverityLevel.CRITICAL]
            assert len(critical) >= 1
            assert any("subprocess.Popen" in f.description for f in critical)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_import_pattern(self):
        payload = b"\x80\x05\x95\x00\x00\x00\x00c__import__\nfoo\n"
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            high = [f for f in result.findings if f.severity == SeverityLevel.HIGH]
            assert len(high) >= 1
            assert any("__import__" in (f.pattern_matched or "") for f in high)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_network_modules(self):
        payload = b"\x80\x05\x95\x00\x00\x00\x00csocket\nconnect\n"
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            network_findings = [f for f in result.findings if "socket" in (f.pattern_matched or "")]
            assert len(network_findings) >= 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_dangerous_eval_call(self):
        # Byte pattern containing the string "eval(" to test detection
        dangerous_bytes = b"eval("
        payload = b"\x80\x05\x95\x00\x00\x00\x00" + dangerous_bytes + b"something)"
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            critical = [f for f in result.findings if f.severity == SeverityLevel.CRITICAL]
            assert len(critical) >= 1
        finally:
            Path(path).unlink(missing_ok=True)


class TestSafetensorsScanner:
    def test_valid_safetensors(self):
        header = json.dumps(
            {"weight": {"dtype": "F32", "shape": [10], "data_offsets": [0, 40]}}
        ).encode()
        data = struct.pack("<Q", len(header)) + header + b"\x00" * 40
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            assert result.is_safe is True
            assert any(
                f.severity == SeverityLevel.INFO and "safe" in f.description.lower()
                for f in result.findings
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_safetensors_too_small(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(b"\x00\x00")
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            assert any("too small" in f.description for f in result.findings)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_corrupted_safetensors_header(self):
        data = struct.pack("<Q", 9999) + b"\x00" * 10
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            assert any(
                "Corrupted" in f.title or "exceeds" in f.description for f in result.findings
            )
        finally:
            Path(path).unlink(missing_ok=True)


class TestBinaryScanner:
    def test_detects_suspicious_strings(self):
        payload = b"\x00\x00/bin/sh\x00\x00some data"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            high = [f for f in result.findings if f.severity == SeverityLevel.HIGH]
            assert len(high) >= 1
            assert any("/bin/sh" in (f.pattern_matched or "") for f in high)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detects_embedded_url(self):
        payload = b"\x00\x00https://evil.example.com/payload\x00\x00"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            url_findings = [f for f in result.findings if "URL" in f.title]
            assert len(url_findings) >= 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clean_binary_no_high_findings(self):
        payload = b"\x00" * 100 + b"just some model weights data" + b"\x00" * 100
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            assert result.is_safe is True
        finally:
            Path(path).unlink(missing_ok=True)


class TestNumpyScanner:
    def test_safe_numpy_file(self):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        np.save(path, arr)
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            high_or_critical = [
                f
                for f in result.findings
                if f.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)
            ]
            assert len(high_or_critical) == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_object_dtype_detected(self):
        import numpy as np

        arr = np.array(["hello", "world"], dtype=object)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        np.save(path, arr)
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            object_findings = [f for f in result.findings if "object" in f.description.lower()]
            assert len(object_findings) >= 1
        finally:
            Path(path).unlink(missing_ok=True)


class TestScanFile:
    def test_file_not_found(self):
        auditor = ModelAuditor()
        with pytest.raises(FileNotFoundError):
            auditor.scan("nonexistent_model.pkl")

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            auditor = ModelAuditor()
            with pytest.raises(ValueError, match="Unsupported"):
                auditor.scan(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_scan_returns_audit_result(self):
        data = pickle.dumps([1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            assert isinstance(result, AuditResult)
            assert result.file_path == path
            assert result.file_type == "pickle"
            assert result.file_size_bytes > 0
            assert result.scan_time_ms >= 0
        finally:
            Path(path).unlink(missing_ok=True)


class TestScanDirectory:
    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            safe = pickle.dumps({"safe": True})
            (Path(tmpdir) / "model_a.pkl").write_bytes(safe)
            (Path(tmpdir) / "model_b.pkl").write_bytes(safe)
            (Path(tmpdir) / "readme.txt").write_text("not a model")

            auditor = ModelAuditor()
            results = auditor.scan_directory(tmpdir)
            assert len(results) == 2
            for r in results:
                assert r.file_type == "pickle"

    def test_scan_directory_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            safe = pickle.dumps({"safe": True})
            (Path(tmpdir) / "top.pkl").write_bytes(safe)
            (subdir / "nested.pkl").write_bytes(safe)

            auditor = ModelAuditor()
            results = auditor.scan_directory(tmpdir, recursive=True)
            assert len(results) == 2

    def test_scan_directory_not_found(self):
        auditor = ModelAuditor()
        with pytest.raises(FileNotFoundError):
            auditor.scan_directory("/nonexistent/path")


class TestPytorchScanner:
    def test_pytorch_uses_pickle_scanner(self):
        attack = b"os" + b"." + b"system"
        payload = b"\x80\x05\x95\x00\x00\x00\x00" + attack + b"('cmd')"
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            assert result.file_type == "pytorch"
            critical = [fi for fi in result.findings if fi.severity == SeverityLevel.CRITICAL]
            assert len(critical) >= 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_torch_load_pattern(self):
        payload = b"\x80\x05\x95\x00\x00\x00\x00torch.load data"
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(payload)
            path = f.name
        try:
            auditor = ModelAuditor()
            result = auditor.scan(path)
            torch_findings = [
                f for f in result.findings if "torch.load" in (f.pattern_matched or "")
            ]
            assert len(torch_findings) >= 1
        finally:
            Path(path).unlink(missing_ok=True)
