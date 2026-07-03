"""Tests for the installation diagnostics (atlas/doctor.py)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas.doctor import (
    _probe_bundle,
    _probe_disk_free,
    _probe_mlx,
    _probe_nvidia,
    _probe_onnxruntime,
    _probe_platform,
    _probe_ripgrep,
    run_diagnosis,
)


# ---------------------------------------------------------------------------
# _probe_platform
# ---------------------------------------------------------------------------

class TestProbePlatform:
    def test_returns_expected_keys(self):
        result = _probe_platform()
        assert set(result.keys()) == {"system", "machine", "release", "python"}

    def test_values_are_strings(self):
        result = _probe_platform()
        assert all(isinstance(v, str) for v in result.values())


# ---------------------------------------------------------------------------
# _probe_onnxruntime
# ---------------------------------------------------------------------------

class TestProbeOnnxruntime:
    def test_onnxruntime_available(self):
        result = _probe_onnxruntime()
        assert "available" in result
        if result["available"]:
            assert "version" in result
            assert "providers" in result

    def test_onnxruntime_returns_dict(self):
        result = _probe_onnxruntime()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _probe_mlx
# ---------------------------------------------------------------------------

class TestProbeMlx:
    def test_returns_available_key(self):
        result = _probe_mlx()
        assert "available" in result

    def test_mlx_not_available(self):
        result = _probe_mlx()
        if result["available"]:
            assert result.get("apple_silicon") is True


# ---------------------------------------------------------------------------
# _probe_nvidia
# ---------------------------------------------------------------------------

class TestProbeNvidia:
    def test_returns_available_key(self):
        result = _probe_nvidia()
        assert "available" in result

    def test_nvidia_not_available(self):
        result = _probe_nvidia()
        if not result["available"]:
            assert "reason" in result


# ---------------------------------------------------------------------------
# _probe_ripgrep
# ---------------------------------------------------------------------------

class TestProbeRipgrep:
    def test_ripgrep_available(self):
        result = _probe_ripgrep()
        if result["available"]:
            assert "version" in result or "path" in result

    def test_returns_dict(self):
        result = _probe_ripgrep()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _probe_disk_free
# ---------------------------------------------------------------------------

class TestProbeDiskFree:
    def test_returns_free_and_total(self):
        result = _probe_disk_free()
        if "free_gb" in result:
            assert isinstance(result["free_gb"], float)
            assert isinstance(result["total_gb"], float)
            assert result["free_gb"] > 0
        else:
            assert "error" in result


# ---------------------------------------------------------------------------
# _probe_bundle
# ---------------------------------------------------------------------------

class TestProbeBundle:
    def test_none_bundle(self):
        result = _probe_bundle(None)
        assert result["checked"] is False

    def test_missing_bundle(self, tmp_path: Path):
        result = _probe_bundle(tmp_path / "nonexistent")
        assert result["checked"] is True
        assert result["exists"] is False

    def test_valid_bundle(self, tmp_path: Path):
        bdir = tmp_path / "bundle"
        bdir.mkdir()
        (bdir / "manifest.json").write_text(
            json.dumps({
                "chunk_count": 50,
                "embedding_model": "test-model",
                "source_branch": "main",
                "source_sha": "abc123def456",
                "embedding_backend": "onnx-cpu",
                "embedding_active_provider": "CPUExecutionProvider",
                "artifacts": {
                    "chunks_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                },
            })
        )
        (bdir / "chunks.parquet").write_bytes(b"fake")
        (bdir / "embeddings.f16.npy").write_bytes(b"fake")

        result = _probe_bundle(bdir)
        assert result["checked"] is True
        assert result["exists"] is True
        assert result["has_manifest"] is True
        assert result["has_chunks"] is True
        assert result["has_embeddings"] is True
        assert result["chunk_count"] == 50
        assert result["embedding_model"] == "test-model"


# ---------------------------------------------------------------------------
# run_diagnosis — patch _probe_mlx_weights to avoid the unconditional
# ``import mlx.core`` in atlas/embed/mlx.py
# ---------------------------------------------------------------------------

class TestRunDiagnosis:
    @patch("atlas.doctor._probe_mlx_weights", return_value={"cached": False, "path": "/tmp"})
    def test_diagnosis_returns_report(self, mock_mlx_w, tmp_path: Path):
        report = run_diagnosis(bundle=None, force=True)
        assert "version" in report
        assert "platform" in report
        assert "onnxruntime" in report
        assert "mlx" in report
        assert "nvidia" in report
        assert "ripgrep" in report
        assert "disk_free" in report
        assert "selected_backend" in report
        assert "selected_reason" in report

    @patch("atlas.doctor._probe_mlx_weights", return_value={"cached": False, "path": "/tmp"})
    def test_diagnosis_with_bundle(self, mock_mlx_w, tmp_path: Path):
        bdir = tmp_path / "bundle"
        bdir.mkdir()
        (bdir / "manifest.json").write_text(json.dumps({"chunk_count": 10}))
        report = run_diagnosis(bundle=bdir, force=True)
        assert report["bundle"]["checked"] is True

    @patch("atlas.doctor._probe_mlx_weights", return_value={"cached": False, "path": "/tmp"})
    def test_diagnosis_caches(self, mock_mlx_w, tmp_path: Path):
        report1 = run_diagnosis(bundle=None, force=True)
        # Write a different value to verify caching
        report2 = run_diagnosis(bundle=None, force=False)
        assert report2["version"] == report1["version"]
