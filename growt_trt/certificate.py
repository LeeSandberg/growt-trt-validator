"""Generate JSON audit certificates for TensorRT engine validation."""

from __future__ import annotations

import datetime
import hashlib
from typing import Any, Optional

from growt_client import AuditResult, MetricsResult


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_certificate(
    onnx_path: str,
    engine_path: str,
    audit: AuditResult,
    metrics: Optional[MetricsResult] = None,
) -> dict[str, Any]:
    """Generate a JSON audit certificate for a TRT engine conversion."""
    return {
        "version": "1.0",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "onnx_model": onnx_path,
        "onnx_sha256": _file_hash(onnx_path),
        "trt_engine": engine_path,
        "trt_sha256": _file_hash(engine_path),
        "diagnosis": audit.diagnosis,
        "safe_to_deploy": audit.safe_to_deploy,
        "transfer_oracle": audit.transfer_oracle,
        "coverage_pct": audit.coverage_pct,
        "sqnr_db": metrics.sqnr_db if metrics else None,
        "cosine_mean": metrics.cosine_mean if metrics else None,
        "rank_correlation": metrics.rank_correlation if metrics else None,
        "n_flagged_samples": audit.n_flagged_samples,
        "classes_at_risk": audit.classes_at_risk,
        "recommendations": audit.recommendations,
        "signed_by": "growt://transferoracle.ai",
    }
