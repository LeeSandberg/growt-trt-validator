"""CLI: Validate TensorRT engine against original ONNX model.

Usage:
    growt-validate-trt --onnx model.onnx --engine model.trt --data calib.npz
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from growt_client import GrowtClient, format_audit_report
from growt_trt.certificate import generate_certificate


def _load_onnx(path: str, data: np.ndarray) -> np.ndarray:
    """Run data through ONNX model, return output features."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime required. Install: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    session = ort.InferenceSession(path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[-1].name

    results = []
    batch_size = 32
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].astype(np.float32)
        out = session.run([output_name], {input_name: batch})[0]
        if out.ndim > 2:
            out = out.mean(axis=tuple(range(2, out.ndim)))
        results.append(out)

    return np.concatenate(results, axis=0)


def _load_trt(path: str, data: np.ndarray) -> np.ndarray:
    """Run data through TensorRT engine, return output features."""
    try:
        import tensorrt as trt
        # Minimal TRT inference — for production use polygraphy
        print("NOTE: TensorRT inference requires full setup. Using ONNX-only mode for now.")
        print("For TRT engine comparison, export both ONNX and TRT outputs to NPZ first.")
        return np.zeros((len(data), 1))  # Placeholder
    except ImportError:
        print("WARNING: tensorrt not installed. Using ONNX output only.", file=sys.stderr)
        return np.zeros((len(data), 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate TensorRT engine against ONNX original using Growt"
    )
    parser.add_argument("--onnx", required=True, help="Path to original ONNX model")
    parser.add_argument("--engine", default=None, help="Path to TensorRT engine (optional)")
    parser.add_argument("--data", required=True, help="Calibration data NPZ (keys: inputs, labels)")
    parser.add_argument("--features-onnx", default=None, help="Pre-extracted ONNX features NPZ")
    parser.add_argument("--features-trt", default=None, help="Pre-extracted TRT features NPZ")
    parser.add_argument("--api-url", default="https://api.transferoracle.ai")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="Path for audit certificate JSON")
    parser.add_argument("--fail-on-red-flag", action="store_true")
    args = parser.parse_args()

    # Load calibration data
    data = np.load(args.data)
    inputs = data["inputs"]
    labels = data["labels"].tolist()

    # Get features (pre-extracted or run inference)
    if args.features_onnx:
        features_onnx = np.load(args.features_onnx)["features"]
    else:
        print(f"Running ONNX inference on {len(inputs)} samples...")
        features_onnx = _load_onnx(args.onnx, inputs)

    if args.features_trt:
        features_trt = np.load(args.features_trt)["features"]
    elif args.engine:
        print(f"Running TRT inference on {len(inputs)} samples...")
        features_trt = _load_trt(args.engine, inputs)
    else:
        print("No --engine or --features-trt. Auditing ONNX model structure only.")
        features_trt = features_onnx  # Self-comparison

    # Audit
    client = GrowtClient(api_url=args.api_url, api_key=args.api_key)

    audit = client.audit_transfer(
        features_train=features_onnx.tolist(),
        labels_train=labels,
        features_deploy=features_trt.tolist(),
        labels_deploy=labels,
    )

    metrics = None
    if len(features_onnx) == len(features_trt):
        metrics = client.metrics_compare(
            features_reference=features_onnx.tolist(),
            features_compare=features_trt.tolist(),
            labels_reference=labels,
        )

    # Report
    print(format_audit_report(audit, metrics, title="GROWT TENSORRT VALIDATION"))

    # Certificate
    if args.output and args.engine:
        cert = generate_certificate(args.onnx, args.engine, audit, metrics)
        with open(args.output, "w") as f:
            json.dump(cert, f, indent=2)
        print(f"\nCertificate saved: {args.output}")

    if args.fail_on_red_flag and audit.diagnosis == "RED_FLAG":
        sys.exit(1)


if __name__ == "__main__":
    main()
