# growt-trt-validator

**Validate [TensorRT](https://developer.nvidia.com/tensorrt) engines against original ONNX models using Growt** — certify before you deploy.

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

> "trtexec builds fast. Growt certifies safe."

## What is this?

CLI tool that compares a TensorRT engine's outputs against the original ONNX model on calibration data. Reports SQNR, per-class coverage, and emits a JSON audit certificate.

## Install

```bash
pip install growt-trt-validator
```

## Usage

```bash
# After building your TRT engine:
trtexec --onnx=model.onnx --fp16 --saveEngine=model.engine

# Validate with Growt:
growt-validate-trt \
  --onnx model.onnx \
  --engine model.engine \
  --data calibration.npz \
  --api-key your-key \
  --output certificate.json \
  --fail-on-red-flag
```

## Certificate Output

```json
{
  "diagnosis": "SAFE",
  "sqnr_db": 22.1,
  "coverage_pct": 0.973,
  "onnx_sha256": "abc123...",
  "trt_sha256": "def456...",
  "signed_by": "growt://transferoracle.ai"
}
```

## License

[MPL-2.0](LICENSE)

## Status & Contributing

This is an early release to get the integration started. The code works but is not battle-tested in production yet. We welcome contributions:

- Bug fixes and improvements — PRs welcome
- New features and endpoint integrations
- Better error handling and edge cases
- Documentation improvements
- Test coverage

Open an issue or submit a PR on GitHub. All contributions must be compatible with the MPL-2.0 license.


## Related

- [Documentation](https://transferoracle.ai/growt/docs) — API reference, all plugins, tiers
- [growt-client](https://github.com/LeeSandberg/growt-client) — Python client (shared by all plugins)
- [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) — NVIDIA ModelOpt
- [growt-quark](https://github.com/LeeSandberg/growt-quark) — AMD Quark
- [growt-nemo](https://github.com/LeeSandberg/growt-nemo) — NeMo / PyTorch Lightning
- [growt-vllm](https://github.com/LeeSandberg/growt-vllm) — vLLM (NVIDIA + AMD)
- [growt-triton](https://github.com/LeeSandberg/growt-triton) — Triton Inference Server
- [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) — TensorRT validator
- [growt-tao](https://github.com/LeeSandberg/growt-tao) — TAO Toolkit

