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

## All Growt Plugins

Open-source plugins and SDKs for the [Transfer Oracle](https://transferoracle.ai) structural AI auditing API.
Plugin code is MPL-2.0; API access is commercial and requires an [API key](https://transferoracle.ai/growt/plugins).

| Plugin | Platform | What it does |
|--------|----------|-------------|
| [growt-client](https://github.com/LeeSandberg/growt-client) | Core | Python client library |
| [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) | NVIDIA | ModelOpt quantization audit |
| [growt-quark](https://github.com/LeeSandberg/growt-quark) | AMD | Quark quantization audit |
| [growt-nemo](https://github.com/LeeSandberg/growt-nemo) | NVIDIA | NeMo / PyTorch Lightning callback |
| [growt-vllm](https://github.com/LeeSandberg/growt-vllm) | NVIDIA + AMD | vLLM inference monitor |
| [growt-triton](https://github.com/LeeSandberg/growt-triton) | NVIDIA | Triton Inference Server monitor |
| [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) | NVIDIA | TensorRT engine validator |
| [growt-tao](https://github.com/LeeSandberg/growt-tao) | NVIDIA | TAO Toolkit pipeline |
| [mlflow-growt](https://github.com/LeeSandberg/mlflow-growt) | MLflow | Evaluator + deployment gate |
| [growt-huggingface](https://github.com/LeeSandberg/growt-huggingface) | HuggingFace | TrainerCallback + Model Card |
| [growt-wandb](https://github.com/LeeSandberg/growt-wandb) | W&B | Callback + artifact + registry gate |
| [growt-airflow](https://github.com/LeeSandberg/growt-airflow) | Airflow | Pre-deployment audit operator |
| [growt-kubeflow](https://github.com/LeeSandberg/growt-kubeflow) | Kubeflow | Pipeline validation component |
| [growt-kserve](https://github.com/LeeSandberg/growt-kserve) | KServe | Pre-serve validation transformer |
| [growt-dagster](https://github.com/LeeSandberg/growt-dagster) | Dagster | Asset + resource for audit |
| [growt-dvc](https://github.com/LeeSandberg/growt-dvc) | DVC | Pre-push model validation |
| [growt-bentoml](https://github.com/LeeSandberg/growt-bentoml) | BentoML | Pre-serve validation hook |
| [growt-argo](https://github.com/LeeSandberg/growt-argo) | Argo | Workflow validation template |
| [growt-prefect](https://github.com/LeeSandberg/growt-prefect) | Prefect | Task + block for audit |
| [growt-clearml](https://github.com/LeeSandberg/growt-clearml) | ClearML | Pipeline step + callback |
| [growt-docker](https://github.com/LeeSandberg/growt-docker) | Docker/OCI | Audit metadata in containers |

**Links:** [API Docs](https://transferoracle.ai/growt/docs) · [Get API Key](https://transferoracle.ai/growt/plugins) · [LLM Benchmark](https://transferoracle.ai/growt/llm-benchmark) · [All Benchmarks](https://transferoracle.ai/benchmarks)
