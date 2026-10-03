# Private LLM Deployment

Purpose: first-step local-only Strata deployment on the existing GPU host. The official Strata source remains installed under `~/ai-stack/strata` at the approved SHA. Its setup cannot resume yet: the current host does not expose CUDA Toolkit 13.0 (`nvcc` and the dpkg package are absent), and non-interactive sudo requires a password.

Target remains Qwen3.8-Flash-Next / IQ3_XXS / 131072-token context, vision off, normal MTP path, and INT8 KV. The intended local API is `127.0.0.1:8081`; the unrelated 8080 listener remains untouched. Review status: `REVIEW_REQUIRED` (waiting for CUDA Toolkit 13.0 to be installed on the checked host).

See [the 2026-10-03/04 review evidence](reviews/strata_qwen38_3090_curl_bootstrap_20261003/README.md).
