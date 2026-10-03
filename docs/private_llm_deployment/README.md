# Private LLM Deployment

Purpose: first-step local-only Strata deployment on the existing GPU host. The official Strata source is cloned under `~/ai-stack/strata`; model installation stopped before downloading model assets because the current upstream release has no ready-made engine and its documented source-build path requires CUDA Toolkit 13.0 installation through sudo.

Target remains Qwen3.8-Flash-Next / IQ3_XXS / 131072-token context, vision off, normal MTP path, and INT8 KV. The intended local API is `127.0.0.1:8081` because the unrelated listener on 8080 remains active. Strata SHA: `99f3dbd0b21d1401b3769e0c0d963913607f380b`. Review status: `REVIEW_REQUIRED` (waiting for the required CUDA toolkit installation).

See [the 2026-10-03 review evidence](reviews/strata_qwen38_3090_curl_bootstrap_20261003/README.md).
