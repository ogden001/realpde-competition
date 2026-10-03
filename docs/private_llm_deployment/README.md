# Private LLM Deployment

Purpose: first-step local-only Strata deployment on the existing GPU host. The approved Strata v0.1.38 source tree remains installed at `~/ai-stack/strata` (SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`) and its CUDA engine compiled successfully with CUDA Toolkit 13.0.88. The model download then stopped because the Strata downloader could not reach `huggingface.co` (`Network is unreachable`).

The target remains Qwen3.8-Flash-Next / IQ3_XXS / 131072-token context, vision off, normal MTP path, and INT8 KV, exposed only at `127.0.0.1:8081`. The model has not downloaded and the server has not started, so API health is not yet established. Review status: `REVIEW_REQUIRED`.

See [the bootstrap review evidence](reviews/strata_qwen38_3090_curl_bootstrap_20261003/README.md).
