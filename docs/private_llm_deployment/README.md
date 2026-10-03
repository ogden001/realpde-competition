# Private LLM Deployment

Purpose: first-step local-only Strata deployment on the existing GPU host. Deployment has not been installed: the 2026-10-03 preflight stopped under the required no-interference gate because an unrelated GPU process already holds about 12 GiB VRAM and port 8080 is already bound.

Target remains Qwen3.8-Flash-Next / IQ3_XXS / 131072-token context, vision off, served by official Strata on 127.0.0.1. No Strata SHA, model path, or launcher has been established. Review status: `REVIEW_REQUIRED` (blocked before installation).

See [the 2026-10-03 preflight review](reviews/strata_qwen38_3090_curl_bootstrap_20261003/README.md).
