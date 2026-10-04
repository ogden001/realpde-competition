# Private LLM Deployment

Purpose: a local-only LLM node on the existing GPU host. The deployment is now running official Strata v0.1.38 at source SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`, with its source-built CUDA engine. Model: Qwen3.8-Flash-Next IQ3_XXS, context 131072, vision off, INT8 KV, normal MTP path.

The OpenAI-compatible API is bound only to `127.0.0.1:8081`. Because direct Hugging Face IPv4/IPv6 access failed, the official Strata downloader used its supported `HF_ENDPOINT=https://hf-mirror.com` setting. Both model shards and MTP preparation completed; health, models, and one Chinese chat request returned HTTP 200.

GPU host: Ubuntu 24.04.4 LTS, RTX 3090 24 GiB, 62 GiB RAM, CUDA Toolkit 13.0.88, NVIDIA driver 595.84. No driver changes, additional web UI, public ingress, Agent, RAG, calibration, or persistent service were added. Review status: `REVIEW_REQUIRED`.

See [bootstrap review evidence](reviews/strata_qwen38_3090_curl_bootstrap_20261003/README.md).
