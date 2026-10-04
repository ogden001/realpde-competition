# Strata Qwen 3.8 Local API Bootstrap — 2026-10-04

Status: `REVIEW_REQUIRED`
Bootstrap gate: `LLM_BOOTSTRAP_PASS`
Continuation commit: `7e5da38acd8791a71bcd85f8fc921807280d161b`
Execution commit: `7e5da38acd8791a71bcd85f8fc921807280d161b`

## Result

The model downloaded and prepared through the existing official Strata v0.1.38 installation, and the API is running locally at `127.0.0.1:8081`. Health, models, and one Chinese chat completion all returned HTTP 200. The server is bound to loopback only.

Direct access to `huggingface.co` failed (IPv4 TLS connection reset; IPv6 connect failed), while `https://hf-mirror.com/` returned HTTP 200. No proxy environment variables were set. The setup shell used the Strata-supported `HF_ENDPOINT=https://hf-mirror.com`; the Strata downloader and its pinned revision and integrity-validation path remained unmodified and enabled. Both shard `.done` completion markers were created by setup. No manual GGUF download, alternate downloader, checksum bypass, or TLS bypass was used.

## Frozen configuration and provenance

- Model: Qwen3.8-Flash-Next / IQ3_XXS; context `131072`; vision OFF; KV `int8`; normal Strata MTP preparation.
- Strata upstream: `https://github.com/Niko1221/Strata.git`; SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`; version `v0.1.38`; source worktree clean.
- CUDA Toolkit: `nvcc` 13.0.88. NVIDIA driver: 595.84, healthy and unchanged.
- Strata CUDA engine build: 129/129 targets passed; executable `~/ai-stack/strata/engine/strata`, 39 MiB. Setup reused this existing engine.
- Model directory: `~/ai-stack/models/models/IQ3_XXS`. Shard 1: 47,039,860,096 bytes; shard 2: 28,800,138,432 bytes; combined: 75,839,998,528 bytes (~75.84 GB). Strata generated completion markers for both shards.
- Prepared MTP path: `~/ai-stack/models/mtp/rt`; pack path: `~/ai-stack/models/packs/iq3_xxs` (includes `dense.bin` 1,538,035,200 bytes, `experts.bin` 707,788,800 bytes, and `index.txt`).
- API endpoint: `http://127.0.0.1:8081`; health reports context 131072 and `images=false`. `/v1/models` returned `qwen3.8-flash-next-iq3_xxs`.

## Commands

Resumed the original setup with the session-local CUDA environment and official HF mirror setting:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
cd "$HOME/ai-stack/strata"
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

Official generated launcher used for startup (run with `BROWSER=/usr/bin/true` so it did not open a browser):

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export BROWSER=/usr/bin/true
cd "$HOME/ai-stack/strata"
./run-iq3_xxs.sh
```

The launcher ran detached for this smoke. Setup log: `~/ai-stack/logs/setup-hf-mirror.log`; server log: `~/ai-stack/logs/server-smoke-20261004.log`. The task-specific process IDs and runtime snapshot are in `runtime.json`. Large logs and model/cache files remain on the GPU host and are not in Git.

## Runtime snapshot after the single chat smoke

- GPU: 23,744 / 24,576 MiB VRAM used; 0% sampled utilization. Strata engine PID 664860 used 23,576 MiB.
- System RAM: 62 GiB total; 45 GiB used; 17 GiB available; swap 8 GiB total / 3 GiB used.
- Strata server PID 664852; listener exactly `127.0.0.1:8081`.
- Existing unrelated wildcard listener on 8080 was left untouched.

## Scope

- RealPDE scientific assets NOT accessed
- Agent framework NOT installed
- RAG NOT deployed
- public ingress NOT enabled
- calibration NOT run
- persistent service NOT configured
- NVIDIA driver NOT changed
- no separate Web UI, proxy service, alternate engine, or model variant added
- no model binaries, caches, credentials, or large logs committed

Final state is `REVIEW_REQUIRED` pending review; no next-stage components were installed.
