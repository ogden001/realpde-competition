# Network, CUDA, and host preflight — 2026-10-04

- Coordination repo: `HEAD == origin/main == 7e5da38acd8791a71bcd85f8fc921807280d161b`; required continuation commit is HEAD; required base is an ancestor; push dry-run passed.
- `getent ahosts huggingface.co` returned IPv4 and IPv6 records. Direct IPv4 probe ended with curl error 35, `Recv failure: Connection reset by peer`, HTTP 000. Direct IPv6 probe ended with curl error 7, `Couldn't connect to server`, HTTP 000.
- `curl -4 -I -L https://hf-mirror.com/` returned HTTP/2 200. Current shell had no HTTP(S)/ALL/NO_PROXY variables. Proxy values were not printed.
- The official Strata-supported `HF_ENDPOINT=https://hf-mirror.com` was set for the resumed setup process only. No host network configuration, DNS, firewall, routing, `/etc/hosts`, proxy service, or TLS verification was changed.
- CUDA Toolkit `cuda-toolkit-13-0` 13.0.3-1; `nvcc` 13.0.88. NVIDIA driver 595.84; `nvidia-smi` healthy, runtime CUDA 13.2. Driver unchanged.
- GPU at startup: RTX 3090, 83 MiB used before model load; no unknown material GPU workload. RAM 62 GiB total / 59 GiB available. Root disk had 1001 GiB free. Port 8081 was free. Existing port 8080 wildcard listeners remained untouched.
- Pinned Strata repo `https://github.com/Niko1221/Strata.git`, SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`, clean. CUDA engine was already built (129/129) and setup reused it.
- Official setup log confirmed `Downloading from https://hf-mirror.com (HF_ENDPOINT)`, completed both configured IQ3_XXS shards, prepared the MTP assets/pack, and ended with `All set.` Both shard completion marker files are present. The pinned Strata setup/downloader ran unmodified; no manual download or validation bypass.
