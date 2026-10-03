# Preflight details

- Coordination Git preflight passed: working tree clean, `HEAD == origin/main == a2c4f30bd928fcd9af417b13926c70a90fd1390a`, required commit ancestor check passed, and push dry-run succeeded.
- Host identity and private network addresses are omitted from this committed evidence. Connection used the preconfigured SSH alias `gpu`.
- Capacity: Ubuntu 24.04.4 LTS; i7-13700KF; RTX 3090 24576 MiB, driver 595.84, CUDA 13.2; RAM 62 GiB total / 58 GiB available; swap 8 GiB total / 3.9 GiB used; root filesystem 1012 GiB free. NVIDIA tools responded successfully.
- The preferred `/opt/ai-stack` root was absent; no install directories were created.
- Conflict: PID 3619115 is running `python3.12 ... uvicorn app.main:app --host 0.0.0.0 --port 5010`, using 12082 MiB VRAM. Port 8080 is already listening on wildcard IPv4 and IPv6 addresses. No process was stopped and no port was changed.
- H7 blocks deployment before any download or startup. Swap activity could not be inferred from a single snapshot.
