# Preflight and setup details

- Coordination repo synchronized at `9cedb64f9713c3861d9172b862aa04c70cb8c8af`; required commit `a2c4f30bd928fcd9af417b13926c70a90fd1390a` is an ancestor. Push dry-run passed.
- User confirmed the earlier GPU memory process had been stopped. Fresh `nvidia-smi`: 83 MiB / 24576 MiB used; only Xorg and gnome-shell. PID 3619115 is absent.
- Capacity: Ubuntu 24.04.4 LTS; i7-13700KF; RTX 3090; driver 595.84; 62 GiB RAM / 59 GiB available; swap 8 GiB / 2.7 GiB used; root filesystem 1012 GiB free. Five `vmstat` samples showed no sustained swap traffic.
- Existing listener on port 8080 remains wildcard IPv4/IPv6. Port 8081 was free; official Strata docs advise 8081 when 8080 is occupied. Setup specifies `--host 127.0.0.1 --port 8081`.
- Installation root: `~/ai-stack`; Strata repository: `~/ai-stack/strata`; model directory: `~/ai-stack/models`; logs: `~/ai-stack/logs`. No pre-existing stack was found at these paths.
- Official Strata upstream: `https://github.com/Niko1221/Strata`; SHA `99f3dbd0b21d1401b3769e0c0d963913607f380b`; branch `main`; repository worktree clean. Current release is v0.1.38.
- Setup installed Python packages in its virtual environment and downloaded llama.cpp source revision `3cf0325`. No model data was downloaded (`~/ai-stack/models`: 12 KiB).
- Ready-made engine lookup returned HTTP 404. Official setup fell back to compile and required NVIDIA CUDA Toolkit 13.0. It downloaded `/tmp/cuda-keyring.deb`, then `sudo dpkg -i` failed: `sudo: a terminal is required to read the password` / `sudo: a password is required`. Setup exited; no system package or driver was changed.
- Full setup command is recorded in the review README. No service start or curl request occurred.
- SSH warning: connection did not use a post-quantum key exchange.
