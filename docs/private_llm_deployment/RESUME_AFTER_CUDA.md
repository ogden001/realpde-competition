# Resume Note — after user-installed CUDA Toolkit 13.0

Status: `IMPLEMENT_AND_EXECUTE_AUTHORIZED / REVIEW_REQUIRED`

This note supplements `docs/private_llm_deployment/NEXT_ACTION.md` for the current Strata bootstrap continuation.

The user has already manually completed the only previously blocked privileged package installation:

```bash
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

Do **not** reinstall CUDA Toolkit unless verification proves the package is actually missing/corrupt. Do not install `nvidia-cuda-toolkit`, `cuda`, `cuda-13-0`, or any NVIDIA driver package.

## Codex responsibilities from here

Codex should now own all non-secret environment/path/config work required to resume Strata, including bounded shell environment adaptation.

### 1. Discover the installed CUDA toolkit

Check at minimum:

```bash
dpkg -l | grep cuda-toolkit-13-0 || true
ls -ld /usr/local/cuda* 2>/dev/null || true
find /usr/local -maxdepth 3 -type f -name nvcc 2>/dev/null || true
which nvcc || true
```

If `/usr/local/cuda-13.0/bin/nvcc` exists but `nvcc` is not on PATH, this is **not** a blocker.

Use bounded environment adaptation such as:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
```

Then verify:

```bash
nvcc --version
nvidia-smi
```

Strata's own setup also searches standard CUDA locations such as `/usr/local/cuda*`; do not install another toolkit simply because the current shell cannot resolve `nvcc` before PATH adaptation.

### 2. Persistent shell environment

Prefer session-local environment changes first so the current bootstrap can proceed with minimal host mutation.

If a persistent user-shell PATH entry is genuinely useful, Codex may update the current deployment user's `~/.bashrc` or equivalent, but only when all of the following are true:

- the detected toolkit path is real and verified;
- the change is user-local, not system-wide;
- the change is idempotent and does not duplicate existing CUDA entries;
- no unrelated shell configuration is rewritten;
- the exact modification is recorded in the deployment evidence.

Do not edit `/etc/environment`, `/etc/profile`, linker configuration, or other system-wide files unless the official Strata build proves it is necessary. If such a system-wide change becomes necessary, treat it as a new privileged action and follow the sudo rule below.

### 3. Sudo handling

The user authorizes Codex to continue and to request interactive sudo **when a necessary command truly requires it**.

Preferred behavior:

1. First try the task without additional privileged changes.
2. If sudo is required, show the exact command and reason.
3. If Codex has an interactive TTY, run the command normally with `sudo` and let the terminal prompt the user for the password.
4. The user may type the password directly into the terminal prompt.
5. Never request that the user paste a sudo password into chat.
6. Never place a password in command-line arguments, shell history, environment variables, files, logs, Git, or evidence.
7. Never use patterns such as `echo PASSWORD | sudo -S` or equivalent secret injection.
8. If no interactive TTY is available, stop at that exact command and ask the user to run it manually; resume afterward.

A convenient interactive validation is allowed:

```bash
sudo -v
```

If it succeeds via the user's terminal password entry, continue only with the specifically required privileged command(s). Do not use the cached sudo authorization to broaden system changes.

### 4. Driver protection

The NVIDIA driver was healthy at `595.84` and remains frozen for this task.

Do not install/upgrade/downgrade any `nvidia-driver-*` package and do not use a CUDA meta-package that pulls a driver.

After any privileged operation, re-check:

```bash
nvidia-smi
```

and record the driver version.

### 5. Resume the existing Strata attempt

Use the existing tree and frozen target from `NEXT_ACTION.md`:

- Strata repo: `~/ai-stack/strata`
- Strata SHA: `99f3dbd0b21d1401b3769e0c0d963913607f380b`
- model: `Qwen3.8-Flash-Next`
- quantization: `IQ3_XXS`
- context: `131072`
- vision: OFF
- KV: INT8
- endpoint: `127.0.0.1:8081`

Resume the same setup rather than starting over:

```bash
cd "$HOME/ai-stack/strata"
./setup.sh --yes --family qwen --model IQ3_XXS --context 131072 \
  --vision no --kv int8 --port 8081 --host 127.0.0.1 \
  --data-dir "$HOME/ai-stack/models" --no-start
```

Then continue exactly as defined in `NEXT_ACTION.md`:

`toolkit verification -> Strata engine build -> model download/preparation -> local start -> curl health/models/chat smoke -> runtime evidence -> commit + push -> REVIEW_REQUIRED`

Do not install Agent/RAG/persistence/calibration/tuning components.
