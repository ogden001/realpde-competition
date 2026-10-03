# curl smoke

Status: `NOT RUN`. Strata's pinned engine compiled successfully, but the official model download stopped with `cannot reach huggingface.co (<urlopen error [Errno 101] Network is unreachable>)`. No model shard was downloaded and no service was started. Health, `/v1/models`, and `/v1/chat/completions` were therefore not queried.
