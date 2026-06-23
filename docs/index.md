# AsyncOPD documentation

Start here if you are evaluating or extending AsyncOPD as an external user.

| Guide | Use it for |
| --- | --- |
| [Quickstart](quickstart.md) | Installation checks, no-GPU sanity commands, first GPU smoke run, and expected outputs. |
| [Architecture guide](architecture.md) | Process topology, queues, ZMQ, NCCL weight sync, scheduling modes, and trust boundaries. |
| [Configuration guide](configuration.md) | Public config sections, canonical vs compatibility keys, and examples for OPD, GRPO/DAPO, and SFT. |
| [Training modes](training-modes.md) | How OPD, GRPO/DAPO, and SFT run, which config to start from, and what knobs to change first. |
| [Loss/logit chunking](loss-chunking.md) | How `trainer.kl_chunk_size` reduces KL/logprob memory, how it relates to chunked CE, and how to tune it. |
| [Evaluation guide](evaluation.md) | Model evaluation, score-only artifacts, Avg@N, code-eval safety, and outputs. |
| [Troubleshooting guide](troubleshooting.md) | vLLM, NCCL, GPU mapping, Ray, tokenizer, checkpoint, and CLI/import issues. |

Start with the curated configs under `configs/examples/`, the documented CLI commands, the lightweight no-GPU checks, and the optional GPU integration runner.
