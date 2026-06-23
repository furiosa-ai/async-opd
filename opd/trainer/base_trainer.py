"""BaseTrainer — shared infrastructure for composition trainers.

Extracts genuinely identical code from OPDTrainer, GRPOTrainer, SFTTrainer:
backend creation, run(), DecPPO prox precomputation, old-logprob alignment.

Each subclass keeps its own train_step, forward_and_loss_fn, and loss_fn
because these differ fundamentally between modes.
"""

import torch

from opd.launch_specs import TrainerLaunchSpec
from opd.trainer.base import BaseBackend, GLOBAL_MINI_PLAN_METADATA_KEYS


class BaseTrainer:
    """Thin base for composition trainers. Shared infrastructure only.

    Provides:
      - __init__: backend creation (FSDP or Megatron)
      - run: delegates to backend command loop
      - _precompute_prox: DecPPO prox precomputation (chunked + fallback)
      - _align_old_logprobs: right-align [B, resp_len] → [B, shifted_len]

    Subclasses must provide:
      - train_step(batch, backend): batch prep + call _run_train_step
      - loss_fn(logits, mb): full-logits loss (fallback path)
    """

    def __init__(self, config, rank_info=None):
        self.launch_spec = config if isinstance(config, TrainerLaunchSpec) else None
        backend_type = config.backend if isinstance(config, TrainerLaunchSpec) else config.get("backend", "fsdp")
        if backend_type == "fsdp":
            from opd.trainer.fsdp import FSDPBackend
            self._backend = FSDPBackend(config, rank_info)
        elif backend_type == "megatron":
            from opd.trainer.megatron import MegatronBackend
            self._backend = MegatronBackend(config, rank_info)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")

        self._use_decoupled_ppo = False
        self._prox_list = []
        self._prox_idx = 0
        self._use_chunked = getattr(self._backend, '_chunked_kl_patched', False)

    def _is_decoupled_ppo_enabled(self):
        """Return DecPPO enablement from explicit overrides or canonical config."""
        explicit = self.__dict__.get("use_decoupled_loss")
        if explicit is not None:
            return bool(explicit)

        config = getattr(self, "config", None)
        if config is not None and hasattr(config, "use_decoupled_loss"):
            return bool(config.use_decoupled_loss)

        kl_config = getattr(self, "kl_config", None)
        if kl_config is not None and hasattr(kl_config, "use_decoupled_loss"):
            return bool(kl_config.use_decoupled_loss)

        return bool(getattr(self, "_use_decoupled_ppo", False))

    def run(self, cmd_queue, result_queue):
        """Start the training command loop."""
        self._backend.run(cmd_queue, result_queue, trainer=self)

    @staticmethod
    def _copy_global_mini_plan_metadata(prepared, flat):
        """Propagate explicit mini-slice metadata into backend train payloads."""
        for key in GLOBAL_MINI_PLAN_METADATA_KEYS:
            if key in prepared:
                flat[key] = prepared[key]

    def _consume_prox_logprobs(self, fallback_logprobs, device=None):
        """Return the next cached pi_prox tensor, or a detached fallback.

        Returns None when decoupled PPO is disabled for the trainer.
        """
        if not self._is_decoupled_ppo_enabled():
            return None

        if device is None:
            device = fallback_logprobs.device

        idx = self._prox_idx
        if idx < len(self._prox_list):
            prox_logprobs = self._prox_list[idx].to(device)
            self._prox_idx = idx + 1
            return prox_logprobs
        return fallback_logprobs.detach()

    def _maybe_precompute_prox(self, flat, backend, *,
                               require_kl_mode=None,
                               allow_megatron=True):
        """Precompute pi_prox when the trainer/run mode actually needs it."""
        if not self._is_decoupled_ppo_enabled():
            return False
        if require_kl_mode is not None and getattr(self, "kl_config", None) is not None:
            if self.kl_config.mode != require_kl_mode:
                return False
        if not allow_megatron and hasattr(backend, "_megatron_model"):
            return False

        self._precompute_prox(flat, backend)
        return True

    def _precompute_prox(self, flat, backend):
        """Compute pi_prox for DecPPO before any optimizer steps.

        Runs no-grad forward on each micro-batch (in eval mode), stores
        per-micro-batch tensors in self._prox_list. Uses chunked LM head
        when available, falls back to full logits + chunked_log_softmax_gather.

        Supports sequence packing when prompt_lengths are present and
        backend.use_sequence_packing is True (only with chunked path).
        """
        from opd.loss.kl import chunked_log_softmax_gather

        model = backend.model
        device = backend.device
        bs = flat["input_ids"].size(0)
        micro_batch_size = backend.micro_batch_size
        kl_chunk_size = getattr(backend, 'kl_chunk_size', 1024)
        use_packing = getattr(backend, 'use_sequence_packing', False)

        use_global_mini_plan = bool(flat.get("_use_global_mini_plan", False))
        if use_global_mini_plan:
            mini_slices = [tuple(x) for x in flat.get("_mini_slices", [])]
            common_micro_counts = list(flat.get("_common_micro_counts", []))
            if len(mini_slices) != len(common_micro_counts):
                raise RuntimeError(
                    "global-mini prox precompute metadata mismatch: "
                    f"{len(mini_slices)} slices vs {len(common_micro_counts)} micro counts"
                )
        else:
            mini_batch_size = getattr(backend, 'mini_batch_size', 0)
            world_size = getattr(backend, 'world_size', 1)
            per_rank_mini_bs = max(1, mini_batch_size // world_size) if mini_batch_size > 0 else 0
            if per_rank_mini_bs > 0 and per_rank_mini_bs < bs:
                n_mini = bs // per_rank_mini_bs
                mini_bs = per_rank_mini_bs
            else:
                n_mini = 1
                mini_bs = bs
            mini_slices = [
                (mini_idx * mini_bs, (mini_idx + 1) * mini_bs)
                for mini_idx in range(n_mini)
            ]
            common_micro_counts = [
                max(1, (mini_bs + micro_batch_size - 1) // micro_batch_size)
                for _ in mini_slices
            ]

        prox_list = []
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for mini_idx, (ms, me) in enumerate(mini_slices):
                local_len = me - ms
                n_micro = common_micro_counts[mini_idx]
                if local_len <= 0:
                    raise RuntimeError(
                        "global-mini prox precompute received an empty local mini-batch"
                    )
                if local_len < n_micro:
                    raise RuntimeError(
                        "global-mini prox precompute cannot create non-empty "
                        f"microsteps: local_len={local_len}, n_micro={n_micro}"
                    )
                for s, e in BaseBackend._split_span_for_micro_steps(ms, me, n_micro):

                    mb_ids = flat["input_ids"][s:e].to(device)
                    mb_attn = flat["attention_mask"][s:e].to(device)
                    mb_pos = flat.get("position_ids")
                    if mb_pos is not None and isinstance(mb_pos, torch.Tensor):
                        mb_pos = mb_pos[s:e].to(device)
                    else:
                        mb_pos = None

                    packing_kwargs = {}
                    if use_packing:
                        mb_prompt_lens = flat.get("prompt_lengths")
                        if mb_prompt_lens is not None and isinstance(mb_prompt_lens, torch.Tensor):
                            from opd.data.packing import pack_micro_batch
                            mb_resp_mask = flat["response_mask"][s:e].to(device)
                            packed = pack_micro_batch(
                                input_ids=mb_ids, attention_mask=mb_attn,
                                response_mask=mb_resp_mask,
                                prompt_lengths=mb_prompt_lens[s:e].to(device),
                            )
                            mb_ids = packed.input_ids
                            mb_pos = packed.position_ids
                            mb_attn = None  # Must be None for FA varlen
                            packing_kwargs = {
                                "cu_seq_lens_q": packed.cu_seq_lens,
                                "cu_seq_lens_k": packed.cu_seq_lens,
                                "max_length_q": packed.max_seq_len,
                                "max_length_k": packed.max_seq_len,
                            }

                    if self._use_chunked:
                        prox = model(
                            input_ids=mb_ids, attention_mask=mb_attn,
                            position_ids=mb_pos,
                            _kl_args={'mode': 'policy_gradient_kl',
                                      'chunk_size': kl_chunk_size},
                            **packing_kwargs,
                        ).squeeze(-1)
                    else:
                        assert not packing_kwargs, \
                            "Packing requires chunked LM head for prox computation"
                        fwd_kwargs = dict(input_ids=mb_ids, attention_mask=mb_attn,
                                          use_cache=False)
                        if mb_pos is not None:
                            fwd_kwargs["position_ids"] = mb_pos
                        out = model(**fwd_kwargs)
                        logits = out.logits if hasattr(out, "logits") else out[0]
                        target_ids = mb_ids[:, 1:]
                        prox = chunked_log_softmax_gather(
                            logits[:, :-1], target_ids.unsqueeze(-1)
                        ).squeeze(-1)
                        del out, logits

                    prox_list.append(prox.cpu())

        if was_training:
            model.train()

        self._prox_list = prox_list
        self._prox_idx = 0

    @staticmethod
    def _align_old_logprobs(tensor, shifted_len, mask, device):
        """Right-align [B, resp_len] tensor to [B, shifted_len].

        Handles truncation when resp_len > shifted_len (trailing padding
        removed by _prepare_batch).
        """
        tensor = tensor.to(device)
        resp_len = tensor.size(1)
        if resp_len > shifted_len:
            actual_resp = int(mask.sum(dim=1).max().item())
            tensor = tensor[:, :actual_resp]
            resp_len = actual_resp
        aligned = torch.zeros(tensor.size(0), shifted_len,
                              device=device, dtype=tensor.dtype)
        aligned[:, -resp_len:] = tensor
        return aligned
