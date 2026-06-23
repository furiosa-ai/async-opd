"""SFT Trainer — Supervised Fine-Tuning via cross-entropy/KL/mixed loss.

Inherits shared infrastructure from BaseTrainer. Provides SFT-specific
loss_fn and inline evaluation handler.
"""

from dataclasses import replace

import torch

from opd.launch_specs import TrainerLaunchSpec
from opd.trainer.config import build_sft_config_from_algorithm_payload
from opd.trainer.base_trainer import BaseTrainer
from opd.utils.config import resolve_trust_remote_code


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer.

    Inherits from BaseTrainer for backend creation and run().

    Loss modes (via SFTConfig.loss_mode):
    - "ce": cross-entropy only (default, no teacher data needed)
    - "kl": KL divergence only (needs teacher logits in batch)
    - "mixed": ce_alpha * CE + (1 - ce_alpha) * KL
    """

    def __init__(self, config: dict, rank_info: dict | None = None):
        if isinstance(config, TrainerLaunchSpec):
            launch_spec = config
            algo = launch_spec.static.algorithm
        else:
            launch_spec = None
            algo = config["algorithm"]
        self.config = build_sft_config_from_algorithm_payload(algo)

        # Set the effective backend loss_mode on a local spec copy before
        # backend construction — backend reads it to
        # decide whether to apply chunked LM head patch. SFT needs full
        # logits for CE loss, so loss_mode must be "sft" (not "kl").
        loss_mode = "kl" if self.config.loss_mode == "kl" else "sft"
        if launch_spec is not None:
            launch_spec = replace(
                launch_spec,
                static=replace(launch_spec.static, loss_mode=loss_mode),
            )
            backend_input = launch_spec
            backend_rank_info = None
        else:
            backend_input = dict(config)
            backend_input["loss_mode"] = loss_mode
            backend_rank_info = rank_info

        super().__init__(backend_input, backend_rank_info)
        self.kl_config = self._backend.kl_config
        self._use_chunked = False  # SFT always uses full logits

    def loss_fn(self, logits, mb):
        """SFT loss: CE, KL, or mixed."""
        from opd.loss.sft import compute_sft_loss

        teacher_valid_mask = mb.get("teacher_valid_mask", mb["response_mask"])
        (loss, n_tok), extras = compute_sft_loss(
            logits,
            mb["input_ids"],
            mb["response_mask"],
            mb.get("teacher_topk_logps"),
            mb.get("teacher_topk_indices"),
            teacher_valid_mask,
            kl_config=self.kl_config,
            sft_config=self.config,
        )
        return loss, n_tok, extras

    def train_step(self, batch, backend):
        """Prepare batch and run training step via backend.

        Follows the same pattern as OPDTrainer: generic batch prep → flatten → _run_train_step.
        """
        prepared = backend._prepare_batch(batch)

        # Flatten: extract tensors, drop metadata
        flat = {k: v for k, v in prepared.items()
                if k not in ("n_mini", "mini_bs", "seq_len", "actual_max_len")}

        return backend._run_train_step(flat, self.loss_fn)

    def command_handlers(self):
        """Extra command handlers for the backend command loop."""
        return {"evaluate": self._handle_evaluate}

    def _handle_evaluate(self, cmd, t_recv, result_queue):
        """Run inline perplexity evaluation (forward pass only)."""
        import torch.nn.functional as F

        backend = self._backend
        if backend.rank == 0:
            eval_cfg = cmd[1]
            from opd.data.sft import SFTDataset, make_sft_collate_fn
            from transformers import AutoTokenizer
            from torch.utils.data import DataLoader

            tokenizer = AutoTokenizer.from_pretrained(
                eval_cfg["tokenizer_path"],
                trust_remote_code=resolve_trust_remote_code(
                    eval_cfg.get("trust_remote_code"),
                    context="SFT inline evaluation tokenizer loading",
                ),
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dataset = SFTDataset(
                eval_cfg["val_files"], tokenizer,
                eval_cfg["max_prompt_length"],
                eval_cfg.get("max_response_length", 256),
                prompt_key=eval_cfg.get("prompt_key", "prompt"),
                completion_key=eval_cfg.get("completion_key", "completion"),
                prompt_template=eval_cfg.get("prompt_template"),
                enable_thinking=eval_cfg.get("enable_thinking"),
                allow_pickle_teacher_logits=eval_cfg.get("allow_pickle_teacher_logits", False),
            )
            loader = DataLoader(dataset, batch_size=len(dataset),
                                shuffle=False,
                                collate_fn=make_sft_collate_fn(tokenizer.pad_token_id))
            batch = next(iter(loader))
        else:
            batch = None

        if getattr(backend, 'use_fsdp', False):
            from opd.data.batch_utils import broadcast_batch as _broadcast_batch
            batch = _broadcast_batch(
                batch if backend.rank == 0 else None,
                backend.rank, backend.world_size, backend.device)

        with torch.no_grad():
            input_ids = batch["input_ids"].to(backend.device)
            attention_mask = batch["attention_mask"].to(backend.device)
            response_mask = batch["response_mask"].to(backend.device)

            bs = input_ids.size(0)
            if backend.world_size > 1 and bs >= backend.world_size:
                per_rank = bs // backend.world_size
                s_ = backend.rank * per_rank
                e_ = s_ + per_rank
                input_ids = input_ids[s_:e_]
                attention_mask = attention_mask[s_:e_]
                response_mask = response_mask[s_:e_]

            eval_mbs = min(8, input_ids.size(0))
            total_masked_loss = torch.tensor(0.0, device=backend.device)
            total_tokens = torch.tensor(0.0, device=backend.device)

            for mb_start in range(0, input_ids.size(0), eval_mbs):
                mb_end = min(mb_start + eval_mbs, input_ids.size(0))
                out = backend.model(
                    input_ids=input_ids[mb_start:mb_end],
                    attention_mask=attention_mask[mb_start:mb_end],
                    use_cache=False)
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = input_ids[mb_start:mb_end, 1:].contiguous()
                shift_mask = response_mask[mb_start:mb_end, 1:].float()
                loss_flat = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), reduction='none')
                total_masked_loss += (loss_flat * shift_mask.view(-1)).sum()
                total_tokens += shift_mask.sum()
                del out, logits, shift_logits, loss_flat

            if getattr(backend, 'use_fsdp', False) and backend.world_size > 1:
                torch.distributed.all_reduce(total_masked_loss)
                torch.distributed.all_reduce(total_tokens)

            n_tokens = int(total_tokens.item())
            val_loss = total_masked_loss.item() / max(n_tokens, 1)

        if backend.rank == 0:
            import math
            perplexity = math.exp(min(val_loss, 20))
            result_queue.put({
                "eval_metrics": {"val_loss": val_loss, "perplexity": perplexity,
                                 "n_tokens": n_tokens},
                "accuracy": val_loss,
            })

    # run() inherited from BaseTrainer


# ============================================================
# Entry point for subprocess spawning
# ============================================================


def sft_trainer_main(config, cmd_queue, result_queue, rank_info):
    """Entry point for SFT training subprocess."""
    trainer = SFTTrainer(config, rank_info)
    trainer.run(cmd_queue, result_queue)
