"""Trainer-side teacher-head recompute for hidden-state teacher artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from opd.utils.config import resolve_trust_remote_code


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported teacher hidden dtype: {name}") from exc


@dataclass
class TeacherRecomputeResult:
    teacher_output: dict[str, torch.Tensor]
    metrics: dict[str, Any]


class TeacherRecomputeHead:
    """Frozen teacher LM head used to recompute teacher logprobs on trainer rank 0.

    The first Stage-2 implementation canonicalizes hidden-state artifacts to the
    existing OPD ``teacher_topk_*`` fields.  For dense KL this uses all vocab
    ids as the support, so the existing sparse KL kernels become exact dense
    KL for the row support.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: torch.device,
        dtype_name: str = "bfloat16",
        hidden_semantics: str = "lm_head_input",
        chunk_size: int = 128,
        trust_remote_code: bool | None = None,
        materialization: str = "lazy",
    ):
        if not model_path:
            raise RuntimeError("teacher_model_path is required for hidden_recompute")
        if hidden_semantics not in {"lm_head_input", "pre_final_norm"}:
            raise ValueError(f"unsupported teacher hidden semantics: {hidden_semantics}")
        if hidden_semantics != "lm_head_input":
            raise RuntimeError(
                "teacher final_norm required for pre_final_norm hidden states but not loaded"
            )
        self.model_path = model_path
        self.device = device
        self.hidden_dtype = resolve_torch_dtype(dtype_name)
        self.hidden_semantics = hidden_semantics
        self.chunk_size = max(int(chunk_size or 128), 1)
        if materialization not in {"lazy", "canonical"}:
            raise ValueError(f"unsupported hidden recompute materialization: {materialization}")
        self.materialization = materialization

        from transformers import AutoModelForCausalLM, AutoConfig

        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code,
            context="teacher hidden-recompute model loading",
        )
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        load_dtype = self.hidden_dtype if self.hidden_dtype != torch.float32 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            dtype=load_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        )
        emb = model.get_output_embeddings()
        if emb is None or not hasattr(emb, "weight"):
            raise RuntimeError("teacher lm_head.weight/output embeddings not found")
        weight = emb.weight.detach().to(device=device, dtype=load_dtype).contiguous()
        self.lm_head_weight = weight
        self.lm_head_weight.requires_grad_(False)
        self.vocab_size = int(weight.size(0))
        self.hidden_size = int(weight.size(1))
        del model

    def _align_hidden_payloads(
        self,
        *,
        gen_output: dict[str, Any],
        hidden_payloads: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        input_ids = gen_output["input_ids"]
        attention_mask = gen_output["attention_mask"]
        bs, seq_len = input_ids.shape
        hidden_lists: list[torch.Tensor] = []
        token_id_lists: list[torch.Tensor | None] = []
        hidden_bytes = 0
        for payload in hidden_payloads:
            semantics = payload.get("teacher_hidden_semantics", self.hidden_semantics)
            if semantics != self.hidden_semantics:
                raise RuntimeError(
                    f"unsupported teacher hidden semantics: payload={semantics!r}, "
                    f"trainer={self.hidden_semantics!r}"
                )
            hidden_rows = payload.get("teacher_hidden_states", [])
            token_rows = payload.get("teacher_hidden_token_ids", [])
            for j, h in enumerate(hidden_rows):
                if not isinstance(h, torch.Tensor):
                    raise RuntimeError("hidden-state payload contains non-tensor row block")
                if h.dim() != 2:
                    raise RuntimeError(f"teacher hidden states must be [C,H], got {tuple(h.shape)}")
                if int(h.size(-1)) != self.hidden_size:
                    raise RuntimeError(
                        f"teacher hidden_size mismatch: payload H={int(h.size(-1))}, "
                        f"lm_head H={self.hidden_size}"
                    )
                hidden_bytes += int(h.numel() * h.element_size())
                hidden_lists.append(h)
                tok = token_rows[j] if j < len(token_rows) else None
                if tok is not None and not isinstance(tok, torch.Tensor):
                    tok = torch.as_tensor(tok, dtype=torch.long)
                token_id_lists.append(tok)
        if len(hidden_lists) != bs:
            raise RuntimeError(
                f"teacher hidden payload count mismatch: got {len(hidden_lists)}, expected {bs}"
            )

        mask_bool = attention_mask.bool().cpu()
        input_ids_cpu = input_ids.cpu()
        eligible = torch.zeros(bs, seq_len, dtype=torch.bool)
        hidden = torch.zeros(bs, seq_len, self.hidden_size, dtype=self.hidden_dtype)

        for i, h in enumerate(hidden_lists):
            # vLLM exposes hidden row r for source/logits position p and stores
            # the corresponding target token id input_ids[p + 1].  Therefore the
            # final real token is never eligible for dense next-token KL.
            real_pos = mask_bool[i].nonzero(as_tuple=True)[0]
            source_pos = real_pos[:-1]
            h_len = int(h.size(0))
            if h_len > int(source_pos.numel()):
                raise RuntimeError(
                    f"teacher hidden payload for sample {i} has {h_len} rows but only "
                    f"{int(source_pos.numel())} source positions with a next-token target"
                )
            tok = token_id_lists[i]
            if tok is not None:
                tok = tok.cpu().long().flatten()
                if int(tok.numel()) != h_len:
                    raise RuntimeError(
                        f"teacher hidden token-id count mismatch for sample {i}: "
                        f"got {int(tok.numel())}, expected {h_len}"
                    )
                expected = input_ids_cpu[i, real_pos[1:1 + h_len]].long()
                if not torch.equal(tok[:h_len], expected):
                    mismatch = (tok[:h_len] != expected).nonzero(as_tuple=True)[0]
                    first = int(mismatch[0].item()) if int(mismatch.numel()) else 0
                    raise RuntimeError(
                        f"teacher hidden token-id alignment mismatch for sample {i} at row {first}: "
                        f"payload={int(tok[first].item())}, expected next token={int(expected[first].item())}"
                    )
            if h_len <= 0:
                continue
            row_pos = source_pos[:h_len]
            eligible[i, row_pos] = True
            hidden[i, row_pos] = h[:h_len].to(dtype=self.hidden_dtype, device="cpu")
        return hidden, eligible, hidden_bytes

    def assemble_lazy_teacher_artifacts(
        self,
        *,
        gen_output: dict[str, Any],
        hidden_payloads: list[dict[str, Any]],
    ) -> TeacherRecomputeResult:
        t0 = time.monotonic()
        hidden, eligible, hidden_bytes = self._align_hidden_payloads(
            gen_output=gen_output,
            hidden_payloads=hidden_payloads,
        )
        return TeacherRecomputeResult(
            teacher_output={
                "teacher_hidden_states": hidden,
                "teacher_hidden_valid_mask": eligible,
            },
            metrics={
                "teacher_hidden_recompute_materialization": "lazy",
                "teacher_hidden_prepare_seconds": time.monotonic() - t0,
                "teacher_hidden_recv_bytes": hidden_bytes,
                "teacher_hidden_vocab_size": self.vocab_size,
                "teacher_hidden_size": self.hidden_size,
            },
        )

    def compute_logps_for_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        t0 = time.monotonic()
        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"teacher hidden states must be [B,S,H], got {tuple(hidden_states.shape)}"
            )
        if int(hidden_states.size(-1)) != self.hidden_size:
            raise RuntimeError(
                f"teacher hidden_size mismatch: payload H={int(hidden_states.size(-1))}, "
                f"lm_head H={self.hidden_size}"
            )
        hidden_dev = hidden_states.to(
            device=self.device,
            dtype=self.hidden_dtype,
            non_blocking=False,
        )
        _, seq_len, _ = hidden_dev.shape
        teacher_logps_chunks = []
        max_bytes = 0
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            logits = F.linear(hidden_dev[:, start:end, :], self.lm_head_weight).float()
            logps = F.log_softmax(logits, dim=-1)
            max_bytes = max(max_bytes, int(logps.numel() * logps.element_size()))
            teacher_logps_chunks.append(logps)
            del logits
        teacher_logps = torch.cat(teacher_logps_chunks, dim=1)
        materialized_bytes = int(teacher_logps.numel() * teacher_logps.element_size())
        del hidden_dev, teacher_logps_chunks
        return teacher_logps, {
            "teacher_hidden_lazy_recompute_seconds": time.monotonic() - t0,
            "teacher_hidden_materialized_bytes": materialized_bytes,
            "teacher_hidden_max_materialized_bytes": max_bytes,
            "teacher_hidden_vocab_size": self.vocab_size,
            "teacher_hidden_size": self.hidden_size,
        }

    def assemble_dense_teacher_output(
        self,
        *,
        gen_output: dict[str, Any],
        hidden_payloads: list[dict[str, Any]],
    ) -> TeacherRecomputeResult:
        t0 = time.monotonic()
        hidden, eligible, hidden_bytes = self._align_hidden_payloads(
            gen_output=gen_output,
            hidden_payloads=hidden_payloads,
        )
        bs, seq_len, _ = hidden.shape
        teacher_logps = torch.empty(bs, seq_len, self.vocab_size, dtype=torch.float32)
        hidden_dev = hidden.to(self.device, non_blocking=False)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            logits = F.linear(hidden_dev[:, start:end, :], self.lm_head_weight).float()
            teacher_logps[:, start:end, :] = F.log_softmax(logits, dim=-1).cpu()
            del logits
        del hidden_dev
        vocab = torch.arange(self.vocab_size, dtype=torch.int32).view(1, 1, -1)
        teacher_indices = vocab.expand(bs, seq_len, self.vocab_size).contiguous()
        metrics = {
            "teacher_hidden_recompute_materialization": "canonical",
            "teacher_hidden_recompute_seconds": time.monotonic() - t0,
            "teacher_hidden_recv_bytes": hidden_bytes,
            "teacher_hidden_vocab_size": self.vocab_size,
            "teacher_hidden_size": self.hidden_size,
        }
        return TeacherRecomputeResult(
            teacher_output={
                "teacher_topk_logps": teacher_logps,
                "teacher_topk_indices": teacher_indices,
                "teacher_valid_mask": eligible.bool(),
            },
            metrics=metrics,
        )
