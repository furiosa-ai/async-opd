"""HF Transformers-based teacher scoring server with ZMQ transport."""

import os
import sys
import time

import torch
import zmq

from opd.launch_specs import TeacherLaunchSpec
from opd.rollout.mc_utils import sample_teacher_multiset_from_log_probs
from opd.utils.config import resolve_trust_remote_code
from opd.worker.teacher.serialization import serialize, deserialize
from opd.utils.trace import timer


class HFTeacherServer:
    """HF Transformers-based teacher scoring server with ZMQ transport."""

    def __init__(self, config: dict):
        self.launch_spec = config if isinstance(config, TeacherLaunchSpec) else None
        if isinstance(config, TeacherLaunchSpec):
            config = config.merged_config()

        model_path = config["model_path"]
        n_logprobs = config["n_logprobs"]
        bind_port = config["bind_port"]
        gpu_ids = config["gpu_ids"]
        scoring_batch_size = config["scoring_batch_size"]
        dtype = config["dtype"]
        use_torch_compile = config["use_torch_compile"]
        bind_address = config["bind_address"]
        seed = config.get("seed")
        trust_remote_code = resolve_trust_remote_code(
            config.get("trust_remote_code"),
            context="HF teacher model loading",
        )
        self.n_logprobs = n_logprobs
        self.scoring_batch_size = scoring_batch_size
        self.seed = seed

        # Deterministic seeding
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        from transformers import AutoModelForCausalLM

        # Map dtype string
        torch_dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}.get(dtype, "auto")

        # Deterministic mode uses eager attention (FA2 crashes with deterministic algorithms)
        attn_impl = "eager" if seed is not None else "flash_attention_2"

        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        ).to(self.device).eval()

        if use_torch_compile:
            print(f"[Teacher-HF] Compiling model with torch.compile...", flush=True)
            self.model = torch.compile(self.model)

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://{bind_address}:{bind_port}")
        print(f"[Teacher-HF] server ready on {bind_address}:{bind_port} "
              f"(gpus={gpu_ids}, dtype={self.model.dtype}, compile={use_torch_compile})",
              flush=True)

    def run(self):
        """ZMQ request loop: receive, score, respond."""
        while True:
            msg = self.sock.recv()
            try:
                req = deserialize(msg)
            except Exception as e:
                self.sock.send(serialize({"status": "error", "reason": str(e)}))
                continue

            if not isinstance(req, dict) or "prompt_token_ids" not in req:
                self.sock.send(serialize({"status": "error", "reason": "bad request"}))
                continue

            prompt_token_ids = req["prompt_token_ids"]
            prompt_lengths = req.get("prompt_lengths")
            query_indices_response = req.get("query_indices_response")
            teacher_mc_n_total_samples = req.get("teacher_mc_n_total_samples")
            query_mode = query_indices_response is not None
            teacher_mc_mode = teacher_mc_n_total_samples is not None
            batch_size = req.get("batch_size", self.scoring_batch_size)

            all_logps, all_indices, all_token_logps, responses = [], [], [], []
            all_query_logps = []
            all_query_indices = []
            t_forward, t_topk, n_sub = 0.0, 0.0, 0

            with timer() as t_score:
                try:
                    for batch_start in range(0, len(prompt_token_ids), batch_size):
                        batch_ids = prompt_token_ids[batch_start:batch_start + batch_size]
                        batch_prompt_lengths = (
                            prompt_lengths[batch_start:batch_start + batch_size]
                            if prompt_lengths is not None else None
                        )
                        batch_query_indices = (
                            query_indices_response[batch_start:batch_start + batch_size]
                            if query_indices_response is not None else None
                        )

                        max_len = max(len(ids) for ids in batch_ids)
                        padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
                        attn_mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
                        for i, ids in enumerate(batch_ids):
                            L = len(ids)
                            padded[i, :L] = torch.tensor(ids, dtype=torch.long)
                            attn_mask[i, :L] = 1

                        padded = padded.to(self.device)
                        attn_mask = attn_mask.to(self.device)

                        t0 = time.monotonic()
                        with torch.no_grad():
                            out = self.model(input_ids=padded, attention_mask=attn_mask)
                        logits = out.logits
                        t1 = time.monotonic()
                        t_forward += t1 - t0

                        log_probs = torch.log_softmax(logits.float(), dim=-1)
                        if teacher_mc_mode:
                            n_total_samples = int(teacher_mc_n_total_samples)
                            for i, ids in enumerate(batch_ids):
                                L = len(ids)
                                teacher_rows = max(L - 1, 0)
                                prompt_len = int(batch_prompt_lengths[i])
                                row_start = max(prompt_len - 1, 0)
                                usable = max(0, teacher_rows - row_start)
                                if usable > 0:
                                    row_log_probs = log_probs[i, row_start:row_start + usable]
                                    sample_idx, sample_logps = sample_teacher_multiset_from_log_probs(
                                        row_log_probs, n_total_samples)
                                    all_query_indices.append(sample_idx.cpu().to(torch.int32))
                                    all_query_logps.append(sample_logps.cpu().to(torch.float32))
                                else:
                                    all_query_indices.append(
                                        torch.zeros(0, n_total_samples, dtype=torch.int32)
                                    )
                                    all_query_logps.append(
                                        torch.zeros(0, n_total_samples, dtype=torch.float32)
                                    )
                                responses.append(torch.tensor([], dtype=torch.int32))
                        elif query_mode:
                            for i, ids in enumerate(batch_ids):
                                L = len(ids)
                                teacher_rows = max(L - 1, 0)
                                prompt_len = int(batch_prompt_lengths[i])
                                row_start = max(prompt_len - 1, 0)
                                q_idx = batch_query_indices[i]
                                if isinstance(q_idx, torch.Tensor):
                                    q_idx = q_idx.to(self.device)
                                else:
                                    q_idx = torch.tensor(q_idx, dtype=torch.long, device=self.device)
                                usable = max(0, min(q_idx.size(0), teacher_rows - row_start))
                                if usable > 0:
                                    gathered = log_probs[i, row_start:row_start + usable].gather(
                                        1, q_idx[:usable].long()
                                    )
                                    all_query_logps.append(gathered.cpu().to(torch.float32))
                                else:
                                    all_query_logps.append(
                                        torch.zeros(0, q_idx.size(1), dtype=torch.float32)
                                    )
                                responses.append(torch.tensor([], dtype=torch.int32))
                        else:
                            topk_vals, topk_idx = torch.topk(log_probs[:, :-1], self.n_logprobs, dim=-1)
                            shifted_ids = padded[:, 1:]
                            actual_lps = log_probs[:, :-1].gather(2, shifted_ids.unsqueeze(2)).squeeze(2)

                            for i, ids in enumerate(batch_ids):
                                L = len(ids)
                                all_logps.append(topk_vals[i, :L-1].cpu().to(torch.float32))
                                all_indices.append(topk_idx[i, :L-1].cpu().to(torch.int32))
                                all_token_logps.append(actual_lps[i, :L-1].cpu().to(torch.float32))
                                responses.append(torch.tensor([], dtype=torch.int32))

                        t_topk += time.monotonic() - t1
                        n_sub += 1

                        del logits, log_probs, out
                        torch.cuda.empty_cache()

                except Exception as e:
                    import traceback
                    err_msg = f"{e}\n{traceback.format_exc()}"
                    print(f"[Teacher-HF] ERROR during scoring: {err_msg}", flush=True)
                    self.sock.send(serialize({"status": "error", "reason": err_msg}))
                    continue

                t_score["mono_end"] = time.monotonic()
                t_ser = time.monotonic()
                resp_payload = {
                    "status": "ok",
                    "responses": responses,
                    "timing": t_score,
                }
                if teacher_mc_mode:
                    resp_payload["teacher_query_logprobs_response"] = all_query_logps
                    resp_payload["teacher_topk_indices"] = all_query_indices
                    resp_payload["teacher_token_logprobs"] = []
                elif query_mode:
                    resp_payload["teacher_query_logprobs_response"] = all_query_logps
                    resp_payload["teacher_topk_logprobs"] = all_query_logps
                    resp_payload["teacher_topk_indices"] = []
                    resp_payload["teacher_token_logprobs"] = []
                else:
                    resp_payload["teacher_topk_logprobs"] = all_logps
                    resp_payload["teacher_topk_indices"] = all_indices
                    resp_payload["teacher_token_logprobs"] = all_token_logps
                resp_data = serialize(resp_payload)
                t_ser = time.monotonic() - t_ser
                self.sock.send(resp_data)
            print(f"[Teacher-HF] {len(prompt_token_ids)} prompts scored (bs={batch_size}) | "
                  f"forward={t_forward:.1f}s  topk={t_topk:.1f}s  "
                  f"serialize={t_ser:.1f}s  ({n_sub} sub-batches, {len(resp_data)/1e6:.1f}MB)",
                  flush=True)


def teacher_hf_server_main(config: dict):
    """Entry point for teacher HF Transformers server subprocess."""
    if isinstance(config, TeacherLaunchSpec):
        config = config.merged_config()
    gpu_ids = config["gpu_ids"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    bin_dir = os.path.dirname(sys.executable)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")

    server = HFTeacherServer(config)
    server.run()
