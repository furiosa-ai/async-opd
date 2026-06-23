"""Dummy rollout backend for testing the backend Protocol.

Demonstrates that a new backend can be added in a single file
without touching coordinator, proxies, or other backends.
"""

import os
import sys
import torch

from opd.rollout.base import BaseRolloutWorker


class DummyRolloutWorker(BaseRolloutWorker):
    """Minimal rollout backend that returns zeros. For testing only."""

    supports_streaming = False
    supports_weight_transfer_nccl = False
    supports_pause_resume = False
    supports_scoring = False
    supports_lora = False

    def __init__(self, config: dict):
        super().__init__(config)
        self._setup_env(self.gpu_ids)
        print(f"[Rollout-Dummy-{self.worker_id}] ready", flush=True)

    def run(self, cmd_queue, result_queue, prompt_queue=None):
        """Command loop — blocks until shutdown."""
        while True:
            cmd = cmd_queue.get()
            if cmd[0] == "shutdown":
                break
            handler = getattr(self, f"handle_{cmd[0]}", None)
            if handler:
                handler(cmd, result_queue)

    def handle_generate(self, cmd, result_queue):
        batch = cmd[1]
        input_ids = batch["input_ids"]
        bs = input_ids.size(0)
        max_prompt_len = input_ids.size(1)
        total_len = max_prompt_len + self.max_response_length
        result_queue.put({
            "input_ids": torch.zeros(bs, total_len, dtype=torch.long),
            "attention_mask": torch.zeros(bs, total_len, dtype=torch.bool),
            "responses": torch.zeros(bs, self.max_response_length, dtype=torch.long),
            "prompt_lengths": torch.ones(bs, dtype=torch.long),
            "response_lengths": torch.ones(bs, dtype=torch.long),
            "full_token_lists": [[] for _ in range(bs)],
            "timing": {"elapsed": 0.0, "generate_seconds": 0.0},
            "_vllm_stats": [],
        })

    def handle_init_weight_transfer(self, cmd, result_queue):
        result_queue.put({"status": "ok"})

    def handle_sync_weights(self, cmd, result_queue):
        result_queue.put({"status": "synced_nccl", "sync_seconds": 0.0})

    def handle_get_vllm_params_info(self, cmd, result_queue):
        result_queue.put({"params_info": []})

    def handle_compute_weight_checksum(self, cmd, result_queue):
        result_queue.put({"checksum": 0.0})


def dummy_rollout_worker_main(config, cmd_queue, result_queue):
    """Entry point for dummy rollout worker subprocess."""
    worker = DummyRolloutWorker(config)
    worker.run(cmd_queue, result_queue)
