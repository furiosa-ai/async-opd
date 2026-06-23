"""Training worker — FSDP backend.

Runs as a subprocess on dedicated training GPU(s).

Commands via mp.Queue:
  ("train", batch_dict)  -> one training step, returns metrics
  ("sync_weights",)      -> NCCL broadcast weights to rollout workers
  ("get_weights",)       -> return weights via queue (CPU fallback)
  ("shutdown",)          -> exits
"""

import os

import torch
import torch.nn as nn

from opd.launch_specs import (
    TrainerLaunchSpec,
    algorithm_is_actor_critic,
    algorithm_uses_token_weighted_backward,
    ensure_trainer_launch_spec,
)
from opd.loss.kl import chunked_lm_head_gather
from opd.trainer.base import BaseBackend, build_lr_scheduler, vllm_trainer_send
from opd.utils.config import (
    LoRAConfig,
    OptimConfig,
    TrainerConfig,
    WeightSyncConfig,
    resolve_trust_remote_code,
)


_OPTIM_DEFAULTS = OptimConfig()
_LORA_DEFAULTS = LoRAConfig()
_TRAINER_DEFAULTS = TrainerConfig()
_WEIGHT_SYNC_DEFAULTS = WeightSyncConfig()
_MAX_DIAGNOSTIC_QUANTILE_ELEMS = 1_000_000


def _diagnostic_quantile(t: torch.Tensor, q: float) -> float:
    """Return a robust quantile for diagnostics.

    Large top-k PPO modes can accumulate tens of millions of raw ratio values
    per step. `torch.quantile()` on those giant tensors can fail on GPU with
    "input tensor is too large". These quantiles are logging-only diagnostics,
    so for very large tensors we compute them on a deterministic subsample and
    run the quantile on CPU.
    """
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    if flat.numel() > _MAX_DIAGNOSTIC_QUANTILE_ELEMS:
        step = (flat.numel() + _MAX_DIAGNOSTIC_QUANTILE_ELEMS - 1) // _MAX_DIAGNOSTIC_QUANTILE_ELEMS
        flat = flat[::step]
        if flat.numel() > _MAX_DIAGNOSTIC_QUANTILE_ELEMS:
            flat = flat[:_MAX_DIAGNOSTIC_QUANTILE_ELEMS]
    return flat.cpu().quantile(q).item()


# ===================== FSDP Backend =====================

class FSDPBackend(BaseBackend):
    """FSDP training backend. Self-contained for @ray.remote wrapping.

    For single-GPU: pipeline spawns one process (fsdp_rank=0, fsdp_world_size=1).
    For multi-GPU: pipeline spawns N processes directly, each with a unique
    fsdp_rank. Rank 0 owns cmd_queue/result_queue; other ranks receive commands
    via torch.distributed.broadcast_object_list.
    """

    def __init__(self, config: dict, rank_info: dict | None = None):
        self.launch_spec = None
        algorithm_launch = None
        if isinstance(config, TrainerLaunchSpec):
            self.launch_spec = ensure_trainer_launch_spec(config, rank_info)
            algorithm_launch = self.launch_spec.static.algorithm
            config = self.launch_spec.merged_config()
            rank_info = self.launch_spec.rank_payload()
        else:
            algorithm_launch = config["algorithm"]

        # Deterministic mode: seed everything before model load
        if config.get("deterministic", False):
            import random
            import numpy as np
            seed = config.get("seed", 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            config["use_torch_compile"] = False
            if config.get("attn_implementation") is None:
                config["attn_implementation"] = "eager"
            print(f"[Backend-FSDP] Deterministic mode: seed={seed}, "
                  f"compile=off, attn={config['attn_implementation']}", flush=True)
        # Unpack config for local use before super().__init__ (which stores them)
        loss_mode = config.get("loss_mode", "kl")
        self.loss_mode = loss_mode
        super().__init__(config, rank_info)
        self.max_grad_norm = float(config.get("max_grad_norm", _OPTIM_DEFAULTS.max_grad_norm))
        self._algorithm_launch = algorithm_launch
        self.pg_token_weighted_backward = algorithm_uses_token_weighted_backward(
            self._algorithm_launch
        )
        self._rank_info = dict(rank_info or {})
        self._fused_hybrid_rollout_cfg = config.get("fused_hybrid_rollout")
        self._fused_hybrid_sync_cfg = config.get("fused_hybrid_sync")
        self._hybrid_adapter = None
        self._trust_remote_code = resolve_trust_remote_code(
            config.get("trust_remote_code"),
            context="FSDP trainer model loading",
        )

        rank = self.rank
        world_size = self.world_size
        model_path = self._model_path
        dtype = self._dtype_str
        gpu_ids = config.get("gpu_ids")
        use_torch_compile = config.get("use_torch_compile", _TRAINER_DEFAULTS.use_torch_compile)
        use_sequence_packing = self.use_sequence_packing
        mini_batch_size = self.mini_batch_size
        total_steps = self.total_steps
        nccl_timeout_hours = int(config.get("nccl_timeout_hours", _WEIGHT_SYNC_DEFAULTS.nccl_timeout_hours))
        attn_implementation = config.get("attn_implementation")
        lora_cfg = config.get("lora")
        lr = float(self.optim_cfg["lr"])

        fsdp_master_port = rank_info.get("fsdp_master_port")
        fsdp_master_addr = rank_info.get("fsdp_master_addr", "127.0.0.1")

        # Each rank sees exactly one GPU
        if gpu_ids is not None:
            gpu_ids_list = gpu_ids.split(",")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_list[rank]
        # else: Ray manages CUDA_VISIBLE_DEVICES via num_gpus allocation
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        print(f"[Backend-FSDP] rank {rank}/{world_size} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()} "
              f"device_name={torch.cuda.get_device_name(0)}", flush=True)

        # Init FSDP process group for multi-GPU
        self.use_fsdp = world_size > 1
        if self.use_fsdp:
            from datetime import timedelta
            nccl_timeout_h = nccl_timeout_hours
            torch.distributed.init_process_group(
                backend="nccl", rank=rank, world_size=world_size,
                init_method=f"tcp://{fsdp_master_addr}:{fsdp_master_port}",
                timeout=timedelta(hours=nccl_timeout_h),
            )

        from transformers import AutoModelForCausalLM, AutoConfig

        gpu_label = gpu_ids_list[rank] if gpu_ids is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "ray")
        print(f"[Backend-FSDP] rank {rank}/{world_size} loading {model_path} on GPU {gpu_label}", flush=True)

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        attn_impl = attn_implementation
        if attn_impl is None:
            if torch_dtype == torch.float32:
                attn_impl = "eager"
            else:
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    raise ImportError(
                        "flash-attn is required for training but not installed. "
                        "Install with: pip install flash-attn --no-build-isolation\n"
                        "Or set training.actor_rollout_ref.actor.attn_implementation: sdpa "
                        "in your config to use PyTorch native attention."
                    )

        self._attn_implementation = attn_impl  # saved for post-eval reload

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=self._trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=hf_config,
            dtype=torch_dtype,
            trust_remote_code=self._trust_remote_code,
            attn_implementation=attn_impl,
        ).to(self.device)
        self._maybe_attach_value_head(hf_config)

        # LoRA: apply PEFT adapters before FSDP wrapping
        self._is_lora = False
        self._use_native_lora = False
        self._peft_config_dict = None
        self._base_weights_info = None
        self._lora_cfg = lora_cfg  # from config["lora"]
        if lora_cfg:
            # Capture base model weights_info BEFORE LoRA wrapping
            # (weight sync expects base model keys, not PEFT-prefixed keys)
            self._base_weights_info = [
                (name, param.shape, param.dtype)
                for name, param in self.model.state_dict().items()
            ]
            self._apply_lora(lora_cfg)
            # Native LoRA: sync only LoRA A/B matrices (skip merge-then-sync)
            if lora_cfg.get("native_lora", False):
                from opd.rollout.vllm.lora import build_peft_config_dict
                self._use_native_lora = True
                self._peft_config_dict = build_peft_config_dict(lora_cfg)

        # Guard: torch.compile + LoRA not yet supported
        if self._is_lora and use_torch_compile:
            print("[Trainer] WARNING: torch.compile disabled for LoRA training", flush=True)
            use_torch_compile = False

        self.model.gradient_checkpointing_enable()
        if not self._is_lora:
            self._patch_gradient_checkpointing()
        # For LoRA: enable_input_require_grads() (called in _apply_lora) +
        # HF gradient_checkpointing_enable() handle this natively

        # Patch model forward to support chunked LM head (avoids [B,S,V] logits).
        # Must be done BEFORE FSDP wrapping so the patched forward runs inside
        # the FSDP lifecycle (where params are unsharded).
        # SFT mode needs full logits for CE loss, so skip the patch.
        self._chunked_kl_patched = False
        if self.loss_mode != "sft":
            if self._is_lora:
                self._patch_lora_model_for_chunked_kl()
                self._chunked_kl_patched = True
            else:
                self._chunked_kl_patched = self._patch_model_for_chunked_kl()

        if self.use_fsdp:
            self.model = self._apply_fsdp2(self.model)

        # torch.compile is not used — FSDP2 + compile crashes during Triton
        # kernel autotuning (cudaErrorIllegalAddress). If single-GPU compile
        # is needed in the future, add it to _run_train_step.
        if use_torch_compile:
            print("[Trainer] WARNING: torch.compile is not supported (FSDP2 "
                  "autotuning crash). Ignoring use_torch_compile=true.", flush=True)

        weight_decay = float(self.optim_cfg.get("weight_decay", _OPTIM_DEFAULTS.weight_decay))
        # Filter to trainable params only (LoRA freezes base model)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999),
                                           weight_decay=weight_decay)
        # Scheduler total_steps will be adjusted on first train call if mini_batch_size
        # causes multiple optimizer steps per train batch (we need actual batch size).
        self.scheduler = self._build_lr_scheduler(total_steps) if total_steps > 0 else None
        self._scheduler_needs_rebuild = mini_batch_size > 0 and total_steps > 0

        # Build weights info for process-separated NCCL sync. The fused hybrid
        # path deliberately avoids this full FSDP state_dict materialization and
        # streams rank-local submodule gathers during refresh instead.
        if self.use_fsdp and self._fused_hybrid_sync_cfg:
            self.weights_info = []
        else:
            self.weights_info = self._build_weights_info()

        # Register FSDP-specific command handlers
        self._command_handlers["compute_weight_checksum"] = self._handle_compute_weight_checksum
        self._command_handlers["compute_lora_checksum"] = self._handle_compute_lora_checksum
        self._command_handlers["finalize_fsdp"] = self._handle_finalize_fsdp

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        extra = []
        if mini_batch_size > 0:
            extra.append(f"mini_batch_size={mini_batch_size}")
        if use_sequence_packing:
            extra.append("sequence_packing=ON")
        extra_str = f", {', '.join(extra)}" if extra else ""
        print(f"[Backend-FSDP] rank {rank} ready. Params: {n_params:.1f}M{extra_str}", flush=True)

    # -------------------- Abstract method implementations --------------------

    @property
    def use_distributed(self) -> bool:
        return self.use_fsdp

    def _get_log_prefix(self) -> str:
        return "Trainer-FSDP"

    def _train_step_impl(self, batch):
        """Deprecated: use composition trainers (OPDTrainer, GRPOTrainer, SFTTrainer).

        All training modes now use composition trainers that call _run_train_step
        directly with their own loss_fn. Use fsdp_trainer_main (which auto-dispatches
        to the correct composition trainer) or the mode-specific entry points
        (opd_trainer_main, grpo_trainer_main, sft_trainer_main).
        """
        raise NotImplementedError(
            f"_train_step_impl is deprecated. Use a composition trainer "
            f"(OPDTrainer, GRPOTrainer, SFTTrainer) instead of calling "
            f"FSDPBackend._train_step_impl directly. "
            f"fsdp_trainer_main now auto-dispatches to the correct trainer."
        )


    def _ensure_hybrid_adapter(self):
        """Create the colocated vLLM external-launcher adapter lazily."""
        if self._hybrid_adapter is not None:
            return self._hybrid_adapter
        rollout_cfg = self._fused_hybrid_rollout_cfg
        sync_cfg = self._fused_hybrid_sync_cfg
        if not rollout_cfg or not sync_cfg:
            raise RuntimeError("fused_hybrid_sync was not configured for this trainer")
        if self.world_size <= 1 and not sync_cfg.get("allow_single_gpu_debug", False):
            raise RuntimeError("fused_hybrid_sync requires multi-rank FSDP")
        rollout_parallelism = str(sync_cfg.get("rollout_parallelism", "spmd_tp"))
        if rollout_parallelism == "data_parallel":
            from opd.hybrid.vllm_dp import FusedHybridVLLMDPAdapter

            adapter_cls = FusedHybridVLLMDPAdapter
        elif rollout_parallelism == "spmd_tp":
            from opd.hybrid.vllm_spmd import FusedHybridVLLMSPMDAdapter

            adapter_cls = FusedHybridVLLMSPMDAdapter
        else:
            raise RuntimeError(
                "unsupported fused_hybrid_sync rollout_parallelism "
                f"{rollout_parallelism!r}"
            )

        common_kwargs = dict(
            model_path=rollout_cfg["model_path"],
            max_response_length=int(rollout_cfg["max_response_length"]),
            temperature=float(rollout_cfg.get("temperature", 1.0)),
            top_p=float(rollout_cfg.get("top_p", 1.0)),
            top_k=int(rollout_cfg.get("top_k", -1)),
            max_num_seqs=int(rollout_cfg.get("max_num_seqs", 64)),
            max_model_len=rollout_cfg.get("max_model_len"),
            max_num_batched_tokens=rollout_cfg.get("max_num_batched_tokens"),
            gpu_memory_utilization=float(rollout_cfg.get("gpu_memory_utilization", 0.1)),
            dtype=str(rollout_cfg.get("dtype", self._dtype_str)),
            seed=int(getattr(self, "seed", 42) if hasattr(self, "seed") else 42),
            sleep_level=int(sync_cfg.get("vllm_sleep_level", 2)),
            bucket_mb=int(sync_cfg.get("update_bucket_mb", 256)),
            weight_update_backend=str(sync_cfg.get("weight_update_backend", "bucketed_inprocess")),
            debug_full_state_sync=bool(sync_cfg.get("debug_full_state_sync", False)),
            verify_weight_checksum=bool(sync_cfg.get("verify_weight_checksum", False)),
            trainer_weights_info=self.weights_info,
            max_logprobs=int(rollout_cfg.get("max_logprobs", 1)),
            rank=int(self.rank),
            world_size=int(self.world_size),
            rank_info=self._rank_info,
        )
        if rollout_parallelism == "data_parallel":
            self._hybrid_adapter = adapter_cls(
                dp_size=int(rollout_cfg.get("dp_size", self.world_size)),
                **common_kwargs,
            )
        else:
            self._hybrid_adapter = adapter_cls(
                tp_size=int(rollout_cfg["tp_size"]),
                **common_kwargs,
            )
        # Start in trainer mode: vLLM allocations are immediately released before
        # the first train phase, then woken for rollout/refresh commands.
        self._hybrid_adapter.sleep(reason="after_hybrid_init")
        if self.use_distributed:
            torch.distributed.barrier()
        if self.rank == 0:
            print(
                "[FusedHybrid] vLLM adapter initialized "
                f"parallelism={rollout_parallelism} "
                f"tp={rollout_cfg['tp_size']} dp={rollout_cfg.get('dp_size', self.world_size)} "
                "backend=bucketed_inprocess",
                flush=True,
            )
        return self._hybrid_adapter

    def _get_state_dict_for_sync(self):
        if self._is_lora:
            if self._use_native_lora:
                # Native LoRA: return only LoRA A/B matrices (skip merge)
                return self._get_lora_state_dict()
            # Merge-then-sync: merge LoRA into base, extract base-model-keyed
            # state dict, then unmerge to continue training
            inner = self._get_peft_inner()
            inner.merge_adapter()
            raw_sd = inner.base_model.model.state_dict()
            sd = {}
            for k, v in raw_sd.items():
                clean = k.replace("base_model.model.", "")
                if "lora_A" in clean or "lora_B" in clean:
                    continue
                clean = clean.replace(".base_layer.", ".")
                sd[clean] = v.clone()
            inner.unmerge_adapter()
            return sd
        if self.use_fsdp:
            return self._gather_fsdp_state_dict()
        return None

    def _get_clean_state_dict(self):
        """Override base: return rollout-loadable weights with regular tensors.

        Merge-then-sync LoRA must export merged base-model weights here so CPU
        sync and collective paths send the exact tensors the rollout loads.
        """
        if getattr(self, '_is_lora', False) and not getattr(self, '_use_native_lora', False):
            sd = self._get_state_dict_for_sync()
        elif self.use_fsdp:
            sd = self._gather_fsdp_state_dict()
        else:
            sd = super()._get_clean_state_dict()
        return {k: v for k, v in sd.items() if not k.startswith("value_head.")}

    def _hybrid_weight_export_modules(self):
        """Return deterministic non-root modules for layered FSDP export."""
        modules = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            has_direct_param = any(True for _ in module.parameters(recurse=False))
            has_direct_buffer = any(True for _ in module.buffers(recurse=False))
            if has_direct_param or has_direct_buffer:
                modules.append((name, module))
        if not modules:
            raise RuntimeError(
                "fused_hybrid_sync could not discover module-level FSDP export roots"
            )
        return modules

    def _iter_hybrid_weight_tensors(self):
        """Yield rollout-loadable tensors through layered FSDP2 state gathers.

        The fused hybrid signoff path must not gather the complete model state
        before loading vLLM.  Instead every rank gathers one parameter-owning
        module at a time through FSDP2/DCP's submodule state_dict API, yields its
        checkpoint-format tensors to the bucket loader, then promptly releases
        that module state before moving to the next module.
        """
        import warnings
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        seen = set()
        modules = self._hybrid_weight_export_modules()
        if self.rank == 0:
            print(
                f"[FusedHybrid] layered FSDP export modules={len(modules)}",
                flush=True,
            )
        for _module_name, module in modules:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Getting submodules only model/optim state_dict is deprecated.*",
                    category=FutureWarning,
                )
                sd = get_model_state_dict(
                    self.model,
                    submodules={module},
                    options=options,
                )
            try:
                for name, tensor in sd.items():
                    clean = name.replace("_orig_mod.", "")
                    if clean.startswith("value_head.") or clean in seen:
                        continue
                    seen.add(clean)
                    yield clean, tensor
            finally:
                del sd
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _gather_full_state_dict(self):
        if self.use_fsdp:
            return self._gather_fsdp_state_dict()
        return None

    def _gather_full_optim_state_dict(self):
        if self.use_fsdp:
            return self._gather_fsdp_optim_state_dict()
        return None

    def _post_save_cleanup(self):
        # Free memory from full state dict gather to reduce fragmentation
        # before next training step's FSDP collectives.
        torch.cuda.empty_cache()

    # -------------------- Command handlers --------------------

    def _handle_compute_weight_checksum(self, cmd, t_recv, result_queue):
        merge_map = cmd[1] if self.rank == 0 and cmd is not None and len(cmd) > 1 else None
        if getattr(self, '_is_lora', False) and not getattr(self, '_use_native_lora', False):
            # Merge-then-sync LoRA must checksum the merged base-model weights,
            # not the PEFT-wrapped state dict with adapter/base_layer keys.
            sd = self._get_state_dict_for_sync()
        elif self.use_fsdp:
            sd = self._gather_fsdp_state_dict()
        else:
            sd = None
        if self.rank == 0:
            if sd is None:
                sd = {}
                for name, param in self.model.state_dict().items():
                    clean = name.replace("_orig_mod.", "")
                    sd[clean] = param
            checksum = 0.0
            phi = 1.6180339887  # must match rollout compute_checksum_fn
            if merge_map is None:
                # CPU weight sync path — no vLLM merge_map, iterate sorted
                filtered = [(n, p) for n, p in sorted(sd.items())
                            if not n.startswith("value_head.")]
                for i, (_, param) in enumerate(filtered):
                    checksum += param.float().abs().sum().item() * (phi ** (i % 32))
            else:
                for i, (_, sources) in enumerate(merge_map):
                    weight = phi ** (i % 32)
                    if len(sources) == 1:
                        checksum += sd[sources[0]].float().abs().sum().item() * weight
                    else:
                        merged = torch.cat([sd[s] for s in sources], dim=0)
                        checksum += merged.float().abs().sum().item() * weight
            result_queue.put({"checksum": checksum})

    def _handle_compute_lora_checksum(self, cmd, t_recv, result_queue):
        """Compute order-sensitive checksum over LoRA adapter params only."""
        if not getattr(self, '_is_lora', False):
            if self.rank == 0:
                result_queue.put({"checksum": 0.0})
            return
        sd = self._get_state_dict_for_sync()  # returns LoRA state dict
        if self.rank == 0:
            checksum = 0.0
            phi = 1.6180339887
            for i, (_, tensor) in enumerate(sorted(sd.items())):
                weight = phi ** (i % 32)
                checksum += tensor.float().abs().sum().item() * weight
            result_queue.put({"checksum": checksum})

    def _handle_finalize_fsdp(self, cmd, t_recv, result_queue):
        """Transition from multi-GPU FSDP to single-GPU mode for post-eval.

        All ranks participate in destroying the FSDP process group. Rank 0
        reloads the model without FSDP wrapping so it can continue handling
        load_checkpoint / sync_weights commands. Ranks 1-N exit after this
        (returns "_break" sentinel to exit the command loop).
        """
        if not self.use_fsdp:
            if self.rank == 0:
                result_queue.put({"status": "already_single_gpu"})
            return

        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        self.use_fsdp = False

        if self.rank == 0:
            # Replace FSDP-wrapped model with plain model for checkpoint load + sync
            del self.model
            torch.cuda.empty_cache()
            from transformers import AutoModelForCausalLM, AutoConfig
            torch_dtype = getattr(torch, self._dtype_str, torch.bfloat16)
            # Post-eval: use flash_attention_2 if available, fall back to sdpa
            attn_impl = getattr(self, '_attn_implementation', None)
            if attn_impl is None:
                if torch_dtype == torch.float32:
                    attn_impl = "eager"
                else:
                    try:
                        import flash_attn  # noqa: F401
                        attn_impl = "flash_attention_2"
                    except ImportError:
                        attn_impl = "sdpa"
            hf_config = AutoConfig.from_pretrained(
                self._model_path, trust_remote_code=self._trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                self._model_path, config=hf_config, dtype=torch_dtype,
                trust_remote_code=self._trust_remote_code, attn_implementation=attn_impl,
            ).to(self.device)
            # No optimizer/scheduler needed for post-eval
            self.optimizer = None
            self.scheduler = None
            print("[Backend-FSDP] Finalized FSDP -> single-GPU mode for post-eval",
                  flush=True)
            result_queue.put({"status": "finalized"})
        else:
            # Ranks 1-N exit the command loop after FSDP teardown
            return "_break"

    # -------------------- Checkpoint helpers --------------------

    def _save_checkpoint(self, checkpoint_dir, step,
                         state_dict=None, save_optimizer=True,
                         optim_state_dict=None):
        """Save model weights and optionally optimizer/scheduler state to disk.

        If async_save=True was set on this trainer, the disk I/O runs in a
        background thread so training can continue immediately.

        For LoRA: saves adapter to adapter/ subdir, merged HF model to hf/ subdir.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # LoRA: clean PEFT prefixes from state dict and save merged HF model
        if self._is_lora:
            # Get or build clean merged state dict
            if state_dict is not None:
                # state_dict was pre-gathered (FSDP) — strip PEFT prefixes
                # and merge LoRA A/B into base weights
                lora_scaling = self._lora_cfg["alpha"] / self._lora_cfg["rank"]
                base_sd = {}
                lora_a = {}  # clean_base_name -> tensor
                lora_b = {}
                for k, v in state_dict.items():
                    clean = k.replace("base_model.model.", "")
                    clean = clean.replace(".base_layer.", ".")
                    val = v.cpu() if hasattr(v, 'cpu') else v
                    if ".lora_A." in clean:
                        # e.g. "model.layers.0.self_attn.q_proj.lora_A.default.weight"
                        base_name = clean.split(".lora_A.")[0]
                        lora_a[base_name] = val.float()
                    elif ".lora_B." in clean:
                        base_name = clean.split(".lora_B.")[0]
                        lora_b[base_name] = val.float()
                    else:
                        base_sd[clean] = val
                # Merge: W = W + B @ A * scaling
                for base_name in lora_a:
                    if base_name in lora_b and base_name + ".weight" in base_sd:
                        key = base_name + ".weight"
                        base_sd[key] = (base_sd[key].float()
                                        + lora_scaling * (lora_b[base_name] @ lora_a[base_name])
                                        ).to(base_sd[key].dtype)
                state_dict = base_sd
            else:
                inner = self._get_peft_inner()
                with torch.no_grad():
                    inner.merge_adapter()
                    raw_sd = inner.base_model.model.state_dict()
                    state_dict = {}
                    for k, v in raw_sd.items():
                        clean = k.replace("base_model.model.", "")
                        if "lora_A" in clean or "lora_B" in clean:
                            continue
                        clean = clean.replace(".base_layer.", ".")
                        state_dict[clean] = v.cpu()
                    inner.unmerge_adapter()

            # Save as HF model (for eval)
            if self.rank == 0:
                hf_dir = os.path.join(checkpoint_dir, "hf")
                os.makedirs(hf_dir, exist_ok=True)
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(
                    self._model_path, trust_remote_code=self._trust_remote_code)
                hf_config.save_pretrained(hf_dir)
                # No tokenizer saved — eval loads from original model path to
                # avoid HF tokenizer serialization bug (corrupted regex patterns).
                import safetensors.torch
                safetensors.torch.save_file(state_dict, os.path.join(hf_dir, "model.safetensors"))

        # Model weights — copy to CPU (synchronous, needs GPU access)
        if state_dict is None:
            state_dict = {}
            for name, param in self.model.state_dict().items():
                clean = name.replace("_orig_mod.", "")
                state_dict[clean] = param.cpu()
        else:
            state_dict = {k: v.cpu() if v.is_cuda else v for k, v in state_dict.items()}

        # Training state — copy to CPU (synchronous)
        train_state = None
        if save_optimizer:
            train_state = {"step": step}
            if self.use_fsdp:
                if optim_state_dict is not None:
                    train_state["optimizer"] = optim_state_dict
            else:
                train_state["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                train_state["scheduler"] = self.scheduler.state_dict()

        # Disk I/O — run in background thread via base class helper
        self._async_write_checkpoint(state_dict, train_state, checkpoint_dir, step)

    def _load_checkpoint(self, checkpoint_dir):
        """Load model weights and optimizer/scheduler state from a checkpoint.

        Returns the step number from the checkpoint, or 0 if not found.
        For LoRA: loads adapter weights from adapter/ subdir if present.
        """
        self._wait_async_save()  # ensure previous save is on disk

        # LoRA: load adapter weights
        if self._is_lora:
            adapter_dir = os.path.join(checkpoint_dir, "adapter")
            if os.path.exists(adapter_dir):
                inner = self._get_peft_inner()
                inner.load_adapter(adapter_dir, adapter_name="default")
                print(f"[Trainer] Loaded LoRA adapter from {adapter_dir}", flush=True)

        model_path = os.path.join(checkpoint_dir, "model.pt")
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        step = 0

        if os.path.exists(model_path) and not self._is_lora:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            # Checkpoints are saved with clean keys (no _orig_mod. prefix).
            # torch.compile adds _orig_mod. prefix; FSDP state_dict_type handles FSDP
            # prefixes but not _orig_mod., so re-add it when the model is compiled.
            if hasattr(self.model, '_orig_mod'):
                sd = {"_orig_mod." + k: v for k, v in sd.items()}
            if self.use_fsdp:
                from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
                options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                set_model_state_dict(self.model, sd, options=options)
                # Broadcast buffers not in state_dict (e.g., rotary embeddings)
                for _, buf in self.model.named_buffers():
                    torch.distributed.broadcast(buf, src=0)
            else:
                self.model.load_state_dict(sd)
            print(f"[Trainer] Loaded model from {model_path}", flush=True)

        if os.path.exists(state_path):
            ts = torch.load(state_path, map_location="cpu", weights_only=False)
            step = ts.get("step", 0)
            if "optimizer" in ts and self.optimizer is not None:
                optim_state = ts["optimizer"]
                # Handle compile mismatch: optimizer state saved with torch.compile
                # has _orig_mod. prefix, but loading without compile doesn't.
                if not hasattr(self.model, '_orig_mod'):
                    def _strip_orig_mod(obj):
                        if isinstance(obj, dict):
                            return {k.replace("_orig_mod.", ""): _strip_orig_mod(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_strip_orig_mod(v) for v in obj]
                        if isinstance(obj, str):
                            return obj.replace("_orig_mod.", "")
                        return obj
                    # Check if state dict has _orig_mod keys
                    if any("_orig_mod." in k for k in optim_state.get("state", {}) if isinstance(k, str)):
                        optim_state = _strip_orig_mod(optim_state)
                        print("[Trainer] Stripped _orig_mod. prefix from optimizer state (compile mismatch)", flush=True)
                _optim_loaded = False
                if self.use_fsdp:
                    try:
                        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions
                        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                        set_optimizer_state_dict(self.model, self.optimizer, optim_state, options=options)
                        _optim_loaded = True
                    except (KeyError, ValueError, RuntimeError) as e:
                        # Only catch format-mismatch errors (legacy FSDP1 checkpoint).
                        # Other exceptions (NCCL failure, OOM, etc.) should propagate.
                        print(f"[Trainer] WARNING: Failed to load optimizer state via FSDP2 API "
                              f"(likely legacy FSDP1 checkpoint format): {e}", flush=True)
                        print("[Trainer] Optimizer and scheduler will restart from scratch.", flush=True)
                else:
                    self.optimizer.load_state_dict(ts["optimizer"])
                    _optim_loaded = True
            if "scheduler" in ts and self.scheduler is not None and self.optimizer is not None and _optim_loaded:
                self.scheduler.load_state_dict(ts["scheduler"])
            print(f"[Trainer] Loaded training state from {state_path} (step={step})", flush=True)

        torch.cuda.synchronize()
        if self.use_fsdp:
            import torch.distributed as dist
            dist.barrier()

        return step

    def _gather_fsdp_optim_state_dict(self):
        """Gather full optimizer state dict from FSDP2 model (all ranks participate)."""
        try:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return get_optimizer_state_dict(self.model, self.optimizer, options=options)
        except Exception as e:
            print(f"[Trainer] WARNING: Failed to gather FSDP optimizer state: {e}", flush=True)
            return None

    def _gather_fsdp_state_dict(self):
        """Gather full (unsharded) state dict from FSDP2 model. All ranks participate in the collective."""
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        sd = get_model_state_dict(self.model, options=options)
        # Clean up torch.compile prefix (FSDP2 doesn't add _fsdp_wrapped_module.)
        cleaned = {}
        for name, param in sd.items():
            clean = name.replace("_orig_mod.", "")
            cleaned[clean] = param
        return cleaned

    # -------------------- Training --------------------

    def _distributed_sum_count(self, value, device):
        """Sum a scalar count across ranks when a process group is active.

        Returns ``(global_value, effective_world_size)``.  Unit tests often
        exercise multi-rank planning without initializing torch.distributed; in
        that case the local value is already the only available denominator and
        no FSDP gradient averaging compensation should be applied.
        """
        if (
            getattr(self, "use_distributed", False)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            t = torch.tensor(float(value), device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            return float(t.item()), torch.distributed.get_world_size()
        return float(value), 1

    def _run_train_step(self, batch, loss_fn, forward_and_loss_fn=None):
        """Generic FSDP training loop with pluggable loss function.

        When forward_and_loss_fn is provided (e.g., OPDTrainer's chunked path),
        it replaces the standard model forward + loss_fn path. The function
        receives raw micro-batch tensors and handles packing + forward + loss
        internally.

        When forward_and_loss_fn returns extras containing '_raw_tensors',
        those tensors are concatenated across micro-batches and used to
        compute percentile stats (kl_p95, kl_p99, r_p95, etc.).
        """
        model = self.model
        optimizer = self.optimizer
        device = self.device
        micro_batch_size = self.micro_batch_size
        scheduler = self.scheduler

        use_forward_and_loss = forward_and_loss_fn is not None

        # Extract required tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bs = input_ids.size(0)

        use_global_mini_plan = bool(batch.get("_use_global_mini_plan", False))
        if use_global_mini_plan:
            mini_slices = [tuple(x) for x in batch.get("_mini_slices", [])]
            common_micro_counts = list(batch.get("_common_micro_counts", []))
            if len(mini_slices) != len(common_micro_counts):
                raise RuntimeError(
                    "global-mini train metadata mismatch: "
                    f"{len(mini_slices)} slices vs {len(common_micro_counts)} micro counts"
                )
            n_mini = len(mini_slices)
            mini_bs = max((e - s for s, e in mini_slices), default=0)
        else:
            # Mini-batch splitting — mini_batch_size is global, divide by world_size
            per_rank_mini_bs = max(1, self.mini_batch_size // self.world_size) if self.mini_batch_size > 0 else 0
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

        total_loss = 0.0
        total_tokens = 0
        all_grad_norms = []
        extras_accum = {}
        raw_tensors_accum = {}  # for _raw_tensors percentile aggregation
        n_optim_steps = 0
        per_mini_r_mean = {}  # per-mini-batch r_mean for decoupled PPO checks
        per_mini_n_tokens = {}
        per_mini_avg_response_length = {}
        per_mini_p90_response_length = {}
        per_mini_n_seqs = {}
        per_mini_raw_tensors_accum = {}

        for mini_idx, (ms, me) in enumerate(mini_slices):
            local_mini_len = me - ms
            if local_mini_len <= 0:
                raise RuntimeError(
                    "global-mini-first produced an empty local mini-batch; "
                    "this shape is not safe for FSDP collectives"
                )

            mini_token_count = None
            mini_sample_count = None
            backward_count_kind = None
            backward_world_factor = 1
            mini_response_mask = batch.get("response_mask")
            if isinstance(mini_response_mask, torch.Tensor):
                mini_response_mask = mini_response_mask[ms:me]
                mini_token_total = int(mini_response_mask.sum().item())
                if n_mini > 1:
                    per_mini_n_tokens[mini_idx] = mini_token_total
                    per_mini_n_seqs[mini_idx] = int(mini_response_mask.size(0))
                    if mini_response_mask.size(0) > 0:
                        per_seq_resp = mini_response_mask.sum(dim=1).float()
                        per_mini_avg_response_length[mini_idx] = (
                            per_seq_resp.mean().item()
                        )
                        per_mini_p90_response_length[mini_idx] = (
                            per_seq_resp.quantile(0.9).item()
                            if per_seq_resp.numel() > 1
                            else per_seq_resp.item()
                        )

            if getattr(self, "pg_token_weighted_backward", False) and mini_response_mask is not None:
                mini_token_count = int(mini_response_mask.sum().item())
                if mini_token_count <= 0:
                    mini_token_count = None

            if use_global_mini_plan:
                if mini_response_mask is not None:
                    local_token_count = int(mini_response_mask.sum().item())
                    global_token_count, backward_world_factor = self._distributed_sum_count(
                        local_token_count, device)
                    if global_token_count > 0:
                        mini_token_count = global_token_count
                        backward_count_kind = "tokens"
                if backward_count_kind is None:
                    global_sample_count, backward_world_factor = self._distributed_sum_count(
                        local_mini_len, device)
                    if global_sample_count <= 0:
                        raise RuntimeError(
                            "global-mini-first cannot scale backward with zero samples"
                        )
                    mini_sample_count = global_sample_count
                    backward_count_kind = "samples"

            n_micro = common_micro_counts[mini_idx]
            if n_micro <= 0:
                raise RuntimeError(
                    f"invalid microstep count {n_micro} for mini {mini_idx}"
                )
            if local_mini_len < n_micro:
                raise RuntimeError(
                    "global-mini-first cannot create non-empty microsteps for "
                    f"mini {mini_idx}: local_len={local_mini_len}, "
                    f"common_n_micro={n_micro}, micro_batch_size={micro_batch_size}"
                )
            micro_slices = BaseBackend._split_span_for_micro_steps(ms, me, n_micro)

            optimizer.zero_grad()

            for s_abs, e_abs in micro_slices:
                # Slice all tensors in batch to micro-batch range, move to device
                mb = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.size(0) == bs:
                        mb[k] = v[s_abs:e_abs].to(device)
                    else:
                        mb[k] = v  # pass through non-tensor or non-batch-dim values

                if use_forward_and_loss:
                    # forward_and_loss_fn handles packing + forward + loss
                    loss, n_tok, extras = forward_and_loss_fn(model, mb, device)
                else:
                    # Standard path: packing + model forward + loss_fn
                    if self.use_sequence_packing:
                        from opd.data.packing import pack_micro_batch
                        pack_kwargs = {
                            "input_ids": mb["input_ids"],
                            "attention_mask": mb["attention_mask"],
                            "response_mask": mb["response_mask"],
                        }
                        if "teacher_logprobs" in mb and isinstance(mb["teacher_logprobs"], torch.Tensor):
                            pack_kwargs["teacher_logprobs"] = mb["teacher_logprobs"]
                        if "teacher_top_indices" in mb and isinstance(mb["teacher_top_indices"], torch.Tensor):
                            pack_kwargs["teacher_top_indices"] = mb["teacher_top_indices"]
                        if "old_logprobs" in mb and isinstance(mb["old_logprobs"], torch.Tensor):
                            pack_kwargs["old_logprobs"] = mb["old_logprobs"]
                        packed = pack_micro_batch(**pack_kwargs)
                        mb.update(packed)

                    fwd_kwargs = dict(
                        input_ids=mb["input_ids"],
                        attention_mask=mb["attention_mask"],
                        use_cache=False,
                    )
                    if "position_ids" in mb and isinstance(mb["position_ids"], torch.Tensor):
                        fwd_kwargs["position_ids"] = mb["position_ids"]

                    out = model(**fwd_kwargs)
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    loss, n_tok, extras = loss_fn(logits, mb)
                    del out, logits

                if use_global_mini_plan and backward_count_kind == "tokens":
                    backward_scale = (
                        backward_world_factor * n_tok / mini_token_count
                        if n_tok > 0 else 0.0
                    )
                elif use_global_mini_plan and backward_count_kind == "samples":
                    micro_sample_count = int(mb["input_ids"].size(0))
                    backward_scale = (
                        backward_world_factor * micro_sample_count / mini_sample_count
                    )
                elif mini_token_count is not None:
                    backward_scale = n_tok / mini_token_count if n_tok > 0 else 0.0
                else:
                    backward_scale = 1.0 / n_micro

                (loss * backward_scale).backward()

                total_loss += loss.detach().item() * n_tok if n_tok > 0 else loss.detach().item()
                total_tokens += n_tok

                # Accumulate raw tensors separately for percentile stats
                raw_tensors = extras.pop("_raw_tensors", None)
                if raw_tensors is not None:
                    for rk, rv in raw_tensors.items():
                        if rk not in raw_tensors_accum:
                            raw_tensors_accum[rk] = []
                        raw_tensors_accum[rk].append(rv)
                        if n_mini > 1:
                            mini_store = per_mini_raw_tensors_accum.setdefault(mini_idx, {})
                            mini_store.setdefault(rk, []).append(rv)

                for ek, ev in extras.items():
                    extras_accum[ek] = extras_accum.get(ek, 0.0) + ev

                # Track per-mini-batch r_mean for decoupled PPO verification
                if n_mini > 1 and "r_mean" in extras:
                    per_mini_r_mean.setdefault(mini_idx, []).append(extras["r_mean"])

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.max_grad_norm)
            optimizer.step()
            n_optim_steps += 1
            all_grad_norms.append(grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm))
            if scheduler is not None:
                scheduler.step()
            if self.rank == 0 and n_mini > 1:
                print(f"  [train] optim step {n_optim_steps}/{n_mini}", flush=True)

        total_steps = sum(common_micro_counts) if use_global_mini_plan else (
            n_mini * max(1, (mini_bs + micro_batch_size - 1) // micro_batch_size)
        )
        result = {
            "kl_loss": total_loss / max(total_tokens, 1),
            "n_tokens": total_tokens,
            "grad_norm": max(all_grad_norms) if all_grad_norms else 0.0,
        }
        result["n_optim_steps"] = n_optim_steps
        if scheduler is not None:
            result["lr"] = scheduler.get_last_lr()[0]
        for ek, ev in extras_accum.items():
            result[ek] = ev / max(total_steps, 1)

        # Compute percentile stats from raw tensors (if collected)
        if raw_tensors_accum:
            def _stats_ready(t):
                return t if t.dtype in (torch.float32, torch.float64) else t.float()

            if "kl_vals" in raw_tensors_accum:
                kl_all = _stats_ready(torch.cat(raw_tensors_accum["kl_vals"]))
                if kl_all.numel() > 0:
                    result["kl_std"] = kl_all.std().item() if kl_all.numel() > 1 else 0.0
                    result["kl_min"] = kl_all.min().item()
                    result["kl_max"] = kl_all.max().item()
                    result["kl_p95"] = _diagnostic_quantile(kl_all, 0.95) if kl_all.numel() > 1 else 0.0
                    result["kl_p99"] = _diagnostic_quantile(kl_all, 0.99) if kl_all.numel() > 1 else 0.0
            if "ratios" in raw_tensors_accum:
                r = _stats_ready(torch.cat(raw_tensors_accum["ratios"]))
                lr = _stats_ready(torch.cat(raw_tensors_accum["log_ratios"]))
                adv = _stats_ready(torch.cat(raw_tensors_accum["advantages"]))
                ch = torch.cat(raw_tensors_accum["clip_high"])
                cl = torch.cat(raw_tensors_accum["clip_low"])
                if r.numel() > 0:
                    result["r_mean"] = r.mean().item()
                    result["r_std"] = r.std().item() if r.numel() > 1 else 0.0
                    result["r_min"] = r.min().item()
                    result["r_max"] = r.max().item()
                    result["r_p95"] = _diagnostic_quantile(r, 0.95) if r.numel() > 1 else 0.0
                    result["r_p99"] = _diagnostic_quantile(r, 0.99) if r.numel() > 1 else 0.0
                    result["r_p999"] = _diagnostic_quantile(r, 0.999) if r.numel() > 1 else 0.0
                    result["logr_mean"] = lr.mean().item()
                    result["logr_std"] = lr.std().item() if lr.numel() > 1 else 0.0
                    result["clip_frac_high"] = ch.float().mean().item()
                    result["clip_frac_low"] = cl.float().mean().item()
                    result["adv_mean"] = adv.mean().item()
                if adv.numel() > 1:
                    result["adv_std"] = adv.std().item()
                    result["adv_p10"] = _diagnostic_quantile(adv, 0.1)
                    result["adv_p50"] = _diagnostic_quantile(adv, 0.5)
                    result["adv_p90"] = _diagnostic_quantile(adv, 0.9)

        # Per-mini-batch r_mean (for decoupled PPO mini-batch divergence checks)
        for mi, vals in per_mini_r_mean.items():
            result[f"r_mean_mini_{mi}"] = sum(vals) / len(vals)
        for mi, n_tok in per_mini_n_tokens.items():
            result[f"n_tokens_mini_{mi}"] = n_tok
        for mi, avg_len in per_mini_avg_response_length.items():
            result[f"avg_response_length_mini_{mi}"] = avg_len
        for mi, p90_len in per_mini_p90_response_length.items():
            result[f"response_length_p90_mini_{mi}"] = p90_len
        for mi, n_seq in per_mini_n_seqs.items():
            result[f"n_seqs_mini_{mi}"] = n_seq
        for mi, raw in per_mini_raw_tensors_accum.items():
            if "ratios" in raw:
                r_mini = _stats_ready(torch.cat(raw["ratios"]))
                if r_mini.numel() > 1:
                    result[f"r_p99_mini_{mi}"] = _diagnostic_quantile(r_mini, 0.99)
            if "advantages" in raw:
                adv_mini = _stats_ready(torch.cat(raw["advantages"]))
                if adv_mini.numel() > 0:
                    result[f"adv_mean_mini_{mi}"] = adv_mini.mean().item()
                    result[f"adv_std_mini_{mi}"] = (
                        adv_mini.std().item() if adv_mini.numel() > 1 else 0.0
                    )
            if "returns" in raw:
                ret_mini = _stats_ready(torch.cat(raw["returns"]))
                if ret_mini.numel() > 0:
                    result[f"return_mean_mini_{mi}"] = ret_mini.mean().item()
                    result[f"return_std_mini_{mi}"] = (
                        ret_mini.std().item() if ret_mini.numel() > 1 else 0.0
                    )
            if "values" in raw:
                val_mini = _stats_ready(torch.cat(raw["values"]))
                if val_mini.numel() > 0:
                    result[f"value_mean_mini_{mi}"] = val_mini.mean().item()
                    result[f"value_std_mini_{mi}"] = (
                        val_mini.std().item() if val_mini.numel() > 1 else 0.0
                    )

        return result

    # -------------------- Model setup helpers --------------------
    # (_fsdp_train_step deleted — OPD KL loss now routes through
    #  OPDTrainer.train_step → _run_train_step + forward_and_loss_fn)

    def _get_lora_state_dict(self):
        """Extract LoRA A/B matrices only, with clean names for vLLM.

        Walks the PEFT model state dict, keeps only lora_A/lora_B weights,
        strips PEFT prefixes (base_model.model., .base_layer.), and casts
        to bfloat16 for consistent dtype in weight transfer.

        For FSDP2: gathers full state dict then filters to LoRA params.
        Returns dict of {clean_name: tensor} on CPU.
        """
        if self.use_fsdp:
            # FSDP2: gather full state dict then filter to LoRA params
            full_sd = self._gather_fsdp_state_dict()
            lora_sd = {}
            for k, v in full_sd.items():
                if "lora_A" not in k and "lora_B" not in k:
                    continue
                clean = k.replace("base_model.model.", "")
                clean = clean.replace(".base_layer.", ".")
                clean = clean.replace(".lora_A.default.", ".lora_A.")
                clean = clean.replace(".lora_B.default.", ".lora_B.")
                lora_sd[clean] = v.to(torch.bfloat16).cpu().clone()
            return lora_sd
        return self._extract_lora_tensors()

    def _extract_lora_tensors(self):
        """Extract LoRA A/B tensors from PEFT model with clean vLLM-compatible names."""
        inner = self._get_peft_inner()
        raw_sd = inner.state_dict()
        lora_sd = {}
        for k, v in raw_sd.items():
            if "lora_A" not in k and "lora_B" not in k:
                continue
            clean = k.replace("base_model.model.", "")
            clean = clean.replace(".base_layer.", ".")
            # Strip PEFT adapter name (e.g. ".default.") — vLLM expects
            # "lora_A.weight" not "lora_A.default.weight"
            clean = clean.replace(".lora_A.default.", ".lora_A.")
            clean = clean.replace(".lora_B.default.", ".lora_B.")
            lora_sd[clean] = v.to(torch.bfloat16).cpu().clone()
        return lora_sd

    def _apply_lora(self, lora_cfg):
        """Wrap model with PEFT LoRA adapters."""
        from peft import get_peft_model, LoraConfig, TaskType

        config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg.get("dropout", _LORA_DEFAULTS.dropout),
            target_modules=lora_cfg.get("target_modules", _LORA_DEFAULTS.target_modules),
            modules_to_save=lora_cfg.get("modules_to_save"),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        # Cast LoRA params to match base model dtype (FSDP requires uniform dtype)
        model_dtype = next(p.dtype for p in self.model.parameters() if not p.requires_grad)
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != model_dtype:
                p.data = p.data.to(model_dtype)
        # Required for gradient checkpointing with frozen base model
        self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()
        self._is_lora = True

    def _get_peft_inner(self):
        """Get the PEFT model. FSDP2 doesn't wrap in .module, so just return model."""
        return self.model

    def _patch_lora_model_for_chunked_kl(self):
        """Patch the PEFT-wrapped model for chunked KL computation.

        The standard _patch_model_for_chunked_kl accesses model.model and
        model.lm_head, but PEFT wraps these. We redirect through the PEFT
        wrapper to access the actual HF model's backbone and lm_head.
        """
        import types

        peft_model = self.model  # PeftModelForCausalLM
        # The actual HF model is at peft_model.base_model.model
        hf_model = peft_model.base_model.model
        original_forward = peft_model.forward

        def _patched_forward(self_model, *args, _kl_args=None, **kwargs):
            if _kl_args is None:
                return original_forward(*args, **kwargs)

            fa_kwargs = {}
            for k in ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k'):
                if k in kwargs:
                    fa_kwargs[k] = kwargs.pop(k)

            # Access backbone through PEFT wrapper (uses LoRA-adapted layers)
            outputs = hf_model.model(
                input_ids=kwargs.get('input_ids', args[0] if args else None),
                attention_mask=kwargs.get('attention_mask'),
                position_ids=kwargs.get('position_ids'),
                use_cache=False,
                **fa_kwargs,
            )
            hidden_states = outputs[0]
            # lm_head may be wrapped by PEFT if in modules_to_save
            lm_weight = hf_model.lm_head.weight

            mode = _kl_args['mode']
            chunk_size = _kl_args.get('chunk_size', 1024)
            return_values = bool(_kl_args.get('return_values', False))

            if mode == 'return_hidden':
                return {'hidden_states': hidden_states, 'lm_head_weight': lm_weight}
            if mode in ('forward_kl', 'reverse_kl', 'reverse_kl_rollout_student_topk', 'thunlp_opd_default_loss', 'skewed_kl'):
                return chunked_lm_head_gather(
                    hidden_states, lm_weight, _kl_args['indices'],
                    chunk_size=chunk_size,
                )
            elif mode in ('multi_sample_policy_gradient_kl', 'multi_sample_forward_kl'):
                return chunked_lm_head_gather(
                    hidden_states[:, :-1], lm_weight, _kl_args['indices'][:, 1:],
                    chunk_size=chunk_size,
                )
            elif mode in ('token_level_kl', 'policy_gradient_kl'):
                input_ids = kwargs.get('input_ids', args[0] if args else None)
                target_ids = input_ids[:, 1:].unsqueeze(-1)
                student_token_logps = chunked_lm_head_gather(
                    hidden_states[:, :-1], lm_weight, target_ids,
                    chunk_size=chunk_size,
                )
                if return_values:
                    if not hasattr(hf_model, "value_head"):
                        raise AttributeError(
                            "Chunked KL forward requested values, but model has no value_head.")
                    values = hf_model.value_head(hidden_states).squeeze(-1)
                    return {
                        "student_token_logps": student_token_logps,
                        "values": values,
                    }
                return student_token_logps
            else:
                return hf_model.lm_head(hidden_states)

        peft_model.forward = types.MethodType(_patched_forward, peft_model)

    def _maybe_attach_value_head(self, hf_config):
        """Attach a trainer-only scalar value head when actor-critic PG-KL is enabled."""
        if not algorithm_is_actor_critic(getattr(self, "_algorithm_launch")):
            return

        hidden_size = getattr(hf_config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model, "lm_head"):
            hidden_size = self.model.lm_head.weight.shape[1]
        if hidden_size is None:
            raise ValueError("Unable to infer hidden_size for actor-critic value head")

        base_param = next(self.model.parameters(), None)
        value_dtype = base_param.dtype if base_param is not None else torch.bfloat16
        value_head = nn.Linear(
            hidden_size,
            1,
            bias=True,
            device=self.device,
            dtype=value_dtype,
        )
        nn.init.zeros_(value_head.weight)
        nn.init.zeros_(value_head.bias)
        self.model.value_head = value_head

    def _apply_fsdp2(self, model):
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp_dtype = getattr(torch, getattr(self, "_dtype_str", "bfloat16"), torch.bfloat16)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
        )
        fsdp_kwargs = {"mp_policy": mp_policy}

        # Find transformer layer classes from HF model config
        transformer_cls_names = getattr(model, "_no_split_modules", None) or []
        # For PEFT models, check the underlying HF model
        if not transformer_cls_names and hasattr(model, "base_model"):
            hf_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
            transformer_cls_names = getattr(hf_model, "_no_split_modules", None) or []

        # Fallback: discover layers dynamically via model.model.layers
        if not transformer_cls_names:
            inner = getattr(model, "model", None)
            # For PEFT: try base_model.model.model
            if inner is None and hasattr(model, "base_model"):
                inner = getattr(model.base_model, "model", None)
                if inner is not None:
                    inner = getattr(inner, "model", None)
            if inner is not None and hasattr(inner, "layers") and len(inner.layers) > 0:
                transformer_cls_names = [type(inner.layers[0]).__name__]

        assert transformer_cls_names, \
            f"Model {model.__class__.__name__} has no _no_split_modules and no model.model.layers; " \
            f"cannot discover transformer layers for per-layer FSDP2 sharding"

        # Apply per-layer sharding to transformer layers
        for module in model.modules():
            if module.__class__.__name__ in transformer_cls_names:
                fully_shard(module, **fsdp_kwargs)

        # Root-level shard
        fully_shard(model, **fsdp_kwargs)
        return model

    # -------------------- Weight helpers --------------------

    def _build_weights_info(self):
        """Build (name, shape, dtype) list matching model.state_dict() order.

        For native LoRA: returns LoRA A/B tensor info (small, ~5M params).
        For merge-then-sync LoRA: returns base model keys (captured before PEFT).
        """
        if getattr(self, '_is_lora', False) and getattr(self, '_use_native_lora', False):
            lora_sd = self._get_lora_state_dict()
            return [
                (name, tuple(t.shape), t.dtype) for name, t in lora_sd.items()
                if not name.startswith("value_head.")
            ]
        if getattr(self, '_is_lora', False) and getattr(self, '_base_weights_info', None) is not None:
            return self._base_weights_info
        if self.use_fsdp:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
            options = StateDictOptions(full_state_dict=True, cpu_offload=False)
            sd = get_model_state_dict(self.model, options=options)
        else:
            sd = self.model.state_dict()
        info = []
        for name, param in sd.items():
            clean = name.replace("_orig_mod.", "")
            if clean.startswith("value_head."):
                continue
            info.append((clean, param.shape, param.dtype))
        return info


def fsdp_trainer_main(config, cmd_queue, result_queue, rank_info):
    """Entry point for FSDP training subprocess.

    Auto-dispatches to the appropriate composition trainer based on loss_mode
    in the config. All training modes now use composition trainers that own
    their loss functions; FSDPBackend is a pure training engine.
    """
    launch_spec = ensure_trainer_launch_spec(config, rank_info) if isinstance(config, TrainerLaunchSpec) else None
    loss_mode = launch_spec.loss_mode if launch_spec is not None else config.get("loss_mode", config.get("algorithm", {}).get("mode", "kl"))
    trainer_input = launch_spec if launch_spec is not None else config
    trainer_rank_info = None if launch_spec is not None else rank_info
    if loss_mode == "sft":
        from opd.trainer.sft import SFTTrainer
        trainer = SFTTrainer(trainer_input, trainer_rank_info)
    elif loss_mode == "grpo":
        from opd.trainer.grpo import GRPOTrainer
        trainer = GRPOTrainer(trainer_input, trainer_rank_info)
    else:
        algo = launch_spec.static.algorithm if launch_spec is not None else config.get("algorithm", {})
        if algorithm_is_actor_critic(algo):
            from opd.trainer.ac_opd import ActorCriticOPDTrainer
            trainer = ActorCriticOPDTrainer(trainer_input, trainer_rank_info)
        else:
            from opd.trainer.opd import OPDTrainer
            trainer = OPDTrainer(trainer_input, trainer_rank_info)
    trainer.run(cmd_queue, result_queue)
