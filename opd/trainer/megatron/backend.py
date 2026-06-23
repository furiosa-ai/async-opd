"""Training worker -- Megatron-LM backend with 3D parallelism (TP/PP/DP).

Runs as multiple processes (one per global rank), coordinated via
torch.distributed. Rank 0 owns cmd_queue/result_queue; other ranks
receive commands via broadcast_object_list.

Commands via mp.Queue (same protocol as FSDPTrainer):
  ("train", batch_dict)            -> one training step, returns metrics
  ("sync_weights", merge_map)      -> NCCL broadcast weights to rollout workers
  ("init_weight_transfer", info)   -> init vLLM NCCL weight transfer engine
  ("get_weights_info",)            -> return (name, shape, dtype) list
  ("save_checkpoint", save_info)   -> save model + optimizer state
  ("load_checkpoint", load_info)   -> load model + optimizer state
  ("compute_weight_checksum", map) -> compute weight checksum for verification
  ("shutdown",)                    -> exits

Architecture:
  - Uses native Megatron-Core GPTModel with real TP/PP/DP sharding,
    fused kernels, pipeline schedule, and DDP.
  - Global rank ordering follows Megatron convention (tp-pp-dp):
    global_rank = dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
  - Weight sync: gather TP-sharded weights to rank 0, broadcast PP stages'
    dicts, then NCCL transfer to vLLM rollout workers.
"""

import os
from datetime import timedelta

import torch

from opd.launch_specs import (
    TrainerLaunchSpec,
    algorithm_mode,
    ensure_trainer_launch_spec,
)
from opd.loss.kl import compute_kl_loss
from opd.trainer.base import BaseBackend, build_lr_scheduler
from opd.trainer.config import (
    build_grpo_config_from_algorithm_payload,
    build_sft_config_from_algorithm_payload,
)
from opd.utils.config import OptimConfig, resolve_trust_remote_code


_OPTIM_DEFAULTS = OptimConfig()


# ===================== Megatron Trainer =====================

class MegatronBackend(BaseBackend):
    """Megatron-LM training backend with 3D parallelism (TP/PP/DP).

    Pipeline spawns tp_size * pp_size * dp_size processes. Rank 0 owns
    cmd_queue/result_queue; other ranks receive commands via
    torch.distributed.broadcast_object_list.

    With use_native_megatron=True (production):
      - Real TP weight sharding (QKV, MLP, embeddings, output_layer)
      - Pipeline parallelism with pre_process/post_process per stage
      - Megatron DDP for gradient sync across DP ranks
      - Vocab-parallel KL loss for all 5 modes

    With use_native_megatron=False (legacy debug):
      - Each rank loads the full HF model (pseudo-TP)
      - No real sharding, but exercises the multi-rank infrastructure
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

        # Unpack loss mode and mode-specific algorithm attributes BEFORE
        # super().__init__ so BaseBackend picks them up via getattr() defaults.
        algo = algorithm_launch
        mode = algorithm_mode(algo)
        self.loss_mode = config["loss_mode"]
        self.sft_config = None
        self.grpo_config = None
        if mode == "sft":
            self.sft_config = build_sft_config_from_algorithm_payload(algo)
        if mode == "grpo":
            self.grpo_config = build_grpo_config_from_algorithm_payload(algo)

        super().__init__(config, rank_info)

        megatron_cfg = config["megatron"]
        self.use_native_megatron = megatron_cfg["use_native_megatron"]
        self._use_transformer_engine = megatron_cfg["use_transformer_engine"]
        self.max_grad_norm = float(config["max_grad_norm"])
        self._trust_remote_code = resolve_trust_remote_code(
            config.get("trust_remote_code"),
            context="Megatron trainer model loading",
        )

        # Unpack from rank_info
        tp_size = megatron_cfg["tp_size"]
        tp_rank = rank_info["tp_rank"]
        pp_size = megatron_cfg["pp_size"]
        pp_rank = rank_info["pp_rank"]
        global_rank = rank_info["global_rank"]
        global_world_size = rank_info["global_world_size"]
        megatron_master_port = rank_info["megatron_master_port"]
        megatron_master_addr = rank_info["megatron_master_addr"]
        nccl_timeout_hours = int(config["nccl_timeout_hours"])
        gpu_ids = config.get("gpu_ids")
        model_path = self._model_path
        dtype = self._dtype_str
        mini_batch_size = self.mini_batch_size
        total_steps = self.total_steps
        lr = float(self.optim_cfg["lr"])
        self._scheduler_lr = lr
        lora_cfg = config.get("lora")

        # Override rank/world_size with Megatron global values
        self.rank = global_rank
        self.world_size = global_world_size

        # Sequence packing requires Transformer Engine (TEDotProductAttention).
        # mcore's DotProductAttention does not support packed_seq_params.
        if self.use_sequence_packing and not self._use_transformer_engine:
            try:
                import transformer_engine  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Megatron sequence packing requires Transformer Engine. "
                    "Either install transformer_engine or set use_sequence_packing=False."
                )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank

        # Each rank sees exactly one GPU
        gpu_ids_list = gpu_ids.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_list[global_rank] if len(gpu_ids_list) > 1 else gpu_ids_list[0]
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # Init Megatron TP process group.
        # vLLM weight transfer uses StatelessProcessGroup (NOT torch.distributed),
        # so there's no conflict with Megatron's parallel_state init.
        self.use_megatron = tp_size > 1 or pp_size > 1
        need_mpu = self.use_megatron or self.use_native_megatron
        if need_mpu:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl", rank=global_rank, world_size=global_world_size,
                    init_method=f"tcp://{megatron_master_addr}:{megatron_master_port}",
                    timeout=timedelta(hours=nccl_timeout_hours),
                )
            from megatron.core import parallel_state as mpu
            if not mpu.model_parallel_is_initialized():
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=tp_size,
                    pipeline_model_parallel_size=pp_size,
                )
                # Update ranks from Megatron's parallel state — Megatron's rank
            # ordering (tp-cp-ep-dp-pp) may differ from our naive computation.
            self.tp_rank = mpu.get_tensor_model_parallel_rank()
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        # Load model
        self._init_native_megatron_model(model_path, dtype)

        # LoRA: add TP-sharded adapters via forward hooks.
        # Must be applied BEFORE DDP wrapping so LoRA params are in DDP's
        # gradient buckets (automatic DP gradient sync, no manual all-reduce).
        self._is_lora = False
        self._lora_params = {}
        self._lora_cfg = lora_cfg
        if lora_cfg:
            from opd.trainer.megatron.lora import apply_megatron_lora
            tp_group = None
            if self.use_megatron:
                from megatron.core import parallel_state as mpu
                tp_group = mpu.get_tensor_model_parallel_group()
            self._lora_params = apply_megatron_lora(
                self._unwrapped_model,
                rank=lora_cfg["rank"],
                alpha=lora_cfg["alpha"],
                target_modules=lora_cfg.get("target_modules"),
                tp_size=tp_size,
                tp_group=tp_group,
                dtype=getattr(torch, dtype, torch.bfloat16),
                hf_config=self._hf_config,
            )
            self._is_lora = True
            # Native LoRA for vLLM rollout
            if lora_cfg.get("native_lora", False):
                from opd.rollout.vllm.lora import build_peft_config_dict
                self._use_native_lora = True
                self._peft_config_dict = build_peft_config_dict(lora_cfg)

        # Wrap in Megatron DDP — required even at world_size=1 because
        # forward_backward_func calls model.finish_grad_sync() and accesses
        # model.ddp_config, which are DDP-only attributes.
        # DDP creates gradient buckets for requires_grad=True params only,
        # so with LoRA applied above, only LoRA params get DP gradient sync.
        if self.use_native_megatron:
            from megatron.core.distributed import DistributedDataParallel as MegatronDDP
            from megatron.core.distributed import DistributedDataParallelConfig
            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
            )
            self.model = MegatronDDP(
                config=self._tf_config,
                ddp_config=ddp_config,
                module=self.model,
            )

        # Optimizer: always use Megatron DistributedOptimizer (ZeRO-1 + fp32
        # master weights). Single code path for both full-model and LoRA.
        from megatron.core.optimizer import OptimizerConfig as MegatronOptimConfig
        from megatron.core.optimizer import get_megatron_optimizer

        weight_decay = float(self.optim_cfg.get("weight_decay", _OPTIM_DEFAULTS.weight_decay))
        param_dtype = getattr(torch, dtype, torch.bfloat16)
        optim_config = MegatronOptimConfig(
            optimizer="adam",
            lr=lr,
            adam_beta1=0.9,
            adam_beta2=float(self.optim_cfg.get("adam_beta2", _OPTIM_DEFAULTS.adam_beta2)),
            adam_eps=float(self.optim_cfg.get("adam_eps", _OPTIM_DEFAULTS.adam_eps)),
            weight_decay=weight_decay,
            bf16=param_dtype is torch.bfloat16,
            fp16=param_dtype is torch.float16,
            params_dtype=param_dtype,
            use_distributed_optimizer=True,
            clip_grad=self.max_grad_norm,
        )
        self.optimizer = get_megatron_optimizer(
            config=optim_config, model_chunks=[self.model])

        # Register training hooks — required for Megatron's
        # forward_backward schedule to handle gradient finalization.
        from megatron.core.distributed import finalize_model_grads
        from megatron.core.utils import get_model_config
        model_config = get_model_config(self.model)
        model_config.grad_scale_func = self.optimizer.scale_loss
        model_config.finalize_model_grads_func = finalize_model_grads

        dp_size = self.world_size // (self.tp_size * self.pp_size)
        print(f"[Backend-Megatron] Megatron DistributedOptimizer "
              f"(ZeRO-1, fp32 master weights, DP={dp_size})", flush=True)

        # LR scheduler — Megatron's ChainedOptimizer isn't a torch.optim.Optimizer
        # subclass, so PyTorch schedulers reject it. Use a dummy optimizer to
        # compute LR schedule, then copy LR to Megatron optimizer each step.
        self._dummy_scheduler_opt = None
        self.scheduler = self._build_lr_scheduler(total_steps) if total_steps > 0 else None
        self._scheduler_needs_rebuild = mini_batch_size > 0 and total_steps > 0

        # Build weights info eagerly (metadata only, no gather).
        # Fused names (qkv_proj, gate_up_proj) matching vLLM model params.
        # For LoRA: returns LoRA A/B param info with HF names.
        self.weights_info = self._build_weights_info()
        self._collective_weights_info = self.weights_info

        # Register Megatron-specific command handlers
        self._command_handlers["compute_weight_checksum"] = self._handle_compute_weight_checksum
        self._command_handlers["compute_lora_checksum"] = self._handle_compute_lora_checksum

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        extra = []
        if mini_batch_size > 0:
            extra.append(f"mini_batch_size={mini_batch_size}")
        extra_str = f", {', '.join(extra)}" if extra else ""
        print(f"[Backend-Megatron] rank {tp_rank}/{tp_size} ready. "
              f"Params: {n_params:.1f}M{extra_str}", flush=True)

    def _build_lr_scheduler(self, total_steps: int):
        """Build an LR scheduler for Megatron's non-PyTorch optimizer wrapper."""
        _dummy_opt = torch.optim.SGD([torch.zeros(1)], lr=self._scheduler_lr)
        self._dummy_scheduler_opt = _dummy_opt
        return build_lr_scheduler(_dummy_opt, self.optim_cfg, total_steps)

    @staticmethod
    def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Match HuggingFace left-padding position IDs for decoder-only models."""
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    @staticmethod
    def _causal_padding_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Build Megatron's boolean [B, 1, S, S] causal + key-padding mask."""
        batch, seq_len = attention_mask.shape
        device = attention_mask.device
        causal = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len)
        key_padding = attention_mask.eq(0).view(batch, 1, 1, seq_len)
        return causal | key_padding

    def _init_native_megatron_model(self, model_path, dtype):
        """Load native Megatron GPTModel with fused kernels (TE when available).

        Creates a Megatron GPTModel via ``make_model_provider`` and loads HF
        checkpoint weights via ``load_hf_weights_to_mcore``.  At TP=1 no real
        tensor parallelism is active, but fused attention/layernorm/MLP kernels
        from Transformer Engine are used when installed.
        """
        from transformers import AutoConfig
        from opd.trainer.megatron.model import (
            hf_config_to_mcore_config, make_model_provider, load_hf_weights_to_mcore,
        )

        # Init CUDA RNG tracker — required by Megatron's VocabParallelEmbedding
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(42)

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        hf_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=self._trust_remote_code)

        # Build Megatron TransformerConfig from HF config.
        # use_transformer_engine: false in config disables sequence_parallel
        # (which requires TE fused layernorms). Without this, TP>1 without TE errors.
        mcore_overrides = {}
        if self._use_transformer_engine:
            mcore_overrides["_require_te"] = True
        else:
            mcore_overrides["sequence_parallel"] = False

        dp_size = self.world_size // (self.tp_size * self.pp_size) if self.tp_size > 0 else 1
        tf_config = hf_config_to_mcore_config(
            hf_config,
            tp_size=self.tp_size,
            dp_size=dp_size,
            pp_size=self.pp_size,
            dtype=torch_dtype,
            **mcore_overrides,
        )

        # Determine max sequence length from config
        max_seq_len = getattr(hf_config, "max_position_embeddings", 8192)

        # Create model provider and instantiate GPTModel
        # With PP: first stage gets embedding (pre_process), last gets output_layer (post_process)
        pre_process = self.pp_rank == 0
        post_process = self.pp_rank == self.pp_size - 1
        provider = make_model_provider(tf_config, hf_config, max_seq_len=max_seq_len)
        self.model = provider(pre_process=pre_process, post_process=post_process)
        self.model = self.model.to(self.device)

        # Load HF weights into Megatron model
        load_hf_weights_to_mcore(
            self.model, model_path,
            tp_rank=self.tp_rank, tp_size=self.tp_size,
            pp_rank=self.pp_rank, pp_size=self.pp_size,
            trust_remote_code=self._trust_remote_code,
        )

        self.model.train()
        self._hf_config = hf_config
        self._tf_config = tf_config

        # NOTE: DDP wrapping is deferred to __init__ (after LoRA is applied)
        # so that LoRA params are included in DDP's gradient buckets.

        print(f"[Backend-Megatron] Native Megatron GPTModel loaded from {model_path}",
              flush=True)

    # -------------------- Abstract method implementations --------------------

    @property
    def _unwrapped_model(self):
        """Get the raw GPTModel, unwrapping DDP if present."""
        return getattr(self.model, 'module', self.model)

    @property
    def use_distributed(self) -> bool:
        return self.use_megatron

    def _get_log_prefix(self) -> str:
        return "Trainer-Megatron"

    @property
    def dp_rank(self) -> int:
        """Data-parallel rank for batch splitting (uses Megatron parallel_state)."""
        if self.use_native_megatron or self.use_megatron:
            from megatron.core import parallel_state as mpu
            if mpu.model_parallel_is_initialized():
                return mpu.get_data_parallel_rank()
        return self.rank

    @property
    def dp_world_size(self) -> int:
        """Data-parallel world size for batch splitting (uses Megatron parallel_state)."""
        if self.use_native_megatron or self.use_megatron:
            from megatron.core import parallel_state as mpu
            if mpu.model_parallel_is_initialized():
                return mpu.get_data_parallel_world_size()
        return self.world_size

    # ================================================================ #
    #  Abstract method implementations                                   #
    # ================================================================ #

    def _train_step_impl(self, batch):
        return self._train_step(batch)

    def _get_state_dict_for_sync(self):
        if self._is_lora:
            # LoRA mode: return only adapter params (TP-gathered, HF names)
            from opd.trainer.megatron.lora import gather_lora_state_dict
            tp_group = None
            if self.use_megatron:
                from megatron.core import parallel_state as mpu
                tp_group = mpu.get_tensor_model_parallel_group()
            num_layers = self._hf_config.num_hidden_layers
            layers_per_stage = num_layers // self.pp_size
            sd = gather_lora_state_dict(
                self._unwrapped_model, self._lora_params,
                tp_size=self.tp_size, tp_rank=self.tp_rank,
                tp_group=tp_group,
                pp_size=self.pp_size, pp_rank=self.pp_rank,
                layers_per_stage=layers_per_stage,
            )
            if self.pp_size > 1:
                # PP gather: combine LoRA from all stages within PP group (DP-safe).
                from megatron.core import parallel_state as _mpu
                pp_group = _mpu.get_pipeline_model_parallel_group()
                pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                full_sd = {}
                for pp_r in range(self.pp_size):
                    src_global = pp_group_ranks[pp_r]
                    stage_data = [sd if self.rank == src_global else None]
                    torch.distributed.broadcast_object_list(
                        stage_data, src=src_global, group=pp_group)
                    if self.rank == 0 and stage_data[0]:
                        full_sd.update(stage_data[0])
                sd = full_sd if self.rank == 0 else {}
            return sd
        # Returns None -- _handle_sync_weights calls _get_clean_state_dict
        # which handles both native megatron (TP/PP gather) and legacy paths.
        return None

    def _gather_full_state_dict(self):
        # Checkpoint gather handled by _save_checkpoint directly.
        return None

    def _gather_full_optim_state_dict(self):
        return None

    def _get_clean_state_dict(self):
        """Get state dict with fused HF-format names for weight sync.

        Uses _gather_hf_state_dict which handles TP all-gather + PP broadcast.
        Returns fused names (qkv_proj, gate_up_proj) that match vLLM's
        model.named_parameters() — used with is_checkpoint_format=False
        for direct param copy (no load_weights stacking overhead).
        """
        return self._gather_hf_state_dict()

    def _defuse_gathered_state_dict(self, sd, *, include_lm_head: bool = False):
        """De-fuse qkv_proj → q/k/v and gate_up_proj → gate/up for load_weights.

        vLLM's model.load_weights() expects standard HF checkpoint names.
        Excludes lm_head.weight (vLLM ties it to embed_tokens.weight).
        """
        hf = self._hf_config
        num_heads = hf.num_attention_heads
        num_kv_heads = getattr(hf, "num_key_value_heads", num_heads)
        head_dim = hf.hidden_size // num_heads
        if hasattr(hf, "head_dim") and hf.head_dim is not None:
            head_dim = hf.head_dim
        heads_per_group = num_heads // num_kv_heads

        out = {}
        for name, tensor in sd.items():
            if ".qkv_proj.weight" in name:
                base = name.replace("qkv_proj.weight", "")
                g = tensor.view(num_kv_heads, heads_per_group + 2, head_dim, -1)
                out[base + "q_proj.weight"] = g[:, :heads_per_group].reshape(num_heads * head_dim, -1)
                out[base + "k_proj.weight"] = g[:, heads_per_group:heads_per_group+1].reshape(num_kv_heads * head_dim, -1)
                out[base + "v_proj.weight"] = g[:, heads_per_group+1:].reshape(num_kv_heads * head_dim, -1)
            elif ".qkv_proj.bias" in name:
                base = name.replace("qkv_proj.bias", "")
                g = tensor.view(num_kv_heads, heads_per_group + 2, head_dim)
                out[base + "q_proj.bias"] = g[:, :heads_per_group].reshape(-1)
                out[base + "k_proj.bias"] = g[:, heads_per_group:heads_per_group+1].reshape(-1)
                out[base + "v_proj.bias"] = g[:, heads_per_group+1:].reshape(-1)
            elif ".gate_up_proj.weight" in name:
                base = name.replace("gate_up_proj.weight", "")
                half = tensor.size(0) // 2
                out[base + "gate_proj.weight"] = tensor[:half]
                out[base + "up_proj.weight"] = tensor[half:]
            elif name == "lm_head.weight" and not include_lm_head:
                continue  # vLLM ties lm_head to embed_tokens
            else:
                out[name] = tensor
        if (
            include_lm_head
            and "lm_head.weight" not in out
            and getattr(self._hf_config, "tie_word_embeddings", False)
            and "model.embed_tokens.weight" in out
        ):
            out["lm_head.weight"] = out["model.embed_tokens.weight"]
        return out

    def _get_hf_state_dict_for_cpu_sync(self):
        """Return a true HuggingFace-format state dict for HF rollout sync.

        ``_get_clean_state_dict`` intentionally returns the fused parameter
        names used by vLLM's direct-copy weight transfer path
        (``qkv_proj``/``gate_up_proj``).  The HF rollout backend uses
        ``model.load_state_dict`` over a multiprocessing queue, so it needs
        standard HF checkpoint names instead (``q_proj``/``k_proj``/``v_proj``
        and ``gate_proj``/``up_proj``), including ``lm_head.weight``.
        """
        fused = self._gather_hf_state_dict()
        return self._defuse_gathered_state_dict(fused, include_lm_head=True)


    # ================================================================ #
    #  Command handlers (Megatron-specific)                              #
    # ================================================================ #

    def _handle_get_clean_state_dict(self, cmd, t_recv, result_queue):
        """Return HF-format weights for CPU queue sync to HF rollout workers."""
        sd = self._get_hf_state_dict_for_cpu_sync()
        if self.rank == 0:
            sd_cpu = {k: v.cpu() for k, v in sd.items()}
            result_queue.put({"state_dict": sd_cpu})

    def _handle_compute_weight_checksum(self, cmd, t_recv, result_queue):
        merge_map = cmd[1] if self.rank == 0 else None
        # All ranks must participate (PP broadcast is a collective)
        sd = self._get_clean_state_dict()
        if self.rank == 0:
            checksum = 0.0
            phi = 1.6180339887  # must match rollout compute_checksum_fn
            for i, (_, sources) in enumerate(merge_map):
                weight = phi ** (i % 32)
                if len(sources) == 1:
                    checksum += sd[sources[0]].float().abs().sum().item() * weight
                else:
                    merged = torch.cat([sd[s] for s in sources], dim=0)
                    checksum += merged.float().abs().sum().item() * weight
            result_queue.put({"checksum": checksum})

    def _handle_compute_lora_checksum(self, cmd, t_recv, result_queue):
        """Compute order-sensitive checksum over LoRA adapter params only.

        All ranks participate in the TP/PP gather. Only rank 0 computes
        and returns the checksum. Uses sorted names to match rollout ordering.
        """
        if not self._is_lora:
            if self.rank == 0:
                result_queue.put({"checksum": 0.0})
            return
        # Reuse _get_state_dict_for_sync which does TP+PP gather for LoRA
        sd = self._get_state_dict_for_sync()
        if self.rank == 0:
            sorted_items = sorted(sd.items())
            checksum = 0.0
            phi = 1.6180339887
            for i, (_, tensor) in enumerate(sorted_items):
                weight = phi ** (i % 32)
                checksum += tensor.float().abs().sum().item() * weight
            result_queue.put({"checksum": checksum})

    # ================================================================ #
    #  Training step                                                     #
    # ================================================================ #

    def _train_step(self, batch):
        """One training step using Megatron's forward/backward schedule.

        Dispatches to the appropriate loss mode:
        - "kl": KL distillation (default OPD pipeline)
        - "sft": Supervised fine-tuning (CE/KL/mixed)
        - "grpo": Group Relative Policy Optimization
        """
        return self._native_megatron_train_step(batch)

    def _run_train_step(self, batch, loss_fn):
        """Megatron training loop with pluggable loss function.

        Uses Megatron's forward_backward_func to handle PP/TP.
        The loss_fn is called inside forward_step.
        """
        from megatron.core.pipeline_parallel import get_forward_backward_func

        model = self.model
        optimizer = self.optimizer
        device = self.device
        micro_batch_size = self.micro_batch_size
        scheduler = self.scheduler

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bs = input_ids.size(0)

        # Mini-batch splitting
        if self.mini_batch_size > 0 and self.mini_batch_size < bs:
            n_mini = bs // self.mini_batch_size
            mini_bs = self.mini_batch_size
        else:
            n_mini = 1
            mini_bs = bs

        total_loss = 0.0
        total_tokens = 0
        all_grad_norms = []
        extras_accum = {}
        n_optim_steps = 0

        forward_backward_func = get_forward_backward_func()

        for mini_idx in range(n_mini):
            ms = mini_idx * mini_bs
            me = ms + mini_bs

            assert mini_bs % micro_batch_size == 0
            n_micro = max(1, mini_bs // micro_batch_size)
            actual_mbs = mini_bs // n_micro

            optimizer.zero_grad()

            # Build micro-batch iterator for Megatron
            micro_batches = []
            for mb_idx in range(n_micro):
                s = mb_idx * actual_mbs
                e = s + actual_mbs
                mb = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.size(0) == bs:
                        mb[k] = v[ms + s:ms + e].to(device)
                    else:
                        mb[k] = v
                micro_batches.append(mb)

            # Nonlocal for capturing loss from forward_step
            step_losses = []
            step_tokens = []
            step_extras = []

            def forward_step(data_iterator, model):
                mb = next(data_iterator)
                tokens = mb["input_ids"]
                mask = mb["attention_mask"]
                out = model(tokens, None, mask)
                logits = out if isinstance(out, torch.Tensor) else out[0]

                loss, n_tok, extras = loss_fn(logits, mb)

                step_losses.append(loss.detach().item())
                step_tokens.append(n_tok)
                step_extras.append(extras)

                # Return loss for backward. Megatron needs num_tokens for averaging.
                return loss, lambda x: torch.Tensor([n_tok]).to(device)

            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=iter(micro_batches),
                model=[model],
                num_microbatches=n_micro,
                seq_length=input_ids.size(1),
                micro_batch_size=actual_mbs,
                forward_only=False,
            )

            # Optimizer step
            update_successful, grad_norm, num_zeros = optimizer.step()
            n_optim_steps += 1
            if self.rank == 0 and n_mini > 1:
                print(f"  [train] optim step {n_optim_steps}/{n_mini}", flush=True)

            if grad_norm is not None:
                all_grad_norms.append(float(grad_norm))

            for sl, st, se in zip(step_losses, step_tokens, step_extras):
                total_loss += sl * st if st > 0 else sl
                total_tokens += st
                for ek, ev in se.items():
                    extras_accum[ek] = extras_accum.get(ek, 0.0) + ev

        # Update LR via dummy scheduler
        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            # Copy LR to Megatron optimizer
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        total_steps = n_mini * max(1, mini_bs // micro_batch_size)
        result = {
            "kl_loss": total_loss / max(total_tokens, 1),
            "n_tokens": total_tokens,
            "grad_norm": max(all_grad_norms) if all_grad_norms else 0.0,
        }
        if n_mini > 1:
            result["n_optim_steps"] = n_optim_steps
        if scheduler is not None:
            result["lr"] = lr
        for ek, ev in extras_accum.items():
            result[ek] = ev / max(total_steps, 1)
        return result

    def _native_megatron_train_step(self, batch):
        """Training step using Megatron's native forward/backward schedule.

        Uses ``get_forward_backward_func()`` which returns:
        - ``forward_backward_no_pipelining`` for PP=1
        - ``forward_backward_pipelining_without_interleaving`` for PP>1

        Supports all 3D parallelism modes (TP/PP/DP) and all loss modes:
        - "kl": KL distillation (teacher logprobs required)
        - "sft": Supervised fine-tuning (CE/KL/mixed via sft_loss_mode)
        - "grpo": Group Relative Policy Optimization (PPO-clip + KL penalty)
        """
        from megatron.core.pipeline_parallel import get_forward_backward_func

        model = self.model
        optimizer = self.optimizer
        device = self.device
        micro_batch_size = self.micro_batch_size
        scheduler = self.scheduler
        loss_mode = self.loss_mode

        # ------------------------------------------------------------------
        # Batch preparation — mode-specific
        # ------------------------------------------------------------------
        if loss_mode == "grpo":
            # GRPO uses its own batch format (no teacher logprobs)
            prep = self._prepare_grpo_batch_megatron(batch)
        elif loss_mode == "sft":
            # SFT: teacher data optional (CE mode doesn't need it)
            prep = self._prepare_sft_batch_megatron(batch)
        else:
            # KL mode: standard OPD batch with teacher logprobs
            prep = self._prepare_train_batch(batch)

        input_ids = prep["input_ids"]
        attention_mask = prep["attention_mask"]
        response_mask = prep["response_mask"]
        prompt_lengths = prep["prompt_lengths"]
        n_mini = prep["n_mini"]
        mini_bs = prep["mini_bs"]
        actual_max_len = prep["actual_max_len"]

        # Mode-specific tensors
        teacher_topk_logps = prep.get("teacher_topk_logps")
        teacher_topk_indices = prep.get("teacher_topk_indices")
        teacher_valid_mask = prep.get("teacher_valid_mask")
        raw_kl_batch = prep.get("batch", {})
        teacher_token_logps = raw_kl_batch.get("teacher_token_logps")
        kl_student_old_logprobs = raw_kl_batch.get("student_logprobs")
        student_old_logprobs = prep.get("student_old_logprobs")
        advantages = prep.get("advantages")
        ref_token_logps = prep.get("ref_token_logps")

        total_loss = 0.0
        total_tokens = 0
        all_grad_norms = []
        n_optim_steps = 0
        stat_accum = {}
        per_mini_n_tokens = {}
        per_mini_avg_response_length = {}
        per_mini_p90_response_length = {}
        per_mini_n_seqs = {}

        forward_backward_func = get_forward_backward_func()

        for mini_idx in range(n_mini):
            ms = mini_idx * mini_bs
            me = ms + mini_bs

            mi_input_ids = input_ids[ms:me]
            mi_attention_mask = attention_mask[ms:me]
            mi_response_mask = response_mask[ms:me]

            if n_mini > 1:
                per_seq_resp = mi_response_mask.sum(dim=1).float()
                per_mini_n_tokens[mini_idx] = int(per_seq_resp.sum().item())
                per_mini_n_seqs[mini_idx] = int(mi_response_mask.size(0))
                if per_seq_resp.numel() > 0:
                    per_mini_avg_response_length[mini_idx] = (
                        per_seq_resp.mean().item()
                    )
                    per_mini_p90_response_length[mini_idx] = (
                        per_seq_resp.quantile(0.9).item()
                        if per_seq_resp.numel() > 1
                        else per_seq_resp.item()
                    )

            assert mini_bs % micro_batch_size == 0
            n_micro = max(1, mini_bs // micro_batch_size)
            actual_mbs = mini_bs // n_micro

            optimizer.zero_grad()

            # Prepare micro-batch iterator for Megatron's schedule
            micro_batches = []
            for mb_idx in range(n_micro):
                s = mb_idx * actual_mbs
                e = s + actual_mbs
                mb_dict = {
                    "input_ids": mi_input_ids[s:e].to(device),
                    "attention_mask": mi_attention_mask[s:e].to(device),
                    "response_mask": mi_response_mask[s:e].to(device),
                }
                # KL mode: add teacher data
                if loss_mode == "kl" or (loss_mode == "sft" and teacher_topk_logps is not None):
                    mb_dict["teacher_logps"] = teacher_topk_logps[ms:me][s:e].to(device)
                    mb_dict["teacher_idx"] = teacher_topk_indices[ms:me][s:e].to(device)
                    if teacher_valid_mask is not None:
                        mb_dict["teacher_valid_mask"] = teacher_valid_mask[ms:me][s:e].to(device)
                    if teacher_token_logps is not None:
                        mb_dict["teacher_token_logps"] = teacher_token_logps[ms:me][s:e].to(device)
                    if kl_student_old_logprobs is not None:
                        mb_dict["student_old_logprobs"] = (
                            kl_student_old_logprobs[ms:me][s:e].to(device)
                        )
                # GRPO mode: add GRPO-specific tensors
                if loss_mode == "grpo":
                    mb_dict["student_old_logprobs"] = student_old_logprobs[ms:me][s:e].to(device)
                    mb_dict["advantages"] = advantages[ms:me][s:e].to(device)
                    if ref_token_logps is not None:
                        mb_dict["ref_token_logps"] = ref_token_logps[ms:me][s:e].to(device)

                # Sequence packing (KL mode only for now — SFT/GRPO packing
                # would need additional pack functions for their batch formats)
                if self.use_sequence_packing and loss_mode == "kl":
                    from opd.data.packing import pack_micro_batch
                    from megatron.core.packed_seq_params import PackedSeqParams
                    packed = pack_micro_batch(
                        input_ids=mb_dict["input_ids"],
                        attention_mask=mb_dict["attention_mask"],
                        teacher_topk_logps=mb_dict["teacher_logps"],
                        teacher_topk_indices=mb_dict["teacher_idx"],
                        response_mask=mb_dict["response_mask"],
                        prompt_lengths=prompt_lengths[ms+s:ms+e].to(device),
                    )
                    packed_seq_params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=packed.cu_seq_lens,
                        cu_seqlens_kv=packed.cu_seq_lens,
                        cu_seqlens_q_padded=packed.cu_seq_lens,
                        cu_seqlens_kv_padded=packed.cu_seq_lens,
                        max_seqlen_q=packed.max_seq_len,
                        max_seqlen_kv=packed.max_seq_len,
                    )
                    mb_dict = {
                        "input_ids": packed.input_ids,
                        "position_ids": packed.position_ids,
                        "packed_seq_params": packed_seq_params,
                        "teacher_logps": packed.teacher_topk_logps,
                        "teacher_idx": packed.teacher_topk_indices,
                        "response_mask": packed.response_mask,
                    }

                micro_batches.append(mb_dict)

            # Accumulator for per-microbatch stats
            mb_losses = []
            mb_tokens = []

            def forward_step(data_iterator, model):
                """Megatron forward_step callback.

                Returns (output_tensor, loss_func) where output_tensor is the
                model logits and loss_func computes loss based on self.loss_mode.
                With PP, loss_func is only called on the last stage.
                """
                mb = next(data_iterator)
                mb_ids = mb["input_ids"]
                mb_resp_mask = mb["response_mask"]

                # Packed mode: position_ids and packed_seq_params from packing
                # Non-packed: compute position_ids from sequence length
                if "packed_seq_params" in mb:
                    position_ids = mb["position_ids"]
                    packed_seq_params = mb["packed_seq_params"]
                    attention_mask = None
                else:
                    mb_attention_mask = mb["attention_mask"]
                    position_ids = self._position_ids_from_attention_mask(
                        mb_attention_mask)
                    attention_mask = self._causal_padding_attention_mask(
                        mb_attention_mask)
                    packed_seq_params = None

                # GPTModel returns logits [B, S, V] on last PP stage,
                # hidden states on intermediate stages
                output_tensor = model(
                    input_ids=mb_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )

                def loss_func(logits):
                    """Compute loss based on loss_mode.

                    Only called on the last PP stage (post_process=True).
                    """
                    if loss_mode == "sft":
                        return self._megatron_sft_loss(
                            logits, mb, mb_resp_mask, mb_losses, mb_tokens)
                    elif loss_mode == "grpo":
                        return self._megatron_grpo_loss(
                            logits, mb, mb_resp_mask, mb_losses, mb_tokens,
                            stat_accum)
                    else:
                        return self._megatron_kl_loss(
                            logits, mb, mb_resp_mask, mb_losses, mb_tokens,
                            stat_accum)

                return output_tensor, loss_func

            # Zero grad buffers if model is Megatron DDP
            if hasattr(model, 'zero_grad_buffer'):
                model.zero_grad_buffer()

            # For packed sequences: batch_size=1, seq_length=total_tokens
            if self.use_sequence_packing and loss_mode == "kl":
                packed_T = micro_batches[0]["input_ids"].size(1)
                fb_seq_len = packed_T
                fb_mbs = 1
            else:
                fb_seq_len = actual_max_len
                fb_mbs = actual_mbs

            forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=iter(micro_batches),
                model=[model],
                num_microbatches=n_micro,
                seq_length=fb_seq_len,
                micro_batch_size=fb_mbs,
                forward_only=False,
            )

            # Megatron optimizer handles grad clipping, fp32 master
            # weight update, and DP gradient all-reduce internally.
            update_ok, grad_norm, _ = optimizer.step()
            if not update_ok:
                print(f"[Backend-Megatron] WARNING: optimizer step skipped "
                      f"(grad overflow?)", flush=True)
            n_optim_steps += 1
            if self.rank == 0 and n_mini > 1:
                print(f"  [train] optim step {n_optim_steps}/{n_mini}", flush=True)
            all_grad_norms.append(
                grad_norm.item() if hasattr(grad_norm, 'item')
                else float(grad_norm)
            )
            if scheduler is not None:
                # Step the dummy scheduler, then copy LR to Megatron optimizer
                scheduler.step()
                new_lr = self._dummy_scheduler_opt.param_groups[0]["lr"]
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr

            # With PP, loss_func only runs on the last stage. Broadcast
            # loss stats from last PP stage to first (rank 0 reports metrics).
            if self.pp_size > 1 and not mb_losses:
                from megatron.core import parallel_state as mpu
                pp_group = mpu.get_pipeline_model_parallel_group()
                last_pp_rank = mpu.get_pipeline_model_parallel_world_size() - 1
                pp_src = torch.distributed.get_global_rank(pp_group, last_pp_rank)
                loss_data = [None]
                torch.distributed.broadcast_object_list(loss_data, src=pp_src, group=pp_group)
                if loss_data[0] is not None:
                    mb_losses = loss_data[0]["losses"]
                    mb_tokens = loss_data[0]["tokens"]
            elif self.pp_size > 1 and mb_losses:
                # Last PP stage: send loss stats to first
                from megatron.core import parallel_state as mpu
                pp_group = mpu.get_pipeline_model_parallel_group()
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                pp_src = torch.distributed.get_global_rank(pp_group, pp_rank)
                loss_data = [{"losses": mb_losses, "tokens": mb_tokens}]
                torch.distributed.broadcast_object_list(loss_data, src=pp_src, group=pp_group)

            total_loss += sum(mb_losses)
            total_tokens += sum(mb_tokens)

        result = {
            "kl_loss": total_loss / max(total_tokens, 1),
            "n_tokens": total_tokens,
            "grad_norm": max(all_grad_norms) if all_grad_norms else 0.0,
        }
        if n_mini > 1 or self.loss_mode == "grpo":
            result["n_optim_steps"] = n_optim_steps
        if scheduler is not None:
            result["lr"] = scheduler.get_last_lr()[0]
        for k, v in stat_accum.items():
            total_steps = n_mini * n_micro if n_mini > 0 else 1
            result[k] = v / max(total_steps, 1)
        for mi, n_tok in per_mini_n_tokens.items():
            result[f"n_tokens_mini_{mi}"] = n_tok
        for mi, avg_len in per_mini_avg_response_length.items():
            result[f"avg_response_length_mini_{mi}"] = avg_len
        for mi, p90_len in per_mini_p90_response_length.items():
            result[f"response_length_p90_mini_{mi}"] = p90_len
        for mi, n_seq in per_mini_n_seqs.items():
            result[f"n_seqs_mini_{mi}"] = n_seq
        return result

    # ================================================================ #
    #  Loss helpers for forward_step dispatch                            #
    # ================================================================ #

    def _megatron_kl_loss(self, logits, mb, mb_resp_mask, mb_losses, mb_tokens,
                          stat_accum=None):
        """KL distillation loss (existing OPD behavior)."""
        mb_t_logps = mb["teacher_logps"]
        mb_t_idx = mb["teacher_idx"]
        kl_kwargs = {
            "student_logits": logits,
            "teacher_topk_logps": mb_t_logps,
            "teacher_topk_indices": mb_t_idx,
            "mask": mb_resp_mask,
            "kl_config": self.kl_config,
        }
        if "teacher_token_logps" in mb:
            kl_kwargs["teacher_token_logps"] = mb["teacher_token_logps"]
            kl_kwargs["input_ids"] = mb["input_ids"]
        if "student_old_logprobs" in mb:
            kl_kwargs["student_old_logprobs"] = mb["student_old_logprobs"]

        if self.tp_size > 1:
            from opd.loss.megatron_kl import vocab_parallel_compute_kl_loss
            loss = vocab_parallel_compute_kl_loss(**kl_kwargs)
        else:
            loss = compute_kl_loss(**kl_kwargs)
        n_tok = mb_resp_mask.sum().item()
        mb_losses.append(loss.detach().item() * n_tok)
        mb_tokens.append(n_tok)
        if stat_accum is not None and hasattr(loss, "pg_stats"):
            stats = loss.pg_stats
            ratios = stats.get("_ratios")
            if ratios is not None and ratios.numel() > 0:
                stat_accum["r_mean"] = stat_accum.get("r_mean", 0.0) + ratios.mean().item()
                stat_accum["r_std"] = stat_accum.get("r_std", 0.0) + (
                    ratios.std().item() if ratios.numel() > 1 else 0.0
                )
            log_ratios = stats.get("_log_ratios")
            if log_ratios is not None and log_ratios.numel() > 0:
                stat_accum["logr_mean"] = (
                    stat_accum.get("logr_mean", 0.0) + log_ratios.mean().item()
                )
            clip_high = stats.get("_clip_high")
            if clip_high is not None and clip_high.numel() > 0:
                stat_accum["clip_frac_high"] = (
                    stat_accum.get("clip_frac_high", 0.0)
                    + clip_high.float().mean().item()
                )
            clip_low = stats.get("_clip_low")
            if clip_low is not None and clip_low.numel() > 0:
                stat_accum["clip_frac_low"] = (
                    stat_accum.get("clip_frac_low", 0.0)
                    + clip_low.float().mean().item()
                )
            advantages = stats.get("_advantages")
            if advantages is not None and advantages.numel() > 0:
                stat_accum["adv_mean"] = (
                    stat_accum.get("adv_mean", 0.0) + advantages.mean().item()
                )
        return loss, {"kl_loss": loss.detach().item()}

    def _megatron_sft_loss(self, logits, mb, mb_resp_mask, mb_losses, mb_tokens):
        """SFT cross-entropy / KL / mixed loss.

        For TP>1 CE mode, uses Megatron's vocab_parallel_cross_entropy
        which handles distributed softmax across TP-sharded logits.
        """
        mb_ids = mb["input_ids"]
        cfg = self.sft_config

        if cfg.loss_mode == "ce":
            # Pure CE — vocab-parallel for TP>1, standard for TP=1
            if self.tp_size > 1:
                loss, n_tok = self._vocab_parallel_sft_loss(logits, mb_ids, mb_resp_mask)
            else:
                from opd.loss.sft import sft_loss
                loss, n_tok = sft_loss(logits, mb_ids, mb_resp_mask)
        elif cfg.loss_mode == "kl":
            mb_t_logps = mb["teacher_logps"]
            mb_t_idx = mb["teacher_idx"]
            if self.tp_size > 1:
                from opd.loss.megatron_kl import vocab_parallel_compute_kl_loss
                loss = vocab_parallel_compute_kl_loss(
                    student_logits=logits,
                    teacher_topk_logps=mb_t_logps,
                    teacher_topk_indices=mb_t_idx,
                    mask=mb_resp_mask,
                    kl_config=self.kl_config,
                )
                n_tok = int(mb_resp_mask.sum().item())
            else:
                from opd.loss.sft import compute_sft_loss
                mb_t_valid = mb.get("teacher_valid_mask", mb_resp_mask)
                (loss, n_tok), _ = compute_sft_loss(
                    logits, mb_ids, mb_resp_mask,
                    mb_t_logps, mb_t_idx, mb_t_valid,
                    sft_loss_mode=cfg.loss_mode, ce_alpha=cfg.ce_alpha,
                    n_kl_logprobs=cfg.n_kl_logprobs, kl_config=self.kl_config)
        else:
            # Mixed: alpha * CE + (1-alpha) * KL
            mb_t_logps = mb["teacher_logps"]
            mb_t_idx = mb["teacher_idx"]
            mb_t_valid = mb.get("teacher_valid_mask", mb_resp_mask)
            if self.tp_size > 1:
                ce_loss, n_tok = self._vocab_parallel_sft_loss(logits, mb_ids, mb_resp_mask)
                from opd.loss.megatron_kl import vocab_parallel_compute_kl_loss
                kl_loss = vocab_parallel_compute_kl_loss(
                    student_logits=logits,
                    teacher_topk_logps=mb_t_logps,
                    teacher_topk_indices=mb_t_idx,
                    mask=mb_resp_mask,
                    kl_config=self.kl_config,
                )
                loss = cfg.ce_alpha * ce_loss + (1 - cfg.ce_alpha) * kl_loss
            else:
                from opd.loss.sft import compute_sft_loss
                (loss, n_tok), _ = compute_sft_loss(
                    logits, mb_ids, mb_resp_mask,
                    mb_t_logps, mb_t_idx, mb_t_valid,
                    sft_loss_mode=cfg.loss_mode, ce_alpha=cfg.ce_alpha,
                    n_kl_logprobs=cfg.n_kl_logprobs, kl_config=self.kl_config)

        mb_losses.append(loss.detach().item() * n_tok)
        mb_tokens.append(n_tok)
        return loss, {"sft_loss": loss.detach().item()}

    def _vocab_parallel_sft_loss(self, logits, input_ids, response_mask):
        """Vocab-parallel cross-entropy for TP>1 SFT.

        Uses Megatron's vocab_parallel_cross_entropy which handles distributed
        softmax across TP-sharded logits [B, S, V/TP].

        Args:
            logits: [B, S, V/TP] TP-sharded logits
            input_ids: [B, S] token IDs
            response_mask: [B, S] bool mask for response tokens

        Returns:
            (loss, n_tokens): scalar loss tensor and token count
        """
        from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

        # Causal LM shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V/TP]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]
        shift_mask = response_mask[:, 1:].float()       # [B, T-1]

        # vocab_parallel_cross_entropy expects [S, B, V/TP] and [S, B]
        per_token_loss = vocab_parallel_cross_entropy(
            shift_logits.transpose(0, 1).contiguous(),
            shift_labels.transpose(0, 1).contiguous(),
        )  # [S, B]
        per_token_loss = per_token_loss.transpose(0, 1)  # [B, T-1]

        masked_loss = per_token_loss * shift_mask
        n_tokens = int(shift_mask.sum().item())
        denom = shift_mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / denom

        return loss, n_tokens

    def _megatron_grpo_loss(self, logits, mb, mb_resp_mask, mb_losses, mb_tokens,
                            stat_accum):
        """GRPO PPO-clip + optional KL penalty loss."""
        from opd.loss.grpo import grpo_clip_loss

        cfg = self.grpo_config
        mb_ids = mb["input_ids"]
        mb_old_lp = mb["student_old_logprobs"]
        mb_adv = mb["advantages"]
        mb_ref = mb.get("ref_token_logps")

        loss, stats = grpo_clip_loss(
            logits, mb_ids, mb_old_lp, mb_adv, mb_resp_mask,
            clip_eps=cfg.clip_eps,
            clip_ratio_low=cfg.clip_ratio_low,
            clip_ratio_high=cfg.clip_ratio_high,
            clip_ratio_c=cfg.clip_ratio_c,
            ref_token_logps=mb_ref,
            kl_beta=cfg.kl_beta,
            kl_type=cfg.kl_type,
            loss_agg_mode=cfg.loss_agg_mode,
        )
        n_tok = int(mb_resp_mask[:, 1:].sum().item())
        mb_losses.append(loss.detach().item() * n_tok)
        mb_tokens.append(n_tok)
        for k, v in stats.items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v):
                if v.numel() != 1:
                    continue
                v = v.item()
            if isinstance(v, (int, float)):
                stat_accum[k] = stat_accum.get(k, 0.0) + float(v)
        return loss, {"grpo_loss": loss.detach().item()}

    # ================================================================ #
    #  Batch preparation helpers (SFT, GRPO)                             #
    # ================================================================ #

    def _prepare_sft_batch_megatron(self, batch):
        """Prepare an SFT batch for Megatron training.

        Similar to _prepare_train_batch but teacher data is optional.
        """
        rank = getattr(self, 'dp_rank', self.rank)
        world_size = getattr(self, 'dp_world_size', self.world_size)
        mini_batch_size = self.mini_batch_size
        max_response_length = self.max_response_length

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lengths = batch["prompt_lengths"]
        needs_teacher = self.sft_config.loss_mode in ("kl", "mixed")

        # Optional teacher tensors
        teacher_topk_logps = batch.get("teacher_topk_logps")
        teacher_topk_indices = batch.get("teacher_topk_indices")
        teacher_valid_mask = batch.get("teacher_valid_mask")

        bs = input_ids.size(0)

        if world_size > 1:
            per_rank = bs // world_size
            rank_start = rank * per_rank
            rank_end = rank_start + per_rank
            input_ids = input_ids[rank_start:rank_end]
            attention_mask = attention_mask[rank_start:rank_end]
            prompt_lengths = prompt_lengths[rank_start:rank_end]
            if needs_teacher and teacher_topk_logps is not None:
                teacher_topk_logps = teacher_topk_logps[rank_start:rank_end]
                teacher_topk_indices = teacher_topk_indices[rank_start:rank_end]
                teacher_valid_mask = teacher_valid_mask[rank_start:rank_end]
            bs = per_rank

        if mini_batch_size > 0 and mini_batch_size < bs:
            n_mini = bs // mini_batch_size
            mini_bs = mini_batch_size
        else:
            n_mini = 1
            mini_bs = bs

        seq_len = input_ids.size(1)
        orig_seq_len = batch.get("_orig_seq_len", seq_len)
        max_prompt = orig_seq_len - max_response_length
        response_mask = attention_mask.clone().bool()
        response_mask[:, :max_prompt] = False

        # Truncate trailing padding
        nonzero_cols = attention_mask.nonzero(as_tuple=True)[1]
        actual_max_len = int(nonzero_cols.max().item()) + 2 if nonzero_cols.numel() > 0 else seq_len
        actual_max_len = min(actual_max_len, seq_len)
        if actual_max_len < seq_len:
            input_ids = input_ids[:, :actual_max_len]
            attention_mask = attention_mask[:, :actual_max_len]
            response_mask = response_mask[:, :actual_max_len]
            if needs_teacher and teacher_topk_logps is not None:
                teacher_topk_logps = teacher_topk_logps[:, :actual_max_len]
                teacher_topk_indices = teacher_topk_indices[:, :actual_max_len]
                teacher_valid_mask = teacher_valid_mask[:, :actual_max_len]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "prompt_lengths": prompt_lengths,
            "n_mini": n_mini,
            "mini_bs": mini_bs,
            "seq_len": seq_len,
            "actual_max_len": actual_max_len,
            "max_prompt": max_prompt,
        }
        if needs_teacher and teacher_topk_logps is not None:
            result["teacher_topk_logps"] = teacher_topk_logps
            result["teacher_topk_indices"] = teacher_topk_indices
            result["teacher_valid_mask"] = teacher_valid_mask
        return result

    def _prepare_grpo_batch_megatron(self, batch):
        """Prepare a GRPO batch for Megatron training.

        Handles rank-splitting, mini-batch splitting, and truncation
        for GRPO-specific batch format.
        """
        rank = getattr(self, 'dp_rank', self.rank)
        world_size = getattr(self, 'dp_world_size', self.world_size)
        mini_batch_size = self.mini_batch_size

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        prompt_lengths = batch["prompt_lengths"]
        student_old_logprobs = batch["student_old_logprobs"]
        advantages = batch["advantages"]
        ref_token_logps = batch.get("ref_token_logps")

        bs = input_ids.size(0)

        if mini_batch_size > 0 and mini_batch_size < bs:
            n_mini = bs // mini_batch_size
            mini_bs = mini_batch_size
        else:
            n_mini = 1
            mini_bs = bs

        # Note: Megatron DP rank-splitting is handled by _prepare_train_batch
        # pattern (dp_rank/dp_world_size), not explicit slicing here.

        # Truncate trailing padding
        nonzero_cols = attention_mask.nonzero(as_tuple=True)[1]
        seq_len = input_ids.size(1)
        actual_max_len = int(nonzero_cols.max().item()) + 2 if nonzero_cols.numel() > 0 else seq_len
        actual_max_len = min(actual_max_len, seq_len)
        if actual_max_len < seq_len:
            input_ids = input_ids[:, :actual_max_len]
            attention_mask = attention_mask[:, :actual_max_len]
            response_mask = response_mask[:, :actual_max_len]
            if ref_token_logps is not None:
                ref_token_logps = ref_token_logps[:, :actual_max_len]
            max_prompt = int(prompt_lengths.max().item())
            actual_resp_len = actual_max_len - max_prompt
            if actual_resp_len < student_old_logprobs.size(1):
                student_old_logprobs = student_old_logprobs[:, :actual_resp_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "prompt_lengths": prompt_lengths,
            "student_old_logprobs": student_old_logprobs,
            "advantages": advantages,
            "ref_token_logps": ref_token_logps,
            "n_mini": n_mini,
            "mini_bs": mini_bs,
            "seq_len": seq_len,
            "actual_max_len": actual_max_len,
        }

    # ================================================================ #
    #  Checkpoint helpers                                                #
    # ================================================================ #

    def _save_checkpoint(self, checkpoint_dir, step, state_dict=None,
                         save_optimizer=True, optim_state_dict=None):
        """Save model weights and optionally optimizer/scheduler state.

        For native Megatron models, converts to HF-format names before saving
        so that checkpoints are compatible with vLLM and AutoModelForCausalLM.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self._mcore_to_hf_state_dict()
            # With PP, each rank only has its stage's layers.
            # Gather from all PP stages within PP group (DP-safe).
            if self.pp_size > 1:
                from megatron.core import parallel_state as _mpu
                pp_group = _mpu.get_pipeline_model_parallel_group()
                pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                full_sd = {}
                for pp_r in range(self.pp_size):
                    src_global = pp_group_ranks[pp_r]
                    stage_data = [state_dict if self.rank == src_global else None]
                    torch.distributed.broadcast_object_list(stage_data, src=src_global, group=pp_group)
                    if self.rank == 0 and stage_data[0]:
                        full_sd.update(stage_data[0])
                state_dict = full_sd if self.rank == 0 else {}

        train_state = None
        if save_optimizer:
            train_state = {"step": step}
            train_state["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                train_state["scheduler"] = self.scheduler.state_dict()

        # Disk I/O — run in background thread via base class helper
        self._async_write_checkpoint(state_dict, train_state, checkpoint_dir, step)

    def _load_checkpoint(self, checkpoint_dir):
        """Load model weights and optimizer/scheduler state.

        For native Megatron models, the checkpoint is in HF format (saved
        by ``_save_checkpoint``).  We re-load via ``load_hf_weights_to_mcore``
        which handles the HF-to-Megatron name conversion (QKV interleaving,
        gate+up fusion, etc.).

        Returns the step number from the checkpoint, or 0 if not found.
        """
        self._wait_async_save()
        model_path = os.path.join(checkpoint_dir, "model.pt")
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        step = 0

        if os.path.exists(model_path):
            self._load_hf_checkpoint_to_mcore(model_path)
            print(f"[Backend-Megatron] Loaded model from {model_path}",
                  flush=True)

        if os.path.exists(state_path):
            ts = torch.load(
                state_path, map_location="cpu", weights_only=False,
            )
            step = ts.get("step", 0)
            if "optimizer" in ts and self.optimizer is not None:
                self.optimizer.load_state_dict(ts["optimizer"])
            if ("scheduler" in ts and self.scheduler is not None
                    and self.optimizer is not None):
                self.scheduler.load_state_dict(ts["scheduler"])
            print(f"[Backend-Megatron] Loaded training state from "
                  f"{state_path} (step={step})", flush=True)

        return step

    # ================================================================ #
    #  Weight helpers                                                    #
    # ================================================================ #

    def _build_weights_info(self):
        """Build (name, shape, dtype) list matching _get_clean_state_dict() order.

        Uses actual model metadata from ``build_megatron_weights_info``
        with fused names (qkv_proj, gate_up_proj) matching both the
        gathered state dict and vLLM's model.named_parameters().

        For PP>1, gathers metadata from all PP stages and remaps local→global
        layer indices.  This ensures weights_info and state_dict have identical
        names and ordering — fixing the PP checksum mismatch caused by the
        old hardcoded approach which had a different per-layer ordering.
        """
        import re

        # LoRA mode: return LoRA param info (tiny, ~20M params)
        if self._is_lora:
            from opd.trainer.megatron.lora import build_lora_weights_info
            num_layers = self._hf_config.num_hidden_layers
            layers_per_stage = num_layers // self.pp_size
            local_info = build_lora_weights_info(
                self._lora_params, tp_size=self.tp_size,
                pp_size=self.pp_size, pp_rank=self.pp_rank,
                layers_per_stage=layers_per_stage,
            )
            if self.pp_size <= 1:
                return local_info
            # PP>1: gather LoRA info from all stages within PP group (DP-safe)
            from megatron.core import parallel_state as _mpu
            pp_group = _mpu.get_pipeline_model_parallel_group()
            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
            full_info = []
            for pp_r in range(self.pp_size):
                src_global = pp_group_ranks[pp_r]
                stage_data = [local_info if self.rank == src_global else None]
                torch.distributed.broadcast_object_list(stage_data, src=src_global, group=pp_group)
                if stage_data[0] is not None:
                    full_info.extend(stage_data[0])
            return full_info

        from opd.trainer.megatron.weight_gather import build_megatron_weights_info

        raw_info = build_megatron_weights_info(self._unwrapped_model, self.tp_size)
        info = []
        for name, shape, dtype_str in raw_info:
            dtype = getattr(torch, dtype_str, torch.bfloat16)
            info.append((name, torch.Size(shape), dtype))

        if self.pp_size <= 1:
            return info

        # PP>1: remap local → global layer indices
        num_layers = self._hf_config.num_hidden_layers
        layers_per_stage = num_layers // self.pp_size
        layer_offset = self.pp_rank * layers_per_stage

        remapped = []
        for name, shape, dtype in info:
            m = re.match(r"^model\.layers\.(\d+)(\..*)", name)
            if m:
                local_idx = int(m.group(1))
                global_idx = local_idx + layer_offset
                name = f"model.layers.{global_idx}{m.group(2)}"
            remapped.append((name, shape, dtype))

        # Gather from all PP stages within PP group (DP-safe).
        from megatron.core import parallel_state as _mpu
        pp_group = _mpu.get_pipeline_model_parallel_group()
        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)

        full_info = []
        for pp_r in range(self.pp_size):
            src_global = pp_group_ranks[pp_r]
            stage_data = [remapped if self.rank == src_global else None]
            torch.distributed.broadcast_object_list(stage_data, src=src_global, group=pp_group)
            if stage_data[0] is not None:
                full_info.extend(stage_data[0])
        return full_info

    def _mcore_to_hf_state_dict(self):
        """Convert native Megatron model weights to true HF state dict.

        De-fuses QKV and gate+up weights back into separate HF-format tensors.
        Returns dict with proper HF names on CPU (for checkpoint saving).
        """
        hf_config = self._hf_config
        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = hf_config.hidden_size // num_heads
        if hasattr(hf_config, "head_dim") and hf_config.head_dim is not None:
            head_dim = hf_config.head_dim
        heads_per_group = num_heads // num_kv_heads

        mcore_sd = {k: v.cpu() for k, v in self._unwrapped_model.state_dict().items()
                    if v is not None and "_extra_state" not in k}
        hf_sd = {}

        # Detect fused layernorm layout
        model_param_names = set(mcore_sd.keys())

        # With PP, this rank only has a subset of layers with local indices.
        # Remap local → global when building the HF state dict.
        num_layers = hf_config.num_hidden_layers
        if self.pp_size > 1:
            layers_per_stage = num_layers // self.pp_size
            local_layers = layers_per_stage
            if self.pp_rank == self.pp_size - 1:
                local_layers = num_layers - self.pp_rank * layers_per_stage
            layer_offset = self.pp_rank * layers_per_stage
        else:
            local_layers = num_layers
            layer_offset = 0

        for local_idx in range(local_layers):
            mc_pre = f"decoder.layers.{local_idx}"
            global_idx = local_idx + layer_offset
            hf_pre = f"model.layers.{global_idx}"

            # De-fuse QKV: split interleaved [Q0,K0,V0,Q1,K1,V1,...] back
            qkv_w = mcore_sd[f"{mc_pre}.self_attention.linear_qkv.weight"]
            # Reshape to [num_kv_heads, heads_per_group+2, head_dim, hidden]
            group_size = (heads_per_group + 2) * head_dim
            qkv_grouped = qkv_w.view(num_kv_heads, heads_per_group + 2, head_dim, -1)
            q_parts = qkv_grouped[:, :heads_per_group, :, :]  # [nkv, hpg, hd, H]
            k_parts = qkv_grouped[:, heads_per_group:heads_per_group+1, :, :]  # [nkv, 1, hd, H]
            v_parts = qkv_grouped[:, heads_per_group+1:heads_per_group+2, :, :]  # [nkv, 1, hd, H]
            q = q_parts.reshape(num_heads * head_dim, -1)
            k = k_parts.reshape(num_kv_heads * head_dim, -1)
            v = v_parts.reshape(num_kv_heads * head_dim, -1)
            hf_sd[f"{hf_pre}.self_attn.q_proj.weight"] = q
            hf_sd[f"{hf_pre}.self_attn.k_proj.weight"] = k
            hf_sd[f"{hf_pre}.self_attn.v_proj.weight"] = v

            # De-fuse QKV bias if present
            qkv_b_key = f"{mc_pre}.self_attention.linear_qkv.bias"
            if qkv_b_key in mcore_sd:
                qkv_b = mcore_sd[qkv_b_key]
                qkv_b_grouped = qkv_b.view(num_kv_heads, heads_per_group + 2, head_dim)
                hf_sd[f"{hf_pre}.self_attn.q_proj.bias"] = (
                    qkv_b_grouped[:, :heads_per_group, :].reshape(-1)
                )
                hf_sd[f"{hf_pre}.self_attn.k_proj.bias"] = (
                    qkv_b_grouped[:, heads_per_group:heads_per_group+1, :].reshape(-1)
                )
                hf_sd[f"{hf_pre}.self_attn.v_proj.bias"] = (
                    qkv_b_grouped[:, heads_per_group+1:heads_per_group+2, :].reshape(-1)
                )

            # De-fuse gate+up
            fc1_w = mcore_sd[f"{mc_pre}.mlp.linear_fc1.weight"]
            half = fc1_w.size(0) // 2
            hf_sd[f"{hf_pre}.mlp.gate_proj.weight"] = fc1_w[:half]
            hf_sd[f"{hf_pre}.mlp.up_proj.weight"] = fc1_w[half:]

            # Simple renames
            hf_sd[f"{hf_pre}.self_attn.o_proj.weight"] = (
                mcore_sd[f"{mc_pre}.self_attention.linear_proj.weight"]
            )
            hf_sd[f"{hf_pre}.mlp.down_proj.weight"] = (
                mcore_sd[f"{mc_pre}.mlp.linear_fc2.weight"]
            )

            # Layer norms
            fused_ln_key = f"{mc_pre}.self_attention.linear_qkv.layer_norm_weight"
            if fused_ln_key in model_param_names:
                hf_sd[f"{hf_pre}.input_layernorm.weight"] = mcore_sd[fused_ln_key]
                hf_sd[f"{hf_pre}.post_attention_layernorm.weight"] = (
                    mcore_sd[f"{mc_pre}.mlp.linear_fc1.layer_norm_weight"]
                )
            else:
                hf_sd[f"{hf_pre}.input_layernorm.weight"] = (
                    mcore_sd[f"{mc_pre}.input_layernorm.weight"]
                )
                hf_sd[f"{hf_pre}.post_attention_layernorm.weight"] = (
                    mcore_sd[f"{mc_pre}.pre_mlp_layernorm.weight"]
                )

            # q_norm / k_norm (Qwen3)
            qn_key = f"{mc_pre}.self_attention.q_layernorm.weight"
            if qn_key in mcore_sd:
                hf_sd[f"{hf_pre}.self_attn.q_norm.weight"] = mcore_sd[qn_key]
                hf_sd[f"{hf_pre}.self_attn.k_norm.weight"] = (
                    mcore_sd[f"{mc_pre}.self_attention.k_layernorm.weight"]
                )

        # Embeddings, final layernorm, LM head
        # With PP: first stage has embedding, last stage has norm + lm_head
        if "embedding.word_embeddings.weight" in mcore_sd:
            hf_sd["model.embed_tokens.weight"] = mcore_sd["embedding.word_embeddings.weight"]
        if "output_layer.weight" in mcore_sd:
            hf_sd["lm_head.weight"] = mcore_sd["output_layer.weight"]
        elif (
            getattr(hf_config, "tie_word_embeddings", False)
            and "model.embed_tokens.weight" in hf_sd
        ):
            hf_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]
        if "decoder.final_layernorm.weight" in mcore_sd:
            hf_sd["model.norm.weight"] = mcore_sd["decoder.final_layernorm.weight"]

        return hf_sd

    def _load_hf_checkpoint_to_mcore(self, model_path):
        """Load an HF-format checkpoint (model.pt) into native Megatron model.

        The checkpoint was saved by ``_save_checkpoint`` in HF naming.
        We create a temporary directory with the checkpoint file, then
        use ``load_hf_weights_to_mcore`` which handles QKV interleaving
        and gate+up fusion.
        """
        from opd.trainer.megatron.model import _load_state_dict_into_mcore, _interleave_qkv

        hf_config = self._hf_config
        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = hf_config.hidden_size // num_heads
        if hasattr(hf_config, "head_dim") and hf_config.head_dim is not None:
            head_dim = hf_config.head_dim

        hf_sd = torch.load(model_path, map_location="cpu", weights_only=True)

        # Collect model names to detect fused vs separate layernorm and tied
        # output-layer allocation.
        model_param_names = set(name for name, _ in self._unwrapped_model.named_parameters())
        model_state_names = set(self._unwrapped_model.state_dict().keys())
        mcore_sd = {}

        # With PP, only load layers assigned to this stage
        num_layers = hf_config.num_hidden_layers
        if self.pp_size > 1:
            layers_per_stage = num_layers // self.pp_size
            pp_layer_start = self.pp_rank * layers_per_stage
            pp_layer_end = pp_layer_start + layers_per_stage
            if self.pp_rank == self.pp_size - 1:
                pp_layer_end = num_layers
        else:
            pp_layer_start = 0
            pp_layer_end = num_layers

        for layer_idx in range(pp_layer_start, pp_layer_end):
            hf_pre = f"model.layers.{layer_idx}"
            local_idx = layer_idx - pp_layer_start
            mc_pre = f"decoder.layers.{local_idx}"

            # QKV fusion with interleaving
            q = hf_sd[f"{hf_pre}.self_attn.q_proj.weight"]
            k = hf_sd[f"{hf_pre}.self_attn.k_proj.weight"]
            v = hf_sd[f"{hf_pre}.self_attn.v_proj.weight"]
            qkv = _interleave_qkv(q, k, v, num_heads, num_kv_heads, head_dim)
            mcore_sd[f"{mc_pre}.self_attention.linear_qkv.weight"] = qkv

            # QKV bias
            qb_key = f"{hf_pre}.self_attn.q_proj.bias"
            if qb_key in hf_sd:
                qb = hf_sd[qb_key]
                kb = hf_sd[f"{hf_pre}.self_attn.k_proj.bias"]
                vb = hf_sd[f"{hf_pre}.self_attn.v_proj.bias"]
                qkv_bias = _interleave_qkv(
                    qb, kb, vb, num_heads, num_kv_heads, head_dim, is_bias=True,
                )
                mcore_sd[f"{mc_pre}.self_attention.linear_qkv.bias"] = qkv_bias

            # gate + up fusion
            gate = hf_sd[f"{hf_pre}.mlp.gate_proj.weight"]
            up = hf_sd[f"{hf_pre}.mlp.up_proj.weight"]
            mcore_sd[f"{mc_pre}.mlp.linear_fc1.weight"] = torch.cat([gate, up], dim=0)

            # Simple renames
            mcore_sd[f"{mc_pre}.self_attention.linear_proj.weight"] = (
                hf_sd[f"{hf_pre}.self_attn.o_proj.weight"]
            )
            mcore_sd[f"{mc_pre}.mlp.linear_fc2.weight"] = (
                hf_sd[f"{hf_pre}.mlp.down_proj.weight"]
            )

            # Layer norms
            fused_ln_key = f"{mc_pre}.self_attention.linear_qkv.layer_norm_weight"
            if fused_ln_key in model_param_names:
                mcore_sd[fused_ln_key] = hf_sd[f"{hf_pre}.input_layernorm.weight"]
                mcore_sd[f"{mc_pre}.mlp.linear_fc1.layer_norm_weight"] = (
                    hf_sd[f"{hf_pre}.post_attention_layernorm.weight"]
                )
            else:
                mcore_sd[f"{mc_pre}.input_layernorm.weight"] = (
                    hf_sd[f"{hf_pre}.input_layernorm.weight"]
                )
                mcore_sd[f"{mc_pre}.pre_mlp_layernorm.weight"] = (
                    hf_sd[f"{hf_pre}.post_attention_layernorm.weight"]
                )

            # q_norm / k_norm (Qwen3)
            qn_key = f"{hf_pre}.self_attn.q_norm.weight"
            if qn_key in hf_sd:
                mcore_sd[f"{mc_pre}.self_attention.q_layernorm.weight"] = hf_sd[qn_key]
                mcore_sd[f"{mc_pre}.self_attention.k_layernorm.weight"] = (
                    hf_sd[f"{hf_pre}.self_attn.k_norm.weight"]
                )

        # Embeddings, final layernorm, LM head — PP-aware
        is_first_stage = self.pp_rank == 0
        is_last_stage = self.pp_rank == self.pp_size - 1
        if is_first_stage:
            mcore_sd["embedding.word_embeddings.weight"] = hf_sd["model.embed_tokens.weight"]
        if is_last_stage:
            if "output_layer.weight" in model_state_names:
                if "lm_head.weight" in hf_sd:
                    mcore_sd["output_layer.weight"] = hf_sd["lm_head.weight"]
                else:
                    mcore_sd["output_layer.weight"] = hf_sd["model.embed_tokens.weight"]
            mcore_sd["decoder.final_layernorm.weight"] = hf_sd["model.norm.weight"]

        _load_state_dict_into_mcore(self._unwrapped_model, mcore_sd)

    def _gather_hf_state_dict(self):
        """Convert native Megatron model state dict to HF format on GPU.

        Uses ``gather_tp_weights`` which handles name conversion and
        (at TP>1) all-gathers sharded params.  With PP>1, gathers from all
        PP stages and remaps local layer indices to global.

        Returns dict with HF-format names and tensors on the model device.
        """
        from opd.trainer.megatron.weight_gather import gather_tp_weights
        tp_group = None
        if self.use_megatron:
            from megatron.core import parallel_state as mpu
            tp_group = mpu.get_tensor_model_parallel_group()
        sd = gather_tp_weights(self._unwrapped_model, tp_group)

        if self.pp_size > 1:
            # With PP, each stage has a subset of layers with local indices.
            # gather_tp_weights returns non-empty only on TP rank 0.
            # Remap local → global layer indices, then broadcast across PP
            # stages within the same DP group using Megatron's PP group.
            import re
            from megatron.core import parallel_state as _mpu
            pp_group = _mpu.get_pipeline_model_parallel_group()

            num_layers = self._hf_config.num_hidden_layers
            layers_per_stage = num_layers // self.pp_size
            layer_offset = self.pp_rank * layers_per_stage

            if self.tp_rank == 0 and sd:
                remapped = {}
                for name, tensor in sd.items():
                    m = re.match(r"^model\.layers\.(\d+)(\..*)", name)
                    if m:
                        local_idx = int(m.group(1))
                        global_idx = local_idx + layer_offset
                        new_name = f"model.layers.{global_idx}{m.group(2)}"
                        remapped[new_name] = tensor.to(self.device)
                    else:
                        remapped[name] = tensor.to(self.device)
            else:
                remapped = {}

            # Build shape/dtype lookup from weights_info (gathered at init)
            wi_lookup = {n: (s, d) for n, s, d in self.weights_info}

            # Collect from all PP stages via per-tensor NCCL broadcast
            # within the PP group (one per DP group, so DP groups are independent).
            # PP group ranks are 0..pp_size-1 (local to the group).
            full_sd = {}
            pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
            for pp_r in range(self.pp_size):
                src_global = pp_group_ranks[pp_r]

                # Broadcast name list (tiny pickle, ~1KB)
                names_list = [list(remapped.keys()) if self.rank == src_global else None]
                torch.distributed.broadcast_object_list(names_list, src=src_global, group=pp_group)
                names = names_list[0]
                if not names:
                    continue

                # Broadcast each tensor via NCCL (dtype-safe)
                for name in names:
                    shape, dtype = wi_lookup[name]
                    if self.rank == src_global:
                        tensor = remapped[name].to(dtype).contiguous()
                    else:
                        tensor = torch.empty(shape, dtype=dtype, device=self.device)
                    torch.distributed.broadcast(tensor, src=src_global, group=pp_group)
                    full_sd[name] = tensor
            # Only rank 0 needs the full state dict for weight sync to vLLM
            return full_sd if self.rank == 0 else {}

        # gather_tp_weights returns CPU tensors; move to device
        return {k: v.to(self.device) for k, v in sd.items()}

    # ================================================================ #
    #  Model setup helpers                                               #
    # ================================================================ #


# ================================================================ #
#  Entry point                                                       #
# ================================================================ #

def megatron_trainer_main(config, cmd_queue, result_queue, rank_info):
    """Entry point for Megatron training subprocess."""
    launch_spec = ensure_trainer_launch_spec(config, rank_info) if isinstance(config, TrainerLaunchSpec) else None
    # Import TE early (before torch.distributed.init_process_group loads NCCL
    # which may pull in system CUDA 13). LD_PRELOAD ensures CUDA 12 loads first.
    megatron_cfg = launch_spec.static.megatron if launch_spec is not None else config["megatron"]
    if megatron_cfg["use_native_megatron"]:
        try:
            import transformer_engine  # noqa: F401
        except ImportError:
            pass
    trainer = MegatronBackend(launch_spec if launch_spec is not None else config,
                              None if launch_spec is not None else rank_info)
    trainer.run(cmd_queue, result_queue)
