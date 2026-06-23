"""Process spawning and lifecycle management for coordinator."""

import atexit
import multiprocessing as mp
import os
import socket
import sys
import time
import warnings

import torch

from opd.launch_specs import (
    RolloutLaunchRuntime,
    RolloutLaunchSpec,
    RolloutLaunchStatic,
    TeacherLaunchRuntime,
    TeacherLaunchSpec,
    TeacherLaunchStatic,
    TrainerLaunchRuntime,
    TrainerLaunchSpec,
    TrainerLaunchStatic,
    build_trainer_algorithm_launch,
    resolve_teacher_n_logprobs,
    serialize_algorithm_payload,
)
from opd.utils.cpu_affinity import (
    get_gpu_topology,
    plan_rollout_cpu_affinities,
    run_rollout_worker_with_affinity,
)
from opd.utils.net import (
    find_free_port,
    port_is_listening,
    kill_tree,
    release_all_port_leases,
)
from opd.utils.config import get_step_off_streaming_config, uses_step_off_streaming
from opd.worker.teacher.client import TeacherClient
from opd.worker.teacher.factory import get_teacher_backend
from opd.rollout.factory import get_rollout_backend
from opd.trainer import fsdp_trainer_main
from opd.worker.proxy import (
    QueueRolloutProxy, QueueTrainerProxy, NCCLWeightSyncEngine,
    CPUWeightSyncEngine,
)


class ProcessLifecycleMixin:
    """Process spawning and lifecycle management for coordinator.

    Required attributes from host class:
        self.config, self.train_cfg, self.teacher_cfg, self.data_cfg,
        self.model_path, self.max_prompt_length, self.max_response_length,
        self.batch_size, self.run_dir, self.step_off,
        self.scheduling_mode, self.staleness_threshold, self.tracer,
        self._need_student_logprobs
    """

    def _apply_rollout_logprob_flags(self, kl_loss_mode):
        use_importance_sampling = bool(self.opd_config.algorithm.opd.use_importance_sampling)
        self._need_student_logprobs = (
            kl_loss_mode == "policy_gradient_kl" and use_importance_sampling
        )
        self._rollout_support_topk_k = (
            int(self.opd_config.algorithm.opd.rollout_student_topk_k)
            if (
                kl_loss_mode == "thunlp_opd_default_loss"
                or (
                    kl_loss_mode == "reverse_kl_rollout_student_topk"
                    and use_importance_sampling
                )
            )
            else 0
        )
        self._mc_n_total_samples = (
            int(self.opd_config.algorithm.opd.pg_kl_n_total_samples)
            if (
                kl_loss_mode == "multi_sample_policy_gradient_kl"
                or (
                    kl_loss_mode == "mof_opd"
                    and int(self.opd_config.algorithm.opd.pg_kl_n_total_samples) > 1
                )
            )
            else 0
        )

    def start(self):
        """Launch all worker processes."""
        oc = self.opd_config
        # Deterministic mode: seed main process before spawning workers
        deterministic = oc.deterministic
        seed = oc.seed
        if deterministic:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"[Pipeline] Deterministic mode: seed={seed}", flush=True)
        self._setup_env()

        lora_dc = oc.trainer.lora
        # Convert LoRA dataclass to dict for subprocess consumption
        lora_cfg = {
            "rank": lora_dc.rank, "alpha": lora_dc.alpha,
            "dropout": lora_dc.dropout, "target_modules": lora_dc.target_modules,
            "modules_to_save": lora_dc.modules_to_save,
            "native_lora": lora_dc.native_lora,
        } if lora_dc is not None else None
        self._native_lora = bool(lora_dc and lora_dc.native_lora)
        pipeline_backend = oc.pipeline.deployment
        self._apply_rollout_logprob_flags(oc.algorithm.opd.kl_loss_mode)

        if pipeline_backend == "ray":
            return self._start_ray()
        ctx = mp.get_context("spawn")

        # --- Layout ---
        ro = oc.rollout
        # Build rollout_cfg dict for downstream methods that still expect it
        rollout_cfg = {
            "gpu_ids": ro.gpu_ids,
            "temperature": ro.temperature,
            "top_p": ro.top_p,
            "top_k": ro.top_k,
            "tensor_model_parallel_size": ro.vllm.tensor_parallel_size,
            "max_model_len": ro.vllm.max_model_len,
            "max_num_seqs": ro.vllm.max_num_seqs,
            "gpu_memory_utilization": ro.vllm.gpu_memory_utilization,
            "colocated_gpu_memory_utilization": ro.vllm.colocated_gpu_memory_utilization,
            "max_num_batched_tokens": ro.vllm.max_num_batched_tokens,
            "block_size": ro.vllm.block_size,
            "enforce_eager": ro.vllm.enforce_eager,
            "quantization": ro.quantization,
            "backend": ro.backend,
            "dtype": ro.dtype,
            "pause_mode": oc.pipeline.fully_async.pause_mode,
        } if ro is not None else {}
        tp = ro.vllm.tensor_parallel_size if ro is not None else 1
        rollout_gpu_str = self._gpu_ids("rollout")
        rollout_gpu_list = rollout_gpu_str.split(",")
        n_rollout_workers = max(len(rollout_gpu_list) // tp, 1)

        # --- Teacher ---
        teacher_gpu_set = self._start_teacher_local(ctx, rollout_gpu_list)

        # --- Rollout worker(s) ---
        self._start_rollout_workers_local(ctx, rollout_cfg, rollout_gpu_list, tp,
                                          n_rollout_workers, teacher_gpu_set, lora_cfg)

        # --- Trainer ---
        self._start_trainer_local(ctx, lora_cfg)

        # --- Construct proxy objects ---
        use_async_rollout = self._uses_async_rollout_workers()
        self._build_proxies(rollout_cfg, tp, n_rollout_workers, use_async_rollout,
                            rollout_gpu_list)

        print("[Pipeline] All workers ready.", flush=True)
        atexit.register(self.shutdown)

    def _setup_env(self):
        """Set up CUDA_HOME/PATH/LD_LIBRARY_PATH/CPATH for child processes."""
        # Ensure all child processes can discover tools from the active Python
        # environment (ninja, etc.) even when CUDA libraries are provided by the
        # system rather than the environment.  Fused-hybrid vLLM is constructed
        # inside trainer ranks, so it relies on this coordinator-level PATH.
        _env_bin = os.path.dirname(sys.executable)
        _env_root = os.path.dirname(_env_bin)
        _env_lib = os.path.join(_env_root, "lib")
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if _env_bin not in path_parts:
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = (
                _env_bin if not old_path else _env_bin + os.pathsep + old_path
            )

        # If the env carries CUDA runtime libraries, prefer the env CUDA toolkit
        # (not system CUDA 13) for JIT compilation and library linking.
        if os.path.isfile(os.path.join(_env_lib, "libcudart.so")):
            os.environ["CUDA_HOME"] = _env_root
            library_path = os.environ.get("LIBRARY_PATH", "")
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LIBRARY_PATH"] = (
                _env_lib if not library_path else _env_lib + os.pathsep + library_path
            )
            os.environ["LD_LIBRARY_PATH"] = (
                _env_lib
                if not ld_library_path
                else _env_lib + os.pathsep + ld_library_path
            )
        # System CUDA headers needed for FlashInfer JIT (cuda_fp16.h, cublasLt.h, etc.)
        for cuda_dir in ["/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-12.8"]:
            cuda_inc = os.path.join(cuda_dir, "include")
            if os.path.isfile(os.path.join(cuda_inc, "cuda_fp16.h")):
                cpath = os.environ.get("CPATH", "")
                os.environ["CPATH"] = f"{cuda_inc}:{cpath}" if cpath else cuda_inc
                if not os.environ.get("CUDA_HOME"):
                    os.environ["CUDA_HOME"] = cuda_dir
                break

    def _uses_async_rollout_workers(self):
        """Whether this coordinator needs streaming/AsyncLLM rollout workers."""
        if self.scheduling_mode == "fully_async":
            return True
        pipeline = getattr(getattr(self, "opd_config", None), "pipeline", None)
        return bool(pipeline and uses_step_off_streaming(pipeline))

    def _uses_direct_teacher_artifacts(self):
        """Whether teacher artifacts should travel on the trainer data channel."""
        oc = getattr(self, "opd_config", None)
        if oc is None:
            return False
        pipeline = getattr(oc, "pipeline", None)
        if pipeline is None:
            return False
        sos = get_step_off_streaming_config(pipeline)
        return bool(
            sos
            and sos.teacher_transport == "direct_trainer"
            and oc.algorithm.opd.teacher_artifact_mode in {"direct", "hidden_recompute"}
        )

    def _build_teacher_launch_spec(self, teacher_gpus, bind_port):
        """Build a typed teacher launch spec with explicit static/runtime split."""
        oc = getattr(self, 'opd_config', None)
        if oc is not None and oc.teacher is not None:
            t = oc.teacher
            teacher_backend = t.backend
            n_logprobs = resolve_teacher_n_logprobs(oc)
            if teacher_backend == "hf":
                static = TeacherLaunchStatic(
                    backend=teacher_backend,
                    model_path=t.path,
                    n_logprobs=n_logprobs,
                    scoring_batch_size=t.scoring_batch_size or 4,
                    dtype=t.dtype,
                    use_torch_compile=t.hf.use_torch_compile,
                    tp_size=1,
                    gpu_memory_utilization=None,
                    max_model_len=None,
                    max_num_seqs=None,
                    enforce_eager=None,
                    disable_fast_logprobs=None,
                    block_size=None,
                    hidden_recompute=(oc.algorithm.opd.teacher_artifact_mode == "hidden_recompute"),
                    teacher_hidden_dtype=oc.algorithm.opd.teacher_hidden_dtype,
                    teacher_hidden_semantics=oc.algorithm.opd.teacher_hidden_semantics,
                    trust_remote_code=t.trust_remote_code,
                )
            else:
                static = TeacherLaunchStatic(
                    backend=teacher_backend,
                    model_path=t.path,
                    n_logprobs=n_logprobs,
                    tp_size=t.vllm.tensor_parallel_size,
                    gpu_memory_utilization=t.vllm.gpu_memory_utilization,
                    max_model_len=t.vllm.max_model_len,
                    max_num_seqs=t.vllm.max_num_seqs,
                    enforce_eager=t.vllm.enforce_eager,
                    scoring_batch_size=t.scoring_batch_size or 32,
                    dtype=t.dtype,
                    disable_fast_logprobs=t.vllm.disable_fast_logprobs,
                    block_size=t.vllm.block_size,
                    hidden_recompute=(oc.algorithm.opd.teacher_artifact_mode == "hidden_recompute"),
                    teacher_hidden_dtype=oc.algorithm.opd.teacher_hidden_dtype,
                    teacher_hidden_semantics=oc.algorithm.opd.teacher_hidden_semantics,
                    trust_remote_code=t.trust_remote_code,
                )
            runtime = TeacherLaunchRuntime(
                gpu_ids=teacher_gpus,
                bind_port=bind_port,
                bind_address=t.bind_address,
                seed=oc.seed if oc.deterministic else None,
            )
            return TeacherLaunchSpec(static=static, runtime=runtime)

    def _start_teacher_local(self, ctx, rollout_gpu_list):
        """Spawn teacher process. Sets self.teacher_port. Returns teacher_gpu_set."""
        oc = getattr(self, 'opd_config', None)
        teacher_gpu_set = set()
        if self._needs_teacher():
            teacher_gpus = self._gpu_ids("teacher")
            teacher_gpu_set = set(teacher_gpus.split(","))
            self.teacher_port = find_free_port("teacher.bind")
            teacher_backend = oc.teacher.backend
            tp_size = oc.teacher.vllm.tensor_parallel_size
            teacher_fn = get_teacher_backend(teacher_backend)["server_main"]
            teacher_spec = self._build_teacher_launch_spec(teacher_gpus, self.teacher_port)
            # Pre-allocate vLLM ports in coordinator so they share _allocated_ports
            # with FSDP/weight-sync ports and can't collide.
            teacher_spec = teacher_spec.with_runtime(
                vllm_port=find_free_port("teacher.vllm"),
                vllm_master_port=find_free_port("teacher.vllm_master"),
            )
            self.teacher_proc = ctx.Process(target=teacher_fn, args=(teacher_spec,))
            self.teacher_proc.start()
            print(f"[Pipeline] Teacher backend: {teacher_backend}", flush=True)

            # Check if teacher and rollout share any GPUs (colocation)
            rollout_gpu_set = set(rollout_gpu_list)
            colocated = bool(rollout_gpu_set & teacher_gpu_set)
            if colocated or tp_size > 1:
                reason = "colocated" if colocated else f"TP={tp_size} (avoid concurrent CUDA init)"
                print(f"[Pipeline] Waiting for teacher to finish loading ({reason})...",
                      flush=True)
                self._wait_for_teacher_ready()
        else:
            print("[Pipeline] Teacher not needed, skipping.", flush=True)
        return teacher_gpu_set

    def _build_rollout_launch_spec(self, rollout_cfg, worker_gpus, tp, worker_id, mem_util,
                                   lora_cfg, prompt_queue=None, pause_mode=None):
        """Build a typed rollout launch spec with explicit static/runtime split."""
        oc = self.opd_config
        rollout_support_topk_k = getattr(self, "_rollout_support_topk_k", 0)
        mc_n_total_samples = getattr(self, "_mc_n_total_samples", 0)
        if rollout_support_topk_k > 0:
            n_logprobs = rollout_support_topk_k
        elif mc_n_total_samples > 0:
            n_logprobs = 1
        else:
            n_logprobs = resolve_teacher_n_logprobs(oc)
        static = RolloutLaunchStatic(
            model_path=self.model_path,
            tp_size=tp,
            max_response_length=self.max_response_length,
            temperature=rollout_cfg.get("temperature", 1.0),
            top_p=rollout_cfg.get("top_p", 0.99),
            top_k=rollout_cfg.get("top_k", -1),
            max_num_seqs=rollout_cfg.get("max_num_seqs", 128),
            use_weight_transfer=self.use_nccl,
            max_model_len=rollout_cfg.get("max_model_len"),
            max_num_batched_tokens=rollout_cfg.get("max_num_batched_tokens"),
            enforce_eager=rollout_cfg.get("enforce_eager", True),
            dtype=rollout_cfg.get("dtype", "auto"),
            quantization=self.rollout_quantization,
            native_lora=bool(self._native_lora),
            lora_rank=lora_cfg["rank"] if self._native_lora else 0,
            lora_cfg=lora_cfg if self._native_lora else None,
            max_logprobs=n_logprobs,
            block_size=rollout_cfg.get("block_size"),
            pin_cpu_affinity=bool(rollout_cfg.get("pin_cpu_affinity", False)),
            bind_numa_memory=bool(rollout_cfg.get("bind_numa_memory", False)),
            trust_remote_code=bool(
                getattr(getattr(oc, "rollout", None), "trust_remote_code", False)
            ),
        )
        runtime = RolloutLaunchRuntime(
            gpu_ids=worker_gpus,
            worker_id=worker_id,
            gpu_memory_utilization=mem_util,
            prompt_queue=prompt_queue,
            pause_mode=pause_mode or "abort",
            cpu_affinity_cpus=(),
            numa_nodes=(),
            seed=oc.seed if oc.deterministic else None,
        )
        return RolloutLaunchSpec(static=static, runtime=runtime)

    def _start_rollout_workers_local(self, ctx, rollout_cfg, rollout_gpu_list, tp,
                                     n_rollout_workers, teacher_gpu_set, lora_cfg):
        """Spawn rollout worker processes."""
        use_async_rollout = self._uses_async_rollout_workers()
        rollout_backend = rollout_cfg.get("backend", "vllm")
        backend_info = get_rollout_backend(rollout_backend)
        if use_async_rollout and not backend_info["supports_streaming"]:
            raise ValueError(
                f"rollout.backend='{rollout_backend}' does not support streaming. "
                "Use step-off scheduling with this backend.")
        if not backend_info["supports_nccl"]:
            self.use_nccl = False
            print(f"[Pipeline] Using {rollout_backend} rollout backend (CPU weight sync)",
                  flush=True)
        if use_async_rollout:
            worker_fn = backend_info["streaming_worker_main"]()
        else:
            worker_fn = backend_info["worker_main"]
        if use_async_rollout:
            print(f"[Pipeline] Using streaming rollout workers ({self.scheduling_mode} mode)", flush=True)
        # Shared prompt queue (pull model): all workers pull from one queue.
        # Self-balancing — fast workers pull more, slow workers pull less.
        use_prompt_queue = use_async_rollout
        if use_prompt_queue:
            # No maxsize — capacity_sem already limits in-flight prompts
            self.rollout_prompt_queue = ctx.Queue()
        else:
            self.rollout_prompt_queue = None
        affinity_plans = {}
        if rollout_cfg.get("pin_cpu_affinity", False):
            gpu_topology = get_gpu_topology()
            affinity_plans = plan_rollout_cpu_affinities(
                rollout_gpu_list, tp, gpu_topology
            )
            if affinity_plans:
                print("[Pipeline] Enabling rollout CPU affinity pinning", flush=True)
            else:
                print(
                    "[Pipeline] WARNING: rollout.pin_cpu_affinity requested but GPU topology was unavailable",
                    flush=True,
                )
        colocated_mem = rollout_cfg.get("colocated_gpu_memory_utilization")
        default_mem = rollout_cfg.get("gpu_memory_utilization", 0.5)
        for i in range(n_rollout_workers):
            worker_gpus = ",".join(rollout_gpu_list[i * tp : (i + 1) * tp])
            cmd_q = ctx.Queue()
            res_q = ctx.Queue()
            self.rollout_cmd_queues.append(cmd_q)
            self.rollout_result_queues.append(res_q)

            # Use lower memory utilization for workers sharing GPUs with teacher
            worker_gpu_set = set(worker_gpus.split(","))
            if colocated_mem is not None and worker_gpu_set & teacher_gpu_set:
                mem_util = colocated_mem
                print(f"[Pipeline] Rollout-{i} colocated with teacher on GPU(s) "
                      f"{worker_gpu_set & teacher_gpu_set}, using gpu_memory_utilization={mem_util}",
                      flush=True)
            else:
                mem_util = default_mem

            prompt_queue = self.rollout_prompt_queue if use_prompt_queue else None
            pause_mode = rollout_cfg.get("pause_mode", "keep") if use_async_rollout else None
            rollout_spec = self._build_rollout_launch_spec(
                rollout_cfg, worker_gpus, tp, i, mem_util, lora_cfg,
                prompt_queue=prompt_queue, pause_mode=pause_mode,
            )
            if i in affinity_plans:
                rollout_spec = rollout_spec.with_runtime(**affinity_plans[i])
            # Pre-allocate vLLM ports in coordinator so they share _allocated_ports
            # with teacher/FSDP/weight-sync ports and can't collide.
            rollout_spec = rollout_spec.with_runtime(
                vllm_port=find_free_port(f"rollout.{i}.vllm"),
                vllm_master_port=find_free_port(f"rollout.{i}.vllm_master"),
            )

            target = (
                run_rollout_worker_with_affinity
                if rollout_spec.static.pin_cpu_affinity
                else worker_fn
            )
            args = (
                (worker_fn, rollout_spec, cmd_q, res_q)
                if target is run_rollout_worker_with_affinity
                else (rollout_spec, cmd_q, res_q)
            )
            p = ctx.Process(target=target, args=args)
            p.start()
            self.rollout_procs.append(p)

    def _build_algorithm_config(self):
        """Build the typed trainer algorithm launch payload.

        The legacy dict serializer remains available for compatibility
        boundaries, but the canonical trainer launch payload is typed.
        """
        oc = getattr(self, 'opd_config', None)
        if oc is not None:
            return build_trainer_algorithm_launch(oc.algorithm)

    def _build_trainer_launch_static(self):
        """Build the trainer's static launch payload."""
        oc = getattr(self, 'opd_config', None)
        backend = self._get_backend()

        if oc is not None:
            tr = oc.trainer
            algo = oc.algorithm
            mode = algo.mode
            # Derive loss_mode from algorithm mode
            loss_mode_map = {"opd": "kl", "opsd": "kl", "grpo": "grpo", "sft": "sft"}
            loss_mode = loss_mode_map.get(mode, "kl")
            # mini_batch_size scaling by grpo_group_size
            mini_bs = (tr.mini_batch_size or 0)
            if mode == "grpo":
                mini_bs *= algo.grpo.group_size
            # Convert optim dataclass to dict
            optim_cfg = {
                "lr": tr.optim.lr,
                "lr_decay_style": tr.optim.lr_decay_style,
                "lr_warmup_steps_ratio": tr.optim.lr_warmup_steps_ratio,
                "weight_decay": tr.optim.weight_decay,
                "min_lr": tr.optim.min_lr,
                "adam_beta2": tr.optim.adam_beta2,
                "adam_eps": tr.optim.adam_eps,
            }
            # Convert lora dataclass to dict if present
            lora = None
            if tr.lora is not None:
                lora = {
                    "rank": tr.lora.rank, "alpha": tr.lora.alpha,
                    "dropout": tr.lora.dropout, "target_modules": tr.lora.target_modules,
                    "modules_to_save": tr.lora.modules_to_save,
                    "native_lora": tr.lora.native_lora,
                }
            megatron_cfg = None
            if backend == "megatron":
                meg = tr.megatron
                megatron_cfg = dict(
                    tp_size=meg.tensor_parallel_size,
                    pp_size=meg.pipeline_parallel_size,
                    use_native_megatron=meg.use_native_megatron,
                    use_transformer_engine=meg.use_transformer_engine,
                )
            fused_rollout = None
            fused_sync = None
            if oc.pipeline.scheduling_mode == "fused_hybrid_sync":
                fhs = oc.pipeline.fused_hybrid_sync
                fused_rollout = {
                    "model_path": self.model_path,
                    "tp_size": oc.rollout.vllm.tensor_parallel_size,
                    "dp_size": (
                        len([gpu for gpu in self._gpu_ids("rollout").split(",") if gpu.strip()])
                        if fhs.rollout_dp_size is None
                        else int(fhs.rollout_dp_size)
                    ),
                    "max_response_length": self.max_response_length,
                    "temperature": oc.rollout.temperature,
                    "top_p": oc.rollout.top_p,
                    "top_k": oc.rollout.top_k,
                    "max_num_seqs": oc.rollout.vllm.max_num_seqs,
                    "max_model_len": oc.rollout.vllm.max_model_len,
                    "max_num_batched_tokens": oc.rollout.vllm.max_num_batched_tokens,
                    "gpu_memory_utilization": oc.rollout.vllm.gpu_memory_utilization,
                    "dtype": oc.rollout.dtype,
                    "max_logprobs": resolve_teacher_n_logprobs(oc),
                    "trust_remote_code": oc.rollout.trust_remote_code,
                }
                fused_sync = {
                    "rollout_parallelism": fhs.rollout_parallelism,
                    "rollout_dp_size": fhs.rollout_dp_size,
                    "weight_update_backend": fhs.weight_update_backend,
                    "debug_full_state_sync": fhs.debug_full_state_sync,
                    "update_bucket_mb": fhs.update_bucket_mb,
                    "vllm_sleep_level": fhs.vllm_sleep_level,
                    "require_vllm_sleep": fhs.require_vllm_sleep,
                    "verify_weight_checksum": fhs.verify_weight_checksum,
                    "refresh_policy": fhs.refresh_policy,
                    "allow_single_gpu_debug": fhs.allow_single_gpu_debug,
                    "log_memory": fhs.log_memory,
                }
            return TrainerLaunchStatic(
                model_path=self.model_path,
                dtype=tr.dtype,
                attn_implementation=tr.attn_implementation,
                optim=optim_cfg,
                micro_batch_size=tr.micro_batch_size,
                mini_batch_size=mini_bs,
                max_response_length=self.max_response_length,
                use_sequence_packing=tr.use_sequence_packing,
                use_torch_compile=tr.use_torch_compile,
                max_grad_norm=tr.optim.max_grad_norm,
                loss_mode=loss_mode,
                lora=lora,
                kl_chunk_size=tr.kl_chunk_size,
                backend=backend,
                algorithm=self._build_algorithm_config(),
                deterministic=oc.deterministic,
                seed=oc.seed,
                megatron=megatron_cfg,
                teacher_model_path=oc.teacher.path if oc.teacher is not None else None,
                teacher_artifact_mode=oc.algorithm.opd.teacher_artifact_mode,
                teacher_hidden_dtype=oc.algorithm.opd.teacher_hidden_dtype,
                teacher_hidden_semantics=oc.algorithm.opd.teacher_hidden_semantics,
                teacher_hidden_recompute_materialization=oc.algorithm.opd.teacher_hidden_recompute_materialization,
                fused_hybrid_rollout=fused_rollout,
                fused_hybrid_sync=fused_sync,
                trust_remote_code=tr.trust_remote_code or oc.model.trust_remote_code,
            )

    def _build_trainer_launch_spec(self, rank_info, *, gpu_ids=None):
        """Build a typed trainer launch spec for a specific rank/runtime payload."""
        runtime = TrainerLaunchRuntime(
            gpu_ids=self._gpu_ids("trainer") if gpu_ids is None else gpu_ids,
            total_steps=self._compute_total_steps(),
            nccl_timeout_hours=self.opd_config.weight_sync.nccl_timeout_hours,
            rank_info=rank_info,
            teacher_artifact_queue=(
                self.teacher_artifact_queue
                if self._uses_direct_teacher_artifacts()
                and int(rank_info.get("fsdp_rank", rank_info.get("global_rank", 0))) == 0
                else None
            ),
        )
        return TrainerLaunchSpec(
            static=self._build_trainer_launch_static(),
            runtime=runtime,
        )

    def _build_fsdp_rank_info(self, fsdp_rank, fsdp_world_size, fsdp_master_port):
        """Build FSDP rank_info dict."""
        return dict(
            fsdp_rank=fsdp_rank,
            fsdp_world_size=fsdp_world_size,
            fsdp_master_port=fsdp_master_port,
            fsdp_master_addr="127.0.0.1",
        )

    def _build_megatron_rank_info(self, global_rank, tp_rank, pp_rank,
                                  global_world_size, megatron_master_port):
        """Build Megatron rank_info dict (superset of FSDP)."""
        return dict(
            fsdp_rank=global_rank,
            fsdp_world_size=global_world_size,
            fsdp_master_port=0,
            fsdp_master_addr="127.0.0.1",
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            global_rank=global_rank,
            global_world_size=global_world_size,
            megatron_master_port=megatron_master_port,
            megatron_master_addr="127.0.0.1",
        )

    def _start_trainer_local(self, ctx, lora_cfg):
        """Spawn trainer process (FSDP or Megatron)."""
        self.trainer_cmd_queue = ctx.Queue()
        self.trainer_result_queue = ctx.Queue()
        self.teacher_artifact_queue = (
            ctx.Queue(maxsize=max(self.step_off + 2, 3) * max(self.batch_size, 1))
            if self._uses_direct_teacher_artifacts()
            else None
        )

        trainer_static = self._build_trainer_launch_static()
        trainer_gpus = self._gpu_ids("trainer")
        backend = trainer_static.backend

        algo_payload = serialize_algorithm_payload(trainer_static.algorithm)
        self._apply_rollout_logprob_flags(algo_payload["kl_loss_mode"])

        if backend == "fsdp":
            trainer_fn = self._get_fsdp_trainer_fn()
            n_trainer_gpus = len(trainer_gpus.split(","))
            self._n_trainer_gpus = n_trainer_gpus
            if n_trainer_gpus > 1:
                fsdp_master_port = find_free_port("trainer.fsdp_master")
                self._trainer_fsdp_procs = []
                for fsdp_rank in range(n_trainer_gpus):
                    rank_info = self._build_fsdp_rank_info(
                        fsdp_rank, n_trainer_gpus, fsdp_master_port)
                    trainer_spec = self._build_trainer_launch_spec(rank_info)
                    cmd_q = self.trainer_cmd_queue if fsdp_rank == 0 else None
                    res_q = self.trainer_result_queue if fsdp_rank == 0 else None
                    p = ctx.Process(
                        target=trainer_fn,
                        args=(trainer_spec, cmd_q, res_q, None),
                    )
                    p.start()
                    self._trainer_fsdp_procs.append(p)
                self.trainer_proc = self._trainer_fsdp_procs[0]
            else:
                self._trainer_fsdp_procs = []
                rank_info = self._build_fsdp_rank_info(0, 1, None)
                trainer_spec = self._build_trainer_launch_spec(rank_info)
                self.trainer_proc = ctx.Process(
                    target=trainer_fn,
                    args=(trainer_spec, self.trainer_cmd_queue,
                          self.trainer_result_queue, None),
                )
                self.trainer_proc.start()

        elif backend == "megatron":
            from opd.trainer.megatron import megatron_trainer_main
            n_trainer_gpus = len(trainer_gpus.split(","))
            self._n_trainer_gpus = n_trainer_gpus
            meg = trainer_static.megatron
            tp_size = meg["tp_size"]
            pp_size = meg["pp_size"]
            dp_size = n_trainer_gpus // (tp_size * pp_size)
            global_world_size = tp_size * pp_size * dp_size
            megatron_master_port = find_free_port("trainer.megatron_master")
            if global_world_size > 1:
                self._trainer_fsdp_procs = []
                for global_rank in range(global_world_size):
                    tp_rank = global_rank % tp_size
                    pp_rank = (global_rank // tp_size) % pp_size
                    rank_info = self._build_megatron_rank_info(
                        global_rank, tp_rank, pp_rank,
                        global_world_size, megatron_master_port)
                    trainer_spec = self._build_trainer_launch_spec(rank_info)
                    cmd_q = self.trainer_cmd_queue if global_rank == 0 else None
                    res_q = self.trainer_result_queue if global_rank == 0 else None
                    p = ctx.Process(
                        target=megatron_trainer_main,
                        args=(trainer_spec, cmd_q, res_q, None),
                    )
                    p.start()
                    self._trainer_fsdp_procs.append(p)
                self.trainer_proc = self._trainer_fsdp_procs[0]
            else:
                self._trainer_fsdp_procs = []
                rank_info = self._build_megatron_rank_info(
                    0, 0, 0, 1, megatron_master_port)
                trainer_spec = self._build_trainer_launch_spec(rank_info)
                self.trainer_proc = ctx.Process(
                    target=megatron_trainer_main,
                    args=(trainer_spec, self.trainer_cmd_queue,
                          self.trainer_result_queue, None),
                )
                self.trainer_proc.start()
            print(f"[Pipeline] Megatron trainer: TP={tp_size}, DP={dp_size}, "
                  f"GPUs={trainer_gpus}", flush=True)

    def _build_proxies(self, rollout_cfg, tp, n_rollout_workers, use_async_rollout,
                       rollout_gpu_list):
        """Construct proxy objects and initialize weight transfer + teacher client."""
        self.rollout_proxy = QueueRolloutProxy(
            cmd_queues=self.rollout_cmd_queues,
            result_queues=self.rollout_result_queues,
            procs=self.rollout_procs,
            need_student_logprobs=self._need_student_logprobs,
            rollout_support_topk_k=getattr(self, "_rollout_support_topk_k", 0),
            mc_n_total_samples=getattr(self, "_mc_n_total_samples", 0),
            prompt_queue=self.rollout_prompt_queue,
        )
        self.trainer_proxy = QueueTrainerProxy(
            cmd_queue=self.trainer_cmd_queue,
            result_queue=self.trainer_result_queue,
            proc=self.trainer_proc,
            fsdp_procs=getattr(self, '_trainer_fsdp_procs', []),
        )

        # --- Wait for workers to initialize ---
        # Explicit readiness check instead of sleep — blocks until both trainer
        # and rollout have finished model loading and entered their command loops.
        # This prevents the NCCL weight transfer init from timing out when the
        # trainer takes >300s to load a large model with FSDP.
        print("[Pipeline] Waiting for workers to initialize...", flush=True)
        # Send readiness pings in parallel (both start model loading concurrently)
        self.trainer_proxy.submit_command_async("get_weights_info")
        self.rollout_proxy.submit_command("get_vllm_params_info")
        # Block until both have responded
        self.rollout_proxy.collect_results(purpose="rollout readiness response")
        self.trainer_proxy.collect_command()
        print("[Pipeline] Workers ready.", flush=True)

        # --- Init vLLM native NCCL weight transfer ---
        if self.use_nccl:
            # TODO: fully_async + keep mode + TP>1 likely hangs because
            # generation may be in-flight on some TP ranks while others
            # enter the NCCL weight broadcast. Pause mode should work.
            if tp > 1 and use_async_rollout:
                pause_mode = rollout_cfg.get("pause_mode", "keep")
                if pause_mode == "keep":
                    raise NotImplementedError(
                        f"Rollout TP={tp} with fully_async + pause_mode=keep is not supported. "
                        f"TP ranks may be mid-generation during weight sync NCCL broadcast. "
                        f"Use pause_mode=pause or step-off scheduling."
                    )
            self._init_weight_transfer(n_rollout_workers, tp_size=tp)
        elif not get_rollout_backend(rollout_cfg.get("backend", "vllm"))["supports_nccl"]:
            # CPU weight sync for non-NCCL backends (e.g. HF)
            self.weight_engine = CPUWeightSyncEngine(
                verify_checksum=self.verify_weight_sync)
            self.weight_engine.initialize(self.trainer_proxy, self.rollout_proxy)

        # --- Teacher client ---
        if self._needs_teacher():
            self.teacher_client = TeacherClient(
                f"tcp://127.0.0.1:{self.teacher_port}",
                n_workers=1,  # always 1 in current architecture
            )
        else:
            self.teacher_client = None

        # Trace metadata for host/gpu_ids in Perfetto spans
        _host = socket.gethostname()
        self._trainer_trace_info = {"host": _host, "gpu_ids": self._gpu_ids("trainer")}
        self._teacher_trace_info = {"host": _host, "gpu_ids": self._gpu_ids("teacher")}
        self._rollout_worker_info = [
            {"host": _host, "gpu_ids": ",".join(rollout_gpu_list[i * tp : (i + 1) * tp])}
            for i in range(n_rollout_workers)
        ]

    def shutdown(self):
        """Shut down all worker processes. Safe to call multiple times."""
        self.stop_trace_monitors()
        self._wait_checkpoint_save()
        # Clean up weight transfer NCCL group before killing rollout —
        # prevents TCPStore EPOLLHUP crash when rollout exits first.
        # Must wait for ack to ensure cleanup completes before rollout exits.
        if self.trainer_proxy:
            try:
                result = self.trainer_proxy.submit_command("shutdown_weight_transfer")
                if result and result.get("status") != "ok":
                    print(f"[Pipeline] WARNING: weight transfer shutdown returned {result}",
                          flush=True)
            except Exception as e:
                print(f"[Pipeline] WARNING: weight transfer shutdown failed: {e}", flush=True)
        if self.rollout_proxy:
            self.rollout_proxy.shutdown()
        if self.trainer_proxy:
            self.trainer_proxy.shutdown()
        # Give workers time to exit gracefully
        for p in self.rollout_procs:
            p.join(timeout=5)
        if self.trainer_proc:
            self.trainer_proc.join(timeout=60)
        # Join extra FSDP trainer procs (rank > 0)
        for p in getattr(self, '_trainer_fsdp_procs', [])[1:]:
            p.join(timeout=5)
        # Force-kill anything still alive — kill entire process tree so
        # grandchild processes (e.g. vLLM EngineCore) are also cleaned up.
        all_procs = self.rollout_procs + ([self.trainer_proc] if self.trainer_proc else [])
        all_procs += getattr(self, '_trainer_fsdp_procs', [])[1:]  # FSDP rank > 0
        all_procs += [self.teacher_proc] if self.teacher_proc else []
        for p in all_procs:
            if p.is_alive():
                kill_tree(p.pid)
                p.join(timeout=5)
        release_all_port_leases()

    def _needs_teacher(self):
        """Whether a teacher/reference model process is needed.

        Delegates to mode_cls if available (mode_cls is set at construction,
        before start() spawns processes). Falls back to config-based detection.
        """
        mode_cls = getattr(self, '_mode_cls', None)
        if mode_cls is not None and hasattr(mode_cls, 'needs_teacher'):
            # Instantiate a lightweight probe — needs_teacher is config-driven
            # so we can call it on a temporary instance or use class-level logic.
            # For modes that need config to decide (e.g. GRPOMode checks kl_beta),
            # we construct a minimal instance via from_coordinator won't work
            # (proxies not ready). Instead, replicate the check from config.
            from opd.coordinator.grpo_mode import GRPOMode
            if issubclass(mode_cls, GRPOMode):
                return self.opd_config.algorithm.grpo.kl_beta > 0
            from opd.coordinator.opd_mode import OPDMode, OPSDMode
            if issubclass(mode_cls, OPSDMode):
                return False
            if issubclass(mode_cls, OPDMode):
                return True
            # SFTMode and others
            try:
                # SFTMode.needs_teacher is a simple False
                tmp = mode_cls.__new__(mode_cls)
                return tmp.needs_teacher()
            except Exception:
                pass
        return True

    def _get_fsdp_trainer_fn(self):
        """Return trainer_main_fn for FSDP trainer subprocess.

        Delegates to mode_cls.get_trainer_fn() if available. Mode classes
        return their specific trainer entry point. All config is in the
        trainer config dict — no extra kwargs needed.
        """
        mode_cls = getattr(self, '_mode_cls', None)
        if mode_cls is not None and hasattr(mode_cls, 'get_trainer_fn'):
            try:
                from opd.coordinator.grpo_mode import GRPOMode
                if issubclass(mode_cls, GRPOMode):
                    from opd.trainer.grpo import grpo_trainer_main
                    return grpo_trainer_main
                from opd.coordinator.opd_mode import OPDMode
                if issubclass(mode_cls, OPDMode):
                    from opd.trainer.opd import opd_trainer_main
                    return opd_trainer_main
            except Exception:
                pass
        return fsdp_trainer_main

    def _get_ray_trainer_cls(self):
        """Return trainer class for Ray actor construction.

        Mode-specific trainers (OPDTrainer, GRPOTrainer) are composition
        wrappers that internally create FSDPBackend/MegatronBackend.
        Falls back to FSDPBackend for backward compatibility.
        """
        mode_cls = getattr(self, '_mode_cls', None)
        if mode_cls is not None:
            try:
                from opd.coordinator.grpo_mode import GRPOMode
                if issubclass(mode_cls, GRPOMode):
                    from opd.trainer.grpo import GRPOTrainer
                    return GRPOTrainer
                from opd.coordinator.opd_mode import OPDMode
                if issubclass(mode_cls, OPDMode):
                    from opd.trainer.opd import OPDTrainer
                    return OPDTrainer
            except Exception:
                pass
        from opd.trainer.fsdp import FSDPBackend
        return FSDPBackend

    def _wait_for_teacher_ready(self, host=None, port_override=None, timeout=1200):
        """Block until the teacher ZMQ port is accepting connections."""
        port = port_override or self.teacher_port
        t0 = time.time()
        while time.time() - t0 < timeout:
            if port_is_listening(port, host=host):
                print(f"[Pipeline] Teacher port is up ({host or 'localhost'}:{port}).",
                      flush=True)
                return
            time.sleep(2)
        raise TimeoutError(f"Teacher did not start within {timeout}s")

    def _start_ray(self):
        """Launch all workers as Ray actors instead of mp.Process."""
        # Build dict aliases from OPDConfig for Ray actor config building.
        # These are minimal dicts with the keys the Ray sub-methods read.
        oc = self.opd_config
        ro = oc.rollout
        t = oc.teacher
        self.teacher_cfg = {
            "model": t.path if t else "",
            "tensor_parallel_size": t.vllm.tensor_parallel_size if t else 1,
            "backend": t.backend if t else "vllm",
            "use_ray_actor": False,  # derived
            "n_dp": 1,
            "n_server_workers": 1,
            "ray_node": t.ray.node if t else None,
            "scoring_batch_size": t.scoring_batch_size if t else 32,
        } if t else {}
        self.train_cfg = {
            "pipeline": {"ray_address": oc.pipeline.ray_address, "backend": "ray"},
            "weight_sync": {
                "backend": oc.weight_sync.backend,
                "nccl_socket_ifname": oc.weight_sync.nccl_socket_ifname,
                "ray_collective": oc.weight_sync.ray_collective,
                "verify_checksum": oc.weight_sync.verify_checksum,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "gpu_ids": ro.gpu_ids if ro else None,
                    "temperature": ro.temperature if ro else 1.0,
                    "top_p": ro.top_p if ro else 1.0,
                    "top_k": ro.top_k if ro else -1,
                    "tensor_model_parallel_size": ro.vllm.tensor_parallel_size if ro else 1,
                    "max_model_len": ro.vllm.max_model_len if ro else None,
                    "max_num_seqs": ro.vllm.max_num_seqs if ro else 512,
                    "gpu_memory_utilization": ro.vllm.gpu_memory_utilization if ro else 0.85,
                    "max_num_batched_tokens": ro.vllm.max_num_batched_tokens if ro else None,
                    "block_size": ro.vllm.block_size if ro else None,
                    "enforce_eager": ro.vllm.enforce_eager if ro else True,
                    "quantization": ro.quantization if ro else None,
                    "backend": ro.backend if ro else "vllm",
                    "dtype": ro.dtype if ro else "auto",
                    "pause_mode": oc.pipeline.fully_async.pause_mode,
                    "ray_node": ro.ray.node if ro else None,
                },
                "actor": {
                    "optim": {"lr": oc.trainer.optim.lr},
                    "lora": None,
                },
            },
            "rollout": {"n_gpus_per_node": (ro.n_gpus or 1) if ro else 1,
                        "gpu_ids": ro.gpu_ids if ro else None},
            "trainer": {
                "n_gpus_per_node": oc.trainer.n_gpus or 1,
                "gpu_ids": oc.trainer.gpu_ids,
                "ray_node": oc.trainer.ray.node,
                "backend": oc.trainer.backend,
                "nccl_timeout_hours": oc.weight_sync.nccl_timeout_hours,
                "tp_size": oc.trainer.megatron.tensor_parallel_size,
                "pp_size": oc.trainer.megatron.pipeline_parallel_size,
            },
            "algorithm": {},
            "mode": oc.algorithm.mode,
        }
        if oc.trainer.lora:
            self.train_cfg["actor_rollout_ref"]["actor"]["lora"] = {
                "rank": oc.trainer.lora.rank, "alpha": oc.trainer.lora.alpha,
                "dropout": oc.trainer.lora.dropout,
                "target_modules": oc.trainer.lora.target_modules,
                "modules_to_save": oc.trainer.lora.modules_to_save,
                "native_lora": oc.trainer.lora.native_lora,
            }
        self.data_cfg = {"train_files": oc.data.train_files}
        import ray as _ray
        from opd.rollout.vllm.ray_actors import VLLMBatchRolloutActor, VLLMStreamingRolloutActor
        from opd.worker.ray_proxy import (
            FSDPTrainerActor, MegatronTrainerActor,
            RayRolloutProxy, RayStreamingRolloutProxy,
            RayTrainerProxy, RayMegatronTrainerProxy,
        )
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ray_address = self.train_cfg.get("pipeline", {}).get("ray_address", "local")
        # NCCL env vars for cross-node weight transfer.
        # NCCL_SOCKET_IFNAME selects the network interface (e.g., "enp" for ethernet).
        # NCCL_IB_DISABLE=1 prevents NCCL from trying InfiniBand when unavailable.
        nccl_env = {}
        ws_cfg = self.train_cfg.get("weight_sync", {})
        nccl_ifname = ws_cfg.get("nccl_socket_ifname")
        if nccl_ifname:
            nccl_env["NCCL_SOCKET_IFNAME"] = nccl_ifname
            nccl_env["NCCL_IB_DISABLE"] = "1"
        if nccl_env:
            os.environ.update(nccl_env)
        runtime_env = {"env_vars": nccl_env} if nccl_env else None
        _ray.init(address=ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
        print(f"[Pipeline] Ray connected: {_ray.cluster_resources()}", flush=True)

        # Resolve this node's routable IP for cross-node ZMQ/NCCL connectivity.
        # socket.gethostbyname may return 127.0.x.x (loopback), so prefer Ray's node IP.
        node_ip = _ray.util.get_node_ip_address()

        # Discover remote nodes for rollout scheduling
        head_node_id = _ray.get_runtime_context().get_node_id()
        all_nodes = _ray.nodes()
        self._remote_node_ids = [
            n["NodeID"] for n in all_nodes
            if n["Alive"] and n["NodeID"] != head_node_id
        ]
        if self._remote_node_ids:
            print(f"[Pipeline] Remote nodes: {len(self._remote_node_ids)} "
                  f"({', '.join(n['NodeManagerAddress'] for n in all_nodes if n['Alive'] and n['NodeID'] != head_node_id)})",
                  flush=True)

        rollout_cfg = self.train_cfg["actor_rollout_ref"]["rollout"]
        tp = rollout_cfg.get("tensor_model_parallel_size", 1)
        # Ray mode: use n_gpus_per_node to determine worker count (no gpu_ids needed).
        n_rollout_gpus = self.train_cfg.get("rollout", {}).get("n_gpus_per_node", 1)
        n_rollout_workers = max(n_rollout_gpus // tp, 1)
        lora_cfg = self.train_cfg.get("actor_rollout_ref", {}).get("actor", {}).get("lora")

        # --- Teacher ---
        teacher_infos = self._start_ray_teacher(
            _ray, head_node_id, node_ip, NodeAffinitySchedulingStrategy)

        # --- Rollout actor(s) ---
        use_async_rollout = self.scheduling_mode == "fully_async"
        rollout_actors = self._start_ray_rollout(
            _ray, head_node_id, node_ip, rollout_cfg, tp, n_rollout_workers,
            use_async_rollout, lora_cfg, VLLMBatchRolloutActor, VLLMStreamingRolloutActor,
            NodeAffinitySchedulingStrategy)

        # --- Trainer actor(s) ---
        trainer_proxy = self._start_ray_trainer(
            _ray, head_node_id, node_ip, FSDPTrainerActor, MegatronTrainerActor,
            RayTrainerProxy, RayMegatronTrainerProxy,
            NodeAffinitySchedulingStrategy)

        # --- Construct proxy objects ---
        if use_async_rollout:
            self.rollout_proxy = RayStreamingRolloutProxy(
                actors=rollout_actors,
                need_student_logprobs=self._need_student_logprobs,
                rollout_support_topk_k=getattr(self, "_rollout_support_topk_k", 0),
            )
        else:
            self.rollout_proxy = RayRolloutProxy(
                actors=rollout_actors,
                need_student_logprobs=self._need_student_logprobs,
                rollout_support_topk_k=getattr(self, "_rollout_support_topk_k", 0),
                prompt_queue=self.rollout_prompt_queue,
            )
        self.trainer_proxy = trainer_proxy

        # Store actor handles for shutdown
        self._ray_rollout_actors = rollout_actors
        # Dummy proc references for shutdown() compatibility
        self.trainer_proc = None
        self.rollout_procs = []

        # --- Wait for actors to initialize ---
        print("[Pipeline] Waiting for Ray actors to initialize...", flush=True)
        time.sleep(2)

        # --- Init weight transfer ---
        if self.use_nccl:
            use_collective = self.train_cfg.get("weight_sync", {}).get(
                "ray_collective", False)  # experimental: Ray collective broadcast
            if use_collective:
                self._init_ray_collective_weight_transfer(
                    self._ray_trainer_actor, self._ray_rollout_actors)
            else:
                self._init_weight_transfer(n_rollout_workers,
                                           master_address=node_ip,
                                           tp_size=tp)

        # --- Teacher client ---
        if isinstance(self._ray_teacher_actor, list):
            # Ray teacher(s): each actor has its own ZMQ endpoint
            teacher_addrs = [f"tcp://{info['ip']}:{info['port']}"
                             for info in teacher_infos]
            self.teacher_client = TeacherClient(teacher_addrs)
        else:
            # Local mp.Process teacher
            self.teacher_client = TeacherClient(
                f"tcp://{node_ip}:{self.teacher_port}",
                n_workers=self.teacher_cfg.get("n_server_workers", 1),
            )
        # Wire up per-worker trace info for TeacherClient
        self.teacher_client.tracer = self.tracer
        if isinstance(self._ray_teacher_actor, list):
            self.teacher_client.teacher_trace_infos = [
                {"host": info["host"], "gpu_ids": info["gpu_ids"]}
                for info in teacher_infos
            ]
        else:
            self.teacher_client.teacher_trace_infos = [
                {"host": self._teacher_host, "gpu_ids": self._teacher_gpu_ids}
            ]

        # Trace metadata for host/gpu_ids in Perfetto spans
        _host = socket.gethostname()
        try:
            t_actor = self._ray_trainer_actor
            if isinstance(t_actor, list):
                t_actor = t_actor[0]
            trainer_info = _ray.get(t_actor.get_worker_info.remote())
            self._trainer_trace_info = trainer_info
        except Exception:
            self._trainer_trace_info = {"host": _host, "gpu_ids": "?"}
        self._teacher_trace_info = {"host": self._teacher_host, "gpu_ids": self._teacher_gpu_ids,
                                     "_remote": self._ray_teacher_actor is not None}

        print("[Pipeline] All Ray actors ready.", flush=True)
        atexit.register(self.shutdown)

    def _start_ray_teacher(self, _ray, head_node_id, node_ip,
                           NodeAffinitySchedulingStrategy):
        """Spawn Ray teacher actor or local teacher process. Returns teacher_infos list."""
        oc = self.opd_config
        teacher_gpus = self._gpu_ids("teacher")
        teacher_tp = oc.teacher.vllm.tensor_parallel_size
        self.teacher_port = find_free_port("ray.teacher.bind")
        teacher_backend = oc.teacher.backend
        use_ray_teacher = bool(oc.teacher.ray.use_ray_actor)

        teacher_fn = get_teacher_backend(teacher_backend)["server_main"]
        teacher_spec = self._build_teacher_launch_spec(teacher_gpus, self.teacher_port)
        if teacher_spec.runtime.bind_address in {"0.0.0.0", "::"}:
            warnings.warn(
                "Ray teacher is configured to bind externally. Ensure the host is on a "
                "trusted network and firewall rules restrict access.",
                UserWarning,
            )

        teacher_infos = []
        if use_ray_teacher:
            # Ray actor(s): teacher can run on any node. Multiple actors = teacher DP.
            from opd.worker.ray_proxy import VLLMTeacherActor
            n_teacher_dp = oc.teacher.ray.n_dp
            RemoteTeacherActor = _ray.remote(VLLMTeacherActor)
            teacher_actors = []
            # ray_node: "head", "remote", or list like ["head", "remote"] for DP
            teacher_ray_node = oc.teacher.ray.node
            if isinstance(teacher_ray_node, str):
                teacher_ray_node = [teacher_ray_node]  # normalize to list
            for t_i in range(n_teacher_dp):
                t_options = dict(num_gpus=teacher_tp, num_cpus=1)
                if teacher_ray_node:
                    node_label = teacher_ray_node[t_i % len(teacher_ray_node)]
                    if node_label == "head":
                        target = head_node_id
                    elif node_label == "remote" and self._remote_node_ids:
                        target = self._remote_node_ids[t_i % len(self._remote_node_ids)]
                    else:
                        target = None
                    if target:
                        t_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                            node_id=target, soft=False)
                actor = RemoteTeacherActor.options(**t_options).remote()
                t_port = find_free_port(f"ray.teacher.{t_i}.bind")
                t_spec = teacher_spec.with_runtime(
                    gpu_ids=",".join(str(g) for g in range(teacher_tp)),
                    bind_port=t_port,
                )
                _ray.get(actor.init.remote(teacher_fn, t_spec))
                info = _ray.get(actor.get_info.remote())
                teacher_actors.append(actor)
                teacher_infos.append(info)
                print(f"[Pipeline] Teacher-{t_i} (Ray actor) on {info['host']}, "
                      f"gpu_ids={info['gpu_ids']}, port={info['port']}", flush=True)
            self._teacher_host = teacher_infos[0]["host"]
            self._teacher_gpu_ids = teacher_infos[0]["gpu_ids"]
            self.teacher_proc = None
            self._ray_teacher_actor = teacher_actors
        else:
            # mp.Process: teacher on head node (default)
            ctx = mp.get_context("spawn")
            self.teacher_proc = ctx.Process(target=teacher_fn, args=(teacher_spec,))
            self.teacher_proc.start()
            self._teacher_host = socket.gethostname()
            self._teacher_gpu_ids = teacher_gpus
            self._ray_teacher_actor = None
            print(f"[Pipeline] Teacher backend: {teacher_backend} (local process, bind={node_ip})",
                  flush=True)

        # Wait for teacher ZMQ port(s) to be ready
        if isinstance(self._ray_teacher_actor, list):
            # Teacher DP: wait for all actors
            print(f"[Pipeline] Waiting for {len(teacher_infos)} teacher(s) to finish loading...",
                  flush=True)
            for info in teacher_infos:
                self._wait_for_teacher_ready(host=info["ip"],
                                             port_override=info["port"])
        elif self._ray_teacher_actor is not None:
            info = _ray.get(self._ray_teacher_actor.get_info.remote())
            print("[Pipeline] Waiting for teacher to finish loading...", flush=True)
            self._wait_for_teacher_ready(host=info["ip"])
        else:
            print("[Pipeline] Waiting for teacher to finish loading...", flush=True)
            self._wait_for_teacher_ready()

        return teacher_infos

    def _start_ray_rollout(self, _ray, head_node_id, node_ip, rollout_cfg, tp,
                           n_rollout_workers, use_async_rollout, lora_cfg,
                           VLLMBatchRolloutActor, VLLMStreamingRolloutActor,
                           NodeAffinitySchedulingStrategy):
        """Spawn Ray rollout actors. Returns list of rollout actor handles."""
        rollout_backend = rollout_cfg.get("backend", "vllm")
        if rollout_backend != "vllm":
            raise ValueError(
                f"Ray mode only supports rollout.backend='vllm', got '{rollout_backend}'. "
                "Use local mode (pipeline.backend='local') for non-vLLM backends.")
        if use_async_rollout:
            RemoteAsyncRolloutActor = _ray.remote(VLLMStreamingRolloutActor)
            print(f"[Pipeline] Using native async rollout actors ({self.scheduling_mode} mode)", flush=True)
        else:
            from opd.rollout.vllm.batch import VLLMBatchRolloutWorker
            worker_cls = VLLMBatchRolloutWorker
            RemoteRolloutActor = _ray.remote(VLLMBatchRolloutActor)

        # Shared prompt queue for fully_async mode
        # Ray actors manage their own local prompt queues — we just pass a
        # sentinel value so the rollout config detects streaming mode.
        use_prompt_queue = self.scheduling_mode == "fully_async"
        if use_prompt_queue:
            # Pass a marker so VLLMBatchRolloutActor creates its internal prompt queue.
            # The actual queue lives inside each actor (not shared across processes).
            self.rollout_prompt_queue = True  # sentinel, not an actual queue
        else:
            self.rollout_prompt_queue = None

        mem_util = rollout_cfg.get("gpu_memory_utilization", 0.5)

        rollout_actors = []
        all_worker_specs = []  # saved for async actor init
        for i in range(n_rollout_workers):
            # Ray mode: gpu_ids=None → worker skips CUDA_VISIBLE_DEVICES override.
            # Ray sets it via num_gpus=tp allocation.
            prompt_queue = self.rollout_prompt_queue if use_prompt_queue else None
            pause_mode = rollout_cfg.get("pause_mode", "keep") if use_async_rollout else None
            rollout_spec = self._build_rollout_launch_spec(
                rollout_cfg, None, tp, i, mem_util, lora_cfg,
                prompt_queue=prompt_queue, pause_mode=pause_mode,
            )

            all_worker_specs.append(rollout_spec)

            # num_gpus=tp so Ray tracks GPU allocation.
            actor_options = dict(num_gpus=tp, num_cpus=1)
            rollout_ray_node = rollout_cfg.get("ray_node")  # "head", "remote", or list
            if isinstance(rollout_ray_node, str):
                rollout_ray_node = [rollout_ray_node]
            if rollout_ray_node:
                node_label = rollout_ray_node[i % len(rollout_ray_node)]
                if node_label == "head":
                    target = head_node_id
                elif node_label == "remote" and self._remote_node_ids:
                    target = self._remote_node_ids[i % len(self._remote_node_ids)]
                else:
                    target = None
                if target:
                    actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                        node_id=target, soft=False)

            if use_async_rollout:
                actor = RemoteAsyncRolloutActor.options(**actor_options).remote()
            else:
                actor = RemoteRolloutActor.options(**actor_options).remote(worker_cls, rollout_spec)
            rollout_actors.append(actor)

        # Initialize async actors (engine init must happen inside the actor)
        if use_async_rollout:
            init_refs = [
                rollout_actors[i].init.remote(all_worker_specs[i])
                for i in range(n_rollout_workers)
            ]
            _ray.get(init_refs)
            print(f"[Pipeline] {n_rollout_workers} async rollout actor(s) initialized", flush=True)

        # Store per-worker trace metadata (host, gpu_ids) for trace spans.
        # In Ray mode, query actors; in local mode, use config gpu_ids.
        self._rollout_worker_info = []
        for i in range(n_rollout_workers):
            self._rollout_worker_info.append({
                "host": node_ip,  # updated below for Ray remote actors
                "gpu_ids": "",
            })
        if rollout_actors:
            import ray as _r
            for i, actor in enumerate(rollout_actors):
                try:
                    info = _r.get(actor.get_worker_info.remote())
                    self._rollout_worker_info[i] = info
                except Exception:
                    pass  # fallback to defaults

        return rollout_actors

    def _start_ray_trainer(self, _ray, head_node_id, node_ip,
                           FSDPTrainerActor, MegatronTrainerActor,
                           RayTrainerProxy, RayMegatronTrainerProxy,
                           NodeAffinitySchedulingStrategy):
        """Spawn Ray trainer actors. Returns trainer_proxy."""
        trainer_gpus = self._gpu_ids("trainer")
        trainer_static = self._build_trainer_launch_static()
        algo_payload = serialize_algorithm_payload(trainer_static.algorithm)
        self._apply_rollout_logprob_flags(
            algo_payload["kl_loss_mode"]
        )

        trainer_backend = self._get_backend()

        if trainer_backend == "megatron":
            # Megatron TP×PP×DP: one Ray actor per global rank
            from opd.utils.net import find_free_port as _find_free_port
            trainer_gpu_list = trainer_gpus.split(",")
            n_trainer_gpus = len(trainer_gpu_list)
            tp_size = trainer_static.megatron["tp_size"]
            pp_size = trainer_static.megatron["pp_size"]
            dp_size = n_trainer_gpus // (tp_size * pp_size)
            global_world_size = tp_size * pp_size * dp_size
            megatron_master_port = _find_free_port("ray.trainer.megatron_master")

            RemoteMegatronTrainerActor = _ray.remote(MegatronTrainerActor)
            megatron_actors = []
            trainer_cfg = self.train_cfg.get("trainer", {})
            trainer_ray_node = trainer_cfg.get("ray_node")
            if isinstance(trainer_ray_node, str):
                trainer_ray_node = [trainer_ray_node]

            def _resolve_trainer_node(label, rank):
                if label == "head":
                    return head_node_id
                if label == "remote":
                    if not self._remote_node_ids:
                        raise RuntimeError(
                            "trainer.ray.node requested a remote Megatron rank "
                            "but Ray reported no remote nodes"
                        )
                    return self._remote_node_ids[rank % len(self._remote_node_ids)]
                return None

            for global_rank in range(global_world_size):
                tp_rank = global_rank % tp_size
                pp_rank = (global_rank // tp_size) % pp_size
                rank_info = self._build_megatron_rank_info(
                    global_rank, tp_rank, pp_rank,
                    global_world_size, megatron_master_port)
                trainer_spec = self._build_trainer_launch_spec(rank_info, gpu_ids=None)
                actor_options = dict(num_gpus=1, num_cpus=1)
                if trainer_ray_node:
                    node_label = trainer_ray_node[global_rank % len(trainer_ray_node)]
                    target_node = _resolve_trainer_node(node_label, global_rank)
                else:
                    target_node = head_node_id
                if target_node:
                    actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                        node_id=target_node, soft=False)
                actor = RemoteMegatronTrainerActor.options(**actor_options).remote(
                    trainer_spec, None)
                megatron_actors.append(actor)

            # Deferred init: get rank-0's IP for multi-node NCCL, then start all
            master_ip = _ray.get(megatron_actors[0].get_node_ip.remote())
            init_futures = [
                a.init.remote(megatron_master_addr=master_ip)
                for a in megatron_actors
            ]
            _ray.get(init_futures)
            try:
                rank_infos = _ray.get([a.get_worker_info.remote() for a in megatron_actors])
                unique_hosts = sorted({info.get("host", "?") for info in rank_infos})
                rank_summary = ", ".join(
                    f"rank{info.get('global_rank', i)}@{info.get('host', '?')}"
                    f"[gpu={info.get('gpu_ids', '?')}]"
                    for i, info in enumerate(rank_infos)
                )
                print(f"[Pipeline] Megatron trainer Ray ranks: {rank_summary}", flush=True)
                print(
                    f"[Pipeline] Megatron trainer spans {len(unique_hosts)} node(s): "
                    f"{', '.join(unique_hosts)}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Pipeline] WARNING: could not query Megatron Ray rank placement: {exc}",
                      flush=True)

            trainer_proxy = RayMegatronTrainerProxy(actors=megatron_actors)
            self._trainer_fsdp_procs = []
            self._ray_trainer_actor = megatron_actors  # list for shutdown
            self._n_trainer_gpus = n_trainer_gpus
            print(f"[Pipeline] Megatron trainer (Ray): TP={tp_size}, PP={pp_size}, "
                  f"DP={dp_size}, {global_world_size} ranks, "
                  f"master={master_ip}:{megatron_master_port}", flush=True)
        else:
            # FSDP: single or multi-node Ray actor(s)
            from opd.utils.net import find_free_port as _find_free_port
            RemoteTrainerActor = _ray.remote(FSDPTrainerActor)

            trainer_cfg = self.train_cfg.get("trainer", {})
            trainer_n_gpus = trainer_cfg.get("n_gpus_per_node", 1)
            trainer_ray_node = trainer_cfg.get("ray_node")
            if isinstance(trainer_ray_node, str):
                trainer_ray_node = [trainer_ray_node]

            # Compute FSDP world size
            if trainer_ray_node and len(trainer_ray_node) > 1:
                fsdp_world_size = len(trainer_ray_node) * trainer_n_gpus
            elif trainer_n_gpus > 1:
                fsdp_world_size = trainer_n_gpus
            else:
                fsdp_world_size = 1

            trainer_cls = self._get_ray_trainer_cls()

            if fsdp_world_size == 1:
                # Single-GPU: single actor, no pg_store needed
                rank_info = self._build_fsdp_rank_info(0, 1, None)
                trainer_spec = self._build_trainer_launch_spec(rank_info, gpu_ids=None)
                trainer_actor = RemoteTrainerActor.options(
                    num_gpus=1, num_cpus=1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=head_node_id, soft=False),
                ).remote(trainer_cls, trainer_spec, None)
                print(f"[Pipeline] Trainer pinned to head node {node_ip}", flush=True)
                trainer_proxy = RayTrainerProxy(actor=trainer_actor)
                self._ray_trainer_actor = trainer_actor
            else:
                # Multi-GPU FSDP: one actor per rank, deferred init
                node_ids = []
                if trainer_ray_node:
                    for label in trainer_ray_node:
                        if label == "head":
                            node_ids.append(head_node_id)
                        elif label == "remote" and self._remote_node_ids:
                            node_ids.append(self._remote_node_ids[0])
                else:
                    node_ids = [head_node_id]

                fsdp_master_port = _find_free_port("ray.trainer.fsdp_master")

                # Phase 1: Create all lightweight actors in parallel
                trainer_actors = []
                rank = 0
                for node_id in node_ids:
                    for _ in range(trainer_n_gpus):
                        rank_info = self._build_fsdp_rank_info(
                            rank, fsdp_world_size, fsdp_master_port)
                        trainer_spec = self._build_trainer_launch_spec(rank_info, gpu_ids=None)
                        actor = RemoteTrainerActor.options(
                            num_gpus=1, num_cpus=1,
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                node_id=node_id, soft=False),
                        ).remote(trainer_cls, trainer_spec, None)
                        trainer_actors.append(actor)
                        rank += 1

                # Phase 2: Discover rank-0 IP + init all actors in parallel
                fsdp_master_addr = _ray.get(trainer_actors[0].get_node_ip.remote())
                _ray.get([a.init.remote(fsdp_master_addr, fsdp_master_port)
                          for a in trainer_actors])
                print(f"[Pipeline] Multi-GPU FSDP trainer: {fsdp_world_size} ranks "
                      f"across {len(node_ids)} node(s), master={fsdp_master_addr}:{fsdp_master_port}",
                      flush=True)

                trainer_proxy = RayTrainerProxy(actors=trainer_actors)
                self._ray_trainer_actor = trainer_actors  # list for shutdown

            self._trainer_fsdp_procs = []

        return trainer_proxy

    def _init_ray_collective_weight_transfer(self, trainer_actors, rollout_actors):
        """Initialize Ray collective broadcast weight transfer.

        Creates a NCCL collective group between trainer rank 0 and rollout
        workers, bypassing vLLM's weight transfer engine entirely.
        """
        from opd.worker.ray_weight_sync import RayCollectiveWeightSyncEngine
        self.weight_engine = RayCollectiveWeightSyncEngine(
            verify_checksum=self.verify_weight_sync,
        )
        # For Megatron with multiple actors, only rank 0 participates in collective
        if isinstance(trainer_actors, list):
            trainer_for_collective = [trainer_actors[0]]
        else:
            trainer_for_collective = [trainer_actors]
        if not isinstance(rollout_actors, list):
            rollout_actors = [rollout_actors]
        self.weight_engine.initialize(
            self.trainer_proxy, self.rollout_proxy,
            trainer_actors=trainer_for_collective,
            rollout_actors=rollout_actors,
        )

    def _init_weight_transfer(self, n_rollout_workers,
                              master_address="127.0.0.1", tp_size=1):
        """Initialize vLLM's native NCCL weight transfer engine."""
        self.weight_engine = NCCLWeightSyncEngine(
            verify_checksum=self.verify_weight_sync,
        )
        # Native LoRA: build peft_config for weight sync
        native_lora = self._native_lora
        lora_cfg = None
        if native_lora:
            oc = self.opd_config
            lora_dc = oc.trainer.lora
            lora_cfg = {
                "rank": lora_dc.rank, "alpha": lora_dc.alpha,
                "dropout": lora_dc.dropout, "target_modules": lora_dc.target_modules,
                "modules_to_save": lora_dc.modules_to_save,
                "native_lora": lora_dc.native_lora,
            }
        peft_config = None
        if native_lora:
            from opd.rollout.vllm.lora import build_peft_config_dict
            peft_config = build_peft_config_dict(lora_cfg)
        self.weight_engine.initialize(self.trainer_proxy, self.rollout_proxy,
                                      master_address=master_address,
                                      tp_size=tp_size,
                                      lora_mode=native_lora,
                                      peft_config=peft_config)
