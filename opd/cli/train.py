#!/usr/bin/env python3
"""On-Policy Distillation Pipeline.

Installed usage:
    opd-train --config configs/examples/opd_qwen3_1.7b.yaml

Module usage:
    python -m opd.cli.train --config configs/examples/opd_qwen3_1.7b.yaml
"""

import argparse
import faulthandler
import json
import os
import resource
import signal
import subprocess
import sys
import threading

_tee_threads = []
_TARGET_NOFILE_SOFT_LIMIT = 650000


def _setup_log_tee(log_path, append=False):
    """Duplicate stdout/stderr to a log file at the fd level.

    All output — print(), subprocess, C extensions — goes to both the
    original destination and the log file. Works by replacing fd 1/2 with
    pipe write-ends and spawning a thread that reads the pipe and writes
    to both the original fd and the log file.
    """
    mode = "ab" if append else "wb"
    log_file = open(log_path, mode)

    def _tee_thread(read_fd, orig_fd, log_f):
        """Read from pipe, write to original fd and log file."""
        try:
            while True:
                data = os.read(read_fd, 8192)
                if not data:
                    break
                os.write(orig_fd, data)
                log_f.write(data)
                log_f.flush()
        except OSError:
            pass
        finally:
            os.close(read_fd)
            os.close(orig_fd)

    # Save original fds
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)

    # Create pipes
    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()

    # Replace stdout/stderr fds with pipe write-ends
    os.dup2(w_out, 1)
    os.dup2(w_err, 2)
    os.close(w_out)
    os.close(w_err)

    # Update Python-level stdout/stderr to use the new fds
    sys.stdout = os.fdopen(1, "w", buffering=1)  # line-buffered
    sys.stderr = os.fdopen(2, "w", buffering=1)

    # Spawn reader threads (daemon so os._exit kills them)
    t1 = threading.Thread(target=_tee_thread, args=(r_out, orig_stdout_fd, log_file), daemon=True)
    t2 = threading.Thread(target=_tee_thread, args=(r_err, orig_stderr_fd, log_file), daemon=True)
    t1.start()
    t2.start()
    _tee_threads.extend([t1, t2])


def _flush_log_tee():
    """Flush and drain tee threads so all output reaches the log file.

    Call before os._exit() to ensure final messages are written.
    """
    if not _tee_threads:
        return
    # Flush Python buffers
    sys.stdout.flush()
    sys.stderr.flush()
    # Close pipe write-ends (fd 1, 2) so reader threads see EOF and drain
    os.close(1)
    os.close(2)
    # Wait for threads to finish draining (short timeout to avoid NCCL hangs)
    for t in _tee_threads:
        t.join(timeout=2)


def _derive_run_dir(config_path):
    """Derive output directory from config path.

    configs/examples/opd_qwen3_1.7b.yaml
      -> results/examples/opd_qwen3_1.7b/
    """
    rel = config_path
    for prefix in ("configs/", "configs\\"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    base, _ = os.path.splitext(rel)
    return os.path.join("results", base)


def _free_target_gpus(gpu_ids):
    """Kill any processes currently using the target GPUs."""
    if not gpu_ids:
        return
    # Map GPU indices to UUIDs
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return
    target_uuids = set()
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        idx, uuid = line.split(",", 1)
        if int(idx.strip()) in gpu_ids:
            target_uuids.add(uuid.strip())

    # Find PIDs on those GPUs
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return
    my_pid = os.getpid()
    pids = set()
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",", 1)
        pid = int(parts[0].strip())
        if pid == my_pid:
            continue
        if len(parts) > 1 and parts[1].strip() in target_uuids:
            pids.add(pid)

    if pids:
        gpu_str = ",".join(str(g) for g in sorted(gpu_ids))
        print(f"[Pipeline] Killing {len(pids)} stale processes on GPUs [{gpu_str}]: {sorted(pids)}", flush=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _git_commit():
    """Return current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_is_dirty():
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _is_debug_config(config_path):
    """Return True if config is under 000_debug/ or 000_test*/ directories."""
    rel = config_path
    for prefix in ("configs/", "configs\\"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return rel.startswith("000_debug/") or rel.startswith("000_test")


def _raise_nofile_limit(target_soft_limit=_TARGET_NOFILE_SOFT_LIMIT):
    """Raise RLIMIT_NOFILE soft limit up to the target when possible.

    Returns (old_soft, old_hard, new_soft, changed).
    """
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(old_hard, max(old_soft, target_soft_limit))
    if new_soft <= old_soft:
        return old_soft, old_hard, old_soft, False
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, old_hard))
    return old_soft, old_hard, new_soft, True


def main():
    # Ignore SIGHUP so the process survives SSH disconnects.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Dump Python traceback on SIGSEGV/SIGABRT/SIGFPE — writes directly to
    # fd 2 (unbuffered), catches C-level crashes that exception handling misses.
    # all_threads=True dumps all threads, not just the crashing one.
    faulthandler.enable(all_threads=True)

    # Always show C++ stack traces and NCCL warnings for crash debugging.
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    try:
        old_soft, old_hard, new_soft, changed = _raise_nofile_limit()
        if changed:
            print(f"[Pipeline] RLIMIT_NOFILE raised {old_soft} -> {new_soft} "
                  f"(hard={old_hard})", flush=True)
    except Exception as e:
        print(f"[Pipeline] WARNING: failed to raise RLIMIT_NOFILE: {e}", flush=True)

    parser = argparse.ArgumentParser(description="On-Policy Distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in results dir")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and run post_allgpu evaluation for existing checkpoints")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Allow running with uncommitted changes (auto-allowed for 000_debug/000_test* configs)")
    parser.add_argument("--log", nargs="*", default=None, metavar="BACKEND",
                        help="Enable remote logging backends (clearml, wandb, aim)")
    parser.add_argument("--set", nargs="*", default=None,
                        help="Override config values: --set trainer.optim.lr=2e-5 eval.n_samples=1")
    args = parser.parse_args()

    if args.eval_only and args.overwrite:
        parser.error("--eval-only cannot be combined with --overwrite because overwrite removes checkpoints")

    # Enforce clean tree for non-debug experiments
    git_commit = _git_commit()
    if git_commit and _git_is_dirty() and not args.allow_dirty and not _is_debug_config(args.config):
        parser.error("Working tree is dirty. Commit changes before running experiments.\n"
                     "  Use --allow-dirty to override, or use a 000_debug/000_test* config.")

    # Import runtime-heavy OPD modules only after argparse has handled --help.
    # This keeps packaged console-script help usable in lightweight wheel smoke
    # environments that do not install the full GPU/runtime dependency stack.
    from opd.utils.config import OPDConfig
    from opd.utils.logger import Logger
    from opd.utils.post_eval import collect_gpu_ids, run_allgpu_post_eval
    from opd.pipeline import create_coordinator

    opd_config = OPDConfig.from_yaml(args.config, overrides=getattr(args, 'set', None))

    # Build a simple dict for Logger provenance (replaces to_internal_dict)
    import dataclasses
    config = dataclasses.asdict(opd_config)
    if git_commit:
        config["git_commit"] = git_commit

    _free_target_gpus(collect_gpu_ids(opd_config))

    run_dir = _derive_run_dir(args.config)

    # Auto-log: duplicate stdout/stderr to run.log at fd level.
    # All output (print, subprocess, C extensions) goes to both screen and file.
    # Works because subprocesses inherit file descriptors.
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    _setup_log_tee(log_path, append=args.resume or args.eval_only)
    # Run name for remote backends: e.g. "001_first_real_test/run_3gpu_one_step_off"
    run_name = os.path.relpath(run_dir, "results")
    results_path = os.path.join(run_dir, "log.jsonl")
    if args.resume:
        # Inject resume_from into OPDConfig for pipeline to pick up
        opd_config.trainer.resume_from = "latest"
    elif os.path.exists(results_path) and not args.overwrite and not args.eval_only:
        parser.error(f"Results file already exists: {results_path}\n"
                     f"  Use --overwrite to replace it.")

    if args.overwrite and os.path.exists(run_dir):
        # Clean stale artifacts that would confuse --resume / post-eval
        import shutil
        for subdir in ("validation_outputs", "checkpoints"):
            p = os.path.join(run_dir, subdir)
            if os.path.exists(p):
                shutil.rmtree(p)
                print(f"[Overwrite] Removed {p}")
        # Remove eval.jsonl so post-eval doesn't skip already-evaluated steps
        eval_jsonl = os.path.join(run_dir, "eval.jsonl")
        if os.path.exists(eval_jsonl):
            os.remove(eval_jsonl)
            print(f"[Overwrite] Removed {eval_jsonl}")

    logger = Logger(
        results_path=results_path,
        config=config,
        backends=args.log,
        run_name=run_name,
        resume=args.resume or args.eval_only,
    )

    eval_modes = list(opd_config.eval.mode)

    # Skip full pipeline if training already done and only post-eval remains.
    # On --resume, check log.jsonl for completion before spawning teacher/rollout/trainer.
    training_ok = False
    pipeline = None
    if args.eval_only:
        print("[Pipeline] --eval-only: skipping training and running requested post-eval.",
              flush=True)
        training_ok = True
    elif args.resume and os.path.exists(results_path):
        total_steps = opd_config.trainer.total_steps
        last_step = 0
        with open(results_path) as f:
            for line in f:
                if '"type": "train"' in line:
                    last_step = json.loads(line).get("step", last_step)
        if last_step >= total_steps:
            print(f"[Pipeline] Training already complete ({last_step}/{total_steps}), "
                  f"skipping to post-eval.", flush=True)
            training_ok = True

    if not training_ok:
        pipeline = create_coordinator(opd_config, logger=logger, run_dir=run_dir)
        try:
            pipeline.start()
            pipeline.run()
            # Verify training actually reached total_steps (daemon thread crashes
            # in streaming mode can cause run() to return without error).
            total_steps = opd_config.trainer.total_steps
            last_step = 0
            if os.path.exists(results_path):
                with open(results_path) as f:
                    for line in f:
                        if '"type": "train"' in line:
                            last_step = json.loads(line).get("step", last_step)
            if last_step < total_steps:
                # Check if final checkpoint exists — step metrics can be lost
                # when a checkpoint save coincides with the final train step.
                final_ckpt = os.path.join(run_dir, "checkpoints", f"step_{total_steps}")
                if os.path.isdir(final_ckpt):
                    print(f"[Pipeline] Training step {total_steps} metrics missing from log "
                          f"(last logged: {last_step}), but checkpoint exists. "
                          f"Treating as complete.", flush=True)
                    training_ok = True
                else:
                    print(f"[Pipeline] Training incomplete: {last_step}/{total_steps} steps.",
                          flush=True)
            else:
                training_ok = True
        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted.", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("[Pipeline] Training failed.", flush=True)
        finally:
            pipeline._wait_checkpoint_save()  # drain final checkpoint before trace save
            pipeline.stop_trace_monitors()
            trace_out = os.path.join(run_dir, "trace.json")
            pipeline.tracer.save(trace_out)
            pipeline.shutdown()
            print("[Pipeline] Shutdown complete.", flush=True)

    # All-GPU post-eval: runs after shutdown so all GPUs are free.
    # Only if training completed successfully — errors must stay at bottom of log.
    eval_ok = True
    tracer = pipeline.tracer if pipeline is not None else None
    if "post_allgpu" in eval_modes and training_ok:
        try:
            run_allgpu_post_eval(opd_config, args.config, run_dir, logger,
                                  tracer=tracer)
            # Re-save trace with eval events included
            if tracer:
                tracer.save(os.path.join(run_dir, "trace.json"))
        except Exception as e:
            print(f"[AllGPU-PostEval] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            eval_ok = False

    try:
        logger.close()
        _flush_log_tee()
    except Exception:
        pass
    # Force-exit: NCCL/torch.distributed background threads can hang
    # indefinitely on futex_wait after clean shutdown, preventing the
    # process from exiting and blocking queue scripts.
    # Exit codes: 0=all done, 1=training failed, 2=training ok but post-eval failed
    if training_ok and eval_ok:
        os._exit(0)
    elif training_ok:
        os._exit(2)
    else:
        os._exit(1)


if __name__ == "__main__":
    main()
