"""Evaluation and validation data loading for coordinator."""

import json
import os

from opd.data.prompt import ValDataset, val_collate_fn
from opd.utils.eval import (
    extract_answer,
    answers_match,
    score_problems,
    should_try_full_response_match,
)
from torch.utils.data import DataLoader


class EvaluationMixin:
    """Evaluation and validation data loading for coordinator.

    Required attributes from host class:
        self.opd_config, self.max_prompt_length,
        self.batch_size, self.run_dir, self._init_tokenizer()
    """

    def _val_data_iterator(self):
        """Iterate over validation data, yielding batches with ground_truth."""
        tokenizer = self._init_tokenizer()
        oc = getattr(self, 'opd_config', None)
        val_files = oc.data.val_files
        if not val_files:
            print("[Pipeline] No val_files configured, skipping eval.", flush=True)
            return
        dataset = ValDataset(
            val_files, tokenizer, self.max_prompt_length,
            prompt_key=oc.data.prompt_key,
            answer_key=oc.data.answer_key or "auto",
            prompt_template=oc.data.prompt_template,
            enable_thinking=oc.data.enable_thinking,
        )
        val_batch_size = oc.eval.batch_size or self.batch_size
        loader = DataLoader(dataset, batch_size=val_batch_size,
                            shuffle=False, drop_last=False, collate_fn=val_collate_fn)
        for batch in loader:
            yield batch

    @staticmethod
    def _extract_answer_default(text):
        return extract_answer(text)

    answers_match = staticmethod(answers_match)

    def _init_answer_extractor(self):
        """Set up self.extract_answer using answer_pattern from config if available."""
        oc = getattr(self, 'opd_config', None)
        pattern = oc.algorithm.reward.answer_pattern
        if pattern:
            self.extract_answer = lambda text: extract_answer(text, pattern=pattern)
        else:
            self.extract_answer = self._extract_answer_default

    def _evaluate(self, global_step):
        """Run evaluation on the validation set.

        Supports two modes:

          - eval_n_samples=1 (default): greedy decoding, simple accuracy
          - eval_n_samples>1: vLLM n= parameter for parallel sampling, Avg@N metric
        """
        self._init_answer_extractor()
        tokenizer = self._init_tokenizer()
        oc = getattr(self, 'opd_config', None)
        n_samples = oc.eval.n_samples
        eval_temperature = oc.eval.temperature
        eval_max_resp = oc.eval.max_response_length
        answer_pattern = oc.algorithm.reward.answer_pattern

        # Prepare validation output file
        val_file = None
        if self.run_dir:
            val_dir = os.path.join(self.run_dir, "validation_outputs")
            os.makedirs(val_dir, exist_ok=True)
            val_path = os.path.join(val_dir, f"step_{global_step}.jsonl")
            val_file = open(val_path, "w")

        problem_results = []
        sample_count = 0

        try:
            if n_samples == 1:
                # Greedy decoding — single sample per prompt
                for batch in self._val_data_iterator():
                    ground_truths = batch.pop("ground_truth")
                    batch["eval"] = True
                    if eval_max_resp:
                        batch["max_response_length"] = eval_max_resp
                    self._async_generate(batch)
                    gen_out = self._wait_generate()

                    responses = gen_out["responses"]
                    for i in range(responses.size(0)):
                        resp_text = tokenizer.decode(responses[i], skip_special_tokens=True)
                        predicted = self.extract_answer(resp_text)
                        gt_raw = ground_truths[i].strip()
                        gt = self.extract_answer(gt_raw) or gt_raw
                        is_correct = self.answers_match(predicted, gt)
                        if (
                            not is_correct
                            and answer_pattern is None
                            and should_try_full_response_match(gt_raw)
                        ):
                            is_correct = self.answers_match(resp_text, gt_raw)
                        problem_results.append({"gt": gt, "n_correct": int(is_correct), "n_total": 1})
                        sample_count += 1

                        if val_file:
                            val_file.write(json.dumps({
                                "problem_id": len(problem_results) - 1,
                                "ground_truth": gt, "predicted": predicted,
                                "correct": is_correct, "response": resp_text,
                            }) + "\n")

                        if len(problem_results) <= 3:
                            resp_preview = resp_text[:200].replace('\n', ' ')
                            print(f"  [Eval sample {len(problem_results)-1}] gt={gt!r} "
                                  f"pred={predicted!r} resp={resp_preview!r}", flush=True)
            else:
                # Avg@N: use vLLM n= parameter for parallel sampling (single call)
                for batch in self._val_data_iterator():
                    ground_truths = batch.pop("ground_truth")
                    batch["eval_n_samples"] = n_samples
                    batch["eval_temperature"] = eval_temperature
                    if eval_max_resp:
                        batch["max_response_length"] = eval_max_resp
                    self._async_generate(batch)
                    gen_out = self._wait_generate()

                    responses_multi = gen_out["responses_multi"]
                    for i, samples in enumerate(responses_multi):
                        gt_raw = ground_truths[i].strip()
                        gt = self.extract_answer(gt_raw) or gt_raw
                        n_correct = 0
                        for s_idx, resp_ids in enumerate(samples):
                            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                            predicted = self.extract_answer(resp_text)
                            is_correct = self.answers_match(predicted, gt)
                            if (
                                not is_correct
                                and answer_pattern is None
                                and should_try_full_response_match(gt_raw)
                            ):
                                is_correct = self.answers_match(resp_text, gt_raw)
                            n_correct += int(is_correct)
                            sample_count += 1

                            if val_file:
                                val_file.write(json.dumps({
                                    "problem_id": len(problem_results),
                                    "sample_idx": s_idx,
                                    "ground_truth": gt, "predicted": predicted,
                                    "correct": is_correct, "response": resp_text,
                                }) + "\n")

                        problem_results.append({"gt": gt, "n_correct": n_correct, "n_total": len(samples)})

                        if len(problem_results) <= 3:
                            print(f"  [Eval problem {len(problem_results)-1}] gt={gt!r} "
                                  f"pass_rate={n_correct}/{len(samples)}", flush=True)
        finally:
            if val_file:
                val_file.close()

        if not problem_results:
            print(f"[Eval @ step {global_step}] No validation samples.", flush=True)
            return

        metrics = score_problems(problem_results, n_samples)
        if n_samples == 1:
            print(f"[Eval @ step {global_step}] Accuracy: {metrics['correct']}/{metrics['total']} = "
                  f"{metrics['accuracy']:.2f}%", flush=True)
        else:
            avg_key = f"avg_at_{n_samples}"
            print(f"[Eval @ step {global_step}] Avg@{n_samples}: {metrics[avg_key]:.2f}% "
                  f"({metrics['correct']}/{metrics['total_samples']} correct, "
                  f"{metrics['n_problems']} problems)", flush=True)

        if self.logger:
            self.logger.log_eval(global_step, metrics)

    def _run_post_eval(self, tracer, test_freq, val_before_train=False):
        """Evaluate all saved checkpoints after training completes.

        Loads each checkpoint into the trainer, syncs weights to rollout
        workers, and runs eval. Results are logged to the same log.jsonl.
        Workers must be in standard dispatch mode (not autonomous).

        If val_before_train is set, evaluates the base model (step 0) first
        before any checkpoint loading.
        """
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")

        # Discover checkpoint steps in order
        steps = []
        if os.path.exists(ckpt_dir):
            for d in os.listdir(ckpt_dir):
                if d.startswith("step_"):
                    try:
                        steps.append(int(d.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
        steps.sort()

        if not steps and not val_before_train:
            print("[PostEval] No checkpoints found, skipping.", flush=True)
            return

        # Skip step 0 (base model): no checkpoint to restore, rollout has trained weights.
        # Skip steps that already have validation outputs (avoids duplicates on --resume).
        val_dir = os.path.join(self.run_dir, "validation_outputs")
        eval_steps = []
        for s in steps:
            val_file = os.path.join(val_dir, f"step_{s}.jsonl")
            if os.path.exists(val_file) and os.path.getsize(val_file) > 0:
                continue  # already evaluated
            # Remove empty/corrupt validation file so eval rewrites it
            if os.path.exists(val_file):
                os.remove(val_file)
                print(f"[PostEval] Removed empty validation file: {val_file}",
                      flush=True)
            eval_steps.append(s)

        if not eval_steps:
            print("[PostEval] All checkpoints already evaluated, skipping.", flush=True)
            return

        print(f"[PostEval] Evaluating {len(eval_steps)} steps: {eval_steps}",
              flush=True)

        # Transition trainer from multi-GPU FSDP to single-GPU mode so that
        # load_checkpoint / sync_weights don't need FSDP collectives with
        # ranks that may have died or timed out.
        if len(getattr(self, '_trainer_fsdp_procs', [])) > 1:
            self._wait_checkpoint_save()
            self.trainer_proxy.submit_command("finalize_fsdp")
            for p in self._trainer_fsdp_procs[1:]:
                p.join(timeout=10)
            print("[PostEval] FSDP peers shut down, trainer in single-GPU mode",
                  flush=True)

        for step in eval_steps:
            if step > 0:
                checkpoint_path = os.path.join(ckpt_dir, f"step_{step}")
                print(f"[PostEval] Loading checkpoint step {step}...",
                      flush=True)
                self._load_checkpoint(checkpoint_path)

            with tracer.span("eval", cat="eval", tid=self.TID_EVAL) as ev:
                ev["step"] = step
                self._evaluate(step)

        print(f"[PostEval] Done — evaluated {len(eval_steps)} checkpoints.",
              flush=True)
