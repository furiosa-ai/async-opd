"""Shared OPDConfig builders for test mocks.

Provides minimal OPDConfig objects that match the dict structures
used by test_streaming_coordinator, test_step_off_coordinator,
test_evaluate, test_coordinator_base, test_checkpoint, etc.
"""

from opd.utils.config import (
    OPDConfig, ModelConfig, TeacherConfig, DataConfig, RolloutConfig,
    TrainerConfig, AlgorithmConfig, PipelineConfig, NStepOffConfig,
    FullyAsyncConfig, EvalConfig, WeightSyncConfig, OptimConfig,
    OPDAlgorithmConfig, GRPOAlgorithmConfig, RewardConfig,
)


def make_test_opd_config(
    *,
    model_path="test-student",
    teacher_path="test-teacher",
    teacher_gpu_ids="0",
    train_files="dummy",
    val_files="dummy_val",
    prompt_template="{problem}",
    answer_key="answer",
    max_prompt_length=128,
    max_response_length=128,
    rollout_gpu_ids="1",
    trainer_gpu_ids="1",
    n_gpus=1,
    batch_size=4,
    total_steps=10,
    total_epochs=1,
    step_off=0,
    scheduling_mode="n_step_off",
    staleness_threshold=2,
    test_freq=-1,
    save_freq=-1,
    val_before_train=True,
    resume_from=None,
    eval_mode=None,
    scoring_batch_size=None,
    mini_batch_size=None,
    enable_thinking=False,
    mode="opd",
):
    """Build a minimal OPDConfig for test coordinator mocks."""
    if eval_mode is None:
        eval_mode = ["inline"]

    teacher = TeacherConfig(path=teacher_path, gpu_ids=teacher_gpu_ids)
    if scoring_batch_size is not None:
        teacher.scoring_batch_size = scoring_batch_size

    trainer = TrainerConfig(
        gpu_ids=trainer_gpu_ids, n_gpus=n_gpus,
        batch_size=batch_size, total_steps=total_steps,
        total_epochs=total_epochs, save_freq=save_freq,
        resume_from=resume_from,
    )
    if mini_batch_size is not None:
        trainer.mini_batch_size = mini_batch_size

    return OPDConfig(
        model=ModelConfig(path=model_path),
        teacher=teacher,
        data=DataConfig(
            train_files=train_files, val_files=val_files,
            prompt_template=prompt_template, answer_key=answer_key,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            enable_thinking=enable_thinking,
        ),
        rollout=RolloutConfig(gpu_ids=rollout_gpu_ids),
        trainer=trainer,
        algorithm=AlgorithmConfig(mode=mode),
        pipeline=PipelineConfig(
            scheduling_mode=scheduling_mode,
            n_step_off=NStepOffConfig(step_off=step_off),
            fully_async=FullyAsyncConfig(staleness_threshold=staleness_threshold),
        ),
        eval=EvalConfig(
            freq=test_freq, before_train=val_before_train, mode=eval_mode,
        ),
    )
