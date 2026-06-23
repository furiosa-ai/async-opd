"""Loss functions — KL divergence, SFT cross-entropy, Megatron vocab-parallel."""

from opd.loss.kl import (
    KLConfig,
    compute_kl_loss,
    chunked_lm_head_gather,
    chunked_log_softmax_gather,
    sparse_forward_kl,
    sparse_reverse_kl,
    KL_CHUNK_SIZE,
    _ChunkedLogSoftmaxGather,
)
from opd.loss.sft import sft_loss

__all__ = [
    "KLConfig",
    "compute_kl_loss",
    "chunked_lm_head_gather",
    "chunked_log_softmax_gather",
    "sparse_forward_kl",
    "sparse_reverse_kl",
    "KL_CHUNK_SIZE",
    "_ChunkedLogSoftmaxGather",
    "sft_loss",
]
