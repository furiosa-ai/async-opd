"""Teacher scoring server package — vLLM and HF backends with ZMQ transport."""

from opd.worker.teacher.vllm import VLLMTeacherServer, teacher_server_main
from opd.worker.teacher.hf import HFTeacherServer, teacher_hf_server_main
from opd.worker.teacher.client import TeacherClient
from opd.worker.teacher.serialization import serialize, deserialize
from opd.worker.teacher.fast_logprobs import captured_topk, install_fast_prompt_logprobs

__all__ = [
    "VLLMTeacherServer",
    "teacher_server_main",
    "HFTeacherServer",
    "teacher_hf_server_main",
    "TeacherClient",
    "serialize",
    "deserialize",
    "captured_topk",
    "install_fast_prompt_logprobs",
]
