"""Native vLLM LoRA adapter support — TensorLoRARequest + hijack.

Enables loading LoRA adapters from in-memory tensors (no disk I/O) by
monkey-patching vLLM's LRUCacheWorkerLoRAManager._load_adapter.

Ported from veRL (verl/utils/vllm/utils.py) with modifications for OPD.
"""

from msgspec import field
from opd.utils.config import LoRAConfig
from vllm.lora.lora_model import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


_LORA_DEFAULTS = LoRAConfig()


# Stable constants for the single LoRA adapter we manage
LORA_NAME = "opd_lora"
LORA_INT_ID = 1
LORA_PATH = "__tensor_lora__"


class TensorLoRARequest(LoRARequest):
    """LoRARequest that carries in-memory tensors instead of a file path.

    Inherits from LoRARequest (msgspec.Struct) so it passes vLLM's type
    checks. The lora_path is set to a sentinel value; the actual weights
    come from lora_tensors.
    """
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


def hijack_lora_manager():
    """Monkey-patch vLLM's LoRA manager to support TensorLoRARequest.

    Must be called BEFORE LLM() init so the patched _load_adapter is
    used when the engine processes add_lora requests.

    Only works with VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process engine).
    """
    import os
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", \
        "hijack_lora_manager requires VLLM_ENABLE_V1_MULTIPROCESSING=0"

    def _hijacked_load_adapter(self, lora_request):
        """Load LoRA adapter — from tensors if TensorLoRARequest, else from disk.

        Adapted for vLLM 0.16 API (model_vocab_size, skip_prefixes, etc.).
        """
        from vllm.lora.peft_helper import PEFTHelper

        # Build expected_lora_modules list
        supported_lora_modules = self._adapter_manager.supported_lora_modules
        packed_modules_mapping = self._adapter_manager.packed_modules_mapping
        expected_lora_lst = []
        for module in supported_lora_modules:
            if module in packed_modules_mapping:
                expected_lora_lst.extend(packed_modules_mapping[module])
            else:
                expected_lora_lst.append(module)
            if module == "experts":
                expected_lora_lst.append(module)
        expected_lora_modules = set(expected_lora_lst)

        # Get model-specific helpers
        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

        if isinstance(lora_request, TensorLoRARequest):
            peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
            peft_helper.validate_legal(self.lora_config)
            # Use "cuda" if tensors are already on GPU (NCCL path),
            # "cpu" if they arrived via other means.
            first_tensor = next(iter(lora_request.lora_tensors.values()))
            tensor_device = "cuda" if first_tensor.is_cuda else "cpu"
            lora = self._lora_model_cls.from_lora_tensors(
                lora_model_id=lora_request.lora_int_id,
                tensors=lora_request.lora_tensors,
                peft_helper=peft_helper,
                device=tensor_device,
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
            )
        else:
            # Fall through to standard disk-based loading
            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                getattr(lora_request, 'tensorizer_config_dict', None),
            )
            peft_helper.validate_legal(self.lora_config)
            lora = self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                tensorizer_config_dict=getattr(lora_request, 'tensorizer_config_dict', None),
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
            )

        return lora

    LRUCacheWorkerLoRAManager._load_adapter = _hijacked_load_adapter


def hijack_update_weights(peft_config_dict):
    """Monkey-patch vLLM's Worker.update_weights to route LoRA updates to add_lora.

    When update_info contains '_lora_update': True, the received NCCL tensors
    are loaded as a LoRA adapter via add_lora(TensorLoRARequest(...)) instead
    of being written to base model parameters.

    Must be called BEFORE LLM() init.
    """
    from vllm.v1.worker.gpu_worker import Worker

    _original_update_weights = Worker.update_weights

    def _patched_update_weights(self, update_info):
        is_lora = update_info.pop("_lora_update", False)
        if not is_lora:
            return _original_update_weights(self, update_info)

        # LoRA path: receive tensors via NCCL, then route to add_lora
        typed_update_info = self.weight_transfer_engine.parse_update_info(
            update_info)

        received = []
        self.weight_transfer_engine.receive_weights(
            typed_update_info,
            load_weights=lambda w: received.extend(w),
        )

        lora_dict = {name: tensor for name, tensor in received}

        # Store cloned tensors for checksum verification (add_lora may
        # modify tensors in-place during LoRA model construction)
        self._last_lora_tensors = {n: t.clone() for n, t in lora_dict.items()}

        # Remove old LoRA if present, then add new one
        try:
            self.model_runner.remove_lora(LORA_INT_ID)
        except Exception:
            pass

        tensor_req = TensorLoRARequest(
            lora_name=LORA_NAME,
            lora_int_id=LORA_INT_ID,
            lora_path=LORA_PATH,
            peft_config=peft_config_dict,
            lora_tensors=lora_dict,
        )
        self.model_runner.add_lora(tensor_req)

    Worker.update_weights = _patched_update_weights


def build_peft_config_dict(lora_cfg):
    """Convert OPD lora config dict to PEFT-format dict for PEFTHelper.

    Args:
        lora_cfg: dict with keys: rank, alpha, target_modules, dropout (optional)

    Returns:
        dict compatible with PEFTHelper.from_dict()
    """
    return {
        "r": lora_cfg["rank"],
        "lora_alpha": lora_cfg["alpha"],
        "lora_dropout": lora_cfg.get("dropout", _LORA_DEFAULTS.dropout),
        "target_modules": lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
