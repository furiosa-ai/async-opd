"""Pipeline compatibility facade.

The stable public package API is deliberately small: use
``opd.create_coordinator`` or ``opd.pipeline.create_coordinator`` to construct a
coordinator, and use the documented CLI/config files for normal operation.

The remaining names in this module are legacy compatibility exports kept for
older source-tree imports.  They are not a commitment to long-term public API
stability.

All orchestration logic lives in:
  - opd/coordinator/base.py      (CoordinatorBase)
  - opd/coordinator/step_off.py  (StepOffCoordinator)
  - opd/coordinator/streaming.py (StreamCoordinator)
  - opd/coordinator/sft.py       (SFTCoordinator)
  - opd/coordinator/factory.py   (factory dispatch)
"""

from opd.coordinator.factory import create_coordinator  # noqa: F401
from opd.coordinator.base import CoordinatorBase  # noqa: F401
from opd.coordinator.step_off import StepOffCoordinator  # noqa: F401
from opd.coordinator.streaming import StreamCoordinator  # noqa: F401
from opd.coordinator.fused_hybrid_sync import FusedHybridSyncCoordinator  # noqa: F401

from opd.data.batch_utils import pad_teacher, split_gen_teacher, broadcast_batch  # noqa: F401
from opd.utils.eval import extract_answer, answers_match  # noqa: F401
from opd.utils.net import find_free_port, port_is_listening, kill_tree  # noqa: F401
from opd.utils.staleness_queue import StalenessQueue  # noqa: F401
from opd.worker.weight_merge import build_weight_merge_map  # noqa: F401
from opd.worker.proxy import (  # noqa: F401
    RolloutProxy, TrainerProxy, WeightSyncEngine,
    QueueRolloutProxy, QueueTrainerProxy, NCCLWeightSyncEngine,
)

# Ray proxies — optional dependency, imported lazily
try:
    from opd.worker.ray_proxy import RayRolloutProxy, RayTrainerProxy  # noqa: F401
except ImportError:
    pass

PUBLIC_API = ("create_coordinator",)

COMPATIBILITY_API = (
    # Coordinator classes retained for existing source-tree imports.
    "FusedHybridSyncCoordinator",
    "CoordinatorBase",
    "StepOffCoordinator",
    "StreamCoordinator",
    # Utility/helper exports retained for legacy tests and scripts.
    "StalenessQueue",
    "find_free_port",
    "port_is_listening",
    "kill_tree",
    "build_weight_merge_map",
    "extract_answer",
    "answers_match",
    "pad_teacher",
    "split_gen_teacher",
    "broadcast_batch",
    # Proxy interfaces retained for existing runtime integrations.
    "RolloutProxy",
    "TrainerProxy",
    "WeightSyncEngine",
    "QueueRolloutProxy",
    "QueueTrainerProxy",
    "NCCLWeightSyncEngine",
)

__all__ = PUBLIC_API + COMPATIBILITY_API
