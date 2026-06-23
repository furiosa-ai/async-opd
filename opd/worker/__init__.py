"""Worker proxies, weight sync, and Ray backends."""

from opd.worker.proxy import (
    RolloutProxy, TrainerProxy, WeightSyncEngine,
    QueueRolloutProxy, QueueTrainerProxy, NCCLWeightSyncEngine,
)

__all__ = [
    "RolloutProxy",
    "TrainerProxy",
    "WeightSyncEngine",
    "QueueRolloutProxy",
    "QueueTrainerProxy",
    "NCCLWeightSyncEngine",
]
