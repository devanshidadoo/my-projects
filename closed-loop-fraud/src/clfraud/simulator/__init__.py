"""The closed loop: decisions censor labels, and the next model trains on what survived."""
from clfraud.simulator.labeling import LabelPipeline, LabelPipelineConfig
from clfraud.simulator.loop import ClosedLoopSimulator, LoopConfig, PolicyArm
from clfraud.simulator.policy import (
    DecisionOutcome,
    ExplorationConfig,
    ThresholdPolicy,
)

__all__ = [
    "DecisionOutcome",
    "ExplorationConfig",
    "ThresholdPolicy",
    "LabelPipeline",
    "LabelPipelineConfig",
    "ClosedLoopSimulator",
    "LoopConfig",
    "PolicyArm",
]
