"""Neural Probabilistic Motor Primitives (Merel et al., ICLR 2019).

Sim-agnostic implementation that distils a set of expert tracking
policies into a single motor-primitive module with a latent
information bottleneck. :class:`NPMPModule` is the architecture
entry point; :func:`npmp_elbo_loss` is the training objective.
"""
from rlworld.imitation.npmp.buffer import NPMPBuffer
from rlworld.imitation.npmp.config import CheckpointRef, T1NPMPDistillConfig
from rlworld.imitation.npmp.decoder import NPMPDecoder
from rlworld.imitation.npmp.encoder import NPMPEncoder
from rlworld.imitation.npmp.expert_dispatch import MultiExpertDispatcher
from rlworld.imitation.npmp.loss import NPMPBatch, NPMPLossInfo, npmp_elbo_loss
from rlworld.imitation.npmp.module import NPMPModule, NPMPStepOutput
from rlworld.imitation.npmp.prior import AR1Prior
from rlworld.imitation.npmp.trainer import NPMPTrainer

__all__ = [
    "AR1Prior",
    "CheckpointRef",
    "MultiExpertDispatcher",
    "NPMPBatch",
    "NPMPBuffer",
    "NPMPDecoder",
    "NPMPEncoder",
    "NPMPLossInfo",
    "NPMPModule",
    "NPMPStepOutput",
    "NPMPTrainer",
    "T1NPMPDistillConfig",
    "npmp_elbo_loss",
]
