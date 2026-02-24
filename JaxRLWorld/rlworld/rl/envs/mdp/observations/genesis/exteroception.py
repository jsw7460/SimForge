from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.utils import MethodEnvStepCache, EnvStepCache

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


@EnvStepCache()
def command(env: GenesisEnv) -> torch.Tensor:
    return env.command_manager.get_commands_tensor()