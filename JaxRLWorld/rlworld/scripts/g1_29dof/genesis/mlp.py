import random
import numpy as np
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.g1_29dof.genesis.mlp import get_config




def main():
    cfgs_for_run = get_config().with_cli_overrides()
    cfgs_for_run.algorithm.obs_normalization = True
    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
