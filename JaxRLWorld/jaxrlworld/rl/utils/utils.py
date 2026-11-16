import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import torch


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def set_seed(seed: int):
    """Seed Python / NumPy / Torch RNGs for reproducibility.

    Also sets the hash and cuBLAS workspace env vars that PyTorch needs
    for deterministic kernels — ``CUBLAS_WORKSPACE_CONFIG`` must be set
    before the first CUDA op, so ``set_seed`` is the right place. We do
    NOT call ``torch.use_deterministic_algorithms`` here because that
    has a measurable perf cost; turn it on per-experiment if residual
    drift remains after seeding warp / Genesis natively (see
    ``GenesisEnv``/``NewtonEnv``/``MujocoEnv`` init).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # ``setdefault`` so the user can override the workspace size via env
    # without us clobbering it. 8 × 4 KiB workspaces is the recommended
    # minimum for deterministic cuBLAS; memory cost is negligible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_log_dir(output_dir: str | None = None) -> tuple[str, str]:
    """
    Create log directory with format: [output_dir/]outputs/YYYY-MM-DD/HH-MM-SS

    Args:
        output_dir: Base directory prefix. If None or "auto", uses EXP_OUTPUT_DIR
                    environment variable if set, otherwise current directory.

    Returns:
        Tuple of (models_log_dir, wandb_log_dir)
    """
    # Get current time in Chicago timezone
    kr_tz = pytz.timezone("America/Chicago")
    now = datetime.now(kr_tz)

    # Create directory path
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    # Determine base path
    if output_dir is None or output_dir == "auto":
        staging_dir = os.environ.get("EXP_OUTPUT_DIR")
        base_path = Path(staging_dir) if staging_dir else Path(".")
    else:
        base_path = Path(output_dir)

    models_log_dir = base_path / "outputs" / "models" / date_str / time_str
    wandb_log_dir = base_path / "outputs" / "logs" / date_str / time_str

    # The timestamp has one-second resolution, so two runners starting in
    # the same second would SHARE the directory — their checkpoint
    # tmp-dir/rename dance then races and one of them crashes mid-write.
    # Claim the models dir atomically (exist_ok=False) and suffix -2, -3,
    # … on collision; the wandb dir follows whatever name won.
    def _claim(base_models: Path, base_wandb: Path) -> tuple[Path, Path]:
        for attempt in range(100):
            suffix = "" if attempt == 0 else f"-{attempt + 1}"
            candidate = base_models.with_name(base_models.name + suffix)
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:  # except-ok: the name was taken by another run, try the next suffix
                continue
            wandb_dir = base_wandb.with_name(base_wandb.name + suffix)
            wandb_dir.mkdir(parents=True, exist_ok=True)
            return candidate, wandb_dir
        raise RuntimeError(f"Could not claim a unique log dir near {base_models}")

    models_log_dir, wandb_log_dir = _claim(models_log_dir, wandb_log_dir)

    return str(models_log_dir), str(wandb_log_dir)


def compare_dicts(config1, config2, name1="eval_env_cfgs", name2="train_cfgs"):
    """Compare two config dictionaries and print their differences in detail.

    Args:
        config1 (dict): First configuration dictionary to compare
        config2 (dict): Second configuration dictionary to compare
        name1 (str): Name identifier for config1 (default: "eval_env_cfgs")
        name2 (str): Name identifier for config2 (default: "train_cfgs")
    """

    def print_dict_differences(d1, d2, name1, name2, prefix=""):
        """Recursively compare two dictionaries and print differences.

        Args:
            d1 (dict): First dictionary
            d2 (dict): Second dictionary
            name1 (str): Name identifier for d1
            name2 (str): Name identifier for d2
            prefix (str): Indentation prefix for nested output

        Returns:
            bool: True if differences were found, False otherwise
        """
        all_keys = set(d1.keys()) | set(d2.keys())
        has_differences = False

        # Check for keys that exist in only one dictionary
        only_in_d1 = set(d1.keys()) - set(d2.keys())
        only_in_d2 = set(d2.keys()) - set(d1.keys())

        if only_in_d1:
            print(f"{prefix}Keys only in {name1}: {sorted(only_in_d1)}")
            has_differences = True

        if only_in_d2:
            print(f"{prefix}Keys only in {name2}: {sorted(only_in_d2)}")
            has_differences = True

        # Compare values for common keys
        common_keys = set(d1.keys()) & set(d2.keys())
        for key in sorted(common_keys):
            val1, val2 = d1[key], d2[key]

            # Recursively compare nested dictionaries
            if isinstance(val1, dict) and isinstance(val2, dict):
                print(f"{prefix}Comparing '{key}':")
                sub_has_diff = print_dict_differences(val1, val2, name1, name2, prefix + "  ")
                if sub_has_diff:
                    has_differences = True
            # Compare non-dictionary values
            elif val1 != val2:
                print(f"{prefix}Different value for key '{key}':")
                print(f"{prefix}  {name1}: {val1}")
                print(f"{prefix}  {name2}: {val2}")
                has_differences = True

        return has_differences

    if config1 != config2:
        print("*****Configurations are different*****")
        print_dict_differences(config1, config2, name1, name2)
    else:
        print("Configurations are identical")
