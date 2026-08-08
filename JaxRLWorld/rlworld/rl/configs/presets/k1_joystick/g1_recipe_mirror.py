"""K1 G1-recipe + left/right mirror-symmetry loss.

Identical to :class:`K1G1RecipeConfig` in every way except that PPO's mirror
(symmetry) auxiliary loss is turned on (``mirror_symmetry_coeff = 1.0``), which
enforces left/right equivariance of the policy — targeting the right-side gait
bias seen in deploy logs and k1_mirror_viser_diag. Works on all three sims via
``sim_type`` (mujoco / newton / genesis).
"""

from __future__ import annotations

from dataclasses import dataclass

from .g1_recipe import K1G1RecipeConfig


@dataclass
class K1G1RecipeMirrorConfig(K1G1RecipeConfig):
    """K1 g1-recipe with the mirror-symmetry loss enabled."""

    mirror_symmetry_coeff: float = 1.0

    # Distinct run names so mirror runs are separable from the baseline.
    _RUN_NAMES = {
        "newton": "K1_Newton_G1Recipe_Mirror",
        "mujoco": "K1_Mujoco_G1Recipe_Mirror",
        "genesis": "K1_Genesis_G1Recipe_Mirror",
    }
