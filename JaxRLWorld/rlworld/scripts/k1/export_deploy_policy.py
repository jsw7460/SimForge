"""Export a trained (JAX/Equinox) K1 actor to a TorchScript module for deploy.

The deploy stack (booster_deploy) consumes a ``torch.jit.ScriptModule``
that maps a raw observation vector to a pre-environment action, with the
observation normalizer baked in. rlworld's policy is JAX + Equinox, so
this script *reimplements* the deterministic actor forward in ``torch.nn``
and copies the weights out of the Equinox pytree, then validates the
port numerically against the original ``PPOActorCritic.act_inference``
before serializing.

Exported forward (matches ``ppo_ac.py:366-372`` + ``normalization.py:46``
+ ``utils.py:_forward_single``)::

    z = (obs - mean) / (sqrt(var) + eps)          # eps=1e-2, ADDED TO STD
    a = ELU(L0(z)); a = ELU(L1(a)); a = ELU(L2(a)); a = L3(a)   # no out-act
    a = tanh(a)                                    # iff squashed_gaussian

The deploy side is responsible for the rest of the pipeline (identity for
K1: clip(-1,1) then *1.0), the obs assembly/order, the gait-phase clock,
and the action-joint permutation. None of that lives in this module.

Run on the training box (needs the sim backend the checkpoint used)::

    jaxpy -m rlworld.scripts.k1.export_deploy_policy \
        --checkpoint /path/to/checkpoint-iter10000 \
        --output /path/to/k1_deploy.pt
"""

from __future__ import annotations

import argparse
import json
import os

import jax

# The validation compares the torch port (CPU full float32) against the JAX
# act_inference. On NVIDIA GPUs JAX defaults matmuls to TF32 (~1e-3 relative
# error), which would make an exact port look wrong by ~2e-3. Force full
# float32 so the numerical-equivalence check is apples-to-apples. Must run
# before any JAX computation.
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.utils.checkpoint import load_checkpoint_metadata


class DeployActor(nn.Module):
    """TorchScript-able port of the deterministic rlworld actor."""

    def __init__(
        self,
        linears: list[tuple[np.ndarray, np.ndarray]],
        norm_mean: np.ndarray | None,
        norm_var: np.ndarray | None,
        num_hidden: int,
        squashed: bool,
        norm_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.squashed = squashed
        self.norm_eps = norm_eps
        self.has_norm = norm_mean is not None

        if norm_mean is not None:
            self.register_buffer("norm_mean", torch.from_numpy(norm_mean.astype(np.float32)))
            self.register_buffer("norm_std", torch.from_numpy(np.sqrt(norm_var).astype(np.float32)))
        else:
            # Registered so TorchScript sees stable attributes; unused.
            self.register_buffer("norm_mean", torch.zeros(1))
            self.register_buffer("norm_std", torch.ones(1))

        layers = []
        for w, b in linears:
            lin = nn.Linear(w.shape[1], w.shape[0])
            # Equinox Linear.weight is (out, in) — same as torch. No transpose.
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(w.astype(np.float32)))
                lin.bias.copy_(torch.from_numpy(b.astype(np.float32)))
            layers.append(lin)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_norm:
            x = (x - self.norm_mean) / (self.norm_std + self.norm_eps)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_hidden:
                x = torch.nn.functional.elu(x)
        if self.squashed:
            x = torch.tanh(x)
        return x


def _extract(model):
    """Pull actor linears + normalizer stats out of the Equinox model."""
    net = model.actor.net
    linears = [(np.asarray(lin.weight), np.asarray(lin.bias)) for lin in net.linears]
    norm = model.actor_obs_normalizer
    if norm is None:
        mean = var = None
    else:
        mean = np.asarray(norm.mean)
        var = np.asarray(norm.var)
    return linears, mean, var, net.num_hidden, bool(model.is_squashed)


def _validate(model, torch_mod, actor_obs_dim, mean, var, n, seed):
    """Assert the torch port matches act_inference on IN-DISTRIBUTION obs.

    Sampling ``mean + sqrt(var) * N(0, 1)`` makes the normalizer output
    O(1) — the regime the policy actually runs in. Feeding raw ``N(0, 3)``
    instead drives near-constant obs dims (var≈0, so the normalizer
    divides by ~eps) into ±100s, where XLA/torch float32 accumulation
    diverges by ~1e-2 and the check would spuriously fail.
    """
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    z = jax.random.normal(sub, (n, actor_obs_dim))
    if mean is not None:
        obs = jax.numpy.asarray(mean) + jax.numpy.sqrt(jax.numpy.asarray(var)) * z
    else:
        obs = z

    keys = jax.random.split(key, n)
    ref = jax.vmap(lambda o, k: model.act_inference(o, key=k)[0])(obs, keys)
    ref = np.asarray(ref)

    with torch.no_grad():
        got = torch_mod(torch.from_numpy(np.asarray(obs).astype(np.float32))).numpy()

    max_abs = float(np.max(np.abs(ref - got)))
    return max_abs


def main() -> None:
    ap = argparse.ArgumentParser(description="Export K1 actor to TorchScript.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Local checkpoint directory.")
    src.add_argument("--wandb_run_path", help="W&B run path, e.g. entity/project/run_id.")
    ap.add_argument(
        "--wandb_checkpoint_iter", type=int, default=None, help="W&B checkpoint iteration (default: latest)."
    )
    ap.add_argument("--output", required=True, help="Output .pt path.")
    ap.add_argument("--num_check", type=int, default=256, help="Validation samples.")
    ap.add_argument("--tol", type=float, default=1e-4, help="Max abs diff tolerance.")
    args = ap.parse_args()

    # Reuse the validated load path: resolves wandb → local dir, builds env
    # (training backend) + runner. No eval_target → native-backend load.
    evaluator = PolicyEvaluator(
        policy_path=args.checkpoint,
        wandb_run_path=args.wandb_run_path,
        wandb_checkpoint_iter=args.wandb_checkpoint_iter,
        num_evals=1,
        seed=42,
        use_logging=False,
        save_data=False,
        record_video=False,
        record_steps=None,
        extra_overrides={"env": {"num_envs": 1}},
    )
    # After construction the evaluator holds the resolved local checkpoint dir.
    metadata = load_checkpoint_metadata(evaluator.policy_path)
    model = evaluator.runner.alg.train_state.model
    actor_obs_dim = int(model.actor_obs_dim)
    num_actions = int(model.actor.num_actions)

    linears, mean, var, num_hidden, squashed = _extract(model)
    # Build the torch model on CPU explicitly. Some sim backends (Genesis)
    # set the default torch device to cuda during env init, which would put
    # the Linear params on GPU while the validation obs stay on CPU. The
    # deploy target is CPU anyway.
    with torch.device("cpu"):
        torch_mod = DeployActor(linears, mean, var, num_hidden, squashed).eval()

    max_abs = _validate(model, torch_mod, actor_obs_dim, mean, var, args.num_check, seed=0)
    print(
        f"[export] actor_obs_dim={actor_obs_dim} num_actions={num_actions} "
        f"num_hidden={num_hidden} squashed={squashed} has_norm={mean is not None}"
    )
    print(f"[export] max|act_inference - torch| = {max_abs:.3e} (tol {args.tol:.0e})")
    if max_abs > args.tol:
        raise SystemExit(f"[export] VALIDATION FAILED: {max_abs:.3e} > {args.tol:.0e}")

    scripted = torch.jit.script(torch_mod)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    scripted.save(args.output)

    # Action post-processing arrays, pulled from the loaded env's action
    # manager (canonical order). Deploy applies, per action index k:
    #   target = clip(a_k, clip_low_k, clip_high_k) * scale_k + offset_k
    # This captures the recipe exactly (pal: scale 1, clip ±1; G1: per-joint
    # scale 0.25*eff/kp, clip ±100), so deploy needs no recipe knowledge.
    am = evaluator.env.act_manager
    action_scale = np.asarray(am._scale.detach().cpu().numpy(), dtype=float).tolist()
    action_clip_low = np.asarray(am._clip_low.detach().cpu().numpy(), dtype=float).tolist()
    action_clip_high = np.asarray(am._clip_high.detach().cpu().numpy(), dtype=float).tolist()
    action_offset = np.asarray(am._offset[0].detach().cpu().numpy(), dtype=float).tolist()

    # Sidecar metadata for the deploy repo (obs is single-frame, no history).
    canonical = metadata.get("canonical_joint_names") or list(am.actuated_joint_names)
    meta = {
        "actor_obs_dim": actor_obs_dim,
        "num_actions": num_actions,
        "squashed_tanh": squashed,
        "obs_normalizer_baked_in": mean is not None,
        "canonical_joint_names": canonical,
        "sim_type": metadata.get("sim_type"),
        "iteration": metadata.get("iteration"),
        "obs_layout_after_linvel_removed": [
            "base_ang_vel(3)",
            "projected_gravity(3)",
            "velocity_command(3)",
            "dof_pos_minus_default(22)",
            "dof_vel(22)",
            "last_action(22)",
            "gait_phase_cos_sin(4)",
        ],
        "action_pipeline": "target_k = clip(a_k, clip_low_k, clip_high_k) * scale_k + offset_k",
        "action_scale": action_scale,
        "action_clip_low": action_clip_low,
        "action_clip_high": action_clip_high,
        "action_offset": action_offset,
        "note": "action index k <-> canonical_joint_names[k]; deploy must "
        "permute to its own joint order and reconstruct the gait "
        "phase clock (init [0,pi]; obs=pre-advance snapshot; "
        "phi += 2*pi*0.02*freq wrapped to [-pi,pi); freeze to "
        "[pi,pi] when ||cmd||<=0.01; freq fixed 1.5). last_action in "
        "obs is the RAW network output a_k (pre clip/scale/offset).",
    }
    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[export] saved TorchScript: {args.output}")
    print(f"[export] saved metadata:    {meta_path}")


if __name__ == "__main__":
    main()
