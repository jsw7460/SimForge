"""Keyed (counter-based) random draws for event terms.

Every random number an event term draws is a pure function of its
coordinates::

    value = f(seed, term, stage, env, episode, extra coords...)

rather than the next value of a shared random stream. Two runs with the
same seed therefore produce the same value at the same coordinate no
matter how many environments exist, which subset of them is resetting,
which other terms run, or in what order; and the three simulators draw
the same values for the same coordinates, which makes domain
randomization a controlled variable in cross-simulator comparisons.

The hash is SplitMix64 over int64 with wrapping arithmetic, evaluated on
the device of ``env_ids`` so a draw never synchronizes with the host.

Coordinate model
----------------
The :class:`EventManager` binds one :class:`TermSampler` per term call
(seed, term name, lifecycle stage, and the per-env counter of that
stage: the episode number for reset stages, the trigger count for
interval stages). The term passes ``env_ids`` and any further
coordinates:

- ``env_ids`` with their counters are the leading output axis (one row
  per environment); a subset of environments yields the rows of the
  full draw.
- A Python ``int`` coordinate (``stream``) separates independent
  quantities of one term (kp vs kd, position vs velocity).
- A tensor coordinate adds output axes: ``coords=(joint_ids[None, :],)``
  draws one value per (env, joint). Tensor coords are broadcast against
  each other after the env axis, numpy style.

Reordering coordinates changes the draws, so a term's coordinate scheme
is fixed once it ships.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_COORD_MUL = 0xD6E8FEB86659FD93
_TWO_POW_53 = float(1 << 53)

STAGE_STARTUP = 1
STAGE_RESET = 2
STAGE_RESET_DR = 3
STAGE_INTERVAL = 4
STAGE_INTERVAL_DR = 5
STAGE_INTERVAL_TIMER = 6

STAGE_IDS: dict[str, int] = {
    "startup": STAGE_STARTUP,
    "reset": STAGE_RESET,
    "reset_dr": STAGE_RESET_DR,
    "interval": STAGE_INTERVAL,
    "interval_dr": STAGE_INTERVAL_DR,
}


def _as_int64(value: int) -> int:
    """A Python int reinterpreted as a signed 64-bit value (torch int64 literal)."""
    value &= _MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def _lsr(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical right shift of an int64 tensor (torch's ``>>`` is arithmetic)."""
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def _splitmix64(z: torch.Tensor) -> torch.Tensor:
    """One SplitMix64 output step on int64 tensors, wrapping on overflow."""
    z = z + _as_int64(_GOLDEN)
    z = (z ^ _lsr(z, 30)) * _as_int64(_MIX1)
    z = (z ^ _lsr(z, 27)) * _as_int64(_MIX2)
    return z ^ _lsr(z, 31)


def _splitmix64_int(z: int) -> int:
    """The same step as :func:`_splitmix64` on a Python int (unsigned 64-bit)."""
    z = (z + _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
    return z ^ (z >> 31)


def _fold_int(h: int, c: int) -> int:
    return _splitmix64_int(h ^ ((c * _COORD_MUL) & _MASK64))


def _split_coords(coords: tuple, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Fold the leading Python ints on the host; return that seed as a device
    scalar plus the remaining coords as int64 tensors (a trailing int becomes
    a 0-d tensor so it folds in place)."""
    h_int = 0
    i = 0
    while i < len(coords) and not isinstance(coords[i], torch.Tensor):
        h_int = _fold_int(h_int, int(coords[i]) & _MASK64)
        i += 1
    h0 = torch.tensor(_as_int64(h_int), dtype=torch.int64, device=device)
    tensors = []
    for c in coords[i:]:
        if isinstance(c, torch.Tensor):
            tensors.append(c.to(device=device, dtype=torch.int64))
        else:
            tensors.append(torch.tensor(_as_int64(int(c)), dtype=torch.int64, device=device))
    return h0, tensors


def _fold_tensors(h: torch.Tensor, *coords: torch.Tensor) -> torch.Tensor:
    for c in coords:
        h = _splitmix64(h ^ (c * _as_int64(_COORD_MUL)))
    return h


def _fold_uniform(h: torch.Tensor, *coords: torch.Tensor) -> torch.Tensor:
    bits = _lsr(_fold_tensors(h, *coords), 11)
    return (bits.to(torch.float64) / _TWO_POW_53).to(torch.float32)


# The fold is ~15 elementwise int64 ops per coordinate; eager torch launches
# each as its own kernel, which on a GPU costs more than the arithmetic
# (0.45 ms for an 8192-env draw, launch-bound). Compiled, one draw is one
# fused kernel. One program per coordinate count, with symbolic shapes so a
# resetting subset of any size reuses it. CPU stays eager (the gates run
# there without a compiler toolchain); the results are identical.
_COMPILED_FOLD_UNIFORM: dict[int, object] = {}


def _fold_uniform_on(device: torch.device, n_coords: int):
    if device.type != "cuda":
        return _fold_uniform
    fn = _COMPILED_FOLD_UNIFORM.get(n_coords)
    if fn is None:
        fn = torch.compile(_fold_uniform, dynamic=True, fullgraph=True)
        _COMPILED_FOLD_UNIFORM[n_coords] = fn
    return fn


def keyed_bits(*coords: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Fold integer ``coords`` into one int64 hash per broadcast element.

    Python ints are folded as scalars; tensors are broadcast together.
    The fold is sequential, so the same coordinates in a different order
    are a different key. Leading Python ints are folded on the host, so
    a draw whose only tensor coords are the env axis launches no
    per-scalar device copies. Eager on every device (the reference path
    the gate checks against NumPy).
    """
    h0, tensors = _split_coords(coords, device)
    return _fold_tensors(h0, *tensors)


def keyed_uniform(*coords: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Uniforms in ``[0, 1)`` from the top 53 bits of :func:`keyed_bits`, as float32.

    On CUDA the fold and the conversion run as one compiled kernel."""
    h0, tensors = _split_coords(coords, device)
    return _fold_uniform_on(device, len(tensors))(h0, *tensors)


def keyed_uniform_eager(*coords: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    """:func:`keyed_uniform` without the compiled path (for checking it)."""
    h0, tensors = _split_coords(coords, device)
    return _fold_uniform(h0, *tensors)


def term_id(name: str) -> int:
    """A stable 64-bit id for a term name (BLAKE2b, independent of ``PYTHONHASHSEED``)."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


@dataclass(frozen=True)
class TermSampler:
    """The keyed draw for one term invocation.

    ``counter`` is the per-env counter of the bound stage over ALL envs
    (indexed by ``env_ids`` at draw time): the episode number for the
    reset stages, the trigger count for interval stages, zeros at
    startup. ``bind`` is what the event manager calls; a term only sees
    ``draw`` / ``draw_int`` / ``bernoulli``.
    """

    base_seed: int
    term_id: int
    stage: int
    counter: torch.Tensor

    @classmethod
    def bind(cls, *, base_seed: int, term_name: str, stage: int, counter: torch.Tensor) -> TermSampler:
        return cls(base_seed=int(base_seed), term_id=term_id(term_name), stage=int(stage), counter=counter)

    def uniforms(
        self,
        env_ids: torch.Tensor,
        coords: Sequence[int | torch.Tensor] = (),
        stream: int = 0,
    ) -> torch.Tensor:
        """Uniforms in ``[0, 1)`` of shape ``[len(env_ids), *broadcast(tensor coords)]``.

        Row ``k`` keys on ``(env_ids[k], counter[env_ids[k]])`` so a subset
        of envs gets the rows the full population would get.
        """
        env_ids = env_ids.to(dtype=torch.int64)
        extra = tuple(coords)
        tensor_rank = max((c.dim() for c in extra if isinstance(c, torch.Tensor)), default=0)
        env_axis = env_ids.reshape((-1,) + (1,) * max(tensor_rank - 1, 0))
        counter_axis = self.counter[env_ids].reshape(env_axis.shape)
        return keyed_uniform(
            self.base_seed, self.term_id, self.stage, stream, env_axis, counter_axis, *extra, device=env_ids.device
        )

    def draw(
        self,
        lo: float,
        hi: float,
        env_ids: torch.Tensor,
        coords: Sequence[int | torch.Tensor] = (),
        distribution: str = "uniform",
        stream: int = 0,
    ) -> torch.Tensor:
        """Samples with the conventions of the former ``dr._utils.sample``:

        - ``uniform``: ``U(lo, hi)``
        - ``log_uniform``: ``exp(U(log lo, log hi))`` with bounds floored at 1e-8
        - ``gaussian``: ``N((lo+hi)/2, ((hi-lo)/4)^2)`` clamped to ``[lo, hi]``
        """
        u = self.uniforms(env_ids, coords, stream)
        if distribution == "uniform":
            return (hi - lo) * u + lo
        if distribution == "log_uniform":
            log_lo = math.log(max(lo, 1e-8))
            log_hi = math.log(max(hi, 1e-8))
            return torch.exp((log_hi - log_lo) * u + log_lo)
        if distribution == "gaussian":
            mean = (lo + hi) / 2.0
            std = (hi - lo) / 4.0
            # Box-Muller from two independent uniforms; the second comes from
            # a stream the caller cannot collide with (it is derived here).
            u2 = self.uniforms(env_ids, coords, _gaussian_partner_stream(stream))
            radius = torch.sqrt(-2.0 * torch.log1p(-u))  # u in [0,1) -> 1-u in (0,1]
            normal = radius * torch.cos(2.0 * math.pi * u2)
            return (mean + std * normal).clamp_(lo, hi)
        raise ValueError(f"Unknown distribution {distribution!r}. Choose from 'uniform', 'log_uniform', 'gaussian'.")

    def draw_shaped(
        self,
        lo: float,
        hi: float,
        env_ids: torch.Tensor,
        shape: Sequence[int],
        distribution: str = "uniform",
        stream: int = 0,
    ) -> torch.Tensor:
        """:meth:`draw` for a ``(len(env_ids), *dims)`` output whose trailing
        dims are positions (a backend's own shape / dof order): each trailing
        dim of size > 1 becomes an index coordinate. A size-1 dim carries no
        position, so ``(E, 1, 1)`` keys exactly as ``(E,)`` does: a shared
        draw is the same value on every backend however it shapes it, and
        ``(E, n, 1)`` equals an ``(E, n)`` draw over the same entities."""
        shape = tuple(int(n) for n in shape)
        if shape[0] != int(env_ids.numel()):
            raise ValueError(f"draw_shaped expects a leading env axis of {env_ids.numel()}, got {shape}")
        wide = [n for n in shape[1:] if n > 1]
        coords: list[torch.Tensor] = []
        for axis, n in enumerate(wide, start=1):
            view = [1] * (len(wide) + 1)
            view[axis] = n
            coords.append(torch.arange(n, device=env_ids.device).reshape(view))
        return self.draw(lo, hi, env_ids, coords, distribution, stream).reshape(shape)

    def draw_int(
        self,
        low: int,
        high: int,
        env_ids: torch.Tensor,
        coords: Sequence[int | torch.Tensor] = (),
        stream: int = 0,
    ) -> torch.Tensor:
        """Integers in ``[low, high)``."""
        if high <= low:
            raise ValueError(f"draw_int needs low < high; got [{low}, {high})")
        u = self.uniforms(env_ids, coords, stream)
        return (low + (u * (high - low)).to(torch.int64)).clamp_(low, high - 1)

    def bernoulli(
        self,
        p: float,
        env_ids: torch.Tensor,
        coords: Sequence[int | torch.Tensor] = (),
        stream: int = 0,
    ) -> torch.Tensor:
        """Booleans true with probability ``p``."""
        return self.uniforms(env_ids, coords, stream) < p


def _gaussian_partner_stream(stream: int) -> int:
    # Far from any small stream number a term would pick by hand.
    return _as_int64((stream * 0x9E3779B1) ^ 0x5BD1E995_7F4A7C15)


def keyed_field_sampler(sampler: TermSampler, env_ids: torch.Tensor, distribution: str, stream: int = 0):
    """A ``(lo, hi, shape, device) -> tensor`` callable for engines that sample
    a model field axis by axis (mjlab's ``Distribution.sample``).

    ``shape`` is ``(len(env_ids), n_entities[, 1])`` or ``(len(env_ids), 1[, 1])``
    for a shared draw. Entity positions become a tensor coordinate; each
    successive call (one per target axis, in the engine's fixed axis order)
    moves to the next stream so axes are independent.
    """
    calls = 0

    def sample(lo, hi, shape, device) -> torch.Tensor:
        nonlocal calls
        if int(shape[0]) != int(env_ids.numel()):
            raise ValueError(f"keyed field sampler expected a leading env axis of {env_ids.numel()}, got shape {shape}")
        values = sampler.draw_shaped(float(lo), float(hi), env_ids, shape, distribution, stream=stream + calls)
        calls += 1
        return values.to(device=device)

    return sample
