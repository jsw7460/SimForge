"""Are event-term draws a pure function of their coordinates?

Every random number an event term draws goes through the keyed sampler
(``rl/envs/mdp/events/keyed_sampler.py``): ``value = f(seed, term, stage,
env, counter, coords)``. This gate pins the properties the design is
for, without a simulator (CPU torch only):

  1. the int64 hash matches an unsigned-64 NumPy reference bit for bit
     (wrapping arithmetic and logical shifts behave on torch int64);
  2. invariances: a resetting subset gets the rows of the full draw, a
     tensor coordinate column equals the scalar coordinate, and the
     number of environments does not change env k's values;
  3. the distributions keep the conventions of the former ``dr._utils``:
     uniform, log-uniform, clamped gaussian, integer, bernoulli;
  4. streams, counters, stages, terms and seeds decorrelate the draws;
  5. the event manager binds the samplers as documented: only terms that
     take ``sampler`` get one, reset stages key on the per-env episode
     count, ``interval`` on the term's per-env fire count, ``interval_dr``
     on its global fire count, ``startup`` on zeros, and the interval
     timers are keyed too (``sampler_for`` / ``run_term`` agree);
  6. the field adapter mjlab's engines sample through maps a per-axis
     call to the right coordinates and is subset-stable;
  7. no event term draws from torch's global stream any more.

    python -m jaxrlworld.scripts.diag.gates.check_keyed_dr_sampler
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import jaxrlworld
from jaxrlworld.rl.configs.common_config_classes import EventConfig
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
from jaxrlworld.rl.envs.managers.common.event import EventManager
from jaxrlworld.rl.envs.mdp.events.keyed_sampler import (
    STAGE_INTERVAL_TIMER,
    STAGE_RESET_DR,
    TermSampler,
    keyed_bits,
    keyed_field_sampler,
    term_id,
)

_MASK = (1 << 64) - 1
_N = 200_000


# ── 1. hash reference ───────────────────────────────────────────────────


def _ref_splitmix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _ref_bits(*coords: int) -> int:
    h = 0
    for c in coords:
        h = _ref_splitmix(h ^ ((int(c) & _MASK) * 0xD6E8FEB86659FD93 & _MASK))
    return h


def check_hash(chk) -> None:
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(3000):
        k = int(rng.integers(1, 6))
        coords = [int(x) for x in rng.integers(-(2**63), 2**63 - 1, size=k, dtype=np.int64)]
        split = int(rng.integers(0, k + 1))
        args = coords[:split] + [torch.tensor(c) for c in coords[split:]]
        got = int(keyed_bits(*args, device=torch.device("cpu")).item()) & _MASK
        mismatches += got != _ref_bits(*coords)
    chk("torch int64 hash == NumPy uint64 reference (3000 coordinate tuples)", mismatches == 0, f"{mismatches} differ")
    vec = keyed_bits(7, torch.arange(10), device=torch.device("cpu"))
    chk("vectorized fold == per-element fold", all(int(vec[i]) & _MASK == _ref_bits(7, i) for i in range(10)))
    chk("term_id is a stable hash of the name", term_id("randomize_friction") == term_id("randomize_friction"))
    chk("term names are distinguished", term_id("randomize_friction") != term_id("randomize_frictio"))


# ── 2. invariances ──────────────────────────────────────────────────────


def _sampler(n: int, seed: int = 42, name: str = "t", stage: int = STAGE_RESET_DR, episode: int = 0) -> TermSampler:
    return TermSampler.bind(
        base_seed=seed, term_name=name, stage=stage, counter=torch.full((n,), episode, dtype=torch.long)
    )


def check_invariances(chk) -> None:
    s = _sampler(4096)
    full = s.draw(0.0, 1.0, torch.arange(4096))
    sub = s.draw(0.0, 1.0, torch.tensor([7, 3, 4000]))
    chk("subset rows == full-population rows", torch.equal(full[[7, 3, 4000]], sub))
    small = _sampler(64).draw(0.0, 1.0, torch.arange(64))
    chk("num_envs does not change env k's value (64 vs 4096)", torch.equal(full[:64], small))
    per_joint = s.draw(0.5, 1.5, torch.arange(8), coords=(torch.arange(5)[None, :],))
    chk("per-joint draw is [E, J]", tuple(per_joint.shape) == (8, 5))
    chk(
        "tensor coordinate column == scalar coordinate",
        torch.equal(per_joint[:, 2], s.draw(0.5, 1.5, torch.arange(8), coords=(2,))),
    )
    shaped = s.draw_shaped(0.0, 1.0, torch.arange(6), (6, 2, 3))
    chk("draw_shaped gives the requested shape", tuple(shaped.shape) == (6, 2, 3))
    chk(
        "draw_shaped is subset-stable",
        torch.equal(s.draw_shaped(0.0, 1.0, torch.tensor([5, 1]), (2, 2, 3)), shaped[[5, 1]]),
    )
    chk("draw_shaped refuses a wrong env axis", _raises(lambda: s.draw_shaped(0.0, 1.0, torch.arange(6), (5, 2))))
    chk("same seed twice == same values", torch.equal(full, _sampler(4096).draw(0.0, 1.0, torch.arange(4096))))


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


# ── 3. distributions ────────────────────────────────────────────────────


def check_distributions(chk) -> None:
    s = _sampler(_N)
    ids = torch.arange(_N)
    u = s.draw(2.0, 5.0, ids)
    chk(
        "uniform within [lo, hi)",
        float(u.min()) >= 2.0 and float(u.max()) < 5.0,
        f"{float(u.min()):.4f}..{float(u.max()):.4f}",
    )
    chk("uniform mean ~ 3.5", abs(float(u.mean()) - 3.5) < 0.01, f"{float(u.mean()):.4f}")
    chk("uniform var ~ 9/12", abs(float(u.var()) - 0.75) < 0.01, f"{float(u.var()):.4f}")
    lu = s.draw(0.1, 10.0, ids, distribution="log_uniform")
    chk("log_uniform within [lo, hi]", float(lu.min()) >= 0.1 - 1e-6 and float(lu.max()) <= 10.0 + 1e-6)
    chk("log_uniform log-mean ~ 0", abs(float(lu.log().mean())) < 0.02, f"{float(lu.log().mean()):.4f}")
    g = s.draw(-1.0, 1.0, ids, distribution="gaussian")
    chk("gaussian mean ~ (lo+hi)/2", abs(float(g.mean())) < 0.01, f"{float(g.mean()):.4f}")
    # N(0, 0.5) clamped to +-1 (two sigma): std 0.4789, 4.55 % at the clamp.
    chk("gaussian std ~ clamped N(0, (hi-lo)/4)", abs(float(g.std()) - 0.479) < 0.01, f"{float(g.std()):.4f}")
    clamp_frac = float(((g <= -1.0) | (g >= 1.0)).float().mean())
    chk("gaussian clamp fraction ~ 4.55 %", abs(clamp_frac - 0.0455) < 0.005, f"{clamp_frac:.4f}")
    di = s.draw_int(3, 10, ids)
    counts = torch.bincount(di - 3, minlength=7)
    chk("draw_int covers [low, high)", int(di.min()) == 3 and int(di.max()) == 9 and bool((counts > 0).all()))
    chk("draw_int is flat", float(counts.max()) / float(counts.min()) < 1.05, f"{counts.tolist()}")
    b = s.bernoulli(0.3, ids)
    chk("bernoulli(0.3) rate ~ 0.3", abs(float(b.float().mean()) - 0.3) < 0.005, f"{float(b.float().mean()):.4f}")
    chk("unknown distribution refused", _raises(lambda: s.draw(0.0, 1.0, ids, distribution="cauchy")))


# ── 4. decorrelation ────────────────────────────────────────────────────


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def check_decorrelation(chk) -> None:
    ids = torch.arange(_N)
    base = _sampler(_N).draw(0.0, 1.0, ids)
    pairs = {
        "stream": _sampler(_N).draw(0.0, 1.0, ids, stream=1),
        "episode counter": _sampler(_N, episode=1).draw(0.0, 1.0, ids),
        "stage": _sampler(_N, stage=STAGE_INTERVAL_TIMER).draw(0.0, 1.0, ids),
        "term name": _sampler(_N, name="u").draw(0.0, 1.0, ids),
        "seed": _sampler(_N, seed=43).draw(0.0, 1.0, ids),
        "neighbouring env": torch.roll(base, 1),
        "tensor coordinate": _sampler(_N).draw(0.0, 1.0, ids, coords=(1,)),
    }
    for label, other in pairs.items():
        c = _corr(base, other)
        chk(f"draws decorrelate across {label}", abs(c) < 0.01 and not torch.equal(base, other), f"corr {c:+.4f}")
    g1 = _sampler(_N).draw(-1.0, 1.0, ids, distribution="gaussian")
    g2 = _sampler(_N).draw(-1.0, 1.0, ids, distribution="gaussian", stream=1)
    chk("gaussian streams decorrelate", abs(_corr(g1, g2)) < 0.01)


# ── 5. event manager binding ────────────────────────────────────────────


def _keyed_term(env, env_ids, sampler, width: int, tag: str) -> None:
    env.log.append((tag, env_ids.clone(), sampler, sampler.draw(0.0, 1.0, env_ids, (torch.arange(width)[None, :],))))


def _plain_term(env, env_ids, value: float) -> None:
    env.log.append(("plain", env_ids.clone(), None, torch.full((len(env_ids),), value)))


@dataclass
class _GateEvents(EventConfig):
    dr = EventTermConfig(func=_keyed_term, mode="reset_dr", params={"width": 3, "tag": "dr"})
    boot = EventTermConfig(func=_keyed_term, mode="startup", params={"width": 1, "tag": "boot"})
    plain = EventTermConfig(func=_plain_term, mode="reset", params={"value": 2.0})
    kick = EventTermConfig(
        func=_keyed_term, mode="interval", interval_range_s=(1.0, 3.0), params={"width": 2, "tag": "kick"}
    )
    global_dr = EventTermConfig(
        func=_keyed_term, mode="interval_dr", interval_dr_period_s=2.0, params={"width": 1, "tag": "gdr"}
    )


def _fake_env(num_envs: int, seed: int = 7):
    return SimpleNamespace(num_envs=num_envs, device=torch.device("cpu"), seed=seed, sim_type="genesis", log=[])


def _entries(env, tag: str):
    return [e for e in env.log if e[0] == tag]


def check_event_manager(chk) -> None:
    env = _fake_env(8)
    mgr = EventManager(env, _GateEvents())
    chk(
        "terms taking `sampler` are detected",
        mgr._takes_sampler == {"dr": True, "boot": True, "plain": False, "kick": True, "global_dr": True},
    )

    mgr.apply("startup")
    boot = _entries(env, "boot")
    chk("startup term gets a sampler keyed on zeros", len(boot) == 1 and bool((boot[0][2].counter == 0).all()))

    all_ids = torch.arange(8)
    mgr.reset(all_ids)
    mgr.apply("reset", env_ids=all_ids)
    mgr.apply("reset_dr", env_ids=all_ids)
    chk("episode counter is 1 after the first reset of every env", bool((mgr.episode_count == 1).all()))
    chk("term without `sampler` is called as before", _entries(env, "plain")[0][3].tolist() == [2.0] * 8)
    first = _entries(env, "dr")[-1][3]

    env.log.clear()
    subset = torch.tensor([2, 5])
    mgr.reset(subset)
    mgr.apply("reset_dr", env_ids=subset)
    chk(
        "episode counter advances for the resetting subset only", mgr.episode_count.tolist() == [1, 1, 2, 1, 1, 2, 1, 1]
    )
    second = _entries(env, "dr")[-1][3]
    chk(
        "a re-reset env draws new values", not torch.equal(second[0], first[2]) and not torch.equal(second[1], first[5])
    )
    env.log.clear()
    mgr.apply("reset_dr", env_ids=torch.tensor([5, 1]))
    third = _entries(env, "dr")[-1][3]
    chk(
        "same (env, episode) -> same values whatever the subset",
        torch.equal(third[0], second[1]) and torch.equal(third[1], first[1]),
    )

    # The manager's binding is what sampler_for / run_term reproduce.
    expected = mgr.sampler_for("dr", "reset_dr").draw(0.0, 1.0, torch.tensor([5, 1]), (torch.arange(3)[None, :],))
    chk("sampler_for reproduces the manager's binding", torch.equal(expected, third))
    env.log.clear()
    mgr.run_term("dr", torch.tensor([5]))
    chk("run_term calls a term with its mode's sampler", torch.equal(_entries(env, "dr")[-1][3][0], third[0]))

    # A fresh manager with the same seed draws the same values; another seed does not.
    env2 = _fake_env(8)
    mgr2 = EventManager(env2, _GateEvents())
    mgr2.reset(all_ids)
    mgr2.apply("reset_dr", env_ids=all_ids)
    chk("same seed -> same first-episode draws", torch.equal(_entries(env2, "dr")[-1][3], first))
    env3 = _fake_env(8, seed=8)
    mgr3 = EventManager(env3, _GateEvents())
    mgr3.reset(all_ids)
    mgr3.apply("reset_dr", env_ids=all_ids)
    chk("other seed -> other draws", not torch.equal(_entries(env3, "dr")[-1][3], first))

    # Environments 0..7 of a 1024-env manager draw what the 8-env one drew.
    env_big = _fake_env(1024)
    mgr_big = EventManager(env_big, _GateEvents())
    mgr_big.reset(torch.arange(1024))
    mgr_big.apply("reset_dr", env_ids=torch.arange(1024))
    chk("num_envs does not change the first 8 envs' DR", torch.equal(_entries(env_big, "dr")[-1][3][:8], first))

    # Interval timers: keyed, resampled per firing, bumped on reset.
    timers0 = mgr._interval_timers[0].clone()
    chk("interval timers within interval_range_s", float(timers0.min()) >= 1.0 and float(timers0.max()) < 3.0)
    chk("interval timers differ per env", len(set(timers0.tolist())) == 8)
    env.log.clear()
    counts_before = mgr._interval_fire_count[0].clone()
    fired_at = {}
    for step in range(400):
        mgr.apply("interval", dt=0.01)
        for tag, ids, _s, _v in env.log:
            if tag == "kick":
                for e in ids.tolist():
                    fired_at.setdefault(e, []).append(step)
        env.log.clear()
    chk("every env fired at least once in 4 s", len(fired_at) == 8)
    fired = mgr._interval_fire_count[0] - counts_before
    chk(
        "fire count == number of firings per env",
        fired.tolist() == [len(fired_at.get(e, [])) for e in range(8)] and int(fired.sum()) > 8,
    )
    env.log.clear()
    mgr.apply("interval", dt=10.0)  # everything fires
    kicks = _entries(env, "kick")[-1]
    env.log.clear()
    mgr.apply("interval", dt=10.0)
    kicks2 = _entries(env, "kick")[-1]
    chk("each firing draws new values (fire count keys the draw)", not torch.equal(kicks[3], kicks2[3]))
    manual = TermSampler.bind(
        base_seed=7, term_name="kick", stage=STAGE_INTERVAL_TIMER, counter=mgr._interval_fire_count[0]
    ).draw(1.0, 3.0, all_ids)
    chk("timer == keyed draw on (term, fire count)", torch.equal(mgr._interval_timers[0], manual))
    before = mgr._interval_fire_count[0].clone()
    mgr.reset(torch.tensor([3]))
    chk(
        "reset bumps the fire count of the reset env",
        mgr._interval_fire_count[0].tolist() == (before + torch.eye(8, dtype=torch.long)[3]).tolist(),
    )

    env.log.clear()
    mgr.apply("interval_dr", dt=0.0)  # timer starts at 0 -> fires
    g1 = _entries(env, "gdr")[-1]
    chk(
        "interval_dr fires for all envs with the global fire count",
        g1[1].tolist() == list(range(8)) and bool((g1[2].counter == 1).all()),
    )
    env.log.clear()
    mgr.apply("interval_dr", dt=2.0)
    g2 = _entries(env, "gdr")[-1]
    chk(
        "next interval_dr firing uses the next count",
        bool((g2[2].counter == 2).all()) and not torch.equal(g1[3], g2[3]),
    )

    chk("sampler_for refuses an unknown mode", _raises(lambda: mgr.sampler_for("dr", "sometimes")))


# ── 6. field adapter ────────────────────────────────────────────────────


def check_field_adapter(chk) -> None:
    s = _sampler(16)
    ids = torch.tensor([3, 0, 9])
    lo, hi = torch.tensor([0.5]), torch.tensor([1.5])
    f = keyed_field_sampler(s, ids, "uniform")
    a = f(lo, hi, (3, 4, 1), "cpu")
    b = f(lo, hi, (3, 4, 1), "cpu")
    chk("field adapter returns the engine's shape", tuple(a.shape) == (3, 4, 1) and tuple(b.shape) == (3, 4, 1))
    chk("successive axis calls are independent draws", not torch.equal(a, b))
    chk("field values within the range", float(a.min()) >= 0.5 and float(a.max()) < 1.5)
    g = keyed_field_sampler(s, torch.arange(16), "uniform")(lo, hi, (16, 4, 1), "cpu")
    chk("field adapter is subset-stable", torch.equal(a, g[[3, 0, 9]]))
    shared = keyed_field_sampler(s, ids, "uniform")(lo, hi, (3, 1, 1), "cpu")
    chk("shared draw keys on the env alone", torch.equal(shared[:, 0, 0], s.draw(0.5, 1.5, ids)))
    chk(
        "(E, n, 1) == (E, n) over the same entities",
        torch.equal(a[..., 0], s.draw(0.5, 1.5, ids, (torch.arange(4)[None, :],))),
    )
    chk("field adapter refuses a wrong env axis", _raises(lambda: f(lo, hi, (2, 4, 1), "cpu")))
    start1 = keyed_field_sampler(s, ids, "uniform", stream=1)(lo, hi, (3, 4, 1), "cpu")
    chk("a stream offset starts at that stream", torch.equal(start1, b))


# ── 7. no raw RNG in the event terms ────────────────────────────────────

_RAW_RNG = re.compile(r"torch\.(rand|randn|randint|randperm|normal)\(|\.uniform_\(|\.normal_\(|\.random_\(")


def check_no_raw_rng(chk) -> None:
    root = Path(jaxrlworld.__file__).parent / "rl" / "envs"
    files = sorted((root / "mdp" / "events").rglob("*.py")) + [root / "managers" / "common" / "event.py"]
    offenders = []
    for f in files:
        if f.name == "keyed_sampler.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            if _RAW_RNG.search(line):
                offenders.append(f"{f.relative_to(root)}:{i}")
    chk(f"no event term draws from torch's global RNG ({len(files)} files)", not offenders, ", ".join(offenders))


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("KEYED DR SAMPLER")
    print("=" * 78)
    for section, fn in (
        ("1. hash", check_hash),
        ("2. invariances", check_invariances),
        ("3. distributions", check_distributions),
        ("4. decorrelation", check_decorrelation),
        ("5. event manager binding", check_event_manager),
        ("6. field adapter", check_field_adapter),
        ("7. no raw RNG", check_no_raw_rng),
    ):
        print(f"-- {section}")
        fn(chk)
    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
