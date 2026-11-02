"""Motion packs: a clip set stored as one directory of concatenated arrays.

A pack holds the frames of many clips along axis 0 of six ``.npy`` arrays
(the fields a ``MotionCommand`` reads: ``joint_pos``, ``joint_vel``,
``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w``) and a
``manifest.json`` with the frame rate, the joint and body names of the
columns, and the clip names and lengths in order. Clip ``i`` occupies frames
``offsets[i] .. offsets[i] + lengths[i]``.

Per-clip npz files (``csv_to_npz``) suit tens of clips; a set of a hundred
thousand clips takes minutes to open file by file and puts every path in
each saved config. The pack is one path, opened as memory maps, and a run
selects clips through a **clip list**: a text file with one clip name per
line (``#`` comments and blank lines ignored).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

MANIFEST_FILE = "manifest.json"
ARRAY_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class PackManifest:
    fps: float
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    clip_names: tuple[str, ...]
    clip_lengths: tuple[int, ...]

    @property
    def num_clips(self) -> int:
        return len(self.clip_names)

    @property
    def num_frames(self) -> int:
        return int(sum(self.clip_lengths))

    @property
    def offsets(self) -> np.ndarray:
        """First frame of every clip, ``(num_clips,)`` int64."""
        lengths = np.asarray(self.clip_lengths, dtype=np.int64)
        return np.concatenate(([0], np.cumsum(lengths)[:-1]))

    def indices_of(self, names: Sequence[str]) -> np.ndarray:
        """Clip indices of ``names`` in the given order; unknown names raise."""
        lookup = {n: i for i, n in enumerate(self.clip_names)}
        missing = [n for n in names if n not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} clip(s) not in the pack, e.g. {missing[:5]}")
        return np.asarray([lookup[n] for n in names], dtype=np.int64)

    def to_json(self) -> dict:
        return {
            "fps": self.fps,
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "clip_names": list(self.clip_names),
            "clip_lengths": [int(n) for n in self.clip_lengths],
        }

    @classmethod
    def from_json(cls, data: dict) -> PackManifest:
        return cls(
            fps=float(data["fps"]),
            joint_names=tuple(data["joint_names"]),
            body_names=tuple(data["body_names"]),
            clip_names=tuple(data["clip_names"]),
            clip_lengths=tuple(int(n) for n in data["clip_lengths"]),
        )


def read_manifest(pack_dir: str | Path) -> PackManifest:
    path = Path(pack_dir) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"not a motion pack (no {MANIFEST_FILE}): {pack_dir}")
    return PackManifest.from_json(json.loads(path.read_text()))


def open_arrays(pack_dir: str | Path) -> dict[str, np.ndarray]:
    """The six arrays as read-only memory maps."""
    pack = Path(pack_dir)
    return {name: np.load(pack / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}


def read_clip_list(path: str | Path) -> tuple[str, ...]:
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    if not names:
        raise ValueError(f"clip list is empty: {path}")
    return tuple(names)


def write_clip_list(path: str | Path, names: Sequence[str], header: str = "") -> None:
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines.extend(names)
    Path(path).write_text("\n".join(lines) + "\n")


def iter_ranges(manifest: PackManifest, clip_indices: np.ndarray) -> Iterator[tuple[int, int]]:
    """``(start, end)`` frame ranges of the clips, in the given order, with
    consecutive clips merged into one range."""
    offsets = manifest.offsets
    lengths = np.asarray(manifest.clip_lengths, dtype=np.int64)
    start = end = None
    for i in clip_indices:
        s, e = int(offsets[i]), int(offsets[i] + lengths[i])
        if end is not None and s == end:
            end = e
            continue
        if start is not None:
            yield start, end
        start, end = s, e
    if start is not None:
        yield start, end


def write_pack(
    pack_dir: str | Path,
    clips: Sequence[tuple[str, str | Path]],
    *,
    body_names: Sequence[str] | None = None,
) -> PackManifest:
    """Concatenate per-clip npz files (``csv_to_npz`` layout) into a pack.

    Args:
        pack_dir: Output directory (created).
        clips: ``(clip name, npz path)`` pairs in pack order.
        body_names: Bodies to keep, in this order; ``None`` keeps the first
            file's bodies. Every file must contain them.

    Two passes: lengths and metadata first, then the frames straight into
    memory-mapped ``.npy`` files, so the host never holds the whole set.
    """
    if not clips:
        raise ValueError("write_pack needs at least one clip")
    out = Path(pack_dir)
    out.mkdir(parents=True, exist_ok=True)

    lengths: list[int] = []
    fps: float | None = None
    joint_names: tuple[str, ...] | None = None
    bodies: tuple[str, ...] | None = None
    for name, path in clips:
        data = np.load(path, allow_pickle=True)
        file_fps = float(np.asarray(data["fps"]).item())
        file_joints = tuple(str(n) for n in np.asarray(data["joint_names"]).tolist())
        file_bodies = tuple(str(n) for n in np.asarray(data["body_names"]).tolist())
        if fps is None:
            fps = file_fps
            joint_names = file_joints
            bodies = tuple(body_names) if body_names is not None else file_bodies
        if file_fps != fps:
            raise ValueError(f"{path}: fps {file_fps} differs from {fps}")
        if file_joints != joint_names:
            raise ValueError(f"{path}: joint order differs from the first clip")
        missing = [b for b in bodies if b not in file_bodies]
        if missing:
            raise ValueError(f"{path}: missing bodies {missing}")
        lengths.append(int(data["joint_pos"].shape[0]))

    manifest = PackManifest(
        fps=fps,
        joint_names=joint_names,
        body_names=bodies,
        clip_names=tuple(name for name, _ in clips),
        clip_lengths=tuple(lengths),
    )
    n = manifest.num_frames
    j = len(joint_names)
    b = len(bodies)
    shapes = {
        "joint_pos": (n, j),
        "joint_vel": (n, j),
        "body_pos_w": (n, b, 3),
        "body_quat_w": (n, b, 4),
        "body_lin_vel_w": (n, b, 3),
        "body_ang_vel_w": (n, b, 3),
    }
    arrays = {
        name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        for name, shape in shapes.items()
    }
    cursor = 0
    for (name, path), length in zip(clips, lengths):
        data = np.load(path, allow_pickle=True)
        file_bodies = [str(x) for x in np.asarray(data["body_names"]).tolist()]
        bidx = [file_bodies.index(x) for x in bodies]
        arrays["joint_pos"][cursor : cursor + length] = data["joint_pos"]
        arrays["joint_vel"][cursor : cursor + length] = data["joint_vel"]
        arrays["body_pos_w"][cursor : cursor + length] = data["body_pos_w"][:, bidx]
        arrays["body_quat_w"][cursor : cursor + length] = data["body_quat_w"][:, bidx]
        arrays["body_lin_vel_w"][cursor : cursor + length] = data["body_lin_vel_w"][:, bidx]
        arrays["body_ang_vel_w"][cursor : cursor + length] = data["body_ang_vel_w"][:, bidx]
        cursor += length
    for arr in arrays.values():
        arr.flush()
    (out / MANIFEST_FILE).write_text(json.dumps(manifest.to_json(), indent=1))
    return manifest
