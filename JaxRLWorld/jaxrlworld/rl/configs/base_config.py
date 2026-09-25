import copy
import dataclasses
import json
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, TypeVar

from colorama import Fore, Style

T = TypeVar("T", bound="BaseConfig")


def parse_override_args() -> Dict[str, Any]:
    override_dict: Dict[str, Dict[str, Any]] = {}
    override_args = [arg for arg in sys.argv[1:] if "=" in arg]

    for arg in override_args:
        try:
            path, value = arg.split("=")
            parts = path.split(".")

            if len(parts) < 2:
                print(f"Warning: Skipping invalid override format: {arg}. Use format: category.parameter=value")
                continue

            category = parts[0]

            # Parse the value. A JSON list or object (``hidden_dims=[512,256]``,
            # ``optimizer_betas=[0.9,0.95]``) comes through as that value.
            try:
                if value.lower() in ("true", "false"):
                    typed_value = value.lower() == "true"
                elif value[:1] in ("[", "{"):
                    typed_value = json.loads(value)
                elif "." in value or "e" in value.lower():
                    typed_value = float(value)
                else:
                    typed_value = int(value)
            except ValueError:
                typed_value = value

            if len(parts) == 2:
                if category not in override_dict:
                    override_dict[category] = {}
                override_dict[category][parts[1]] = typed_value
            else:
                if category not in override_dict:
                    override_dict[category] = {}

                current = override_dict[category]
                for i in range(1, len(parts) - 1):
                    if parts[i] not in current:
                        current[parts[i]] = {}
                    current = current[parts[i]]

                current[parts[-1]] = typed_value

        except Exception as e:
            print(f"Warning: Failed to parse override argument: {arg}. Error: {e}")

    return override_dict


def _print_override_changes(overrides, config):
    print("\n" + f"{Fore.CYAN}{'=' * 50}")
    print(f"{Fore.YELLOW}Applying command line overrides:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")

    def current(config_obj, param):
        """The value about to be replaced. ``_type`` names the class of a
        nested config that a ``{"_type": ...}`` override swaps out."""
        if isinstance(config_obj, dict):
            return config_obj.get(param)
        if param == "_type":
            return type(config_obj).__name__
        return getattr(config_obj, param)

    def print_nested_changes(params, current_config, prefix=""):
        for param, value in params.items():
            current_value = current(current_config, param)
            if isinstance(value, dict):
                print(f"{prefix}{param}:")
                print_nested_changes(value, current_value, prefix + "  ")
            else:
                print(f"{prefix}{param}:")
                print(f"{prefix}  {Fore.RED}- From: {current_value}{Style.RESET_ALL}")
                print(f"{prefix}  {Fore.GREEN}+ To: {value}{Style.RESET_ALL}")

    for category, params in overrides.items():
        print(f"{Fore.GREEN}{category}:{Style.RESET_ALL}")
        print_nested_changes(params, getattr(config, category))

    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")


from collections.abc import Iterable, Mapping, Sized

import numpy as np

# ── Term discovery (IsaacLab pattern) ──────────────────────────────────────


def iter_terms(cfg: Any, term_type: type) -> dict:
    """Discover named term attributes on *cfg* that are instances of *term_type*.

    Walks the MRO to find class-level defaults, then checks instance overrides.
    Terms set to ``None`` are considered disabled and excluded.
    """
    result = {}
    # Class-level attributes (term defaults defined on the class body)
    for cls in type(cfg).__mro__:
        if cls is object:
            continue
        for name, val in vars(cls).items():
            if name.startswith("_") or name in result:
                continue
            # Get instance value (may override class default)
            instance_val = getattr(cfg, name, val)
            if isinstance(instance_val, term_type):
                result[name] = instance_val
    # Instance-level: check for None overrides (disabling a term)
    for name, val in getattr(cfg, "__dict__", {}).items():
        if name.startswith("_"):
            continue
        if isinstance(val, term_type):
            result[name] = val
        elif val is None and name in result:
            del result[name]
    return result


# ── Serialization (object → dict) ──────────────────────────────────────────

_YAML_SAFE_TYPES = (str, int, float, bool, type(None))


def _convert_value(v: Any) -> Any:
    """Recursively convert a value to a YAML-safe representation.

    Callables are automatically converted to ``"module:qualname"`` strings.
    """
    # StrEnum / IntEnum / Enum: collapse to the underlying primitive value
    # *before* the str/int isinstance check below — otherwise
    # ``isinstance(StrEnum.MEMBER, str)`` returns True and we'd pass the
    # enum instance straight through to ``yaml.dump``, which then emits
    # ``!!python/object/apply:...`` tags that ``yaml.safe_load`` refuses
    # to construct (post-strict-typed-NN-config migration symptom).
    from enum import Enum

    if isinstance(v, Enum):
        return v.value
    if isinstance(v, _YAML_SAFE_TYPES):
        return v
    if isinstance(v, BaseConfig):
        return v.recursive_to_dict()
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _dataclass_to_dict(v)
    if isinstance(v, dict):
        return {str(dk): _convert_value(dv) for dk, dv in v.items()}
    if isinstance(v, list | tuple):
        return [_convert_value(item) for item in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer | np.floating | np.bool_):
        return v.item()
    if callable(v):
        from jaxrlworld.rl.utils.resolve import callable_to_string

        return callable_to_string(v)
    return str(v)


def _dataclass_to_dict(obj: Any) -> Dict:
    """Convert a non-BaseConfig dataclass to a plain dict (no _type metadata)."""
    result = {}
    for f in dataclasses.fields(obj):
        if f.name.startswith("_"):
            continue
        result[f.name] = _convert_value(getattr(obj, f.name))
    return result


def _recursive_to_dict(obj: "BaseConfig") -> Dict:
    """Convert a BaseConfig hierarchy to a serializable dict.

    Walks class-level attributes (for named terms) and instance attributes.
    """
    exclude = set(getattr(obj, "_EXCLUDE_FROM_SERIALIZATION", ()))
    result = {}
    # Collect class-level attributes first (named terms, class defaults)
    for cls in type(obj).__mro__:
        if cls is object:
            continue
        for k, v in vars(cls).items():
            if k.startswith("_") or k in exclude or k in result:
                continue
            # Skip methods, properties, classmethods, ClassVar-like things
            if isinstance(v, property | classmethod | staticmethod):
                continue
            if callable(v) and not dataclasses.is_dataclass(v):
                # Skip plain methods but keep callable term configs (dataclasses are callable)
                continue
            # Get actual instance value (may override class default)
            actual = getattr(obj, k, v)
            if actual is None:
                continue
            result[k] = _convert_value(actual)
    # Instance-level attributes override/add
    for k, v in obj.__dict__.items():
        if k.startswith("_") or k in exclude:
            continue
        result[k] = _convert_value(v)
    return result


# ── Deserialization (dict → object, in-place update) ───────────────────────


def update_from_dict(obj: Any, data: dict, _ns: str = "") -> None:
    """Update *obj* in-place from *data*, following IsaacLab's pattern.

    - Nested Mapping → recurse into existing member.
    - Iterable with nested Mappings → recurse element-wise.
    - Callable attribute + string value → keep as string (resolved lazily).
    - Simple value → assign directly.
    """
    for key, value in data.items():
        key_ns = f"{_ns}/{key}"

        # Check key exists
        if isinstance(obj, dict):
            if key not in obj:
                # For dicts, allow new keys (e.g. entities dict)
                obj[key] = value
                continue
            obj_mem = obj[key]
        elif hasattr(obj, key):
            obj_mem = getattr(obj, key)
        else:
            # Skip unknown keys silently (fields removed, _EXCLUDE_FROM_SERIALIZATION, etc.)
            continue

        # 1) Nested mapping → recurse
        if isinstance(value, Mapping):
            if obj_mem is not None and (hasattr(obj_mem, "__dict__") or isinstance(obj_mem, dict)):
                update_from_dict(obj_mem, value, _ns=key_ns)
                continue
            # obj_mem is None → assign the dict directly
            # (will be a plain dict; consumer code should handle it)

        # 2) Iterable (non-string)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            # 2a) Flat iterable (no nested Mappings) → assign
            if all(not isinstance(el, Mapping) for el in value):
                value = tuple(value) if isinstance(obj_mem, tuple) else value
            # 2b) Iterable with nested Mappings
            elif obj_mem is not None and isinstance(obj_mem, Sized) and len(obj_mem) == len(value):
                for i in range(len(obj_mem)):
                    if isinstance(value[i], Mapping):
                        update_from_dict(obj_mem[i], value[i], _ns=key_ns)
                continue
            # else: length mismatch or obj_mem is None → assign directly

        # 3) Callable attribute + string → keep string for lazy resolution
        elif callable(obj_mem) and isinstance(value, str):
            pass  # value stays as string, resolved_func handles it

        # Assign
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)


@dataclass
class BaseConfig:
    def __init_subclass__(cls, **kwargs) -> None:
        """Reject a field overridden without its type annotation.

        Config settings are dataclass fields, so a subclass that writes
        ``enable_corruption = False`` without the annotation overrides
        nothing: the generated ``__init__`` assigns the INHERITED default
        over it. The config reads as correct and runs as the opposite,
        which is the worst way for one to be wrong — and there is no
        error, no warning, and no wrong-looking number to notice. So
        refuse the class outright rather than trust a convention to hold
        across every preset.

        Terms are unaffected: a term is a plain class attribute, not a
        field, and stays the way every preset writes it.
        """
        super().__init_subclass__(**kwargs)
        overridable = {f.name for f in dataclasses.fields(cls)}
        annotated = cls.__dict__.get("__annotations__", {})
        for name, value in cls.__dict__.items():
            if name in overridable and name not in annotated:
                field_type = cls.__dataclass_fields__[name].type
                type_name = field_type if isinstance(field_type, str) else field_type.__name__
                raise TypeError(
                    f"{cls.__name__} sets {name} = {value!r} without a type annotation, "
                    f"which a dataclass ignores. Write '{name}: {type_name} = {value!r}'."
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for attribute access with default."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    # Fields listed here will be excluded from serialization (e.g. sim-specific objects).
    # ClassVar so dataclass does NOT treat this as an instance field.
    _EXCLUDE_FROM_SERIALIZATION: ClassVar[tuple[str, ...]] = ()

    def recursive_to_dict(self) -> Dict:
        return _recursive_to_dict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create a default instance and update it in-place from *config_dict*."""
        obj = cls()
        update_from_dict(obj, config_dict)
        return obj

    def __repr__(self) -> str:
        """Pretty print the config object with proper indentation and colors"""
        return self._pretty_repr(self.recursive_to_dict())

    @classmethod
    def from_dict_with_overrides(cls, config_dict: Dict) -> "BaseConfig":
        """Create from dict, then apply CLI overrides."""
        config = cls.from_dict(config_dict)
        return config.with_cli_overrides()

    def with_cli_overrides(self: T) -> T:
        """Apply command-line overrides to this config instance."""
        overrides = parse_override_args()
        if overrides:
            _print_override_changes(overrides, self)
            self.apply_overrides(**overrides)
        return self

    def _pretty_repr(self, obj: Any, indent: int = 4, use_colors: bool = True) -> str:
        """
        Create a pretty, readable representation of complex nested data structures.

        Args:
            obj: The object to represent
            indent: Number of spaces for indentation
            use_colors: Whether to add terminal color codes

        Returns:
            String representation of the object
        """
        # Terminal color codes
        BLUE = "\033[94m" if use_colors else ""
        GREEN = "\033[92m" if use_colors else ""
        YELLOW = "\033[93m" if use_colors else ""
        CYAN = "\033[96m" if use_colors else ""
        ENDC = "\033[0m" if use_colors else ""

        # Helper function to add colors to json strings
        def colorize_json(json_str: str) -> str:
            if not use_colors:
                return json_str

            lines = []
            for line in json_str.split("\n"):
                # Check if this is a dict key line
                if ": " in line and '"' in line:
                    parts = line.split(": ", 1)
                    key_part = parts[0]
                    value_part = parts[1] if len(parts) > 1 else ""

                    # Color the key
                    if '"' in key_part:
                        key_part = key_part.replace('"', f'{CYAN}"', 1)
                        if key_part.endswith('"'):
                            key_part = key_part[:-1] + f'"{ENDC}'
                        else:
                            key_part += ENDC

                    # Color values based on type
                    if value_part:
                        if value_part.startswith('"'):
                            value_part = f"{GREEN}{value_part}{ENDC}"
                        elif value_part.strip() in ("true", "false", "null"):
                            value_part = f"{YELLOW}{value_part}{ENDC}"
                        elif value_part[0].isdigit() or value_part.startswith("-"):
                            value_part = f"{BLUE}{value_part}{ENDC}"

                    line = f"{key_part}: {value_part}"
                lines.append(line)
            return "\n".join(lines)

        # Handle serialization of complex objects
        def json_serializer(o):
            if isinstance(o, set | frozenset):
                return list(o)
            return str(o)

        try:
            formatted = json.dumps(obj, indent=indent, default=json_serializer)
            return colorize_json(formatted)
        except (TypeError, ValueError):
            # Fallback to standard representation if JSON serialization fails
            return str(obj)

    def apply_overrides(self, **kwargs):
        """
        Apply specific overrides to configuration settings.

        Example:
            cfg.apply_overrides(
                env={'num_envs': 16},
                algorithm={'learning_rate': 0.0003}
            )
        """
        immutable = getattr(self, "IMMUTABLE_SETTINGS", {})

        for config_type, params in kwargs.items():
            if not hasattr(self, config_type):
                raise ValueError(f"Unknown config type: {config_type}")
            for param_name in params:
                if config_type in immutable and param_name in immutable[config_type]:
                    raise ValueError(f"Cannot override immutable setting: {config_type}.{param_name}")
            _apply_override_params(getattr(self, config_type), params, config_type)


def _is_config_object(obj: Any) -> bool:
    """A nested config that dotted overrides descend into: a ``BaseConfig``
    or a plain dataclass such as a reward/event/observation term."""
    return isinstance(obj, BaseConfig) or (dataclasses.is_dataclass(obj) and not isinstance(obj, type))


def _apply_override_params(config_obj: Any, params: Dict[str, Any], path: str) -> None:
    """Write ``params`` onto ``config_obj``, descending into nested configs.

    Dotted overrides reach any depth (``nn.actor.activation=relu``,
    ``reward.track_lin_vel.weight=3.5``): a dict value is applied field by
    field into a nested config object, whether a ``BaseConfig`` or a plain
    term dataclass, and merges into a dict attribute, descending likewise
    into dict entries that are config objects (``scene.entities.robot``).
    A dict carrying ``_type`` replaces the nested object outright and is
    hydrated by the parent's ``__post_init__``, which also re-coerces
    plain strings (an activation name) the way construction does.
    """
    for param_name, value in params.items():
        if not hasattr(config_obj, param_name):
            raise ValueError(f"Unknown parameter: {path}.{param_name}")
        current = getattr(config_obj, param_name)
        if isinstance(value, dict) and isinstance(current, dict):
            merged = current.copy()
            for key, entry in value.items():
                if (
                    isinstance(entry, dict)
                    and key in merged
                    and _is_config_object(merged[key])
                    and "_type" not in entry
                ):
                    _apply_override_params(merged[key], entry, f"{path}.{param_name}.{key}")
                else:
                    merged[key] = entry
            setattr(config_obj, param_name, merged)
        elif isinstance(value, dict) and _is_config_object(current) and "_type" not in value:
            _apply_override_params(current, value, f"{path}.{param_name}")
        else:
            setattr(config_obj, param_name, value)
    post_init = getattr(type(config_obj), "__post_init__", None)
    if post_init is not None:
        post_init(config_obj)


# ── Saved-config reconciliation ─────────────────────────────────────────────


def diff_config_dicts(saved: Dict[str, Any], current: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """The nested subset of ``saved`` whose leaves differ from ``current``.

    Both are ``recursive_to_dict()`` outputs (``saved`` typically after a
    YAML round trip), so callables are ``"module:qualname"`` strings and
    tuples are lists on both sides. The result has the shape
    ``apply_overrides`` takes: a nested dict of the differing leaves, or
    the whole saved subtree where its ``_type`` names another class than
    the current one (that subtree is then hydrated by ``_type``). A key
    absent from ``current`` is kept, so applying the diff fails loudly on
    a field the current preset no longer has. A saved ``None`` against a
    nested config disables it (a term switched off for the run). A saved
    dict facing a plain value, or a sequence of nested mappings that
    differs, cannot be expressed as an override and raises. Subtrees are
    copied, so applying the result never mutates ``saved`` (hydration of
    a ``_type`` dict consumes it).
    """
    diff: Dict[str, Any] = {}
    for key, saved_value in saved.items():
        leaf_path = f"{path}.{key}" if path else key
        if key not in current:
            diff[key] = copy.deepcopy(saved_value)
            continue
        current_value = current[key]
        if isinstance(saved_value, dict) and isinstance(current_value, dict):
            if saved_value.get("_type") != current_value.get("_type"):
                diff[key] = copy.deepcopy(saved_value)
            else:
                sub = diff_config_dicts(saved_value, current_value, leaf_path)
                if sub:
                    diff[key] = sub
        elif saved_value is None and current_value is not None:
            # A nested config (a named term) disabled by the saved run.
            diff[key] = None
        elif isinstance(saved_value, dict) or isinstance(current_value, dict):
            raise ValueError(
                f"{leaf_path}: saved {type(saved_value).__name__} against current {type(current_value).__name__}; "
                "a nested config cannot be reconciled with a plain value"
            )
        elif isinstance(saved_value, list) or isinstance(current_value, list):
            if saved_value != current_value:
                if any(isinstance(el, dict) for el in (saved_value or [])) or any(
                    isinstance(el, dict) for el in (current_value or [])
                ):
                    raise ValueError(
                        f"{leaf_path}: a sequence of nested configs differs from the saved one; "
                        "it cannot be reconciled as an override"
                    )
                diff[key] = list(saved_value)
        elif saved_value != current_value:
            diff[key] = saved_value
    return diff


def flatten_leaf_paths(nested: Dict[str, Any], path: str = "") -> list[str]:
    """Dotted paths of every leaf in a nested override dict."""
    paths: list[str] = []
    for key, value in nested.items():
        leaf_path = f"{path}.{key}" if path else key
        if isinstance(value, dict) and value:
            paths.extend(flatten_leaf_paths(value, leaf_path))
        else:
            paths.append(leaf_path)
    return paths
