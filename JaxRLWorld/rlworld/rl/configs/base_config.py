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

            # Parse the value
            try:
                if value.lower() in ("true", "false"):
                    typed_value = value.lower() == "true"
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

    def print_nested_changes(params, current_config, prefix=""):
        for param, value in params.items():
            if isinstance(value, dict):
                current_value = (
                    current_config[param] if isinstance(current_config, dict) else getattr(current_config, param)
                )
                print(f"{prefix}{param}:")
                print_nested_changes(value, current_value, prefix + "  ")
            else:
                current_value = (
                    current_config[param] if isinstance(current_config, dict) else getattr(current_config, param)
                )
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
        from rlworld.rl.utils.resolve import callable_to_string

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

            config_obj = getattr(self, config_type)
            for param_name, value in params.items():
                # Check immutable settings if defined
                if config_type in immutable and param_name in immutable[config_type]:
                    raise ValueError(f"Cannot override immutable setting: {config_type}.{param_name}")

                if not hasattr(config_obj, param_name):
                    raise ValueError(f"Unknown parameter: {config_type}.{param_name}")

                if isinstance(value, dict) and isinstance(getattr(config_obj, param_name), dict):
                    current_dict = getattr(config_obj, param_name).copy()
                    current_dict.update(value)
                    setattr(config_obj, param_name, current_dict)
                else:
                    setattr(config_obj, param_name, value)
