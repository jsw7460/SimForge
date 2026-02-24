import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

from colorama import Fore, Style

if TYPE_CHECKING:
    from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun
    from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun


def parse_override_args() -> Dict[str, Any]:
    override_dict: Dict[str, Dict[str, Any]] = {}
    override_args = [arg for arg in sys.argv[1:] if '=' in arg]

    for arg in override_args:
        try:
            path, value = arg.split('=')
            parts = path.split('.')

            if len(parts) < 2:
                print(f"Warning: Skipping invalid override format: {arg}. Use format: category.parameter=value")
                continue

            category = parts[0]

            # Parse the value
            try:
                if value.lower() in ('true', 'false'):
                    typed_value = value.lower() == 'true'
                elif '.' in value or 'e' in value.lower():
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
                current_value = current_config[param] if isinstance(current_config, dict) else getattr(
                    current_config, param
                )
                print(f"{prefix}{param}:")
                print_nested_changes(value, current_value, prefix + "  ")
            else:
                current_value = current_config[param] if isinstance(current_config, dict) else getattr(
                    current_config, param
                )
                print(f"{prefix}{param}:")
                print(f"{prefix}  {Fore.RED}- From: {current_value}{Style.RESET_ALL}")
                print(f"{prefix}  {Fore.GREEN}+ To: {value}{Style.RESET_ALL}")

    for category, params in overrides.items():
        print(f"{Fore.GREEN}{category}:{Style.RESET_ALL}")
        print_nested_changes(params, getattr(config, category))

    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")


@dataclass
class BaseConfig:

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for attribute access with default."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def recursive_to_dict(self) -> Dict:
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            result[k] = self._to_serializable(v)
        return result

    @staticmethod
    def _to_serializable(v):
        """Recursively convert a value to a pickle-safe form."""
        if isinstance(v, BaseConfig):
            return v.recursive_to_dict()
        if callable(v) and not isinstance(v, type):
            return repr(v)
        if hasattr(v, '__dataclass_fields__'):
            return {
                k: BaseConfig._to_serializable(val)
                for k, val in v.__dict__.items()
                if not k.startswith('_')
            }
        if isinstance(v, dict):
            return {k: BaseConfig._to_serializable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [BaseConfig._to_serializable(item) for item in v]
        return v

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # Collect all field names from the full class hierarchy
        all_fields = set()
        for klass in cls.__mro__:
            all_fields.update(getattr(klass, '__annotations__', {}).keys())

        return cls(
            **{
                k: v for k, v in config_dict.items()
                if k in all_fields
            }
        )

    def __repr__(self) -> str:
        """Pretty print the config object with proper indentation and colors"""
        return self._pretty_repr(self.recursive_to_dict())

    @classmethod
    def from_dict_with_overrides(cls, config_dict: Dict) -> "GenesisConfigsForRun":
        """admit command line overrides"""
        # Base config
        config = cls.from_dict(config_dict)

        # Parsing and applying overrides
        overrides = parse_override_args()
        if overrides:
            _print_override_changes(overrides, config)
            config.apply_overrides(**overrides)

        return config

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
        BLUE = '\033[94m' if use_colors else ''
        GREEN = '\033[92m' if use_colors else ''
        YELLOW = '\033[93m' if use_colors else ''
        CYAN = '\033[96m' if use_colors else ''
        ENDC = '\033[0m' if use_colors else ''

        # Helper function to add colors to json strings
        def colorize_json(json_str: str) -> str:
            if not use_colors:
                return json_str

            lines = []
            for line in json_str.split('\n'):
                # Check if this is a dict key line
                if ': ' in line and '"' in line:
                    parts = line.split(': ', 1)
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
                            value_part = f'{GREEN}{value_part}{ENDC}'
                        elif value_part.strip() in ('true', 'false', 'null'):
                            value_part = f'{YELLOW}{value_part}{ENDC}'
                        elif value_part[0].isdigit() or value_part.startswith('-'):
                            value_part = f'{BLUE}{value_part}{ENDC}'

                    line = f"{key_part}: {value_part}"
                lines.append(line)
            return '\n'.join(lines)

        # Handle serialization of complex objects
        def json_serializer(o):
            if isinstance(o, (set, frozenset)):
                return list(o)
            return str(o)

        try:
            formatted = json.dumps(obj, indent=indent, default=json_serializer)
            return colorize_json(formatted)
        except (TypeError, ValueError) as e:
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
        immutable = getattr(self, 'IMMUTABLE_SETTINGS', {})

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
