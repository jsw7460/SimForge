from __future__ import annotations

from functools import wraps
from typing import Any, Callable


class EnvStepCache:
    """Decorator class for caching global function results based on cache generation."""

    _global_cache: dict[int, dict[str, tuple[int, Any]]] = {}

    def __call__(self, func: Callable) -> Callable:
        # Module-qualified: backends define same-named readers (``dof_pos``
        # in common/ and in each backend's observations), and a bare
        # ``__name__`` key would hand one of them the other's result.
        func_name = f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def wrapper(env, *args, **kwargs):
            env_id = id(env)
            args_key = str(hash(tuple(args) + tuple(sorted(kwargs.items()))))
            cache_key = f"{func_name}:{args_key}"

            current_gen = env._cache_generation

            if env_id not in self._global_cache:
                self._global_cache[env_id] = {}

            if cache_key in self._global_cache[env_id]:
                cached_gen, cached_result = self._global_cache[env_id][cache_key]
                if cached_gen == current_gen:
                    return cached_result

            result = func(env, *args, **kwargs)
            self._global_cache[env_id][cache_key] = (current_gen, result)

            return result

        return wrapper

    @classmethod
    def clear_cache(cls, env=None):
        if env is None:
            cls._global_cache.clear()
        else:
            env_id = id(env)
            if env_id in cls._global_cache:
                del cls._global_cache[env_id]


# For methods that should invalidate the cache when called
class LearningIterationObserver:
    """Deprecated: kept as empty mixin for backward compatibility.
    Learning iteration is now tracked directly by the runner."""

    @property
    def learning_iteration(self):
        raise AttributeError(
            "LearningIterationObserver singleton removed. Use runner.current_learning_iteration instead."
        )


class NumStepCallsObserver:
    """Deprecated: kept as empty mixin for backward compatibility.
    Env step count is now accessed via env._env_step_counter."""

    @property
    def env_step_calls(self):
        raise AttributeError("NumStepCallsObserver singleton removed. Use env._env_step_counter instead.")
