from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

import torch
from matplotlib import pyplot as plt
from scipy.signal import butter


class EnvStepCache:
    """Decorator class for caching global function results based on cache generation."""

    _global_cache: dict[int, dict[str, tuple[int, Any]]] = {}

    def __call__(self, func: Callable) -> Callable:
        func_name = func.__name__

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


class MethodEnvStepCache:
    """Decorator class for caching method results based on env step counter."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._current_step_call: int = 0

    def __call__(self, func: Callable) -> Callable:
        """Creates a wrapper function that implements the caching logic.

        Args:
            func: The method to be cached

        Returns:
            A wrapper function that implements caching
        """
        func_name = func.__name__

        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            # Create a unique cache key for this instance and method
            instance_id = id(instance)
            cache_key = f"{instance_id}:{func_name}"

            # Get current env_step_counter
            current_step_calls = getattr(instance, '_env_step_counter', 0)

            # Check if we have a valid cached result
            if cache_key in self._cache:
                cached_step_calls, cached_result = self._cache[cache_key]
                if cached_step_calls == current_step_calls:
                    return cached_result

            # If no valid cache, compute result and cache it
            result = func(instance, *args, **kwargs)
            self._cache[cache_key] = (current_step_calls, result)

            return result

        return wrapper

    def has_cache(self, instance, func_name):
        """Check if there is a valid cache for the given instance and function"""
        instance_id = id(instance)
        cache_key = f"{instance_id}:{func_name}"

        if cache_key in self._cache:
            cached_step_calls, _ = self._cache[cache_key]
            current_step_calls = getattr(instance, '_env_step_counter', 0)
            return cached_step_calls == current_step_calls

        return False

    def get_cache(self, instance, func_name):
        """Get cached value if it exists"""
        instance_id = id(instance)
        cache_key = f"{instance_id}:{func_name}"

        if self.has_cache(instance, func_name):
            _, cached_result = self._cache[cache_key]
            return cached_result

        return None


# For methods that should invalidate the cache when called
def invalidate_cache(func: Callable) -> Callable:
    """Decorator to invalidate the cache when certain methods are called.

    Args:
        func: The method that should trigger cache invalidation

    Returns:
        A wrapper function that clears the cache before executing the method
    """

    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        # Clear all caches associated with this instance
        instance_id = id(instance)
        if hasattr(instance, '_step_calls_cache'):
            cache = instance._step_calls_cache._cache
            # Remove all cached entries for this instance
            cache_keys = [k for k in cache.keys() if k.startswith(f"{instance_id}:")]
            for key in cache_keys:
                del cache[key]
        return func(instance, *args, **kwargs)

    return wrapper


class LearningIterationObserver:
    """Deprecated: kept as empty mixin for backward compatibility.
    Learning iteration is now tracked directly by the runner."""

    @property
    def learning_iteration(self):
        raise AttributeError(
            "LearningIterationObserver singleton removed. "
            "Use runner.current_learning_iteration instead."
        )


class NumStepCallsObserver:
    """Deprecated: kept as empty mixin for backward compatibility.
    Env step count is now accessed via env._env_step_counter."""

    @property
    def env_step_calls(self):
        raise AttributeError(
            "NumStepCallsObserver singleton removed. "
            "Use env._env_step_counter instead."
        )
