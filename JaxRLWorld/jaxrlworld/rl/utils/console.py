"""Centralized ANSI console helpers for colored output."""

import os
from typing import Any


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# Convenience aliases
GREEN = Colors.GREEN
YELLOW = Colors.YELLOW
RED = Colors.RED
BLUE = Colors.BLUE
CYAN = Colors.CYAN
BOLD = Colors.BOLD
DIM = Colors.DIM
RESET = Colors.RESET
MAGENTA = Colors.HEADER


def print_header(text: str, width: int = 70):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * width}{Colors.RESET}")


def print_subheader(text: str, width: int = 70):
    """Print a formatted subheader."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─' * width}{Colors.RESET}")


def print_key_value(key: str, value: Any, key_width: int = 20):
    """Print a key-value pair with formatting."""
    print(f"  {Colors.DIM}{key:<{key_width}}{Colors.RESET}: {Colors.GREEN}{value}{Colors.RESET}")


def print_path(label: str, path: str, key_width: int = 20):
    """Print a path with both relative and absolute versions."""
    abs_path = os.path.abspath(path)
    print(f"  {Colors.DIM}{label:<{key_width}}{Colors.RESET}: {Colors.GREEN}{path}{Colors.RESET}")
    print(f"  {Colors.DIM}{'':<{key_width}}{Colors.RESET}  {Colors.DIM}({abs_path}){Colors.RESET}")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", width: int = 40):
    """Print a progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix} {Colors.CYAN}│{bar}│{Colors.RESET} {percent * 100:5.1f}% {suffix}", end="", flush=True)
