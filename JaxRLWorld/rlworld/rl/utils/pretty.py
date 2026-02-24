"""Pretty printing utilities using Rich library for JaxRLWorld.

This module provides Rich-based pretty printing functions for displaying
environment and manager information in a colorful, structured format.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def get_console() -> Console:
    """Get a Rich console instance."""
    return Console()


def format_shape(shape: tuple | int) -> str:
    """Format shape tuple for display.

    Args:
        shape: Shape tuple or integer dimension.

    Returns:
        Formatted string like "[3]" or "[3, 12]".
    """
    if isinstance(shape, int):
        return f"[{shape}]"
    if len(shape) == 1:
        return f"[{shape[0]}]"
    return f"[{', '.join(str(d) for d in shape)}]"


def format_weight(weight: Any) -> str:
    """Format reward weight for display.

    Handles constant weights and various schedule types.

    Args:
        weight: Weight value or WeightSchedule instance.

    Returns:
        Formatted string representation of the weight.
    """
    from rlworld.rl.configs.rewards import (
        WeightSchedule, LinearSchedule, ExponentialDecay, StepSchedule
    )

    if isinstance(weight, (int, float)):
        if abs(weight) < 0.0001 and weight != 0:
            return f"{weight:.2e}"
        return f"{weight}"

    if isinstance(weight, LinearSchedule):
        return f"Linear({weight.initial}→{weight.final} @{weight.total_steps})"

    if isinstance(weight, ExponentialDecay):
        return f"ExpDecay({weight.initial}, rate={weight.decay_rate})"

    if isinstance(weight, StepSchedule):
        if weight.milestones:
            first_step, first_val = weight.milestones[0]
            return f"Step({weight.default}→{first_val} @{first_step})"
        return f"Step({weight.default})"

    if isinstance(weight, WeightSchedule):
        return f"Schedule({type(weight).__name__})"

    return str(weight)


def create_env_panel(
    title: str,
    rows: list[tuple[str, str]],
    subtitle: str | None = None,
    border_style: str = "blue"
) -> Panel:
    """Create an environment info panel with key-value pairs.

    Args:
        title: Panel title.
        rows: List of (key, value) tuples to display.
        subtitle: Optional subtitle.
        border_style: Border color style.

    Returns:
        Rich Panel object.
    """
    # Create two-column layout for key-value pairs
    content_lines = []

    # Process rows in pairs for side-by-side display
    for i in range(0, len(rows), 2):
        left_key, left_val = rows[i]
        line = f"  [bold cyan]{left_key:<12}[/] {left_val:<16}"

        if i + 1 < len(rows):
            right_key, right_val = rows[i + 1]
            line += f"[bold cyan]{right_key:<12}[/] {right_val}"

        content_lines.append(line)

    content = "\n".join(content_lines)

    return Panel(
        content,
        title=f"[bold]{title}[/]",
        subtitle=subtitle,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def create_manager_table(
    title: str,
    columns: list[str],
    rows: list[list[Any]],
    footer: str | None = None,
    title_style: str = "bold white on blue",
    header_style: str = "bold cyan",
) -> Table:
    """Create a styled table for manager information.

    Args:
        title: Table title.
        columns: Column headers.
        rows: List of row data (each row is a list of values).
        footer: Optional footer text (e.g., "Total: 48 dims").
        title_style: Style for the title.
        header_style: Style for column headers.

    Returns:
        Rich Table object.
    """
    table = Table(
        title=title,
        title_style=title_style,
        box=box.ROUNDED,
        show_header=True,
        header_style=header_style,
        padding=(0, 1),
        collapse_padding=True,
    )

    # Add columns with appropriate justification
    for i, col in enumerate(columns):
        if col.lower() in ("idx", "index", "#"):
            table.add_column(col, justify="center", style="dim")
        elif col.lower() in ("shape", "dims", "dim"):
            table.add_column(col, justify="center", style="green")
        elif col.lower() in ("weight", "scale"):
            table.add_column(col, justify="right", style="yellow")
        elif col.lower() in ("name", "joint", "link"):
            table.add_column(col, justify="left", style="white")
        else:
            table.add_column(col, justify="left")

    # Add rows
    for row in rows:
        table.add_row(*[str(val) for val in row])

    # Add footer if provided
    if footer:
        table.caption = f"[dim]{footer}[/]"
        table.caption_justify = "right"

    return table


def table_to_string(table: Table) -> str:
    """Convert Rich Table to string for __str__ methods.

    Args:
        table: Rich Table object.

    Returns:
        String representation of the table.
    """
    console = Console(file=StringIO(), width=100)
    console.print(table)
    return console.file.getvalue()


def panel_to_string(panel: Panel) -> str:
    """Convert Rich Panel to string for __str__ methods.

    Args:
        panel: Rich Panel object.

    Returns:
        String representation of the panel.
    """
    console = Console(file=StringIO(), force_terminal=True, width=100)
    console.print(panel)
    return console.file.getvalue()


def print_env_summary(env: "World") -> None:
    """Print complete environment summary with all managers.

    This is the main function to call when displaying environment info.
    Prints panels and tables for all registered managers.

    Args:
        env: The environment instance (GenesisEnv or NewtonEnv).
    """
    console = get_console()

    # Environment header panel
    env_rows = [
        ("Simulator", env.sim_name),
        ("Seed", str(env.seed)),
        ("Num Envs", str(env.num_envs)),
        ("Device", str(env.device)),
        ("Physics dt", f"{env.physics_dt:.4f}s"),
        ("Control dt", f"{env.control_dt:.4f}s"),
    ]

    if hasattr(env, 'decimation'):
        env_rows.append(("Decimation", str(env.decimation)))

    if hasattr(env, 'task_name'):
        env_rows.append(("Task", env.task_name))

    panel = create_env_panel(
        title=f"{env.sim_name} Environment",
        rows=env_rows,
        border_style="blue",
    )
    console.print(panel)
    console.print()

    # Print each manager
    managers = [
        ("obs_manager", "Observation Manager"),
        ("act_manager", "Action Manager"),
        ("reward_manager", "Reward Manager"),
        ("termination_manager", "Termination Manager"),
        ("contact_manager", "Contact Manager"),
        ("command_manager", "Command Manager"),
        ("event_manager", "Event Manager"),
    ]

    for attr_name, display_name in managers:
        if hasattr(env, attr_name):
            manager = getattr(env, attr_name)
            if manager is not None and hasattr(manager, '__str__'):
                manager_str = str(manager)
                if manager_str.strip():  # Only print if non-empty
                    console.print(manager_str)
                    console.print()