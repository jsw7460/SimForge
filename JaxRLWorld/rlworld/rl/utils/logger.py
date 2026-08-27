from __future__ import annotations

import atexit
import os
import statistics
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
import wandb
from colorama import Fore, Style, init

from rlworld.rl.algorithms.metrics import BaseMetrics, MetricType

if TYPE_CHECKING:
    from rlworld.rl.runners.iteration_data import IterationData

init(autoreset=True)


class ConsoleWriter:
    """
    Standalone console writer that always prints training metrics.
    Independent of any logger backend (WandB, TensorBoard, etc.)
    """

    # Type-based color mapping
    TYPE_COLORS = {
        MetricType.LOSS: Fore.CYAN,
        MetricType.ENTROPY: Fore.MAGENTA,
        MetricType.COEFFICIENT: Fore.YELLOW,
        MetricType.VALUE: Fore.GREEN,
        MetricType.RATIO: Fore.BLUE,
        MetricType.COUNT: Fore.WHITE,
        MetricType.STRING: Fore.YELLOW,
    }

    def __init__(self):
        self.pad = 35
        # Rich live dashboard: one in-place-updating panel instead of a
        # ~40-line block scrolling by every iteration. Only when stdout
        # is a real terminal — log files, tee pipes, and CHTC condor
        # jobs keep the plain scrolling blocks (they need parseable,
        # append-only text). JAXRLWORLD_PLAIN_LOG=1 forces plain.
        self._live = None
        self._live_disabled = os.environ.get("JAXRLWORLD_PLAIN_LOG", "0") == "1" or not sys.stdout.isatty()
        # Rolling window for the dashboard's timing statistics.
        self._roll: deque = deque(maxlen=50)
        self._prev_rewards: Dict[str, float] = {}

    def _ensure_live(self):
        """Start the rich Live region on first use (TTY only)."""
        if self._live is not None or self._live_disabled:
            return self._live
        try:
            from rich.console import Console
            from rich.live import Live
        except ImportError:
            print("[ConsoleWriter] rich not installed — falling back to plain block logging.")
            self._live_disabled = True
            return None
        self._console = Console()
        self._live = Live(console=self._console, refresh_per_second=4, transient=False)
        self._live.start()
        atexit.register(self._stop_live)
        return self._live

    def _stop_live(self):
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    def write_metrics(
        self,
        data: Dict,
        metrics: BaseMetrics | None,
        mode: str,
        width: int = 100,
        print_reward_stats: bool = True,
        last_eval_stats: Dict | None = None,
    ):
        """Write formatted metrics to console."""
        log_string = []

        # Header section
        if mode == "train":
            iteration = data.get("iteration", "?")
            total_iterations = data.get("total_iterations", "?")
            header = f" Learning iteration {Fore.GREEN}{iteration}/{total_iterations}{Style.RESET_ALL} "
        else:
            header = " Evaluation Results "

        log_string.extend(self._create_section_header(width, header))

        # Run info section
        log_string.extend(self._format_run_info_section(data))

        # Performance metrics (training only)
        if mode == "train":
            perf_metrics = {
                "fps": data.get("fps", 0),
                "collection_time": data.get("collection_time", 0),
                "learning_time": data.get("learning_time", 0),
                "total_time": data.get("total_time", 0),
                "total_timesteps": data.get("total_timesteps", 0),
            }
            log_string.extend(self._format_performance_section(perf_metrics))
            log_string.extend(self._format_algorithm_metrics(metrics))

        # Episode statistics
        log_string.extend(self._format_episode_stats(data))

        # Reward statistics
        if print_reward_stats and "reward_stats" in data:
            log_string.extend(self._format_reward_stats(data))

        # Persistent eval stats section
        if last_eval_stats is not None:
            log_string.extend(self._format_eval_stats(last_eval_stats))

        # Summary (training only)
        if mode == "train":
            log_string.extend(self._format_summary(data, perf_metrics))

        # Footer
        log_string.append(f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}")

        text = "\n".join(log_string)
        if self._live is not None:
            # Print above the live dashboard (rich keeps the region at
            # the bottom), so eval blocks leave a persistent record.
            from rich.text import Text

            self._live.console.print(Text.from_ansi(text))
        else:
            print(text)

    def write_iteration(
        self,
        data: IterationData,
        context: Dict[str, Any],
        width: int = 100,
        last_eval_stats: Dict | None = None,
    ):
        """Write formatted IterationData to console."""
        # Build a compat dict for the existing section formatters
        ep = data.episode_stats
        compat = {
            "iteration": data.iteration,
            "total_iterations": context.get("total_iterations", "?"),
            "fps": data.fps,
            "collection_time": data.collection_time,
            "learning_time": data.learning_time,
            "total_time": data.total_time,
            "total_timesteps": data.total_timesteps,
            "mean_return": ep.mean_return,
            "mean_episode_length": ep.mean_episode_length,
            "success_rate": ep.success_rate,
            "reward_stats": ep.reward_stats,
        }
        # Merge display context (wandb_url, simulator, etc.)
        compat.update(context)

        live = self._ensure_live()
        if live is not None:
            try:
                self._roll.append((data.fps, data.collection_time, data.learning_time))
                live.update(self._build_dashboard(compat, data.metrics, last_eval_stats))
                return
            except Exception:
                print("[ConsoleWriter] rich dashboard failed — falling back to plain block logging.")
                traceback.print_exc()
                self._live_disabled = True
                self._stop_live()

        self.write_metrics(
            data=compat,
            metrics=data.metrics,
            mode="train",
            print_reward_stats=True,
            last_eval_stats=last_eval_stats,
        )

    def _build_dashboard(self, data: Dict, metrics: BaseMetrics | None, last_eval_stats: Dict | None):
        """One rich renderable holding the whole iteration view.

        Replaces the scrolling per-iteration block in TTY sessions: the
        panel updates in place, eval blocks print above it (persistent),
        and the timing row carries rolling statistics over the last 50
        iterations — information the scrollback used to provide.
        """
        from rich.console import Group
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.table import Table
        from rich.text import Text

        rows = []

        def section(title: str, style: str) -> None:
            # A labelled horizontal rule separates each category.
            rows.append(Text(""))
            rows.append(Rule(f"[bold {style}]{title}[/]", style="dim", align="left"))

        def indent(renderable) -> None:
            rows.append(Padding(renderable, (0, 0, 0, 2)))

        # ── run info ────────────────────────────────────────────────
        info_bits = [f"[yellow]{data.get('simulator', 'N/A')}[/] · [yellow]{data.get('task_name', 'N/A')}[/]"]
        if "wandb_url" in data:
            info_bits.append(f"[dim]{data['wandb_url']}[/]")
        rows.append(Text.from_markup("   ".join(info_bits)))
        rows.append(Text(""))

        # ── progress ────────────────────────────────────────────────
        iteration = data.get("iteration", 0)
        total = data.get("total_iterations", 0)
        frac = (iteration / total) if isinstance(total, int | float) and total else 0.0
        bar_w = 44
        filled = int(round(frac * bar_w))
        bar = Text.assemble(
            (f"iter {iteration:,}/{total:,}  " if isinstance(total, int | float) else f"iter {iteration}  ", "bold"),
            ("█" * filled, "cyan"),
            ("░" * (bar_w - filled), "grey35"),
            (f"  {frac * 100:5.1f}%", "cyan"),
            (f"   elapsed {_format_time(data.get('total_time', 0))}", "white"),
            (f"   ETA {self._calculate_eta(data, data.get('total_time', 0))}", "yellow"),
        )
        rows.append(bar)

        # ── throughput / timing with rolling stats ──────────────────
        fps = data.get("fps", 0)
        col = data.get("collection_time", 0.0)
        lrn = data.get("learning_time", 0.0)
        if len(self._roll) >= 2:
            med_fps = statistics.median(r[0] for r in self._roll)
            med_col = statistics.median(r[1] for r in self._roll)
            med_lrn = statistics.median(r[2] for r in self._roll)
            roll_txt = f"   [dim]last {len(self._roll)} median: {med_fps:,.0f} st/s · {med_col:.3f}s/{med_lrn:.3f}s[/]"
        else:
            roll_txt = ""
        section("Performance", "yellow")
        indent(
            Text.from_markup(
                f"[bold green]{fps:,.0f}[/] steps/s   "
                f"collect [cyan]{col:.3f}s[/] │ learn [cyan]{lrn:.3f}s[/]{roll_txt}"
            )
        )

        # ── algorithm metrics ───────────────────────────────────────
        if metrics is not None:
            section("Algorithm", "red")
            grid = Table.grid(padding=(0, 3))
            items = []
            for m in metrics.get_console_metrics():
                value = f"{m.value:.4f}" if isinstance(m.value, float) else str(m.value)
                items.append(f"[dim]{m.display_name}[/] [cyan]{value}[/]")
            per_row = 4
            for i in range(0, len(items), per_row):
                grid.add_row(*items[i : i + per_row])
            indent(grid)

        # ── episode stats ───────────────────────────────────────────
        mean_return = data.get("mean_return", 0.0)
        ret_style = "green" if mean_return >= 0 else "red"
        ep_line = (
            f"return [{ret_style}]{mean_return:.2f}[/]   " f"ep len [cyan]{data.get('mean_episode_length', 0.0):.1f}[/]"
        )
        success = data.get("success_rate")
        if success is not None:
            ep_line += f"   success [yellow]{success * 100:.1f}%[/]"
        ep_line += f"   steps [white]{data.get('total_timesteps', 0):,}[/]"
        section("Episode", "green")
        indent(Text.from_markup(ep_line))

        # ── rewards, |value| descending, with change arrows ─────────
        reward_stats = data.get("reward_stats") or {}
        if reward_stats:
            section("Rewards (|value| desc)", "blue")
            items = sorted(
                ((k, v["mean"]) for k, v in reward_stats.items()),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            table = Table.grid(padding=(0, 1))
            per_row = 3
            cells = []
            for name, mean in items:
                prev = self._prev_rewards.get(name)
                if prev is None or abs(mean - prev) < 1e-6:
                    arrow = " "
                elif mean > prev:
                    arrow = "[green]▲[/]"
                else:
                    arrow = "[red]▼[/]"
                style = "green" if mean >= 0 else "red"
                cells.append(f"[white]{name[:24]:<24}[/][{style}]{mean:>9.4f}[/]{arrow}")
            for i in range(0, len(cells), per_row):
                table.add_row(*cells[i : i + per_row])
            indent(table)
            self._prev_rewards = {k: v for k, v in items}

        # ── pinned eval ─────────────────────────────────────────────
        if last_eval_stats is not None:
            section("Eval", "magenta")
            indent(
                Text.from_markup(
                    f"[magenta]@iter {last_eval_stats.get('eval/iteration', '?')}:[/] "
                    f"R [bold]{last_eval_stats.get('eval/mean_return', 0.0):.2f}[/]"
                    f" ± {last_eval_stats.get('eval/std_return', 0.0):.2f}   "
                    f"len {last_eval_stats.get('eval/mean_episode_length', 0.0):.1f}   "
                    f"({last_eval_stats.get('eval/num_episodes', 0)} eps)"
                )
            )

        title = data.get("wandb_run_name") or data.get("task_name", "training")
        return Panel(Group(*rows), title=f"[bold]{title}[/]", border_style="cyan")

    def _create_section_header(self, width: int, title: str) -> List[str]:
        """Create a formatted section header."""
        return [
            f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}",
            title.center(width, " ").rstrip(),
            f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}",
            "",
        ]

    def _format_run_info_section(self, data: Dict) -> List[str]:
        """Format run information section with WandB, simulator, and task info."""
        lines = [f"{Fore.MAGENTA}🚀 Run Info:{Style.RESET_ALL}"]

        # WandB run name and URL
        if "wandb_url" in data:
            wandb_url = data["wandb_url"]
            run_name = data["wandb_run_name"]
            lines.append(
                f"  {Fore.WHITE}Run{Style.RESET_ALL}".ljust(self.pad + 9) + f"{Fore.CYAN}{run_name}{Style.RESET_ALL}"
            )
            lines.append(
                f"  {Fore.WHITE}WandB{Style.RESET_ALL}".ljust(self.pad + 9) + f"{Fore.BLUE}{wandb_url}{Style.RESET_ALL}"
            )

        if "wandb_run_path" in data:
            lines.append(
                f"  {Fore.WHITE}Run Path{Style.RESET_ALL}".ljust(self.pad + 9)
                + f"{Fore.GREEN}{data['wandb_run_path']}{Style.RESET_ALL}"
            )

        # Simulator and task
        simulator = data.get("simulator", "N/A")
        task_name = data.get("task_name", "N/A")

        lines.append(
            f"  {Fore.WHITE}Simulator{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"{Fore.YELLOW}{simulator}{Style.RESET_ALL}"
        )
        lines.append(
            f"  {Fore.WHITE}Task{Style.RESET_ALL}".ljust(self.pad + 9) + f"{Fore.YELLOW}{task_name}{Style.RESET_ALL}"
        )

        # Log directory
        if "log_dir" in data:
            lines.append(
                f"  {Fore.WHITE}Log Dir{Style.RESET_ALL}".ljust(self.pad + 9)
                + f"{Fore.WHITE}{data['log_dir']}{Style.RESET_ALL}"
            )

        lines.append("")
        return lines

    def _format_performance_section(self, perf_metrics: Dict) -> List[str]:
        """Format performance metrics section."""
        if not perf_metrics:
            return []

        lines = [f"{Fore.YELLOW}⚡ Performance:{Style.RESET_ALL}"]

        fps = perf_metrics.get("fps", 0)
        collection_time = perf_metrics.get("collection_time", 0)
        learning_time = perf_metrics.get("learning_time", 0)

        lines.append(
            f"  {Fore.WHITE}Throughput{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"{Fore.GREEN}{fps:,.0f}{Style.RESET_ALL} steps/sec"
        )
        lines.append(
            f"  {Fore.WHITE}Timing{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"collect {Fore.CYAN}{collection_time:.3f}s{Style.RESET_ALL} │ "
            f"learn {Fore.CYAN}{learning_time:.3f}s{Style.RESET_ALL}"
        )

        return lines

    def _format_algorithm_metrics(self, metrics: BaseMetrics | None) -> List[str]:
        """Format algorithm metrics from BaseMetrics object."""
        lines = ["", f"{Fore.RED}📉 Algorithm:{Style.RESET_ALL}"]

        if metrics is None:
            lines.append("  No metrics available")
            return lines

        console_metrics = metrics.get_console_metrics()
        metric_items = []
        for m in console_metrics:
            color = self.TYPE_COLORS.get(m.metric_type, Fore.WHITE)
            metric_items.append((m.display_name, m.value, color))

        lines.extend(self._format_metric_rows(metric_items))
        return lines

    def _format_episode_stats(self, data: Dict) -> List[str]:
        """Format episode statistics section."""
        lines = ["", f"{Fore.GREEN}📈 Episode Stats:{Style.RESET_ALL}"]

        if "mean_return" in data:
            mean_return = data["mean_return"]
            color = Fore.GREEN if mean_return >= 0 else Fore.RED
            lines.append(
                f"  {Fore.WHITE}Mean Return{Style.RESET_ALL}".ljust(self.pad + 9)
                + f"{color}{mean_return:.2f}{Style.RESET_ALL}"
            )

        if "mean_episode_length" in data:
            lines.append(
                f"  {Fore.WHITE}Mean Episode Length{Style.RESET_ALL}".ljust(self.pad + 9)
                + f"{Fore.CYAN}{data['mean_episode_length']:.1f}{Style.RESET_ALL}"
            )

        if "success_rate" in data and data["success_rate"] is not None:
            success_rate = data["success_rate"] * 100
            color = Fore.GREEN if success_rate >= 50 else Fore.YELLOW if success_rate >= 20 else Fore.RED
            lines.append(
                f"  {Fore.WHITE}Success Rate{Style.RESET_ALL}".ljust(self.pad + 9)
                + f"{color}{success_rate:.1f}%{Style.RESET_ALL}"
            )

        return lines

    def _format_reward_stats(self, data: Dict) -> List[str]:
        """Format reward statistics section - compact multi-column layout."""
        lines = ["", f"{Fore.BLUE}💰 Reward Breakdown:{Style.RESET_ALL}"]

        reward_stats = data["reward_stats"]
        items = [(k, v["mean"]) for k, v in reward_stats.items()]
        # # Sort by absolute value (most impactful first)
        # items.sort(key=lambda x: abs(x[1]), reverse=True)

        # Sort by name
        items.sort(key=lambda x: x[0])

        # Settings
        num_columns = 3
        name_width = 25
        value_width = 10

        # Build rows
        for i in range(0, len(items), num_columns):
            row_items = items[i : i + num_columns]
            segments = []
            for name, mean in row_items:
                # Truncate long names
                display_name = name[: name_width - 2] + ".." if len(name) > name_width else name
                # Color based on sign
                color = Fore.GREEN if mean >= 0 else Fore.RED
                segment = f"{Fore.WHITE}{display_name:<{name_width}}{Style.RESET_ALL} {color}{mean:>{value_width}.4f}{Style.RESET_ALL}"
                segments.append(segment)

            lines.append("  " + "   ".join(segments))

        return lines

    def _format_eval_stats(self, eval_stats: Dict) -> List[str]:
        """Format persistent eval stats section."""
        eval_iter = eval_stats.get("eval/iteration", "?")
        lines = [
            "",
            f"{Fore.MAGENTA}🎯 Eval (iter {eval_iter}):{Style.RESET_ALL}",
        ]

        mean_ret = eval_stats.get("eval/mean_return", 0.0)
        std_ret = eval_stats.get("eval/std_return", 0.0)
        min_ret = eval_stats.get("eval/min_return", 0.0)
        max_ret = eval_stats.get("eval/max_return", 0.0)
        mean_len = eval_stats.get("eval/mean_episode_length", 0.0)
        n_eps = eval_stats.get("eval/num_episodes", 0)

        color = Fore.GREEN if mean_ret >= 0 else Fore.RED
        lines.append(
            f"  {Fore.WHITE}Return{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"{color}{mean_ret:.2f}{Style.RESET_ALL} ± {std_ret:.2f}  "
            f"[{min_ret:.2f}, {max_ret:.2f}]"
        )
        lines.append(
            f"  {Fore.WHITE}Episode Length{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"{Fore.CYAN}{mean_len:.1f}{Style.RESET_ALL}  "
            f"({n_eps} episodes)"
        )

        # Success rate (only present for envs that report success, e.g. ManiSkill)
        if "eval/success_rate" in eval_stats:
            sr = eval_stats["eval/success_rate"] * 100
            sr_color = Fore.GREEN if sr >= 50 else Fore.YELLOW if sr >= 20 else Fore.RED
            lines.append(
                f"  {Fore.WHITE}Success Rate{Style.RESET_ALL}".ljust(self.pad + 9)
                + f"{sr_color}{sr:.1f}%{Style.RESET_ALL}"
            )

        # Per-reward-type breakdown
        reward_keys = sorted(k for k in eval_stats if k.startswith("eval/reward/"))
        if reward_keys:
            lines.append(f"  {Fore.WHITE}Reward Breakdown:{Style.RESET_ALL}")
            # Multi-column layout matching training reward stats
            items = [(k.split("eval/reward/")[1], eval_stats[k]) for k in reward_keys]
            num_columns = 3
            name_width = 25
            value_width = 10
            for i in range(0, len(items), num_columns):
                row_items = items[i : i + num_columns]
                segments = []
                for name, mean in row_items:
                    display_name = name[: name_width - 2] + ".." if len(name) > name_width else name
                    color = Fore.GREEN if mean >= 0 else Fore.RED
                    segment = f"{Fore.WHITE}{display_name:<{name_width}}{Style.RESET_ALL} {color}{mean:>{value_width}.4f}{Style.RESET_ALL}"
                    segments.append(segment)
                lines.append("    " + "   ".join(segments))

        return lines

    def _format_summary(self, data: Dict, perf_metrics: Dict) -> List[str]:
        """Format summary section for training mode."""
        if "iteration" not in data:
            return []

        lines = [
            "",
            f"{Fore.WHITE}📊 Summary:{Style.RESET_ALL}",
        ]

        total_timesteps = data.get("total_timesteps", 0)
        total_time = perf_metrics.get("total_time", 0)
        eta = self._calculate_eta(data, total_time)

        lines.append(
            f"  {Fore.WHITE}Timesteps{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"{Fore.GREEN}{total_timesteps:,}{Style.RESET_ALL}"
        )
        lines.append(
            f"  {Fore.WHITE}Elapsed{Style.RESET_ALL}".ljust(self.pad + 9)
            + f"{Fore.CYAN}{_format_time(total_time)}{Style.RESET_ALL} │ "
            f"ETA: {Fore.YELLOW}{eta}{Style.RESET_ALL}"
        )
        lines.append("")

        return lines

    def _format_metric_rows(
        self,
        items: List[tuple],
        num_columns: int = 2,
    ) -> List[str]:
        """Format metric items into rows with specified columns."""
        lines = []
        for i in range(0, len(items), num_columns):
            row = items[i : i + num_columns]
            segments = []
            for label, value, color in row:
                if isinstance(value, str):
                    # STRING type: no formatting
                    segments.append(f"{Fore.WHITE}{label:<20}{Style.RESET_ALL} {color}{value:>10}{Style.RESET_ALL}")
                else:
                    # Numeric type: format as float
                    segments.append(f"{Fore.WHITE}{label:<20}{Style.RESET_ALL} {color}{value:>10.4f}{Style.RESET_ALL}")
            lines.append("  " + "   ".join(segments))
        return lines

    def _calculate_eta(self, data: Dict, total_time: float) -> str:
        """Calculate estimated time remaining."""
        iteration = data.get("iteration", 0)
        total_iterations = data.get("total_iterations", 0)

        if iteration == 0 or total_iterations == 0:
            return "N/A"

        seconds = total_time / (iteration + 1) * (total_iterations - iteration)
        return _format_time(seconds)


def _format_time(seconds: float) -> str:
    """Convert seconds to days, hours, minutes, seconds format."""
    days = int(seconds // (24 * 3600))
    seconds %= 24 * 3600
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


class WandbLogger:
    """
    WandB logger for remote tracking.
    Console printing is handled separately by ConsoleWriter.
    """

    def __init__(
        self,
        log_dir: str,
        project_name: str,
        group_name: str,
        run_name: str,
        cfg: Dict = None,
    ):
        if run_name is None:
            utc_now = datetime.now(timezone.utc)
            import pytz

            central = pytz.timezone("America/Chicago")
            ct_now = utc_now.astimezone(central)

            run_name = ct_now.strftime("run_%Y%m%d_%H%M%S_CT")

        # No Settings(start_method=...): wandb deprecated it as
        # non-functional, and newer releases (CHTC images) reject the
        # field outright with a pydantic extra_forbidden error.
        self.run = wandb.init(
            project=project_name,
            dir=log_dir,
            config=cfg,
            group=group_name,
            name=run_name,
        )
        # ``get_url`` is deprecated (removed after the warning stage);
        # ``run.url`` is the long-standing property both old and new
        # wandb releases serve.
        self.wandb_url = self.run.url
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def log_iteration(self, data: IterationData, step: int):
        """Log typed IterationData to WandB."""
        log_dict = data.to_wandb_dict()

        # Action distribution logging
        if data.action_distribution:
            action_dist = data.action_distribution
            for stat_name in ["mean", "std", "min", "max"]:
                if stat_name in action_dist:
                    values = action_dist[stat_name]
                    for i, v in enumerate(values):
                        val = v.item() if hasattr(v, "item") else v
                        log_dict[f"ActionDist/{stat_name}/dim_{i}"] = val

            if "raw" in action_dist:
                raw_actions = action_dist["raw"]
                for i in range(raw_actions.shape[-1]):
                    log_dict[f"ActionDist/histogram/dim_{i}"] = wandb.Histogram(raw_actions[:, i].flatten())

        wandb.log(log_dict, step=step)

    def log_training_data(
        self,
        training_data: Dict[str, Union[float, torch.Tensor, List, Dict]],
        step: int,
    ):
        """Log training metrics to WandB only."""
        log_dict = {}

        # Episode info
        if "ep_infos" in training_data:
            log_dict.update(self._process_episode_info(training_data["ep_infos"]))

        # Returns and lengths
        if "return_buffer" in training_data and training_data["return_buffer"]:
            log_dict["Train/mean_return"] = statistics.mean(training_data["return_buffer"])

        if "length_buffer" in training_data and training_data["length_buffer"]:
            log_dict["Train/mean_episode_length"] = statistics.mean(training_data["length_buffer"])

        if "success_rate" in training_data:
            log_dict["Train/success_rate"] = training_data["success_rate"]

        # Reward breakdown
        if "reward_breakdown_stats" in training_data:
            for reward_name, stats in training_data["reward_breakdown_stats"].items():
                for category, val in stats.items():
                    log_dict[f"Rewards/{category}/{reward_name}"] = val

        # Training metrics
        metrics_mapping = {
            "value_loss": "Loss/value_function",
            "surrogate_loss": "Loss/surrogate",
            "entropy": "Loss/entropy",
            "actor_loss": "Loss/actor",
            "critic_loss": "Loss/critic",
            "estimator_loss": "Loss/estimator",
        }

        for key, metric_name in metrics_mapping.items():
            if key in training_data:
                value = training_data[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                log_dict[metric_name] = value

        # Action distribution logging
        if "action_distribution" in training_data:
            action_dist = training_data["action_distribution"]

            # Per-dimension statistics
            for stat_name in ["mean", "std", "min", "max"]:
                if stat_name in action_dist:
                    values = action_dist[stat_name]
                    for i, v in enumerate(values):
                        val = v.item() if hasattr(v, "item") else v
                        log_dict[f"ActionDist/{stat_name}/dim_{i}"] = val

            # Histograms per dimension
            if "raw" in action_dist:
                raw_actions = action_dist["raw"]  # (num_steps * num_envs, action_dim)
                for i in range(raw_actions.shape[-1]):
                    log_dict[f"ActionDist/histogram/dim_{i}"] = wandb.Histogram(raw_actions[:, i].flatten())

        # Performance
        if "collection_time" in training_data and "learning_time" in training_data:
            total_time = training_data["collection_time"] + training_data["learning_time"]
            fps = training_data.get("fps") or int(
                training_data.get("num_steps", 0) * training_data.get("num_envs", 1) / total_time
            )
            log_dict["Performance/fps"] = fps
            log_dict["Performance/collection_time"] = training_data["collection_time"]
            log_dict["Performance/learning_time"] = training_data["learning_time"]

        # Curriculum
        if "curriculum_info" in training_data:
            log_dict["Curriculum/current_level"] = training_data["curriculum_info"]["current_level"]
            log_dict["Curriculum/steps_in_level"] = training_data["curriculum_info"]["steps_in_level"]

        if "wandb_extra" in training_data:
            extra_dict = training_data["wandb_extra"]
            if isinstance(extra_dict, dict):
                log_dict.update(self._flatten_dict(extra_dict))

        wandb.log(log_dict, step=step)

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
        """Flatten nested dictionary for wandb logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    items.append((new_key, v.item()))
                else:
                    # Individual values for line plots
                    for i, val in enumerate(v.flatten()):
                        items.append((f"{new_key}{sep}{i}", val.item()))
                    # Histogram for distribution view
                    items.append((f"{new_key}_hist", wandb.Histogram(v.detach().cpu().numpy())))
            else:
                items.append((new_key, v))

        return dict(items)

    def _process_episode_info(self, ep_infos: List[Dict], prefix: str = "Episode") -> Dict[str, float]:
        """Process episode information."""
        log_dict = {}
        if not ep_infos:
            return log_dict

        for key in ep_infos[0]:
            infos = []
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                infos.append(value)

            if infos:
                mean_value = sum(infos) / len(infos)
                metric_name = f"{prefix}/{key}" if "/" not in key else key
                log_dict[metric_name] = mean_value
        return log_dict

    def log_eval_data(self, eval_stats: Dict[str, Any], step: int) -> None:
        """Log evaluation metrics to WandB with structured keys."""
        log_dict = {}

        # Core eval metrics
        key_mapping = {
            "eval/mean_return": "Eval/mean_return",
            "eval/std_return": "Eval/std_return",
            "eval/min_return": "Eval/min_return",
            "eval/max_return": "Eval/max_return",
            "eval/mean_episode_length": "Eval/mean_episode_length",
            "eval/num_episodes": "Eval/num_episodes",
            "eval/time": "Eval/time",
            "eval/success_rate": "Eval/success_rate",
        }
        for src, dst in key_mapping.items():
            if src in eval_stats:
                log_dict[dst] = eval_stats[src]

        # Per-reward-type breakdown
        for key, val in eval_stats.items():
            if key.startswith("eval/reward/"):
                reward_name = key.split("eval/reward/")[1]
                log_dict[f"Eval/Rewards/{reward_name}"] = val

        wandb.log(log_dict, step=step)

    def upload_checkpoint_artifact(self, checkpoint_dir: str, iteration: int, metadata: dict | None = None) -> None:
        """Upload a checkpoint directory as a wandb Artifact."""
        artifact = wandb.Artifact(
            name=f"checkpoint-iter{iteration}",
            type="checkpoint",
            metadata=metadata,
        )
        artifact.add_dir(checkpoint_dir)
        self.run.log_artifact(artifact)

    def close(self):
        """Finish the WandB run."""
        wandb.finish()
