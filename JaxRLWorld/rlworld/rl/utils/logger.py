import os
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Union, Any, Optional

import torch
import wandb
from colorama import Fore, Style, init

from rlworld.rl.algorithms.metrics import MetricType, ConsoleMetric, BaseMetrics

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

    def write_metrics(
        self,
        data: Dict,
        metrics: Optional[BaseMetrics],
        mode: str,
        width: int = 100,
        print_reward_stats: bool = True,
    ):
        """Write formatted metrics to console."""
        log_string = []

        # Header section
        if mode == "train":
            iteration = data.get("iteration", "?")
            total_iterations = data.get("total_iterations", "?")
            header = f" Learning iteration {Fore.GREEN}{iteration}/{total_iterations}{Style.RESET_ALL} "
        else:
            header = f" Evaluation Results "

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

        # Summary (training only)
        if mode == "train":
            log_string.extend(self._format_summary(data, perf_metrics))

        # Footer
        log_string.append(f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}")

        print("\n".join(log_string))

    def _create_section_header(self, width: int, title: str) -> List[str]:
        """Create a formatted section header."""
        return [
            f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}",
            f"{Fore.CYAN}║{Style.RESET_ALL}" + title.center(width - 2, " ") + f"{Fore.CYAN}║{Style.RESET_ALL}",
            f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}",
            ""
        ]

    def _format_run_info_section(self, data: Dict) -> List[str]:
        """Format run information section with WandB, simulator, and task info."""
        lines = [f"{Fore.MAGENTA}🚀 Run Info:{Style.RESET_ALL}"]

        # WandB run name and URL
        if "wandb_url" in data:
            wandb_url = data["wandb_url"]
            run_name = data["wandb_run_name"]
            lines.append(
                f"  {Fore.WHITE}Run{Style.RESET_ALL}".ljust(self.pad + 9) +
                f"{Fore.CYAN}{run_name}{Style.RESET_ALL}"
            )
            lines.append(
                f"  {Fore.WHITE}WandB{Style.RESET_ALL}".ljust(self.pad + 9) +
                f"{Fore.BLUE}{wandb_url}{Style.RESET_ALL}"
            )

        # Simulator and task
        simulator = data.get("simulator", "N/A")
        task_name = data.get("task_name", "N/A")

        lines.append(
            f"  {Fore.WHITE}Simulator{Style.RESET_ALL}".ljust(self.pad + 9) +
            f"{Fore.YELLOW}{simulator}{Style.RESET_ALL}"
        )
        lines.append(
            f"  {Fore.WHITE}Task{Style.RESET_ALL}".ljust(self.pad + 9) +
            f"{Fore.YELLOW}{task_name}{Style.RESET_ALL}"
        )

        # Log directory
        if "log_dir" in data:
            lines.append(
                f"  {Fore.WHITE}Log Dir{Style.RESET_ALL}".ljust(self.pad + 9) +
                f"{Fore.WHITE}{data['log_dir']}{Style.RESET_ALL}"
            )

        lines.append("")
        return lines

    def _format_performance_section(self, perf_metrics: Dict) -> List[str]:
        """Format performance metrics section."""
        if not perf_metrics:
            return []

        lines = [f"{Fore.YELLOW}⚡ Performance:{Style.RESET_ALL}"]

        fps = perf_metrics.get('fps', 0)
        collection_time = perf_metrics.get('collection_time', 0)
        learning_time = perf_metrics.get('learning_time', 0)

        lines.append(
            f"  {Fore.WHITE}Throughput{Style.RESET_ALL}".ljust(self.pad + 9) +
            f"{Fore.GREEN}{fps:,.0f}{Style.RESET_ALL} steps/sec"
        )
        lines.append(
            f"  {Fore.WHITE}Timing{Style.RESET_ALL}".ljust(self.pad + 9) +
            f"collect {Fore.CYAN}{collection_time:.3f}s{Style.RESET_ALL} │ "
            f"learn {Fore.CYAN}{learning_time:.3f}s{Style.RESET_ALL}"
        )

        return lines

    def _format_algorithm_metrics(self, metrics: Optional[BaseMetrics]) -> List[str]:
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
        lines = [
            "",
            f"{Fore.GREEN}📈 Episode Stats:{Style.RESET_ALL}"
        ]

        if "mean_return" in data:
            mean_return = data['mean_return']
            color = Fore.GREEN if mean_return >= 0 else Fore.RED
            lines.append(
                f"  {Fore.WHITE}Mean Return{Style.RESET_ALL}".ljust(self.pad + 9) +
                f"{color}{mean_return:.2f}{Style.RESET_ALL}"
            )

        if "mean_episode_length" in data:
            lines.append(
                f"  {Fore.WHITE}Mean Episode Length{Style.RESET_ALL}".ljust(self.pad + 9) +
                f"{Fore.CYAN}{data['mean_episode_length']:.1f}{Style.RESET_ALL}"
            )

        if "success_rate" in data and data["success_rate"] is not None:
            success_rate = data['success_rate'] * 100
            color = Fore.GREEN if success_rate >= 50 else Fore.YELLOW if success_rate >= 20 else Fore.RED
            lines.append(
                f"  {Fore.WHITE}Success Rate{Style.RESET_ALL}".ljust(self.pad + 9) +
                f"{color}{success_rate:.1f}%{Style.RESET_ALL}"
            )

        return lines

    def _format_reward_stats(self, data: Dict) -> List[str]:
        """Format reward statistics section - compact multi-column layout."""
        lines = [
            "",
            f"{Fore.BLUE}💰 Reward Breakdown:{Style.RESET_ALL}"
        ]

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
            row_items = items[i:i + num_columns]
            segments = []
            for name, mean in row_items:
                # Truncate long names
                display_name = name[:name_width - 2] + ".." if len(name) > name_width else name
                # Color based on sign
                color = Fore.GREEN if mean >= 0 else Fore.RED
                segment = f"{Fore.WHITE}{display_name:<{name_width}}{Style.RESET_ALL} {color}{mean:>{value_width}.4f}{Style.RESET_ALL}"
                segments.append(segment)

            lines.append("  " + "   ".join(segments))

        return lines

    def _format_summary(self, data: Dict, perf_metrics: Dict) -> List[str]:
        """Format summary section for training mode."""
        if "iteration" not in data:
            return []

        lines = [
            "",
            f"{Fore.WHITE}📊 Summary:{Style.RESET_ALL}",
        ]

        total_timesteps = data.get('total_timesteps', 0)
        total_time = perf_metrics.get('total_time', 0)
        eta = self._calculate_eta(data, total_time)

        lines.append(
            f"  {Fore.WHITE}Timesteps{Style.RESET_ALL}".ljust(self.pad + 9) +
            f"{Fore.GREEN}{total_timesteps:,}{Style.RESET_ALL}"
        )
        lines.append(
            f"  {Fore.WHITE}Elapsed{Style.RESET_ALL}".ljust(self.pad + 9) +
            f"{Fore.CYAN}{_format_time(total_time)}{Style.RESET_ALL} │ "
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
            row = items[i:i + num_columns]
            segments = []
            for label, value, color in row:
                if isinstance(value, str):
                    # STRING type: no formatting
                    segments.append(
                        f"{Fore.WHITE}{label:<20}{Style.RESET_ALL} {color}{value:>10}{Style.RESET_ALL}"
                    )
                else:
                    # Numeric type: format as float
                    segments.append(
                        f"{Fore.WHITE}{label:<20}{Style.RESET_ALL} {color}{value:>10.4f}{Style.RESET_ALL}"
                    )
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
    seconds %= (24 * 3600)
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
            central = pytz.timezone('America/Chicago')
            ct_now = utc_now.astimezone(central)

            run_name = ct_now.strftime("run_%Y%m%d_%H%M%S_CT")

        self.run = wandb.init(
            project=project_name,
            dir=log_dir,
            config=cfg,
            group=group_name,
            name=run_name,
            settings=wandb.Settings(start_method="fork")
        )
        self.wandb_url = self.run.get_url()
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

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
                        val = v.item() if hasattr(v, 'item') else v
                        log_dict[f"ActionDist/{stat_name}/dim_{i}"] = val

            # Histograms per dimension
            if "raw" in action_dist:
                raw_actions = action_dist["raw"]  # (num_steps * num_envs, action_dim)
                for i in range(raw_actions.shape[-1]):
                    log_dict[f"ActionDist/histogram/dim_{i}"] = wandb.Histogram(
                        raw_actions[:, i].flatten()
                    )

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

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
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

    def close(self):
        """Finish the WandB run."""
        wandb.finish()