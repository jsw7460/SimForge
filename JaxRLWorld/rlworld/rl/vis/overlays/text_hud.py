from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .base import Base2DOverlay, Overlay2DConfig
from .hud_items import HUDItem

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class TextHUDConfig(Overlay2DConfig):
    """Configuration for text HUD overlay."""

    enabled: bool = True

    # Position
    position: str = "top_left"  # top_left, top_right, bottom_left, bottom_right
    margin_x: int = 20
    margin_y: int = 20
    line_spacing: int = 18

    # Font settings
    font_size: int = 16
    font_color: tuple[int, int, int] = (255, 255, 255)  # RGB
    font_path: str | None = None  # None = use default monospace font
    background_color: tuple[int, int, int] = (0, 0, 0)  # RGB
    background_alpha: float = 0.6

    # Bar graph defaults
    default_bar_width: int = 60
    default_bar_height: int = 12

    # Item spacing
    add_spacer_between_items: bool = True


class TextHUDOverlay(Base2DOverlay):
    """
    HUD overlay with plugin-based display items.

    Uses PIL for high-quality font rendering.

    Usage:
        hud = TextHUDOverlay()
        hud.add_item(BaseHeightItem())
        hud.add_item(JointPositionItem())

        # Render
        frame = hud.render(env, frame, env_idx=0)
    """

    def __init__(self, config: TextHUDConfig | None = None):
        if not HAS_PIL:
            raise ImportError("Pillow (PIL) is required for TextHUDOverlay")

        config = config or TextHUDConfig()
        super().__init__(config)
        self.config: TextHUDConfig = config

        self._items: dict[str, HUDItem] = {}
        self._item_order: list[str] = []

        # Load font
        self._font = self._load_font()
        self._small_font = self._load_font(size_ratio=0.8)

    def _load_font(self, size_ratio: float = 1.0) -> ImageFont.FreeTypeFont:
        """Load font with fallback to default."""
        size = int(self.config.font_size * size_ratio)

        if self.config.font_path:
            try:
                return ImageFont.truetype(self.config.font_path, size)
            except OSError:
                pass

        # Try common monospace fonts
        mono_fonts = [
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
            "UbuntuMono-R.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/Menlo.ttc",  # macOS
            "C:/Windows/Fonts/consola.ttf",  # Windows
        ]

        for font_path in mono_fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue

        # Fallback to default
        return ImageFont.load_default()

    # =========================================================================
    # Item Management
    # =========================================================================

    def add_item(self, item: HUDItem) -> "TextHUDOverlay":
        """Add a HUD item."""
        name = item.name
        if name in self._items:
            raise ValueError(f"Item '{name}' already exists. Use replace_item().")

        self._items[name] = item
        self._item_order.append(name)
        return self

    def remove_item(self, name: str) -> "TextHUDOverlay":
        """Remove a HUD item by name."""
        if name in self._items:
            del self._items[name]
            self._item_order.remove(name)
        return self

    def replace_item(self, item: HUDItem) -> "TextHUDOverlay":
        """Replace an existing item or add if not exists."""
        name = item.name
        if name in self._items:
            self._items[name] = item
        else:
            self.add_item(item)
        return self

    def get_item(self, name: str) -> HUDItem | None:
        """Get item by name."""
        return self._items.get(name)

    def enable_item(self, name: str) -> None:
        """Enable a specific item."""
        if name in self._items:
            self._items[name].enable()

    def disable_item(self, name: str) -> None:
        """Disable a specific item."""
        if name in self._items:
            self._items[name].disable()

    def clear_items(self) -> "TextHUDOverlay":
        """Remove all items."""
        self._items.clear()
        self._item_order.clear()
        return self

    def reorder_items(self, names: list[str]) -> "TextHUDOverlay":
        """Reorder items."""
        new_order = []
        for name in names:
            if name in self._items:
                new_order.append(name)

        for name in self._item_order:
            if name not in new_order:
                new_order.append(name)

        self._item_order = new_order
        return self

    # =========================================================================
    # Core Methods
    # =========================================================================

    def update(self, state: dict) -> None:
        """Not used in new design. Kept for interface compatibility."""
        pass

    def render(
        self,
        env: "GenesisEnv",
        frame: np.ndarray,
        env_idx: int = 0,
    ) -> np.ndarray:
        """
        Render HUD overlay on frame.

        Args:
            env: GenesisEnv instance
            frame: RGB image as numpy array (H, W, 3)
            env_idx: Environment index to render

        Returns:
            Modified frame with overlay
        """
        if not self.enabled:
            return frame

        # Build all lines
        all_lines = self._build_all_lines(env, env_idx)
        if not all_lines:
            return frame

        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        h, w = frame.shape[:2]

        # Calculate dimensions
        max_width, total_height = self._calculate_dimensions(all_lines)

        bg_width = max_width + 40
        bg_height = total_height + 20

        # Get background position
        bg_x, bg_y = self._get_background_position(w, h, bg_width, bg_height)

        # Create overlay for semi-transparent background
        overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw semi-transparent background
        bg_color = self.config.background_color + (int(255 * self.config.background_alpha),)
        overlay_draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=bg_color)

        # Composite background
        pil_image = Image.alpha_composite(pil_image.convert("RGBA"), overlay)

        # Draw content
        draw = ImageDraw.Draw(pil_image)
        y_offset = bg_y + 10
        x_offset = bg_x + 15

        for line in all_lines:
            if isinstance(line, str):
                draw.text(
                    (x_offset, y_offset),
                    line,
                    font=self._font,
                    fill=self.config.font_color,
                )
            elif isinstance(line, dict) and line.get("type") == "bar":
                self._draw_bar(draw, x_offset, y_offset, line)

            y_offset += self.config.line_spacing

        # Convert back to RGB numpy array
        return np.array(pil_image.convert("RGB"))

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _build_all_lines(
        self,
        env: "GenesisEnv",
        env_idx: int,
    ) -> list[str | dict]:
        """Build all lines from enabled items."""
        all_lines: list[str | dict] = []

        for name in self._item_order:
            item = self._items.get(name)
            if item is None or not item.enabled:
                continue

            lines = item.build_lines(env, env_idx)
            if not lines:
                continue

            if self.config.add_spacer_between_items and all_lines:
                all_lines.append("")

            all_lines.extend(lines)

        return all_lines

    def _calculate_dimensions(self, lines: list[str | dict]) -> tuple[int, int]:
        """Calculate max width and total height of content."""
        max_width = 0
        total_height = 0

        for line in lines:
            if isinstance(line, str):
                bbox = self._font.getbbox(line)
                text_w = bbox[2] - bbox[0]
                max_width = max(max_width, text_w)
            elif isinstance(line, dict) and line.get("type") == "bar":
                bar_width = line.get("bar_width", self.config.default_bar_width)
                total_bar_width = 50 + bar_width + 70
                max_width = max(max_width, total_bar_width)

            total_height += self.config.line_spacing

        return max_width, total_height

    def _get_background_position(self, w: int, h: int, bg_width: int, bg_height: int) -> tuple[int, int]:
        """Get background rectangle position."""
        pos = self.config.position

        if pos == "top_left":
            return self.config.margin_x, self.config.margin_y
        elif pos == "top_right":
            return w - bg_width - self.config.margin_x, self.config.margin_y
        elif pos == "bottom_left":
            return self.config.margin_x, h - bg_height - self.config.margin_y
        else:  # bottom_right
            return w - bg_width - self.config.margin_x, h - bg_height - self.config.margin_y

    def _draw_bar(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        bar_info: dict,
    ) -> None:
        """Draw a bar graph line."""
        label = bar_info["label"]
        value = bar_info["value"]
        max_value = bar_info["max_value"]
        bar_width = bar_info.get("bar_width", self.config.default_bar_width)
        bar_height = bar_info.get("bar_height", self.config.default_bar_height)

        # Draw label
        draw.text(
            (x, y),
            f"{label}:",
            font=self._small_font,
            fill=self.config.font_color,
        )

        # Bar position
        bar_x = x + 45
        bar_y = y + 2

        # Background bar
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], fill=(60, 60, 60))

        # Filled portion
        fill_ratio = min(1.0, max(0.0, value / max_value))
        fill_width = int(bar_width * fill_ratio)

        # Color gradient: green -> yellow -> red
        if fill_ratio < 0.5:
            g = 255
            r = int(255 * fill_ratio * 2)
        else:
            r = 255
            g = int(255 * (1 - fill_ratio) * 2)
        color = (r, g, 0)  # RGB

        if fill_width > 0:
            draw.rectangle([bar_x, bar_y, bar_x + fill_width, bar_y + bar_height], fill=color)

        # Value text
        draw.text(
            (bar_x + bar_width + 5, y),
            f"{value:.2f}m",
            font=self._small_font,
            fill=self.config.font_color,
        )
