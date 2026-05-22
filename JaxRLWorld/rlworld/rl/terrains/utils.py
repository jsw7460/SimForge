"""Height-field helpers (backend-agnostic, numpy only)."""

from __future__ import annotations

import numpy as np


def bilinear_upsample(coarse: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample ``coarse`` to ``out_shape`` (numpy only).

    Replaces IsaacLab's ``scipy.interpolate.RectBivariateSpline`` so the
    terrain package depends only on numpy. Bilinear (vs. cubic spline)
    avoids overshoot at the discrete random heights, which is what we want
    for ``random_uniform`` rough terrain.
    """
    h0, w0 = coarse.shape
    h, w = out_shape
    yi = np.linspace(0.0, h0 - 1, h)
    xi = np.linspace(0.0, w0 - 1, w)
    y0 = np.floor(yi).astype(np.int64)
    x0 = np.floor(xi).astype(np.int64)
    y1 = np.minimum(y0 + 1, h0 - 1)
    x1 = np.minimum(x0 + 1, w0 - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    c00 = coarse[np.ix_(y0, x0)]
    c01 = coarse[np.ix_(y0, x1)]
    c10 = coarse[np.ix_(y1, x0)]
    c11 = coarse[np.ix_(y1, x1)]
    top = c00 * (1.0 - wx) + c01 * wx
    bot = c10 * (1.0 - wx) + c11 * wx
    return top * (1.0 - wy) + bot * wy
