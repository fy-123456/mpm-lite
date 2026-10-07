"""Small Viser adapter used by the 3D demos.

Viser keeps the simulation process headless and streams the particle cloud to
any browser connected to the configured HTTP port.  Large MPM scenes are
optionally downsampled for interactive browser performance.
"""

from __future__ import annotations

import numpy as np


class ParticleViewer:
    """Stream an MPM particle cloud through a Viser web server."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        max_points: int = 250_000,
        point_size: float = 0.005,
        title: str = "MPM Lite",
    ) -> None:
        try:
            import viser
        except ImportError as exc:  # pragma: no cover - depends on optional UI
            raise RuntimeError(
                "Viser is required for browser visualization. Install it with "
                "`uv add viser`."
            ) from exc

        self.max_points = max(1, int(max_points))
        self.server = viser.ViserServer(host=host, port=int(port), label=title)
        self.server.scene.set_up_direction("+z")
        self._cloud = None
        self._point_size = float(point_size)
        self._title = title
        self._status = self.server.gui.add_markdown(
            f"**{title}**\n\nWaiting for the first simulation frame."
        )
        print(f"[{title}] Viser: open http://{host}:{port}", flush=True)

    def _sample(self, points: np.ndarray, colors: np.ndarray):
        points = np.ascontiguousarray(points, dtype=np.float32).reshape((-1, 3))
        colors = np.asarray(colors)
        if colors.ndim == 1:
            colors = np.broadcast_to(colors, (len(points), 3))
        colors = colors.reshape((-1, 3))
        if colors.size and np.nanmax(colors) <= 1.0:
            colors = colors * 255.0
        colors = np.ascontiguousarray(np.clip(colors, 0, 255), dtype=np.uint8)
        if len(colors) != len(points):
            raise ValueError("particle colors and positions must have the same length")
        if len(points) > self.max_points:
            stride = int(np.ceil(len(points) / self.max_points))
            points = points[::stride]
            colors = colors[::stride]
        return points, colors

    def update(self, points: np.ndarray, colors: np.ndarray, *, step: int) -> None:
        points, colors = self._sample(points, colors)
        if self._cloud is None:
            self._cloud = self.server.scene.add_point_cloud(
                "/particles",
                points=points,
                colors=colors,
                point_size=self._point_size,
                point_shape="circle",
            )
        else:
            with self.server.atomic():
                self._cloud.points = points
                self._cloud.colors = colors
        self._status.content = (
            f"**{self._title}**  \n"
            f"Simulation step: `{step}`  \n"
            f"Displayed particles: `{len(points):,}`"
        )
