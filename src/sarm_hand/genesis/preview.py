"""OpenCV live preview windows for Genesis cameras."""

from __future__ import annotations

import numpy as np

# Default tile for recording camera previews (name → top-left pixel).
# Spread so front / top / arm are not stacked on launch (macOS OpenCV default).
DEFAULT_WINDOW_LAYOUT: dict[str, tuple[int, int]] = {
    "front": (32, 32),
    "top": (688, 32),
    "arm": (32, 544),
}


class CameraPreview:
    """Show named OpenCV windows for each Genesis camera feed."""

    def __init__(
        self,
        enabled: bool = True,
        *,
        layout: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        self.enabled = enabled
        self._layout = layout or DEFAULT_WINDOW_LAYOUT
        self._windows: set[str] = set()

    def show(self, frames: dict[str, np.ndarray]) -> None:
        if not self.enabled or not frames:
            return
        import cv2

        for name, rgb in frames.items():
            title = f"sarm-hand: {name}"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            cv2.imshow(title, bgr)
            if name in self._layout:
                x, y = self._layout[name]
                cv2.resizeWindow(title, w, h)
                cv2.moveWindow(title, x, y)
            self._windows.add(title)
        cv2.waitKey(1)

    def close(self) -> None:
        if not self._windows:
            return
        import cv2

        for title in list(self._windows):
            try:
                cv2.destroyWindow(title)
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        cv2.waitKey(1)
        self._windows.clear()
