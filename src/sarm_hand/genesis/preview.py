"""OpenCV live preview windows for Genesis cameras."""

from __future__ import annotations

import numpy as np


class CameraPreview:
    """Show named OpenCV windows for each Genesis camera feed."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._windows: set[str] = set()

    def show(self, frames: dict[str, np.ndarray]) -> None:
        if not self.enabled or not frames:
            return
        import cv2

        for name, rgb in frames.items():
            title = f"sarm-hand: {name}"
            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(title, bgr)
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
