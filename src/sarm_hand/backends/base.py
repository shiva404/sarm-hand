"""Robot backend protocol for hardware and simulation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    """Subset of LeRobot Robot API used by sarm-hand."""

    name: str
    robot_type: str
    action_features: dict[str, type]
    observation_features: dict
    cameras: dict[str, Any]
    is_connected: bool

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...
