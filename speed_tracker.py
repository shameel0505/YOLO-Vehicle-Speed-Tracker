"""
Vehicle Speed Estimation & Traffic Analysis Engine.
Calculates real-world vehicle velocities using frame-index timing and perspective-aware distance calibration.
"""

from typing import Dict, Optional, Tuple, Set
import numpy as np


class SpeedEstimator:
    """
    Computes vehicle velocity based on video frame intervals rather than wall-clock time.
    Prevents inference latency / GPU frame drops from distorting physical speed calculations.
    """

    def __init__(
        self,
        fps: float,
        calibration_distance_meters: float = 18.0,
        y_entry_down: int = 350,
        y_exit_down: int = 450,
        y_entry_up: int = 450,
        y_exit_up: int = 350,
        x_divider: int = 614,
    ):
        """
        Args:
            fps: Video frames per second.
            calibration_distance_meters: Known real-world physical distance between entry and exit lines.
            y_entry_down: Y-coordinate for downward traffic entry gate.
            y_exit_down: Y-coordinate for downward traffic exit gate.
            y_entry_up: Y-coordinate for upward traffic entry gate.
            y_exit_up: Y-coordinate for upward traffic exit gate.
            x_divider: X-coordinate dividing opposing traffic lanes.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        self.fps = fps
        self.distance = calibration_distance_meters
        self.y_entry_down = y_entry_down
        self.y_exit_down = y_exit_down
        self.y_entry_up = y_entry_up
        self.y_exit_up = y_exit_up
        self.x_divider = x_divider

        # Track vehicle frame timestamps {track_id: frame_index}
        self.entry_frames: Dict[int, int] = {}
        self.exit_frames: Dict[int, int] = {}
        self.estimated_speeds: Dict[int, float] = {}

    def update_vehicle(self, track_id: int, cx: int, cy: int, frame_idx: int) -> float:
        """
        Updates tracking gates for a vehicle and returns estimated speed in km/h.
        """
        # Downward Lane (Left of divider)
        if cx < self.x_divider:
            if track_id not in self.entry_frames and cy > self.y_entry_down:
                self.entry_frames[track_id] = frame_idx
            elif track_id in self.entry_frames and track_id not in self.exit_frames and cy > self.y_exit_down:
                self.exit_frames[track_id] = frame_idx
                self.estimated_speeds[track_id] = self._calculate_velocity(track_id)

        # Upward Lane (Right of divider)
        else:
            if track_id not in self.entry_frames and cy < self.y_entry_up:
                self.entry_frames[track_id] = frame_idx
            elif track_id in self.entry_frames and track_id not in self.exit_frames and cy < self.y_exit_up:
                self.exit_frames[track_id] = frame_idx
                self.estimated_speeds[track_id] = self._calculate_velocity(track_id)

        return self.estimated_speeds.get(track_id, 0.0)

    def _calculate_velocity(self, track_id: int) -> float:
        """
        Calculates velocity: v = (d / dt) * 3.6 [km/h]
        dt = (frame_exit - frame_entry) / fps
        """
        if track_id not in self.entry_frames or track_id not in self.exit_frames:
            return 0.0

        delta_frames = abs(self.exit_frames[track_id] - self.entry_frames[track_id])
        if delta_frames == 0:
            return 0.0

        delta_seconds = delta_frames / self.fps
        speed_kmh = (self.distance / delta_seconds) * 3.6
        return round(float(speed_kmh), 1)

    @staticmethod
    def calculate_average_speed(speeds: Dict[int, float], min_valid: float = 10.0, max_valid: float = 160.0) -> Optional[float]:
        """
        Filters noise / outliers and computes mean traffic flow velocity.
        """
        valid_speeds = [s for s in speeds.values() if s is not None and min_valid <= s <= max_valid]
        if not valid_speeds:
            return None
        return float(np.mean(valid_speeds))


class TrafficFlowAnalyzer:
    """
    Categorizes traffic congestion levels and maintains lane counts.
    """

    @staticmethod
    def classify_density(active_vehicle_count: int) -> Tuple[str, Tuple[int, int, int]]:
        """
        Returns traffic status label and BGR color.
        """
        if active_vehicle_count > 20:
            return "Heavy Congestion", (0, 0, 255)      # Red
        if active_vehicle_count >= 8:
            return "Moderate Traffic", (0, 255, 255)    # Yellow
        return "Free Flow Traffic", (0, 255, 0)         # Green
