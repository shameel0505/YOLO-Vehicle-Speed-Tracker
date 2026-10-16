"""
Unit tests for vehicle speed calculation, frame interval physics, and outlier rejection.
"""

import pytest
from speed_tracker import SpeedEstimator, TrafficFlowAnalyzer


def test_speed_calculation_known_distance_and_time():
    """
    Test scenario:
    - Distance: 18.0 meters
    - Video FPS: 30.0 fps
    - Vehicle takes 30 frames to cross the two gate lines.
    - dt = 30 / 30 = 1.0 second
    - velocity = (18m / 1s) * 3.6 = 64.8 km/h
    """
    estimator = SpeedEstimator(
        fps=30.0,
        calibration_distance_meters=18.0,
        y_entry_down=350,
        y_exit_down=450,
        x_divider=600
    )

    track_id = 1
    # Frame 10: Vehicle enters gate (cy > 350)
    speed = estimator.update_vehicle(track_id, cx=300, cy=360, frame_idx=10)
    assert speed == 0.0  # Not exited yet

    # Frame 25: Vehicle in transit
    speed = estimator.update_vehicle(track_id, cx=300, cy=400, frame_idx=25)
    assert speed == 0.0

    # Frame 40: Vehicle crosses exit gate (cy > 450)
    speed = estimator.update_vehicle(track_id, cx=300, cy=460, frame_idx=40)
    assert speed == 64.8


def test_speed_calculation_fast_vehicle():
    """
    Vehicle crosses 18 meters in 15 frames at 30 fps (0.5 seconds).
    velocity = (18 / 0.5) * 3.6 = 129.6 km/h
    """
    estimator = SpeedEstimator(fps=30.0, calibration_distance_meters=18.0)
    estimator.update_vehicle(42, cx=200, cy=360, frame_idx=100)
    speed = estimator.update_vehicle(42, cx=200, cy=460, frame_idx=115)
    assert speed == 129.6


def test_zero_or_negative_fps_raises():
    with pytest.raises(ValueError):
        SpeedEstimator(fps=0.0)
    with pytest.raises(ValueError):
        SpeedEstimator(fps=-25.0)


def test_average_speed_outlier_filtering():
    speeds = {
        1: 60.0,
        2: 70.0,
        3: 80.0,
        4: 0.0,     # Incomplete track
        5: 350.0,   # Tracker jitter outlier
        6: -10.0,   # Invalid
    }
    avg = SpeedEstimator.calculate_average_speed(speeds, min_valid=20.0, max_valid=150.0)
    # Mean of 60, 70, 80 = 70.0
    assert avg == pytest.approx(70.0, rel=1e-3)


def test_traffic_density_classification():
    label, color = TrafficFlowAnalyzer.classify_density(5)
    assert "Free Flow" in label

    label, color = TrafficFlowAnalyzer.classify_density(12)
    assert "Moderate" in label

    label, color = TrafficFlowAnalyzer.classify_density(25)
    assert "Congestion" in label
