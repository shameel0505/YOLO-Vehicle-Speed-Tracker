"""
Production Entrypoint for YOLO Vehicle Tracking & Velocity Estimation.
Supports CLI arguments, frame-rate synchronized speed estimation, and robust error handling.
"""

import argparse
import collections as cl
import logging
from typing import Dict, List, Set
import cv2
from ultralytics import YOLO

from speed_tracker import SpeedEstimator, TrafficFlowAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("traffic-tracker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Vehicle Detection, Tracking, and Speed Estimation.")
    parser.add_argument("--source", type=str, default="sample.mp4", help="Path to input video file or stream RTSP URL.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model checkpoint name or path.")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold for detections.")
    parser.add_argument("--classes", nargs="+", type=int, default=[2, 3, 5, 7], help="COCO classes to track: 2=car, 3=motorcycle, 5=bus, 7=truck.")
    parser.add_argument("--distance", type=float, default=18.0, help="Calibration distance in meters between speed lines.")
    parser.add_argument("--divider-x", type=int, default=614, help="X pixel coordinate dividing opposing lanes.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path to save annotated video.")
    parser.add_argument("--no-show", action="store_true", help="Run in headless mode without displaying GUI window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Opening video stream: {args.source}")
    vd = cv2.VideoCapture(args.source)
    if not vd.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        return

    fps = vd.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        logger.warning(f"Invalid FPS detected ({fps}). Defaulting to 30.0 FPS.")
        fps = 30.0

    width = int(vd.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vd.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vd.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video initialized: {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)")

    # Initialize YOLO tracker
    logger.info(f"Loading YOLO weights: {args.model}")
    model = YOLO(args.model)

    # Initialize Speed & Flow Analyzers
    estimator = SpeedEstimator(
        fps=fps,
        calibration_distance_meters=args.distance,
        x_divider=args.divider_x
    )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        logger.info(f"Saving output rendering to: {args.output}")

    left_lane_counts: Dict[str, int] = cl.defaultdict(int)
    right_lane_counts: Dict[str, int] = cl.defaultdict(int)
    counted_ids: Set[int] = set()

    frame_idx = 0
    try:
        while True:
            ret, frame = vd.read()
            if not ret or frame is None:
                logger.info("Video stream completed or end of file reached.")
                break

            frame_idx += 1

            # Run tracker with persistence
            results = model.track(
                frame,
                persist=True,
                classes=args.classes,
                conf=args.conf,
                verbose=False
            )

            active_vehicles: Set[int] = set()
            class_names = model.names

            # Draw lane demarcation lines
            cv2.line(frame, (args.divider_x, 0), (args.divider_x, height), (0, 0, 255), 2)
            cv2.line(frame, (55, 350), (args.divider_x - 50, 350), (0, 165, 255), 2)
            cv2.line(frame, (55, 450), (args.divider_x - 50, 450), (0, 165, 255), 2)
            cv2.line(frame, (args.divider_x + 50, 350), (width - 50, 350), (0, 165, 255), 2)
            cv2.line(frame, (args.divider_x + 50, 450), (width - 50, 450), (0, 165, 255), 2)

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    active_vehicles.add(track_id)
                    class_name = class_names.get(class_id, "vehicle")

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Update frame-accurate speed
                    speed_kmh = estimator.update_vehicle(track_id, cx, cy, frame_idx)

                    # Update lane crossing tallies
                    if track_id not in counted_ids:
                        if cx < args.divider_x and 350 < cy < 450:
                            counted_ids.add(track_id)
                            left_lane_counts[class_name] += 1
                        elif cx > args.divider_x and 350 < cy < 450:
                            counted_ids.add(track_id)
                            right_lane_counts[class_name] += 1

                    # Render bounding box and telemetry
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (76, 175, 80), 2)
                    label = f"#{track_id} {class_name}"
                    if speed_kmh > 0:
                        label += f" {speed_kmh} km/h"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Render Traffic Congestion Status
            status_text, status_color = TrafficFlowAnalyzer.classify_density(len(active_vehicles))
            cv2.putText(frame, f"STATUS: {status_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"ACTIVE VEHICLES: {len(active_vehicles)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Left Lane HUD
            cv2.putText(frame, "DOWNWARD LANE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            y_off = 145
            for v_type, count in left_lane_counts.items():
                cv2.putText(frame, f" - {v_type}: {count}", (60, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                y_off += 22

            # Right Lane HUD
            cv2.putText(frame, "UPWARD LANE", (width - 250, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            y_off = 145
            for v_type, count in right_lane_counts.items():
                cv2.putText(frame, f" - {v_type}: {count}", (width - 240, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                y_off += 22

            if writer:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("YOLO Vehicle Speed & Flow Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Termination key received.")
                    break

    finally:
        vd.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline finalized successfully.")


if __name__ == "__main__":
    main()
