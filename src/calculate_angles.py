import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from src.pose_extractor import angle_degrees


# MediaPipe Pose landmark indices
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def calculate_joint_angles(landmarks):
    """Calculate selected 2D joint angles from MediaPipe landmarks."""

    points = {
        name: np.array(
            [landmarks[index].x, landmarks[index].y],
            dtype=float,
        )
        for name, index in LANDMARKS.items()
    }

    angles = {
        "left_knee": angle_degrees(
            points["left_hip"],
            points["left_knee"],
            points["left_ankle"],
        ),
        "right_knee": angle_degrees(
            points["right_hip"],
            points["right_knee"],
            points["right_ankle"],
        ),
        "left_elbow": angle_degrees(
            points["left_shoulder"],
            points["left_elbow"],
            points["left_wrist"],
        ),
        "right_elbow": angle_degrees(
            points["right_shoulder"],
            points["right_elbow"],
            points["right_wrist"],
        ),
    }

    return angles


def main():
    parser = argparse.ArgumentParser(
        description="Calculate joint angles from a yoga image."
    )
    parser.add_argument("image", help="Path to a yoga image")
    parser.add_argument(
        "--output",
        default="reports/joint_angles.csv",
        help="CSV output path",
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

    if not result.pose_landmarks:
        raise RuntimeError(
            "No pose detected. Try an image where the person's body "
            "is clearly visible."
        )

        # This is 2D image-plane geometry, not validated 3D anatomy.
    angles = calculate_joint_angles(result.pose_landmarks.landmark)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "joint", "angle_degrees"])

        for joint, angle in angles.items():
            writer.writerow(
                [image_path.name, joint, f"{angle:.2f}"]
            )
            print(f"{joint}: {angle:.2f} degrees")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()