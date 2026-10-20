import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


JOINTS = {
    "left_knee": ("Left Hip", "Left Knee", "Left Ankle"),
    "right_knee": ("Right Hip", "Right Knee", "Right Ankle"),
    "left_elbow": ("Left Shoulder", "Left Elbow", "Left Wrist"),
    "right_elbow": ("Right Shoulder", "Right Elbow", "Right Wrist"),
}


def calculate_angle(a, b, c):
    """Calculate the 2D angle ABC in degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def annotate_image(image_path, output_file):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot open image: {image_path}")
        return

    original = image.copy()
    height, width = image.shape[:2]

    # Scale the image to fit the screen.
    scale = min(1100 / width, 750 / height, 1.0)
    display = cv2.resize(
        image,
        (int(width * scale), int(height * scale))
    )

    points = []
    current_joint = "left_knee"

    window = "Joint Angle Annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((int(x / scale), int(y / scale)))

    cv2.setMouseCallback(window, mouse_callback)

    joint_names = list(JOINTS.keys())
    joint_index = 0

    while joint_index < len(joint_names):
        current_joint = joint_names[joint_index]
        required_points = JOINTS[current_joint]

        canvas = cv2.resize(
            original,
            (int(width * scale), int(height * scale))
        )

        for index, point in enumerate(points):
            x, y = int(point[0] * scale), int(point[1] * scale)
            cv2.circle(canvas, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        instruction = (
            f"{current_joint}: click "
            f"{required_points[min(len(points), 2)]}"
            if len(points) < 3
            else f"{current_joint}: press S to save angle"
        )

        cv2.putText(
            canvas,
            instruction,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        angle = None

        if len(points) == 3:
            angle = calculate_angle(*points)
            cv2.putText(
                canvas,
                f"Reference angle: {angle:.2f} degrees",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 128, 0),
                2,
            )

        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            points.clear()

        elif key == ord("s") and len(points) == 3:
            angle = calculate_angle(*points)

            if angle is not None:
                writer = csv.writer(output_file)
                writer.writerow([
                    image_path.name,
                    current_joint,
                    f"{angle:.2f}",
                    "",
                    "manual 2D point annotation",
                ])
                output_file.flush()

                print(
                    f"{image_path.name} | {current_joint} | "
                    f"{angle:.2f} degrees"
                )

                points.clear()
                joint_index += 1

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Manually annotate joint angles in yoga images."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to one yoga image",
    )

    parser.add_argument(
        "--output",
        default="reports/reference_angles.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    with output_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "image_id",
                "angle_name",
                "reference_angle_deg",
                "predicted_angle_deg",
                "notes",
            ])

        annotate_image(image_path, file)


if __name__ == "__main__":
    main()