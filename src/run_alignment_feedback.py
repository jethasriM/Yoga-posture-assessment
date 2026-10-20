"""Generate preliminary alignment feedback from calculated joint angles."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.alignment_feedback import get_alignment_feedback


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV file containing MediaPipe joint angles",
    )

    parser.add_argument(
        "--pose",
        required=True,
        help="Yoga pose name",
    )

    parser.add_argument(
        "--output",
        default="reports/alignment_feedback.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"image", "joint", "angle_degrees"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing)}"
        )

    results = []

    for image_name, group in df.groupby("image"):
        angles = {}

        for _, row in group.iterrows():
            angles[row["joint"]] = row["angle_degrees"]

        feedback = get_alignment_feedback(args.pose, angles)

        results.append({
            "image": str(image_name),
            "pose": args.pose,
            "joint_angles": angles,
            "feedback": feedback,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nSaved feedback to: {output_path}")


if __name__ == "__main__":
    main()