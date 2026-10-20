import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Calculate joint-angle Mean Absolute Error."
    )

    parser.add_argument(
        "--input",
        default="validation_templates/angle_annotations_template.csv",
        help="CSV containing reference and predicted angles",
    )

    args = parser.parse_args()
    csv_path = Path(args.input)

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = [
        "image_id",
        "angle_name",
        "reference_angle_deg",
        "predicted_angle_deg",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["reference_angle_deg"] = pd.to_numeric(
        df["reference_angle_deg"], errors="coerce"
    )
    df["predicted_angle_deg"] = pd.to_numeric(
        df["predicted_angle_deg"], errors="coerce"
    )

    df = df.dropna(
        subset=["image_id", "angle_name",
                "reference_angle_deg", "predicted_angle_deg"]
    )

    if df.empty:
        raise ValueError(
            "No valid measurements found. Enter actual reference "
            "and predicted angles in the CSV first."
        )

    df["absolute_error_deg"] = np.abs(
        df["reference_angle_deg"] - df["predicted_angle_deg"]
    )

    print(f"\nValid angle measurements: {len(df)}")
    print(f"Mean Absolute Error (MAE): {df['absolute_error_deg'].mean():.2f}°")
    print(f"Median Absolute Error: {df['absolute_error_deg'].median():.2f}°")
    print(f"90th percentile error: {df['absolute_error_deg'].quantile(0.90):.2f}°")

    print("\nMAE by joint:")
    print(
        df.groupby("angle_name")["absolute_error_deg"]
        .mean()
        .round(2)
        .to_string()
    )

    output_path = Path("reports/angle_error_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()