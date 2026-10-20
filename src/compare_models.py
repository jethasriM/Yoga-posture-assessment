"""Compare two classifiers on paired predictions from the same held-out examples."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

REQUIRED = {"sample_id", "true_label", "baseline_prediction", "improved_prediction"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/model_comparison.json")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=list(REQUIRED))
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique: one row per held-out test example.")
    if len(df) == 0:
        raise ValueError("No paired prediction rows found.")
    baseline_ok = df["baseline_prediction"].astype(str) == df["true_label"].astype(str)
    improved_ok = df["improved_prediction"].astype(str) == df["true_label"].astype(str)
    baseline_error = float((~baseline_ok).mean())
    improved_error = float((~improved_ok).mean())
    reduction = None if baseline_error == 0 else float((baseline_error - improved_error) / baseline_error * 100)
    result = {
        "n_test_samples": int(len(df)),
        "baseline_accuracy": float(accuracy_score(df["true_label"], df["baseline_prediction"])),
        "improved_accuracy": float(accuracy_score(df["true_label"], df["improved_prediction"])),
        "baseline_error_rate_percent": baseline_error * 100,
        "improved_error_rate_percent": improved_error * 100,
        "absolute_error_rate_change_percentage_points": (improved_error - baseline_error) * 100,
        "relative_error_reduction_percent": reduction,
        "meets_25_percent_reduction_target": bool(reduction is not None and reduction >= 25),
        "baseline_report": classification_report(df["true_label"], df["baseline_prediction"], output_dict=True, zero_division=0),
        "improved_report": classification_report(df["true_label"], df["improved_prediction"], output_dict=True, zero_division=0)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_report")}, indent=2))
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
