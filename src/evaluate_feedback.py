"""Evaluate binary alignment-issue detection against expert-reviewed labels."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

REQUIRED = {"sample_id", "issue_name", "reference_issue", "predicted_issue", "review_status"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/feedback_validation.json")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    uncertain = df["review_status"].astype(str).str.lower().eq("uncertain")
    usable = df.loc[~uncertain].copy()
    usable["reference_issue"] = pd.to_numeric(usable["reference_issue"], errors="coerce")
    usable["predicted_issue"] = pd.to_numeric(usable["predicted_issue"], errors="coerce")
    usable = usable.dropna(subset=["reference_issue", "predicted_issue"])
    if usable.empty:
        raise ValueError("No reviewed rows with binary 0/1 labels. Replace template examples with expert-reviewed data.")
    for col in ("reference_issue", "predicted_issue"):
        if not usable[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 or 1.")
    y, pred = usable["reference_issue"].astype(int), usable["predicted_issue"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "evaluated_rows": int(len(usable)),
        "excluded_uncertain_rows": int(uncertain.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "report": classification_report(y, pred, output_dict=True, zero_division=0)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "report"}, indent=2))
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
