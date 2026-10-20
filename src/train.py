import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import accuracy_score

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.data_utils import build_features


def main():
    X, y = build_features()

    labels, counts = np.unique(y, return_counts=True)

    print(
        f"Classes with detected poses: {len(labels)}; "
        f"usable images: {len(y)}"
    )

    if counts.min() < 3:
        raise ValueError(
            "Some classes have fewer than 3 detected images. "
            "Collect more usable images before tuning."
        )

    # Preserve the existing stratified hold-out split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Cross-validation is performed only on the training data.
    min_train_count = np.unique(y_train, return_counts=True)[1].min()
    cv_splits = min(5, int(min_train_count))

    if cv_splits < 2:
        raise ValueError(
            "Not enough training examples per class for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    base_model = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("\nTuning Random Forest using training data only...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print(f"\nBest cross-validation accuracy: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    # This is a one-time check on the held-out split.
    holdout_accuracy = accuracy_score(
        y_test,
        best_model.predict(X_test),
    )
    print(f"Hold-out accuracy: {holdout_accuracy:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "yoga_classifier.joblib"

    joblib.dump(
        {
            "model": best_model,
            "feature_count": X.shape[1],
        },
        model_path,
    )

    print(f"\nSaved tuned model: {model_path}")
    print(
        "Note: Treat the hold-out result as an evaluation result. "
        "Do not repeatedly tune against this same test set."
    )


if __name__ == "__main__":
    main()