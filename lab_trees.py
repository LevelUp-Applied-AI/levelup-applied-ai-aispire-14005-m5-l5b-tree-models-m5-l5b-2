"""
Module 5 Week B — Applied Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the Petra
Telecom churn dataset. Handle class imbalance honestly (class_weight as an
operating-point tool at a fixed threshold), evaluate with PR-AUC and
calibration, and demonstrate what tree models capture that linear models
cannot.

Complete the 12 functions below. See the lab guide for task-by-task detail.
Run with:  python lab_trees.py
Tests:     pytest tests/ -v
"""

import os

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             classification_report, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    # TODO: Load the CSV, select NUMERIC_FEATURES into X, use `churned` as y,
    #       split with test_size=0.2 and stratify=y.
    pass


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    # TODO: Fit a DecisionTreeClassifier with the given max_depth and seed.
    pass


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error using equal-count (quantile) binning.

    Sort samples by predicted probability, split into `n_bins` equal-size
    chunks, and sum the bin-weighted absolute difference between each bin's
    mean predicted probability and its fraction of true positives.

    A perfectly calibrated model has ECE = 0. Higher ECE means predicted
    probabilities don't correspond to empirical rates.

    Args:
        y_true: 1D array-like of true binary labels (0 or 1).
        y_prob: 1D array-like of predicted probabilities for class 1.
        n_bins: Number of equal-count bins.

    Returns:
        ECE as a float in [0, 1].
    """
    # TODO: Sort indices by y_prob ascending; use np.array_split to make
    #       n_bins equal-size bins; for each bin compute
    #       (bin_size / total) * abs(mean_prob - fraction_positive); sum.
    pass


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Teaches that pure-leaf trees (unbounded depth) produce extreme
    probabilities → poor calibration; depth-constrained trees smooth
    probabilities → better calibration.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    # TODO: Fit a DecisionTreeClassifier with max_depth=None; compute ECE on
    #       its test-set predict_proba. Fit another with max_depth=5; same.
    #       Return both as a dict.
    pass


def build_random_forest(X_train, y_train, n_estimators=100, max_depth=10,
                        class_weight=None, random_state=42):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss
            so minority-class samples count more during training.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    # TODO: Fit a RandomForestClassifier with the given parameters.
    pass


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    # TODO: Zip feature_names with model.feature_importances_, sort by
    #       importance descending, return as a regular dict.
    pass


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Standard .predict() uses threshold 0.5. Passing a different threshold
    lets you observe how recall responds to operating-point choice — which
    is what `class_weight='balanced'` effectively shifts.

    Returns:
        Recall as a float in [0, 1].
    """
    # TODO: Get predict_proba(X_test)[:, 1], threshold it, compute
    #       recall_score(y_test, y_pred, zero_division=0).
    pass


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class.

    Threshold-independent: measures the model's ability to rank positives
    above negatives across all thresholds. Unlike recall at a specific
    threshold, PR-AUC does not change when you apply class_weight='balanced'
    in a way that merely shifts predicted probabilities uniformly — the
    ranking is what matters.

    Returns:
        Float in [0, 1].
    """
    # TODO: Get predict_proba(X_test)[:, 1] and call average_precision_score.
    pass


def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG.

    Args:
        output_path: Destination path (e.g., 'results/pr_curves.png').
    """
    # TODO: Create a matplotlib figure. Use PrecisionRecallDisplay.from_estimator
    #       for each model on the same axes. Title the plot. Save to
    #       output_path with plt.savefig. Close the figure.
    pass


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    # TODO: Create a figure. Use CalibrationDisplay.from_estimator for each
    #       model on the same axes. Save to output_path. Close the figure.
    pass


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Linear models need their inputs on a common scale, otherwise features
    with larger numeric ranges (total_charges ~ 0-9000) swamp features with
    smaller ranges (binary indicators at 0/1). Apply StandardScaler to the
    training features BEFORE calling this function.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    # TODO: Fit a LogisticRegression(max_iter=1000, random_state=random_state).
    pass


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test_raw,
                                     X_test_scaled, y_test, feature_names,
                                     min_diff=0.15):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    The tree-vs-linear capability demonstration. The random forest can
    capture feature interactions, non-monotonic relationships, and threshold
    effects that a linear model cannot express with per-feature coefficients.
    Finding a sample where the two models disagree — and explaining WHY in
    structural terms — is the lab's evidence that trees have capabilities
    linear models don't, regardless of aggregate PR-AUC.

    Args:
        rf_model: Trained RF (takes raw features).
        lr_model: Trained LR (takes scaled features).
        X_test_raw: Unscaled test features (what RF consumes).
        X_test_scaled: Scaled test features (what LR consumes).
        y_test: True labels for the test set.
        feature_names: List of feature name strings.
        min_diff: Minimum probability difference to count as disagreement.

    Returns:
        Dict with keys:
          - sample_idx (int): test-set row index of the selected sample
          - feature_values (dict): {name: value} for the sample's features
          - rf_proba (float): RF's predicted P(churn=1)
          - lr_proba (float): LR's predicted P(churn=1)
          - prob_diff (float): |rf_proba - lr_proba|
          - true_label (int): 0 or 1
    """
    # TODO: Compute predict_proba(:, 1) for both models on their respective
    #       X_test inputs. Take absolute difference. Find the sample index
    #       with the MAXIMUM difference (must be >= min_diff). Return the
    #       dict with all six fields populated.
    pass


def main():
    """Orchestrate all 7 lab tasks. Run with: python lab_trees.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_split()
    if not result:
        print("load_and_split not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}")

    # Task 2: Decision tree + calibration comparison
    dt = build_decision_tree(X_train, y_train)
    if dt is not None:
        print(f"\n--- Decision Tree (max_depth=5) ---")
        print(classification_report(y_test, dt.predict(X_test), zero_division=0))
        # Plot tree (first 3 levels)
        plt.figure(figsize=(14, 8))
        plot_tree(dt, feature_names=NUMERIC_FEATURES, max_depth=3,
                  filled=True, fontsize=8)
        plt.savefig("results/decision_tree.png", dpi=100, bbox_inches="tight")
        plt.close()

    cal = compare_dt_calibration(X_train, X_test, y_train, y_test)
    if cal:
        print(f"DT ECE (max_depth=None): {cal['ece_unbounded']:.3f}")
        print(f"DT ECE (max_depth=5):    {cal['ece_depth_5']:.3f}")

    # Task 3: Random forest + feature importances
    rf = build_random_forest(X_train, y_train)
    if rf is not None:
        print(f"\n--- Random Forest (max_depth=10) ---")
        imp = get_feature_importances(rf, NUMERIC_FEATURES)
        if imp:
            print("Feature importances:")
            for name, value in imp.items():
                print(f"  {name:<22s} {value:.3f}")

    # Task 4: Balanced RF + recall@0.5 comparison + PR-AUC
    rf_bal = build_random_forest(X_train, y_train, class_weight="balanced")
    if rf is not None and rf_bal is not None:
        r_def = evaluate_recall_at_threshold(rf, X_test, y_test, threshold=0.5)
        r_bal = evaluate_recall_at_threshold(rf_bal, X_test, y_test, threshold=0.5)
        print(f"\n--- class_weight effect at default 0.5 threshold ---")
        print(f"  RF default recall@0.5:  {r_def:.3f}")
        print(f"  RF balanced recall@0.5: {r_bal:.3f}  (ratio: {r_bal / max(r_def, 1e-9):.2f}x)")

        auc_def = compute_pr_auc(rf, X_test, y_test)
        auc_bal = compute_pr_auc(rf_bal, X_test, y_test)
        print(f"\n--- PR-AUC (threshold-independent ranking quality) ---")
        print(f"  RF default:  {auc_def:.3f}")
        print(f"  RF balanced: {auc_bal:.3f}")
        print("Note: class_weight='balanced' shifts the operating point at a fixed "
              "threshold; it does not improve the underlying ranking (PR-AUC).")

        # Task 5: PR curves + calibration curves
        plot_pr_curves(rf, rf_bal, X_test, y_test, "results/pr_curves.png")
        plot_calibration_curves(rf, rf_bal, X_test, y_test, "results/calibration_curves.png")

    # Task 6: Tree-vs-linear disagreement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = build_logistic_regression(X_train_scaled, y_train)
    if rf is not None and lr is not None:
        d = find_tree_vs_linear_disagreement(
            rf, lr, X_test, X_test_scaled, y_test, NUMERIC_FEATURES
        )
        if d:
            print(f"\n--- Tree-vs-linear disagreement (sample idx={d['sample_idx']}) ---")
            print(f"  RF P(churn=1)={d['rf_proba']:.3f}  LR P(churn=1)={d['lr_proba']:.3f}")
            print(f"  |diff| = {d['prob_diff']:.3f}   true label = {d['true_label']}")
            print(f"  Feature values: {d['feature_values']}")


if __name__ == "__main__":
    main()
