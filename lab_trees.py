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


# These are the columns we use as input features (X) for our models.
# We only use numeric columns — the model can't handle text directly.
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


# ---------------------------------------------------------------------------
# TASK 1 — Load and split
# ---------------------------------------------------------------------------

def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    # Read the CSV file into a pandas DataFrame (like opening an Excel sheet)
    df = pd.read_csv(filepath)

    # X = input features (the columns the model learns from)
    X = df[NUMERIC_FEATURES]

    # y = the target we want to predict (1 = churned, 0 = stayed)
    y = df['churned']

    # Split into 80% training and 20% test sets.
    # stratify=y ensures both splits keep the same ~16% churn rate.
    # random_state=42 makes the split reproducible every time we run this.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    return (X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# TASK 2 — Decision tree and calibration comparison
# ---------------------------------------------------------------------------

def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    # Create the decision tree model with the given depth limit
    model = DecisionTreeClassifier(
        max_depth=max_depth,      # How many questions the tree can ask
        random_state=random_state
    )

    # Train the model — it learns the decision rules from the training data
    model.fit(X_train, y_train)

    return model


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
    n = len(y_prob)

    # Step 1: Get the sort order from lowest to highest predicted probability
    order = np.argsort(y_prob)

    # Reorder both arrays so they go from least to most confident prediction
    y_prob = y_prob[order]
    y_true = y_true.iloc[order] if hasattr(y_true, 'iloc') else y_true[order]

    # Step 2: Split the sorted indices into n_bins equal-sized groups
    # Each group contains customers with similar predicted probabilities
    bins = np.array_split(np.arange(n), n_bins)

    # Step 3: For each bin, measure how far off the model's confidence is
    ece = 0.0
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue

        # What the model predicted on average for this group
        mean_predicted = y_prob[bin_indices].mean()

        # What actually happened in this group (true positive rate)
        if hasattr(y_true, 'iloc'):
            fraction_actual = y_true.iloc[bin_indices].mean()
        else:
            fraction_actual = y_true[bin_indices].mean()

        # Weight this bin's error by how many samples it contains
        weight = len(bin_indices) / n
        ece += weight * abs(mean_predicted - fraction_actual)

    return ece


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Teaches that pure-leaf trees (unbounded depth) produce extreme
    probabilities → poor calibration; depth-constrained trees smooth
    probabilities → better calibration.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    # Train a tree with no depth limit — it will memorize the training data
    dt_unbounded = build_decision_tree(X_train, y_train, max_depth=None)

    # Train a tree limited to 5 levels — smoother, more honest probabilities
    dt_depth5 = build_decision_tree(X_train, y_train, max_depth=5)

    # Get predicted probabilities on the test set for each tree
    # predict_proba returns [[P(stay), P(churn)], ...] — we take column 1
    prob_unbounded = dt_unbounded.predict_proba(X_test)[:, 1]
    prob_depth5 = dt_depth5.predict_proba(X_test)[:, 1]

    # Compute ECE for both and return as a dictionary
    return {
        "ece_unbounded": compute_ece(y_test, prob_unbounded),
        "ece_depth_5":   compute_ece(y_test, prob_depth5)
    }


# ---------------------------------------------------------------------------
# TASK 3 — Random forest and feature importances
# ---------------------------------------------------------------------------

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
    # A random forest trains many decision trees and combines their votes.
    # n_estimators = number of trees in the forest
    # max_depth = depth limit on each individual tree
    # class_weight = how much to weight the minority class (churners)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    # Zip each feature name with its importance score from the trained model
    pairs = zip(feature_names, model.feature_importances_)

    # Sort by importance value (highest first)
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    return dict(sorted_pairs)


# ---------------------------------------------------------------------------
# TASK 4 — class_weight at the default 0.5 threshold
# ---------------------------------------------------------------------------

def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Standard .predict() uses threshold 0.5. Passing a different threshold
    lets you observe how recall responds to operating-point choice — which
    is what `class_weight='balanced'` effectively shifts.

    Returns:
        Recall as a float in [0, 1].
    """
    # Get the probability of churn for each customer
    y_prob = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to binary predictions using the given threshold
    # Any probability >= threshold is predicted as churn (1)
    y_pred = (y_prob >= threshold).astype(int)

    # Recall = how many actual churners did we correctly identify?
    return recall_score(y_test, y_pred, zero_division=0)


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
    # Get predicted probabilities for the churn class
    y_prob = model.predict_proba(X_test)[:, 1]

    # average_precision_score computes PR-AUC (area under precision-recall curve)
    return average_precision_score(y_test, y_prob)


# ---------------------------------------------------------------------------
# TASK 5 — PR curves and calibration curves
# ---------------------------------------------------------------------------

def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG.

    Args:
        output_path: Destination path (e.g., 'results/pr_curves.png').
    """
    fig, ax = plt.subplots()

    # Plot the default RF's precision-recall curve
    PrecisionRecallDisplay.from_estimator(
        rf_default, X_test, y_test, ax=ax, name="RF Default"
    )

    # Plot the balanced RF's curve on the same axes for comparison
    PrecisionRecallDisplay.from_estimator(
        rf_balanced, X_test, y_test, ax=ax, name="RF Balanced"
    )

    ax.set_title("Precision-Recall Curves")
    plt.savefig(output_path)
    plt.close()  # Free memory — always close after saving


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    fig, ax = plt.subplots()

    # A well-calibrated model's curve lies close to the diagonal (perfect line)
    CalibrationDisplay.from_estimator(
        rf_default, X_test, y_test, ax=ax, n_bins=10, name="RF Default"
    )
    CalibrationDisplay.from_estimator(
        rf_balanced, X_test, y_test, ax=ax, n_bins=10, name="RF Balanced"
    )

    ax.set_title("Calibration Curves")
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# TASK 6 — Tree-vs-linear capability demonstration
# ---------------------------------------------------------------------------

def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Linear models need their inputs on a common scale, otherwise features
    with larger numeric ranges (total_charges ~ 0-9000) swamp features with
    smaller ranges (binary indicators at 0/1). Apply StandardScaler to the
    training features BEFORE calling this function.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    # Logistic regression learns a weighted sum of features.
    # max_iter=1000 gives the optimizer enough steps to converge.
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


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
    # Get churn probability from each model
    rf_proba = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Compute the absolute difference between the two models for every customer
    diffs = np.abs(rf_proba - lr_proba)

    # Find the customer where the two models disagree the most
    best_idx = np.argmax(diffs)

    # If even the biggest disagreement is smaller than min_diff, return None
    if diffs[best_idx] < min_diff:
        return None

    # Get the actual DataFrame index (row label) for this customer
    sample_idx = X_test_raw.index[best_idx]

    # Build a dictionary of feature name → feature value for this customer
    feature_values = {
        name: X_test_raw.iloc[best_idx][name]
        for name in feature_names
    }

    return {
        "sample_idx":     sample_idx,
        "feature_values": feature_values,
        "rf_proba":       float(rf_proba[best_idx]),
        "lr_proba":       float(lr_proba[best_idx]),
        "prob_diff":      float(diffs[best_idx]),
        "true_label":     int(y_test.iloc[best_idx])
    }


# ---------------------------------------------------------------------------
# MAIN — runs all 7 tasks end to end
# ---------------------------------------------------------------------------

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
    print("\n--- Random Forest BALANCED ---")
    print(classification_report(y_test, rf_bal.predict(X_test), zero_division=0))
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