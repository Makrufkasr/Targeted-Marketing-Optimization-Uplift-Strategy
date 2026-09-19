import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report
)

FEATURE_COLS = [
    # Demographics
    "demographics_missing", "age_imputed", "income_imputed", "tenure_days",
    "gender_F", "gender_M", "gender_O", "gender_Missing",
    # Offer metadata
    "offer_type_bogo", "offer_type_discount", "offer_type_informational",
    "difficulty", "reward", "duration_days", "num_channels",
    "channel_web", "channel_email", "channel_mobile", "channel_social",
    # Customer historical behavior
    "cust_total_received", "cust_total_viewed", "cust_total_completed",
    "cust_view_rate", "cust_completion_rate",
    "cust_total_trans_count", "cust_total_trans_amount", "cust_avg_trans_amount"
]

def split_by_customer(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits data at customer (person_id) level to prevent customer-level data leakage.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["person_id"]))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df

def train_calibrated_model(train_df: pd.DataFrame):
    """
    Trains a HistGradientBoostingClassifier with Isotonic Calibration.
    """
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]

    # Base gradient boosting model
    base_model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )

    # Calibrate probability using 3-fold cross validation
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    return calibrated_model

def evaluate_and_score(model, test_df: pd.DataFrame):
    """
    Evaluates calibrated model and scores the test dataset with deciles.
    """
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print("=== MODEL PERFORMANCE METRICS ===")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"Brier Score (Calib):    {brier:.4f}")
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, preds))

    scored_df = test_df.copy()
    scored_df["calibrated_propensity"] = probs
    
    # 10 Deciles: Decile 1 is highest propensity, Decile 10 is lowest
    scored_df["decile"] = pd.qcut(
        scored_df["calibrated_propensity"].rank(method="first", ascending=False),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)]
    )

    return scored_df, {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier}

def compute_decile_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard marketing Lift & Gain decile table.
    """
    baseline_rate = scored_df["target"].mean()
    total_converters = scored_df["target"].sum()

    grouped = scored_df.groupby("decile", observed=True).agg(
        total_customers=("target", "count"),
        actual_converters=("target", "sum"),
        avg_propensity=("calibrated_propensity", "mean"),
        actual_reward_spent=("actual_reward_incurred", "sum"),
        window_revenue=("window_trans_amount", "sum")
    ).reset_index()

    grouped["conversion_rate"] = grouped["actual_converters"] / grouped["total_customers"]
    grouped["cum_customers"] = grouped["total_customers"].cumsum()
    grouped["cum_converters"] = grouped["actual_converters"].cumsum()
    grouped["gain_pct"] = (grouped["cum_converters"] / total_converters) * 100.0
    grouped["cum_conv_rate"] = grouped["cum_converters"] / grouped["cum_customers"]
    grouped["decile_lift"] = grouped["conversion_rate"] / baseline_rate
    grouped["cum_lift"] = grouped["cum_conv_rate"] / baseline_rate

    return grouped
