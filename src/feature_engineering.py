import os
import pandas as pd
import numpy as np

def compute_customer_historical_behavior(transcript: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Computes historical behavior metrics per customer:
    - cust_total_received
    - cust_total_viewed
    - cust_total_completed
    - cust_view_rate
    - cust_completion_rate
    - cust_total_trans_count
    - cust_total_trans_amount
    - cust_avg_trans_amount
    """
    # Group event counts
    event_counts = (
        transcript.groupby(["person_id", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["offer received", "offer viewed", "offer completed", "transaction"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    event_counts = event_counts.rename(columns={
        "offer received": "cust_total_received",
        "offer viewed": "cust_total_viewed",
        "offer completed": "cust_total_completed",
        "transaction": "cust_total_trans_count"
    })

    # Transaction statistics per customer
    trans_df = transcript[transcript["event"] == "transaction"]
    trans_stats = trans_df.groupby("person_id")["amount"].agg(
        cust_total_trans_amount="sum",
        cust_avg_trans_amount="mean"
    ).reset_index()

    cust_beh = pd.merge(event_counts, trans_stats, on="person_id", how="left")
    cust_beh["cust_total_trans_amount"] = cust_beh["cust_total_trans_amount"].fillna(0.0)
    cust_beh["cust_avg_trans_amount"] = cust_beh["cust_avg_trans_amount"].fillna(0.0)

    # Derived rates
    cust_beh["cust_view_rate"] = (
        cust_beh["cust_total_viewed"] / cust_beh["cust_total_received"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)

    cust_beh["cust_completion_rate"] = (
        cust_beh["cust_total_completed"] / cust_beh["cust_total_viewed"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)

    return cust_beh

def build_features(
    funnel_df: pd.DataFrame,
    profile: pd.DataFrame,
    portfolio: pd.DataFrame,
    transcript: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the feature matrix at customer-offer interaction level.
    """
    # 1. Customer Demographics
    prof = profile.copy()
    
    # Impute missing demographics (age 118 anomaly)
    median_age = prof.loc[prof["demographics_missing"] == 0, "clean_age"].median()
    median_income = prof.loc[prof["demographics_missing"] == 0, "income"].median()
    
    prof["age_imputed"] = prof["clean_age"].fillna(median_age)
    prof["income_imputed"] = prof["income"].fillna(median_income)
    
    # Gender one-hot
    prof["gender"] = prof["gender"].fillna("Missing")
    prof["gender_F"] = (prof["gender"] == "F").astype(int)
    prof["gender_M"] = (prof["gender"] == "M").astype(int)
    prof["gender_O"] = (prof["gender"] == "O").astype(int)
    prof["gender_Missing"] = (prof["gender"] == "Missing").astype(int)
    
    demo_cols = [
        "person_id", "demographics_missing", "age_imputed", "income_imputed",
        "tenure_days", "gender_F", "gender_M", "gender_O", "gender_Missing"
    ]
    prof_subset = prof[demo_cols]
    
    # 2. Portfolio Offer Features
    port = portfolio.copy()
    port["offer_type_bogo"] = (port["offer_type"] == "bogo").astype(int)
    port["offer_type_discount"] = (port["offer_type"] == "discount").astype(int)
    port["offer_type_informational"] = (port["offer_type"] == "informational").astype(int)
    
    port_cols = [
        "offer_id", "offer_type_bogo", "offer_type_discount", "offer_type_informational",
        "difficulty", "reward", "duration_days", "num_channels",
        "channel_web", "channel_email", "channel_mobile", "channel_social"
    ]
    port_subset = port[port_cols]
    
    # 3. Customer Historical Behavior
    cust_beh = compute_customer_historical_behavior(transcript, portfolio)
    
    # Merge all into funnel_df
    df = pd.merge(funnel_df, prof_subset, on="person_id", how="left")
    df = pd.merge(df, port_subset, on="offer_id", how="left", suffixes=("", "_port"))
    df = pd.merge(df, cust_beh, on="person_id", how="left")
    
    return df

if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    from src.funnel_builder import align_customer_offer_funnel
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    portfolio, profile, transcript = load_all_raw_data(data_dir)
    funnel = align_customer_offer_funnel(transcript, portfolio)
    
    feature_df = build_features(funnel, profile, portfolio, transcript)
    print(f"Feature dataset shape: {feature_df.shape}")
    print(f"Columns:\n{feature_df.columns.tolist()}")
