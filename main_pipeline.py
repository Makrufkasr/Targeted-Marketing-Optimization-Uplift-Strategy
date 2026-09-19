import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_all_raw_data
from src.funnel_builder import align_customer_offer_funnel
from src.feature_engineering import build_features
from src.model import split_by_customer, train_calibrated_model, evaluate_and_score, compute_decile_table
from src.financial_sim import run_financial_simulation, plot_and_save_visualizations

def main():
    print("=" * 70, flush=True)
    print("TARGETED MARKETING OPTIMIZATION & UPLIFT STRATEGY PIPELINE", flush=True)
    print("=" * 70, flush=True)

    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Ingestion & Preprocessing
    print("\n[STEP 1] Ingesting and Preprocessing Raw Datasets...", flush=True)
    portfolio, profile, transcript = load_all_raw_data(raw_dir)
    print(f"  -> Portfolio:  {portfolio.shape[0]} offers", flush=True)
    print(f"  -> Profiles:   {profile.shape[0]} customers ({profile['demographics_missing'].sum()} missing demographics)", flush=True)
    print(f"  -> Transcript: {transcript.shape[0]} logged interaction events", flush=True)

    # 2. Causal Funnel Alignment
    print("\n[STEP 2] Performing Causal Event Funnel Alignment...", flush=True)
    funnel_df = align_customer_offer_funnel(transcript, portfolio)
    print(f"  -> Total Customer-Offer Interactions: {len(funnel_df):,}", flush=True)
    print("\nFunnel Segmentation:", flush=True)
    for cat, count in funnel_df["funnel_category"].value_counts().items():
        pct = (count / len(funnel_df)) * 100
        print(f"  * {cat:32s}: {count:6d} ({pct:5.2f}%)", flush=True)

    # Quantify wasted subsidies
    wasted_subsidy = funnel_df.loc[
        funnel_df["funnel_category"] == "unintended_wasteful_conversion",
        "actual_reward_incurred"
    ].sum()
    valid_subsidy = funnel_df.loc[
        funnel_df["funnel_category"] == "valid_conversion",
        "actual_reward_incurred"
    ].sum()
    print(f"\n  -> Valid Influenced Subsidy:     ${valid_subsidy:,.2f}", flush=True)
    print(f"  -> Wasted Cannibalized Subsidy:  ${wasted_subsidy:,.2f} ({(wasted_subsidy/(wasted_subsidy+valid_subsidy+1e-5))*100:.1f}% of total rewards)", flush=True)

    # 3. Feature Engineering
    print("\n[STEP 3] Engineering Customer-Offer Features...", flush=True)
    features_df = build_features(funnel_df, profile, portfolio, transcript)
    print(f"  -> Feature Matrix Shape: {features_df.shape}", flush=True)

    processed_path = os.path.join(processed_dir, "customer_offer_features.csv")
    features_df.to_csv(processed_path, index=False)
    print(f"  -> Processed dataset saved to: {processed_path}", flush=True)

    # 4. Model Training & Evaluation
    print("\n[STEP 4] Training Calibrated Propensity Model (HistGradientBoosting + Isotonic)...", flush=True)
    train_df, test_df = split_by_customer(features_df, test_size=0.20, random_state=42)
    print(f"  -> Train Interactions: {len(train_df):,} ({train_df['person_id'].nunique():,} unique customers)", flush=True)
    print(f"  -> Test Interactions:  {len(test_df):,} ({test_df['person_id'].nunique():,} unique customers)", flush=True)

    model = train_calibrated_model(train_df)
    scored_test_df, metrics = evaluate_and_score(model, test_df)

    # 5. Financial Simulation & Decile Analysis
    print("\n[STEP 5] Running Business Simulation & Unit Economics Analysis...", flush=True)
    dec_df, comparison_summary = run_financial_simulation(
        scored_test_df,
        gross_margin_rate=0.60,
        cost_per_message=0.05
    )

    print("\nDecile Summary Performance:", flush=True)
    display_cols = ["decile", "customers", "converters", "conv_rate", "decile_lift", "cum_lift", "revenue", "reward_cost", "net_profit", "roi"]
    print(dec_df[display_cols].to_string(index=False), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("EXECUTIVE STRATEGY COMPARISON: MASS BLAST vs TARGETED OPTIMIZATION", flush=True)
    print("=" * 70, flush=True)
    print(comparison_summary.to_string(index=False), flush=True)

    dec_table_path = os.path.join(artifacts_dir, "decile_summary.csv")
    dec_df.to_csv(dec_table_path, index=False)

    comp_table_path = os.path.join(artifacts_dir, "executive_comparison.csv")
    comparison_summary.to_csv(comp_table_path, index=False)

    # 6. Generate Visualizations
    print("\n[STEP 6] Plotting and Saving Lift & Gain Visualizations...", flush=True)
    plot_and_save_visualizations(dec_df, artifacts_dir)

    print("\nPipeline execution complete successfully!", flush=True)

if __name__ == "__main__":
    main()
