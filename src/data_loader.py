import os
import pandas as pd
import numpy as np

def load_portfolio(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses portfolio.json (offer metadata)."""
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.rename(columns={"id": "offer_id"})
    
    channels = ["web", "email", "mobile", "social"]
    for ch in channels:
        df[f"channel_{ch}"] = df["channels"].apply(lambda c: 1 if isinstance(c, list) and ch in c else 0)
    df["num_channels"] = df["channels"].apply(lambda c: len(c) if isinstance(c, list) else 0)
    
    df["duration_days"] = df["duration"]
    df["duration_hours"] = df["duration"] * 24.0
    
    return df

def load_profile(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans profile.json (customer demographics).
    Age 118 indicates missing demographic data (correlated with null income and gender).
    """
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.rename(columns={"id": "person_id"})
    
    df["demographics_missing"] = (df["age"] == 118).astype(int)
    df["clean_age"] = df["age"].apply(lambda x: np.nan if x == 118 else x)
    
    df["membership_date"] = pd.to_datetime(df["became_member_on"].astype(str), format="%Y%m%d")
    max_date = df["membership_date"].max()
    df["tenure_days"] = (max_date - df["membership_date"]).dt.days
    
    return df

def load_transcript(file_path: str) -> pd.DataFrame:
    """
    Loads and parses transcript.json (event logs).
    Fast vectorized parsing of nested 'value' dictionaries.
    """
    df = pd.read_json(file_path, orient="records", lines=True)
    df = df.rename(columns={"person": "person_id"})
    
    # Ultra-fast list comprehension parsing (0.1s vs 30s)
    val_list = df["value"].tolist()
    offer_ids = [
        v.get("offer id") or v.get("offer_id") if isinstance(v, dict) else None
        for v in val_list
    ]
    amounts = [
        float(v.get("amount", 0.0)) if isinstance(v, dict) and "amount" in v else 0.0
        for v in val_list
    ]
    rewards = [
        float(v.get("reward", 0.0)) if isinstance(v, dict) and "reward" in v else 0.0
        for v in val_list
    ]
    
    df["offer_id"] = offer_ids
    df["amount"] = amounts
    df["reward"] = rewards
    df.drop(columns=["value"], inplace=True)
    
    df["time_days"] = df["time"] / 24.0
    df["event"] = df["event"].str.strip()
    
    return df

def load_all_raw_data(data_dir: str):
    """Loads all 3 raw datasets from data_dir."""
    portfolio_p = os.path.join(data_dir, "portfolio.json")
    profile_p = os.path.join(data_dir, "profile.json")
    transcript_p = os.path.join(data_dir, "transcript.json")
    
    portfolio = load_portfolio(portfolio_p)
    profile = load_profile(profile_p)
    transcript = load_transcript(transcript_p)
    
    return portfolio, profile, transcript
