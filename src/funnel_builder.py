import os
import pandas as pd
import numpy as np

def align_customer_offer_funnel(transcript: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the causal customer-offer interaction records with strict temporal validation.
    Optimized for high-throughput vectorized binary search.
    
    Target definition (y in {0, 1}):
    - y = 1 (Valid Conversion): received -> viewed -> completed (viewed_time <= completed_time within validity window)
    - y = 0 (Wasteful / Non-Conversion):
        - Kasus A: completed without viewed (Sure Things / Cannibalization)
        - Kasus B: expired without completed
        - Kasus C: viewed but expired without completed
    """
    portfolio_lookup = portfolio.set_index("offer_id").to_dict(orient="index")
    
    received_df = transcript[transcript["event"] == "offer received"].copy()
    viewed_df = transcript[transcript["event"] == "offer viewed"].copy()
    completed_df = transcript[transcript["event"] == "offer completed"].copy()
    trans_df = transcript[transcript["event"] == "transaction"].copy()
    
    # Pre-aggregate views & completions: (person_id, offer_id) -> list of times
    views_by_pair = viewed_df.groupby(["person_id", "offer_id"])["time"].apply(list).to_dict()
    completed_by_pair = completed_df.groupby(["person_id", "offer_id"])["time"].apply(list).to_dict()
    
    # Pre-sort and store transactions in numpy arrays per customer for O(log N) lookup
    trans_by_person = {}
    trans_sorted = trans_df.sort_values("time")
    for p_id, grp in trans_sorted.groupby("person_id"):
        trans_by_person[p_id] = (grp["time"].to_numpy(), grp["amount"].to_numpy())
        
    records = []
    used_views = set()
    used_completions = set()
    
    # Sort received by time and iterate with itertuples for 50x speedup
    received_sorted = received_df.sort_values("time")
    
    for row in received_sorted.itertuples(index=False):
        person_id = row.person_id
        offer_id = row.offer_id
        rec_time = row.time
        
        offer_info = portfolio_lookup.get(offer_id, {})
        duration_hours = offer_info.get("duration_hours", 0.0)
        offer_type = offer_info.get("offer_type", "unknown")
        reward_value = offer_info.get("reward", 0.0)
        difficulty = offer_info.get("difficulty", 0.0)
        
        end_time = rec_time + duration_hours
        
        # Check views within window [rec_time, end_time]
        cand_views = views_by_pair.get((person_id, offer_id), [])
        valid_view_time = None
        for v_time in cand_views:
            view_key = (person_id, offer_id, v_time)
            if rec_time <= v_time <= end_time and view_key not in used_views:
                valid_view_time = v_time
                used_views.add(view_key)
                break
                
        # Check completions within window [rec_time, end_time]
        cand_completions = completed_by_pair.get((person_id, offer_id), [])
        valid_comp_time = None
        for c_time in cand_completions:
            comp_key = (person_id, offer_id, c_time)
            if rec_time <= c_time <= end_time and comp_key not in used_completions:
                valid_comp_time = c_time
                used_completions.add(comp_key)
                break
                
        # Binary search for transactions in window [rec_time, end_time]
        if person_id in trans_by_person:
            t_times, t_amts = trans_by_person[person_id]
            left = np.searchsorted(t_times, rec_time, side="left")
            right = np.searchsorted(t_times, end_time, side="right")
            window_trans_amount = float(np.sum(t_amts[left:right]))
            window_trans_count = int(right - left)
        else:
            window_trans_amount = 0.0
            window_trans_count = 0
            
        is_viewed = 1 if valid_view_time is not None else 0
        is_completed = 1 if valid_comp_time is not None else 0
        
        # Target y definition
        if offer_type in ["bogo", "discount"]:
            if is_completed == 1 and is_viewed == 1 and valid_view_time <= valid_comp_time:
                target = 1
                funnel_category = "valid_conversion"
            elif is_completed == 1 and (is_viewed == 0 or valid_view_time > valid_comp_time):
                target = 0
                funnel_category = "unintended_wasteful_conversion"
            elif is_viewed == 1 and is_completed == 0:
                target = 0
                funnel_category = "viewed_uncompleted"
            else:
                target = 0
                funnel_category = "expired_ignored"
        elif offer_type == "informational":
            if is_viewed == 1 and window_trans_count > 0:
                target = 1
                funnel_category = "informational_influenced"
            elif is_viewed == 1 and window_trans_count == 0:
                target = 0
                funnel_category = "informational_viewed_no_trans"
            else:
                target = 0
                funnel_category = "informational_ignored"
        else:
            target = 0
            funnel_category = "other"
            
        actual_reward_incurred = reward_value if is_completed == 1 else 0.0
        
        records.append({
            "person_id": person_id,
            "offer_id": offer_id,
            "offer_type": offer_type,
            "received_time": rec_time,
            "valid_until_time": end_time,
            "duration_days": duration_hours / 24.0,
            "difficulty": difficulty,
            "reward": reward_value,
            "is_viewed": is_viewed,
            "view_time": valid_view_time,
            "is_completed": is_completed,
            "complete_time": valid_comp_time,
            "window_trans_amount": round(window_trans_amount, 2),
            "window_trans_count": window_trans_count,
            "actual_reward_incurred": actual_reward_incurred,
            "funnel_category": funnel_category,
            "target": target
        })
        
    funnel_df = pd.DataFrame(records)
    return funnel_df
