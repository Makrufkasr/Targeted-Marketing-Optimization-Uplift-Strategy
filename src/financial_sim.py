import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_financial_simulation(
    scored_df: pd.DataFrame,
    gross_margin_rate: float = 0.60,
    cost_per_message: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates financial outcomes across deciles and compares Mass Campaign vs Optimized Targeting.
    
    Parameters:
    - scored_df: DataFrame containing 'decile', 'target', 'window_trans_amount', 'actual_reward_incurred'
    - gross_margin_rate: F&B gross margin on transactions (e.g., 60%)
    - cost_per_message: Communication transmission cost per offer (e.g., $0.05)
    """
    total_population = len(scored_df)
    baseline_converters = scored_df["target"].sum()
    
    decile_summary = []
    
    for decile_label in [f"D{i}" for i in range(1, 11)]:
        d_df = scored_df[scored_df["decile"] == decile_label]
        n_cust = len(d_df)
        converters = d_df["target"].sum()
        cr = converters / n_cust if n_cust > 0 else 0.0
        
        revenue = d_df["window_trans_amount"].sum()
        reward_cost = d_df["actual_reward_incurred"].sum()
        msg_cost = n_cust * cost_per_message
        total_cost = reward_cost + msg_cost
        
        gross_profit = revenue * gross_margin_rate
        net_profit = gross_profit - total_cost
        roi = (net_profit / total_cost) if total_cost > 0 else 0.0
        
        decile_summary.append({
            "decile": decile_label,
            "customers": n_cust,
            "converters": converters,
            "conv_rate": cr,
            "revenue": round(revenue, 2),
            "reward_cost": round(reward_cost, 2),
            "msg_cost": round(msg_cost, 2),
            "total_cost": round(total_cost, 2),
            "gross_profit": round(gross_profit, 2),
            "net_profit": round(net_profit, 2),
            "roi": round(roi, 2)
        })
        
    dec_df = pd.DataFrame(decile_summary)
    
    # Cumulative metrics
    dec_df["cum_customers"] = dec_df["customers"].cumsum()
    dec_df["cum_pct_audience"] = (dec_df["cum_customers"] / total_population) * 100.0
    dec_df["cum_converters"] = dec_df["converters"].cumsum()
    dec_df["cum_gain_pct"] = (dec_df["cum_converters"] / baseline_converters) * 100.0
    dec_df["cum_conv_rate"] = dec_df["cum_converters"] / dec_df["cum_customers"]
    baseline_cr = baseline_converters / total_population
    dec_df["decile_lift"] = dec_df["conv_rate"] / baseline_cr
    dec_df["cum_lift"] = dec_df["cum_conv_rate"] / baseline_cr
    
    dec_df["cum_revenue"] = dec_df["revenue"].cumsum()
    dec_df["cum_total_cost"] = dec_df["total_cost"].cumsum()
    dec_df["cum_reward_cost"] = dec_df["reward_cost"].cumsum()
    dec_df["cum_gross_profit"] = dec_df["gross_profit"].cumsum()
    dec_df["cum_net_profit"] = dec_df["net_profit"].cumsum()
    dec_df["cum_roi"] = (dec_df["cum_net_profit"] / dec_df["cum_total_cost"]).round(2)
    
    # Mass Campaign (All Deciles)
    mass_row = dec_df.iloc[-1]
    top30_row = dec_df.iloc[2] # D1-D3 (30%)
    top50_row = dec_df.iloc[4] # D1-D5 (50%)
    
    comparison_summary = pd.DataFrame([
        {
            "Strategy": "Mass Blast (Baseline - 100% Audience)",
            "Targeted Audience": int(mass_row["cum_customers"]),
            "Audience Pct": "100.0%",
            "Captured Converters": int(mass_row["cum_converters"]),
            "Captured Conv Pct": "100.0%",
            "Effective Conv Rate": f"{mass_row['cum_conv_rate']*100:.2f}%",
            "Total Revenue ($)": f"${mass_row['cum_revenue']:,.2f}",
            "Reward Subsidy ($)": f"${mass_row['cum_reward_cost']:,.2f}",
            "Saved Subsidy ($)": "$0.00",
            "Net Profit ($)": f"${mass_row['cum_net_profit']:,.2f}",
            "ROI": f"{mass_row['cum_roi']*100:.1f}%"
        },
        {
            "Strategy": "Targeted Top 30% (High Precision D1-D3)",
            "Targeted Audience": int(top30_row["cum_customers"]),
            "Audience Pct": f"{top30_row['cum_pct_audience']:.1f}%",
            "Captured Converters": int(top30_row["cum_converters"]),
            "Captured Conv Pct": f"{top30_row['cum_gain_pct']:.1f}%",
            "Effective Conv Rate": f"{top30_row['cum_conv_rate']*100:.2f}%",
            "Total Revenue ($)": f"${top30_row['cum_revenue']:,.2f}",
            "Reward Subsidy ($)": f"${top30_row['cum_reward_cost']:,.2f}",
            "Saved Subsidy ($)": f"${(mass_row['cum_reward_cost'] - top30_row['cum_reward_cost']):,.2f}",
            "Net Profit ($)": f"${top30_row['cum_net_profit']:,.2f}",
            "ROI": f"{top30_row['cum_roi']*100:.1f}%"
        },
        {
            "Strategy": "Targeted Top 50% (Balanced Reach D1-D5)",
            "Targeted Audience": int(top50_row["cum_customers"]),
            "Audience Pct": f"{top50_row['cum_pct_audience']:.1f}%",
            "Captured Converters": int(top50_row["cum_converters"]),
            "Captured Conv Pct": f"{top50_row['cum_gain_pct']:.1f}%",
            "Effective Conv Rate": f"{top50_row['cum_conv_rate']*100:.2f}%",
            "Total Revenue ($)": f"${top50_row['cum_revenue']:,.2f}",
            "Reward Subsidy ($)": f"${top50_row['cum_reward_cost']:,.2f}",
            "Saved Subsidy ($)": f"${(mass_row['cum_reward_cost'] - top50_row['cum_reward_cost']):,.2f}",
            "Net Profit ($)": f"${top50_row['cum_net_profit']:,.2f}",
            "ROI": f"{top50_row['cum_roi']*100:.1f}%"
        }
    ])
    
    return dec_df, comparison_summary

def plot_and_save_visualizations(dec_df: pd.DataFrame, output_dir: str):
    """
    Generates and saves:
    1. lift_curve.png
    2. gain_chart.png
    3. decile_profit_simulation.png
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Cumulative Lift Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = range(1, 11)
    ax.plot(x, dec_df["cum_lift"], marker="o", color="#006241", linewidth=2.5, label="Calibrated Model Lift")
    ax.axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.8, label="Baseline (Random Mass Blast = 1.0x)")
    ax.set_title("Cumulative Lift Curve across Deciles", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Decile (D1 = Highest Propensity)", fontsize=11)
    ax.set_ylabel("Cumulative Lift (vs Baseline)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dec_df["decile"])
    for i, txt in enumerate(dec_df["cum_lift"]):
        ax.annotate(f"{txt:.2f}x", (x[i], txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    lift_path = os.path.join(output_dir, "lift_curve.png")
    fig.savefig(lift_path)
    plt.close(fig)
    
    # 2. Cumulative Gain Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot([0] + list(dec_df["cum_pct_audience"]), [0] + list(dec_df["cum_gain_pct"]), marker="s", color="#1e3932", linewidth=2.5, label="Model Cumulative Gain")
    ax.plot([0, 100], [0, 100], color="#999999", linestyle="--", linewidth=1.8, label="Random Selection (Diagonal)")
    ax.set_title("Cumulative Gain Chart", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("% of Total Audience Contacted", fontsize=11)
    ax.set_ylabel("% of Total Converters Captured", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    for i, txt in enumerate(dec_df["cum_gain_pct"]):
        ax.annotate(f"{txt:.1f}%", (dec_df["cum_pct_audience"].iloc[i], txt), textcoords="offset points", xytext=(-5, 6), fontsize=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    gain_path = os.path.join(output_dir, "gain_chart.png")
    fig.savefig(gain_path)
    plt.close(fig)
    
    # 3. Decile Profit & Cost Simulation
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2 = ax1.twinx()
    
    bar_width = 0.35
    indices = np.arange(10)
    
    bars1 = ax1.bar(indices - bar_width/2, dec_df["net_profit"], width=bar_width, color="#00754a", label="Net Profit ($)")
    bars2 = ax1.bar(indices + bar_width/2, dec_df["total_cost"], width=bar_width, color="#cba258", label="Total Cost (Subsidy+Msg) ($)")
    
    line = ax2.plot(indices, dec_df["conv_rate"] * 100, color="#d9534f", marker="o", linewidth=2.2, label="Conversion Rate (%)")
    
    ax1.set_title("Decile Economics: Net Profit, Cost, and Conversion Rate", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Propensity Decile", fontsize=11)
    ax1.set_ylabel("Financial Value ($)", fontsize=11)
    ax2.set_ylabel("Conversion Rate (%)", fontsize=11, color="#d9534f")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(dec_df["decile"])
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    
    plt.tight_layout()
    profit_path = os.path.join(output_dir, "decile_profit_simulation.png")
    fig.savefig(profit_path)
    plt.close(fig)
    
    print(f"Visualizations saved to: {output_dir}")
