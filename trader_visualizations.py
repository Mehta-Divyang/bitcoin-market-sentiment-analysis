import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_path = "visualizations/historical"
os.makedirs(output_path, exist_ok=True)

trader = pd.read_csv("data/historical_data.csv")

trader["date"] = pd.to_datetime(
    trader["Timestamp IST"],
    dayfirst=True,
    errors="coerce"
)

trader["date_only"] = trader["date"].dt.date
trader["hour"] = trader["date"].dt.hour
trader["year"] = trader["date"].dt.year
trader["month"] = trader["date"].dt.month

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

daily_pnl = trader.groupby("date_only")["Closed PnL"].sum().reset_index()
daily_pnl["date_only"] = pd.to_datetime(daily_pnl["date_only"])

daily_pnl = daily_pnl.sort_values("date_only")
daily_pnl["cumulative"] = daily_pnl["Closed PnL"].cumsum()

plt.figure(figsize=(14,6))

plt.plot(
    daily_pnl["date_only"],
    daily_pnl["cumulative"],
    linewidth=2
)

plt.axhline(0,color="red",linestyle="--")

plt.title("Cumulative Closed PnL Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")

plt.grid(alpha=0.3)

plt.savefig(
    output_path + "/trader_01_cumulative_pnl.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

top_accounts = (
    trader.groupby("Account")["Closed PnL"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10,7))

plt.barh(
    top_accounts.index.astype(str),
    top_accounts.values,
    color="green"
)

plt.title("Top 15 Accounts by Total PnL")
plt.xlabel("Closed PnL")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_02_top_accounts.png",
    dpi=300
)

plt.close()

pnl = trader["Closed PnL"]

plt.figure(figsize=(10,6))

plt.hist(
    pnl.clip(-5000,5000),
    bins=80,
    color="steelblue"
)

plt.axvline(0,color="red")

plt.title("Distribution of Closed PnL")
plt.xlabel("Closed PnL")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_03_pnl_distribution.png",
    dpi=300
)

plt.close()

coin_volume = (
    trader.groupby("Coin")["Size USD"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))

plt.bar(
    coin_volume.index,
    coin_volume.values/1e6
)

plt.title("Top 10 Coins by Volume")
plt.ylabel("Volume (Million USD)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    output_path + "/trader_04_coin_volume.png",
    dpi=300
)

plt.close()

acc_stats = trader.groupby("Account").agg(
    trades=("Closed PnL","count"),
    wins=("Closed PnL",lambda x:(x>0).sum()),
    total_pnl=("Closed PnL","sum")
)

acc_stats["win_rate"] = (
    acc_stats["wins"] /
    acc_stats["trades"]
) * 100

plt.figure(figsize=(10,6))

scatter = plt.scatter(
    acc_stats["trades"],
    acc_stats["win_rate"],
    c=acc_stats["total_pnl"],
    cmap="RdYlGn",
    alpha=0.7
)

plt.colorbar(scatter,label="Total PnL")

plt.title("Win Rate vs Total Trades")
plt.xlabel("Trades")
plt.ylabel("Win Rate (%)")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_05_winrate_scatter.png",
    dpi=300
)

plt.close()

hourly = trader.groupby("hour").size()

plt.figure(figsize=(10,5))

plt.bar(
    hourly.index,
    hourly.values
)

plt.title("Trading Activity by Hour")
plt.xlabel("Hour")
plt.ylabel("Trades")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_06_hourly_activity.png",
    dpi=300
)

plt.close()

monthly_pnl = (
    trader.groupby(["year","month"])["Closed PnL"]
    .sum()
    .unstack(fill_value=0)
)

plt.figure(figsize=(12,6))

plt.imshow(
    monthly_pnl,
    aspect="auto",
    cmap="RdYlGn"
)

plt.colorbar(label="PnL")

plt.xticks(
    range(12),
    months
)

plt.yticks(
    range(len(monthly_pnl.index)),
    monthly_pnl.index
)

plt.title("Monthly PnL Heatmap")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_07_monthly_pnl_heatmap.png",
    dpi=300
)

plt.close()

direction_pnl = (
    trader.groupby("Direction")["Closed PnL"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,6))

plt.barh(
    direction_pnl.index,
    direction_pnl.values
)

plt.title("PnL by Trade Direction")
plt.xlabel("PnL")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_08_direction_pnl.png",
    dpi=300
)

plt.close()

fees = (
    trader.groupby("Account")["Fee"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10,6))

plt.bar(
    range(len(fees)),
    fees.values
)

plt.xticks(
    range(len(fees)),
    [x[:8] for x in fees.index],
    rotation=45
)

plt.title("Top Accounts by Total Fees Paid")
plt.ylabel("Fees")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_09_fee_drag.png",
    dpi=300
)

plt.close()

coin_pnl = (
    trader.groupby("Coin")["Closed PnL"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10,7))

plt.barh(
    coin_pnl.index,
    coin_pnl.values
)

plt.title("Top Coins by Total PnL")
plt.xlabel("PnL")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_10_coin_pnl.png",
    dpi=300
)

plt.close()

trader["size_bucket"] = pd.cut(
    trader["Size USD"],
    bins=[0,100,500,1000,5000,10000,50000,1e12],
    labels=[
        "<100",
        "100-500",
        "500-1K",
        "1K-5K",
        "5K-10K",
        "10K-50K",
        ">50K"
    ]
)

bucket_pnl = (
    trader.groupby(
        "size_bucket",
        observed=True
    )["Closed PnL"]
    .mean()
)

plt.figure(figsize=(10,5))

plt.bar(
    bucket_pnl.index.astype(str),
    bucket_pnl.values
)

plt.axhline(0,color="red")

plt.title("Average PnL by Trade Size Bucket")
plt.ylabel("Average Closed PnL")

plt.tight_layout()

plt.savefig(
    output_path + "/trader_11_size_bucket_pnl.png",
    dpi=300
)

plt.close()

print("All trader visualizations saved successfully.")