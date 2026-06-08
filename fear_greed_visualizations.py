import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

output_path = "visualizations/fear_grid"
os.makedirs(output_path, exist_ok=True)

fg = pd.read_csv("data/fear_greed_index.csv")

fg["date"] = pd.to_datetime(fg["date"])
fg = fg.sort_values("date").reset_index(drop=True)

fg["rolling_30"] = fg["value"].rolling(30).mean()
fg["year"] = fg["date"].dt.year
fg["month"] = fg["date"].dt.month

SENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

COLORS = {
    "Extreme Fear": "red",
    "Fear": "orange",
    "Neutral": "yellow",
    "Greed": "green",
    "Extreme Greed": "blue"
}

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# =============================================================================
# 1. DAILY FEAR & GREED INDEX
# =============================================================================

plt.figure(figsize=(15,6))

for cls in SENT_ORDER:
    mask = fg["classification"] == cls
    plt.scatter(
        fg.loc[mask,"date"],
        fg.loc[mask,"value"],
        s=6,
        alpha=0.6,
        color=COLORS[cls],
        label=cls
    )

plt.plot(
    fg["date"],
    fg["rolling_30"],
    color="black",
    linewidth=2,
    label="30-Day Moving Average"
)

plt.title("Bitcoin Fear & Greed Index Over Time")
plt.ylabel("Index Value")
plt.xlabel("Date")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    output_path + "/fg_01_timeseries.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =============================================================================
# 2. SENTIMENT DISTRIBUTION
# =============================================================================

counts = fg["classification"].value_counts().reindex(SENT_ORDER)

plt.figure(figsize=(8,5))

bars = plt.bar(
    counts.index,
    counts.values,
    color=[COLORS[x] for x in counts.index]
)

plt.title("Sentiment Distribution")
plt.ylabel("Number of Days")
plt.xticks(rotation=20)

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height(),
        f"{int(bar.get_height())}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    output_path + "/fg_02_distribution.png",
    dpi=300
)
plt.close()

# =============================================================================
# 3. YEARLY AVERAGE INDEX
# =============================================================================

yearly_avg = fg.groupby("year")["value"].mean()

plt.figure(figsize=(10,5))

bars = plt.bar(
    yearly_avg.index.astype(str),
    yearly_avg.values,
    color="steelblue"
)

plt.axhline(50, linestyle="--", color="red")

plt.title("Yearly Average Fear & Greed Index")
plt.ylabel("Average Index")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.5,
        f"{bar.get_height():.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    output_path + "/fg_03_yearly_average.png",
    dpi=300
)
plt.close()

# =============================================================================
# 4. MONTHLY SEASONALITY
# =============================================================================

monthly_avg = fg.groupby("month")["value"].mean()

plt.figure(figsize=(10,5))

plt.bar(
    months,
    monthly_avg.values,
    color="orange"
)

plt.axhline(50,color="red",linestyle="--")

plt.title("Monthly Average Fear & Greed Index")
plt.ylabel("Average Value")

plt.tight_layout()
plt.savefig(
    output_path + "/fg_04_monthly_seasonality.png",
    dpi=300
)
plt.close()

# =============================================================================
# 5. YEAR-MONTH HEATMAP
# =============================================================================

pivot = fg.groupby(
    ["year","month"]
)["value"].mean().unstack()

plt.figure(figsize=(12,6))

plt.imshow(
    pivot,
    aspect="auto",
    cmap="RdYlGn",
    vmin=0,
    vmax=100
)

plt.colorbar(label="Index Value")

plt.xticks(
    range(12),
    months
)

plt.yticks(
    range(len(pivot.index)),
    pivot.index
)

plt.title("Fear & Greed Heatmap")

plt.tight_layout()
plt.savefig(
    output_path + "/fg_05_heatmap.png",
    dpi=300
)
plt.close()

# =============================================================================
# 6. SENTIMENT TRANSITION MATRIX
# =============================================================================

fg["next_sentiment"] = fg["classification"].shift(-1)

transition = pd.crosstab(
    fg["classification"],
    fg["next_sentiment"],
    normalize="index"
) * 100

transition = transition.reindex(
    index=SENT_ORDER,
    columns=SENT_ORDER
)

plt.figure(figsize=(8,6))

plt.imshow(
    transition,
    cmap="Blues"
)

plt.colorbar(label="%")

plt.xticks(
    range(len(SENT_ORDER)),
    SENT_ORDER,
    rotation=45
)

plt.yticks(
    range(len(SENT_ORDER)),
    SENT_ORDER
)

for i in range(len(SENT_ORDER)):
    for j in range(len(SENT_ORDER)):
        val = transition.iloc[i,j]
        if not np.isnan(val):
            plt.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center"
            )

plt.title("Sentiment Transition Matrix (%)")

plt.tight_layout()
plt.savefig(
    output_path + "/fg_06_transition_matrix.png",
    dpi=300
)
plt.close()

# =============================================================================
# 7. STREAK ANALYSIS
# =============================================================================

fg["streak_id"] = (
    fg["classification"]
    != fg["classification"].shift()
).cumsum()

streaks = (
    fg.groupby(
        ["streak_id","classification"]
    )
    .size()
    .reset_index(name="length")
)

avg_streak = (
    streaks.groupby("classification")["length"]
    .mean()
    .reindex(SENT_ORDER)
)

plt.figure(figsize=(8,5))

plt.bar(
    avg_streak.index,
    avg_streak.values,
    color=[COLORS[x] for x in avg_streak.index]
)

plt.title("Average Sentiment Streak Length")
plt.ylabel("Days")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig(
    output_path + "/fg_07_streak_length.png",
    dpi=300
)
plt.close()

# =============================================================================
# 8. VIOLIN DISTRIBUTION
# =============================================================================

data = [
    fg.loc[
        fg["classification"] == sentiment,
        "value"
    ]
    for sentiment in SENT_ORDER
]

plt.figure(figsize=(10,5))

parts = plt.violinplot(
    data,
    showmedians=True
)

plt.xticks(
    range(1,6),
    SENT_ORDER,
    rotation=20
)

plt.ylabel("Index Value")
plt.title("Fear & Greed Distribution by Sentiment")

plt.tight_layout()
plt.savefig(
    output_path + "/fg_08_violin_distribution.png",
    dpi=300
)
plt.close()

# =============================================================================
# 9. YEARLY SENTIMENT COMPOSITION
# =============================================================================

year_sent = (
    fg.groupby(
        ["year","classification"]
    )
    .size()
    .unstack(fill_value=0)
)

year_sent = year_sent[SENT_ORDER]

year_pct = year_sent.div(
    year_sent.sum(axis=1),
    axis=0
) * 100

plt.figure(figsize=(12,6))

bottom = np.zeros(len(year_pct))

for sentiment in SENT_ORDER:

    plt.bar(
        year_pct.index.astype(str),
        year_pct[sentiment],
        bottom=bottom,
        color=COLORS[sentiment],
        label=sentiment
    )

    bottom += year_pct[sentiment]

plt.title("Yearly Sentiment Composition")
plt.ylabel("Percentage")
plt.legend()

plt.tight_layout()

plt.savefig(
    output_path + "/fg_09_yearly_composition.png",
    dpi=300
)

plt.close()

print("All Fear & Greed visualizations saved successfully.")