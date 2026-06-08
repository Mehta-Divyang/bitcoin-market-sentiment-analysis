# Bitcoin Market Sentiment and Trader Behavior Analysis

## Overview

This project explores the relationship between Bitcoin market sentiment and trader performance using historical trading data and the Bitcoin Fear & Greed Index.

The objective is to understand how market psychology influences trading behavior, profitability, risk-taking, and overall trading outcomes. By combining sentiment data with real-world trading activity, the analysis uncovers patterns that can help inform data-driven trading strategies.

---

## Dataset Description

### 1. Bitcoin Fear & Greed Index

The sentiment dataset contains daily Bitcoin market sentiment classifications based on market indicators such as volatility, momentum, and social sentiment.

**Features:**

* Date
* Fear & Greed Score (0–100)
* Sentiment Classification

  * Extreme Fear
  * Fear
  * Neutral
  * Greed
  * Extreme Greed

---

### 2. Historical Trader Data

The trading dataset contains historical trade execution records from a cryptocurrency derivatives trading platform.

**Features:**

* Account
* Coin
* Execution Price
* Trade Size
* Side (Buy/Sell)
* Direction
* Closed PnL
* Fee
* Timestamp
* Trade ID
* Order ID

---

## Project Objectives

* Analyze Bitcoin market sentiment trends over time.
* Explore trader profitability and trading behavior.
* Investigate the relationship between market sentiment and trader performance.
* Identify patterns in trading activity during Fear and Greed market conditions.
* Generate actionable insights using exploratory data analysis.

---

## Project Structure

```text
.
├── data/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
│
├── visualizations/
│   ├── fear_grid/
│   │   └── ...
│   ├── historical/
│   │   └── ...
│   └── sentiment_trader/
│   │   └── ...
│
├── fear_greed_visualizations.py
├── trader_visualizations.py
├── sentiment_trader_analysis.py
└── README.md
```

---

## Analysis Performed

### Market Sentiment Analysis

* Sentiment Distribution
* Fear & Greed Time Series Analysis
* Yearly Sentiment Trends
* Monthly Seasonality
* Sentiment Transition Matrix
* Sentiment Streak Analysis
* Heatmap Visualizations

### Trader Performance Analysis

* Cumulative Profit & Loss Analysis
* PnL Distribution
* Top Performing Accounts
* Trading Volume by Asset
* Win Rate Analysis
* Trading Activity by Hour
* Fee Impact Analysis
* Trade Size Analysis
* Coin-wise Profitability

### Sentiment vs Trader Performance

* Average PnL by Sentiment
* Win Rate by Sentiment
* Trade Volume by Sentiment
* Trading Activity by Sentiment
* Long vs Short Performance Analysis
* Risk Behavior under Different Market Conditions

---

## Key Insights

Some of the questions explored include:

* Do traders perform better during Fear or Greed markets?
* How does market sentiment affect win rates?
* Does trading activity increase during periods of extreme sentiment?
* Which assets generate the highest profitability?
* How does trade size influence performance?
* What behavioral patterns emerge during market uncertainty?

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bitcoin-market-sentiment-analysis.git
cd bitcoin-market-sentiment-analysis
```

Run the visualization scripts:

```bash
python fear_greed_visualizations.py
python trader_visualizations.py
```

Run file: sentiment_trader_analysis.ipynb

---

## Visualization Outputs

The project automatically generates and stores all visualizations inside the `visualizations/` directory.

Examples include:

* Fear & Greed Time Series
* Sentiment Distribution
* Monthly Heatmaps
* Cumulative PnL Analysis
* Top Trader Analysis
* Coin Volume Analysis
* Win Rate Analysis
* Sentiment-Based Performance Comparisons

---

## Future Improvements

* Trader clustering using machine learning.
* Sentiment-driven trading strategy backtesting.
* Predictive modeling for trader profitability.
* Interactive dashboards using Plotly or Streamlit.
* Real-time sentiment monitoring.

---
