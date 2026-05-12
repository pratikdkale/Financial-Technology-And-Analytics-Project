# QWIM Regime Allocation Dashboard

## Forward-Looking Regime Detection Using LSI and IRI for Dynamic Portfolio Allocation

This project builds an interactive investment analytics dashboard for the Bank of America QWIM project on market regimes, changepoints, bubbles, and crashes.

The core idea is simple: markets do not behave the same way all the time. A portfolio that works well in calm markets may become weak during stress. This dashboard uses market stress indicators to detect regimes and then connects those regimes to portfolio allocation decisions.

The framework uses two stress signals:

- **Liquidity Stress Index (LSI)** — captures funding-side and liquidity stress.
- **Implied Risk Index (IRI)** — captures forward-looking risk perception from market-implied signals.

These indicators are used to identify four market regimes:

- **Deep Calm**
- **Calm**
- **Elevated Stress**
- **Crisis**

The dashboard then compares regime-aware allocation strategies against traditional static benchmarks.

---

## Project Objective

The main objective is to test whether regime-aware portfolio allocation can improve performance compared with static allocation.

The dashboard answers four practical questions:

1. Can stress indicators identify changing market conditions?
2. Do assets behave differently across regimes?
3. Can regime-based allocation improve Sharpe ratio and drawdown control?
4. Does the strategy still work after transaction costs?

---

## Asset Universe

The project uses four liquid ETFs:

| Ticker | Asset Class | Role in Portfolio |
|---|---|---|
| SPY | U.S. Equities | Growth and risk-on exposure |
| TLT | Long-Term Treasuries | Duration hedge and crisis protection |
| GLD | Gold | Real-asset hedge and defensive exposure |
| HYG | High-Yield Credit | Credit carry and risk appetite exposure |

These assets were selected because they respond differently to liquidity stress, volatility, inflation fears, and crisis conditions.

---

## Dashboard Features

The dashboard includes the following sections:

### 1. Data Retrieval

Shows the data pipeline used in the project:

- ETF price data
- Weekly return calculation
- LSI and IRI stress indicators
- Sample period
- Burn-in window
- Out-of-sample backtest period

---

### 2. Exploratory Data Analysis

Includes:

- Growth of $1 chart
- Weekly return distributions
- Correlation heatmap
- Basic asset behavior comparison

This section helps show why a static allocation may not be enough.

---

### 3. Statistical Tests

Includes:

- ADF stationarity test
- Jarque-Bera normality test
- Skewness
- Excess kurtosis

The goal is to show that returns are stationary enough for modeling, but not normally distributed. This supports the need for regime-based analysis.

---

### 4. Stress Indicators

Visualizes:

- LSI over time
- IRI over time
- Rolling z-scores
- Stress thresholds
- Major historical stress periods

This section shows how market stress builds before or during major drawdowns.

---

### 5. Regime Detection

Uses a Hidden Markov Model to classify each week into one of four regimes.

Includes:

- Regime timeline
- Regime probability chart
- Regime frequency chart
- LSI vs. IRI regime scatter plot
- HMM vs. GMM comparison

The HMM is preferred because it captures persistence. Markets have memory; stress does not usually disappear in one week.

---

### 6. Conditional Asset Behavior

Shows how asset returns and volatility change across regimes.

This is one of the most important parts of the project because it shows that asset leadership rotates.

For example:

- SPY performs well in calm regimes.
- TLT and GLD become more useful in crisis regimes.
- HYG behaves like credit carry in calm periods but becomes risky during stress.

---

### 7. Allocation Engine

The project compares two regime-aware allocation methods:

#### Phase II: Hand-Picked Regime Weights

Weights are chosen using economic intuition.

The logic:

- More SPY during calm regimes
- More TLT and GLD during stress
- Lower equity exposure during crisis

#### Phase III: Optimized Regime Weights

Weights are selected using max-Sharpe optimization.

The optimizer uses:

- Regime-conditional expected returns
- Regime-conditional covariance
- Ledoit-Wolf covariance shrinkage
- Long-only constraint
- Fully invested portfolio
- 60% maximum weight cap per asset

---

## Strategies Compared

The dashboard compares six strategies:

| Strategy | Description |
|---|---|
| Optimized Regime | Max-Sharpe regime-conditional allocation |
| Hand Regime | Economic-intuition regime allocation |
| 60/40 Benchmark | Static 60% SPY and 40% TLT |
| Equal Weight | 25% allocation to each ETF |
| Risk Parity | Inverse-volatility allocation |
| Minimum Volatility | Minimum-variance benchmark |

All strategies are tested under the same backtesting assumptions.

---

## Backtesting Framework

The backtest uses:

- Weekly observations
- 156-week burn-in window
- Out-of-sample testing from May 2010 to February 2026
- Weekly rebalancing logic
- Transaction cost assumptions
- Drift threshold to reduce unnecessary trading

Transaction costs tested:

- 0 bps
- 10 bps
- 25 bps
- 50 bps

---

## Key Performance Metrics

The dashboard reports:

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Calmar ratio
- Annual turnover

Main result:

| Metric | Optimized Regime | 60/40 Benchmark |
|---|---:|---:|
| Total Return | 919.1% | 350.9% |
| Annualized Return | 15.7% | 9.9% |
| Annualized Volatility | 8.5% | 10.1% |
| Sharpe Ratio | 1.85 | 0.99 |
| Max Drawdown | -23.7% | -27.3% |
| Calmar Ratio | 0.66 | 0.36 |

The optimized regime strategy improves return, Sharpe ratio, and drawdown control versus the 60/40 benchmark.

---

## Transaction Cost Robustness

The dashboard includes a transaction cost sensitivity test.

Even when transaction costs rise, the optimized regime strategy remains ahead of the 60/40 benchmark.

| Cost Level | 60/40 | Hand Regime | Optimized Regime |
|---|---:|---:|---:|
| 0 bps | 352.6% | 692.7% | 1016.9% |
| 10 bps | 350.9% | 650.8% | 919.1% |
| 25 bps | 348.3% | 592.1% | 788.0% |
| 50 bps | 344.0% | 504.2% | 605.6% |

The advantage becomes smaller as costs increase, but it does not disappear.

---