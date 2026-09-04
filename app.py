# ============================================================
# STOCK LEADER -> FOLLOWER ANALYSIS DASHBOARD
# ============================================================
#
# Purpose:
#   Given a currently booming "leader" stock, identify S&P 500
#   stocks that historically performed strongly after similar
#   movements in the leader.
#
# Main outputs:
#   - Top 5 historical follower stocks
#   - Probability of outperforming leader by >= 1 percentage point
#   - Probability of staying within +/- 1 percentage point
#   - Probability of underperforming leader by >= 1 percentage point
#   - Positive-return probability
#   - Mean / median returns
#   - Log returns
#   - Standard deviation
#   - Annualized volatility
#   - Downside risk
#   - Maximum drawdown
#   - Correlation
#   - Beta
#   - Historical peak return
#   - Typical peak day
#   - 30-trading-day forecast
#   - 95% prediction interval
#   - Interactive charts
#
# IMPORTANT:
#   This is a research/decision-support tool, NOT financial advice.
#   Forecasts are statistical estimates and are not guaranteed.
#
# ============================================================

import io
import math
import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Leader → Follower Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    .small-text {
        font-size: 0.8rem;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

TRADING_DAYS_MONTH = 21
TRADING_DAYS_YEAR = 252

DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2025, 12, 31)

MIN_HISTORY_REQUIRED = 100

# Number of candidate S&P 500 companies
DEFAULT_TOP_CANDIDATES = 5

# Event similarity tolerance
DEFAULT_SIMILARITY_TOLERANCE = 0.03


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_ticker(ticker: str) -> str:
    """
    Convert common ticker formats to Yahoo Finance format.
    """
    ticker = ticker.strip().upper()

    # Yahoo uses '-' instead of '.' for some tickers.
    ticker = ticker.replace(".", "-")

    return ticker


def safe_float(value, default=np.nan):
    """
    Convert a value to float safely.
    """
    try:
        return float(value)
    except Exception:
        return default


def annualized_return(monthly_return):
    """
    Approximate annualized return from a 21-trading-day return.
    """
    if pd.isna(monthly_return):
        return np.nan

    if monthly_return <= -1:
        return np.nan

    return (1 + monthly_return) ** (252 / TRADING_DAYS_MONTH) - 1


# ============================================================
# S&P 500 CONSTITUENTS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_constituents():
    """
    Download current S&P 500 constituents.

    We use requests with a browser-like User-Agent because
    direct pandas.read_html() requests to Wikipedia can sometimes
    receive HTTP 403 responses.
    """

    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies?oldformat=true"
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/149.0.0.0 Safari/537.36"
        )
    }

    last_error = None

    for url in urls:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )

            response.raise_for_status()

            tables = pd.read_html(io.StringIO(response.text))

            if len(tables) == 0:
                continue

            table = tables[0].copy()

            # Identify ticker column
            ticker_col = None

            for candidate in ["Symbol", "Ticker"]:
                if candidate in table.columns:
                    ticker_col = candidate
                    break

            if ticker_col is None:
                continue

            # Identify company column
            company_col = None

            for candidate in ["Security", "Company"]:
                if candidate in table.columns:
                    company_col = candidate
                    break

            if company_col is None:
                company_col = ticker_col

            table["Ticker"] = (
                table[ticker_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)
            )

            table["Company"] = (
                table[company_col]
                .astype(str)
                .str.strip()
            )

            table = table[
                ["Ticker", "Company"]
            ].drop_duplicates()

            table = table[
                table["Ticker"].str.len() > 0
            ]

            if len(table) >= 400:
                return table

        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Could not download the S&P 500 constituent list. "
        f"Last error: {last_error}"
    )


# ============================================================
# PRICE DOWNLOAD
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def download_price_data(
    tickers,
    start_date,
    end_date
):
    """
    Download adjusted daily close prices from Yahoo Finance.

    We add a buffer before the requested start date because
    rolling 21-day calculations need previous observations.
    """

    tickers = list(dict.fromkeys(tickers))

    start = pd.Timestamp(start_date) - pd.Timedelta(days=60)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=2)

    all_frames = []

    # Download in batches to reduce the chance of Yahoo throttling.
    batch_size = 75

    for i in range(0, len(tickers), batch_size):

        batch = tickers[i:i + batch_size]

        try:

            data = yf.download(
                tickers=batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
                multi_level_index=True
            )

            if data is None or data.empty:
                continue

            # Handle modern yfinance MultiIndex format.
            if isinstance(data.columns, pd.MultiIndex):

                # Usually:
                # level 0 = Price field
                # level 1 = Ticker
                if "Close" in data.columns.get_level_values(0):

                    close = data["Close"].copy()

                elif "Close" in data.columns.get_level_values(1):

                    close = data.xs(
                        "Close",
                        axis=1,
                        level=1
                    ).copy()

                else:
                    continue

            else:

                # Single ticker fallback
                if "Close" in data.columns:
                    close = data[["Close"]].copy()

                    if len(batch) == 1:
                        close.columns = [batch[0]]

                else:
                    continue

            close.index = pd.to_datetime(close.index)
            close = close.sort_index()

            # Ensure columns are ticker names
            close.columns = [
                str(c).upper().replace(".", "-")
                for c in close.columns
            ]

            all_frames.append(close)

        except Exception as e:

            st.warning(
                f"Yahoo download warning for batch "
                f"{i + 1}-{min(i + batch_size, len(tickers))}: {e}"
            )

        # Small pause between batches
        time.sleep(0.15)

    if not all_frames:
        raise RuntimeError(
            "No market data was downloaded from Yahoo Finance."
        )

    prices = pd.concat(
        all_frames,
        axis=1
    )

    # Remove duplicate columns
    prices = prices.loc[
        :,
        ~prices.columns.duplicated()
    ]

    prices = prices.sort_index()

    # Remove duplicate dates
    prices = prices[
        ~prices.index.duplicated(keep="last")
    ]

    # Only requested period + historical buffer
    prices = prices[
        (prices.index >= start)
        & (prices.index <= end)
    ]

    return prices


# ============================================================
# CURRENT LEADER DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_current_leader(ticker):
    """
    Download enough recent data to determine the current
    21-trading-day movement of the selected leader.
    """

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=90)

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):

        if "Close" in data.columns.get_level_values(0):
            series = data["Close"]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

        else:
            series = data.iloc[:, 0]

    else:

        if "Close" in data.columns:
            series = data["Close"]

        else:
            series = data.iloc[:, 0]

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    series.index = pd.to_datetime(series.index)

    return series.sort_index()


# ============================================================
# RETURN CALCULATIONS
# ============================================================

def calculate_returns(prices):
    """
    Calculate daily simple and log returns.
    """

    simple_returns = prices.pct_change()

    log_returns = np.log(
        prices / prices.shift(1)
    )

    return simple_returns, log_returns


def calculate_rolling_returns(prices, window=21):
    """
    Calculate rolling 21-trading-day returns.
    """

    return prices.pct_change(window)


# ============================================================
# IDENTIFY HISTORICAL LEADER EVENTS
# ============================================================

def identify_leader_events(
    leader_prices,
    analysis_start,
    analysis_end,
    current_return,
    similarity_tolerance=DEFAULT_SIMILARITY_TOLERANCE,
    min_events=8,
    max_events=40
):
    """
    Identify historical leader events.

    Primary criterion:
        historical 21D return is similar to the current
        leader's 21D return.

    If insufficient observations are available, we fall back
    to the strongest positive 21D returns.
    """

    series = leader_prices.dropna().copy()

    series.index = pd.to_datetime(series.index)

    historical = series.loc[
        (series.index >= pd.Timestamp(analysis_start))
        & (series.index <= pd.Timestamp(analysis_end))
    ]

    rolling_return = historical.pct_change(
        TRADING_DAYS_MONTH
    )

    # Need enough future observations for a 21-day event study.
    valid = rolling_return.dropna()

    if valid.empty:
        return pd.DataFrame()

    # Candidate events need positive momentum.
    positive = valid[valid > 0]

    if positive.empty:
        return pd.DataFrame()

    # Current 21-day return may not be available.
    if pd.isna(current_return):
        current_return = positive.quantile(0.90)

    # Similarity threshold.
    lower = current_return - similarity_tolerance
    upper = current_return + similarity_tolerance

    similar = positive[
        (positive >= lower)
        & (positive <= upper)
    ].copy()

    # Remove events too close to one another.
    selected = []

    # Sort by similarity to current event
    similar = (
        (similar - current_return)
        .abs()
        .sort_values()
    )

    for event_date in similar.index:

        event_date = pd.Timestamp(event_date)

        if all(
            abs(
                (event_date - existing).days
            ) >= 30
            for existing in selected
        ):
            selected.append(event_date)

        if len(selected) >= max_events:
            break

    # If insufficient similar events, use strongest historical events.
    if len(selected) < min_events:

        strongest = positive.sort_values(
            ascending=False
        )

        for event_date in strongest.index:

            event_date = pd.Timestamp(event_date)

            if all(
                abs(
                    (event_date - existing).days
                ) >= 30
                for existing in selected
            ):
                selected.append(event_date)

            if len(selected) >= max_events:
                break

    selected = sorted(selected)

    rows = []

    for event_date in selected:

        loc = historical.index.get_indexer(
            [event_date]
        )[0]

        if loc < TRADING_DAYS_MONTH:
            continue

        if loc + TRADING_DAYS_MONTH >= len(historical):
            continue

        event_return = rolling_return.loc[event_date]

        rows.append(
            {
                "event_date": event_date,
                "leader_21d_return": event_return
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EVENT STUDY
# ============================================================

def create_event_study(
    prices,
    leader,
    candidate,
    events,
    horizon=TRADING_DAYS_MONTH
):
    """
    For every historical leader event, measure the subsequent
    follower and leader returns over the next 21 trading days.
    """

    if leader not in prices.columns:
        return pd.DataFrame()

    if candidate not in prices.columns:
        return pd.DataFrame()

    leader_series = prices[leader].dropna()
    candidate_series = prices[candidate].dropna()

    common = pd.concat(
        [leader_series, candidate_series],
        axis=1,
        join="inner"
    )

    common.columns = [
        "leader",
        "candidate"
    ]

    common = common.dropna()

    if common.empty:
        return pd.DataFrame()

    rows = []

    for _, event in events.iterrows():

        event_date = pd.Timestamp(
            event["event_date"]
        )

        dates = common.index

        # Find first available date >= event date
        positions = np.where(
            dates >= event_date
        )[0]

        if len(positions) == 0:
            continue

        start_pos = positions[0]
        end_pos = start_pos + horizon

        if end_pos >= len(common):
            continue

        leader_start = common["leader"].iloc[start_pos]
        leader_end = common["leader"].iloc[end_pos]

        candidate_start = common["candidate"].iloc[start_pos]
        candidate_end = common["candidate"].iloc[end_pos]

        if (
            leader_start <= 0
            or candidate_start <= 0
        ):
            continue

        leader_return = (
            leader_end / leader_start
        ) - 1

        candidate_return = (
            candidate_end / candidate_start
        ) - 1

        relative_return = (
            candidate_return - leader_return
        )

        # Maximum favorable excursion
        future_candidate = common[
            "candidate"
        ].iloc[start_pos:end_pos + 1]

        future_leader = common[
            "leader"
        ].iloc[start_pos:end_pos + 1]

        candidate_path = (
            future_candidate / candidate_start
        ) - 1

        leader_path = (
            future_leader / leader_start
        ) - 1

        max_candidate_gain = candidate_path.max()
        max_candidate_gain_day = (
            candidate_path.idxmax()
        )

        max_leader_gain = leader_path.max()

        peak_position = (
            common.index.get_loc(
                max_candidate_gain_day
            )
            - start_pos
        )

        rows.append(
            {
                "event_date": event_date,
                "leader_return": leader_return,
                "candidate_return": candidate_return,
                "relative_return": relative_return,
                "max_gain": max_candidate_gain,
                "max_gain_day": peak_position,
                "leader_max_gain": max_leader_gain
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STATISTICAL METRICS
# ============================================================

def calculate_stock_statistics(
    price_series,
    leader_series=None
):
    """
    Calculate historical risk and return statistics.
    """

    price_series = price_series.dropna()

    if len(price_series) < MIN_HISTORY_REQUIRED:
        return {}

    simple_returns = price_series.pct_change().dropna()

    log_returns = np.log(
        price_series / price_series.shift(1)
    ).dropna()

    if simple_returns.empty:
        return {}

    mean_return = simple_returns.mean()
    median_return = simple_returns.median()

    mean_log_return = log_returns.mean()

    std_daily = simple_returns.std()

    annualized_volatility = (
        std_daily * np.sqrt(TRADING_DAYS_YEAR)
    )

    downside_returns = simple_returns[
        simple_returns < 0
    ]

    if len(downside_returns) > 1:

        downside_deviation = (
            downside_returns.std()
            * np.sqrt(TRADING_DAYS_YEAR)
        )

    else:
        downside_deviation = np.nan

    cumulative = (
        1 + simple_returns
    ).cumprod()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    # Approximate annualized Sharpe ratio
    annualized_mean = (
        mean_return * TRADING_DAYS_YEAR
    )

    if std_daily > 0:
        sharpe = (
            annualized_mean
            / annualized_volatility
        )
    else:
        sharpe = np.nan

    result = {
        "Mean daily return": mean_return,
        "Median daily return": median_return,
        "Mean log return": mean_log_return,
        "Daily std": std_daily,
        "Annualized volatility": annualized_volatility,
        "Downside volatility": downside_deviation,
        "Maximum drawdown": max_drawdown,
        "Sharpe ratio": sharpe
    }

    if leader_series is not None:

        common = pd.concat(
            [price_series, leader_series],
            axis=1,
            join="inner"
        ).dropna()

        if len(common) > 20:

            stock_returns = common.iloc[:, 0].pct_change()
            leader_returns = common.iloc[:, 1].pct_change()

            aligned = pd.concat(
                [stock_returns, leader_returns],
                axis=1
            ).dropna()

            if len(aligned) > 20:

                correlation = (
                    aligned.iloc[:, 0]
                    .corr(aligned.iloc[:, 1])
                )

                leader_variance = (
                    aligned.iloc[:, 1].var()
                )

                if leader_variance > 0:
                    beta = (
                        aligned.iloc[:, 0]
                        .cov(aligned.iloc[:, 1])
                        / leader_variance
                    )
                else:
                    beta = np.nan

                result["Correlation"] = correlation
                result["Beta"] = beta

    return result


# ============================================================
# FORECAST
# ============================================================

def calculate_event_based_forecast(
    event_study,
    historical_daily_returns,
    horizon=TRADING_DAYS_MONTH
):
    """
    Forecast 21-trading-day return using the historical
    event-conditioned returns.

    The prediction interval combines:
      - empirical historical event distribution
      - recent volatility
    """

    if event_study.empty:
        return {}

    returns = event_study[
        "candidate_return"
    ].dropna()

    if len(returns) < 3:
        return {}

    mean_event_return = returns.mean()
    median_event_return = returns.median()

    std_event_return = returns.std()

    # Recent daily volatility
    daily_std = historical_daily_returns.std()

    if pd.isna(daily_std):
        daily_std = 0

    # Approximate 21-day volatility
    horizon_volatility = (
        daily_std
        * np.sqrt(horizon)
    )

    # Combine event uncertainty and recent volatility.
    combined_std = np.sqrt(
        max(std_event_return, 0) ** 2
        + max(horizon_volatility, 0) ** 2
    )

    lower_95 = (
        mean_event_return
        - 1.96 * combined_std
    )

    upper_95 = (
        mean_event_return
        + 1.96 * combined_std
    )

    return {
        "expected_return": mean_event_return,
        "median_expected_return": median_event_return,
        "forecast_std": combined_std,
        "lower_95": lower_95,
        "upper_95": upper_95
    }


# ============================================================
# FOLLOWER RANKING
# ============================================================

def rank_followers(
    prices,
    leader,
    events,
    candidate_tickers,
    top_n=5
):
    """
    Rank candidate stocks according to event-based
    follower behaviour.
    """

    results = []

    leader_series = prices[leader].dropna()

    for ticker in candidate_tickers:

        if ticker == leader:
            continue

        if ticker not in prices.columns:
            continue

        candidate_prices = prices[ticker].dropna()

        if len(candidate_prices) < MIN_HISTORY_REQUIRED:
            continue

        event_study = create_event_study(
            prices=prices,
            leader=leader,
            candidate=ticker,
            events=events,
            horizon=TRADING_DAYS_MONTH
        )

        if len(event_study) < 3:
            continue

        relative = event_study[
            "relative_return"
        ].dropna()

        candidate_returns = event_study[
            "candidate_return"
        ].dropna()

        leader_returns = event_study[
            "leader_return"
        ].dropna()

        if len(relative) < 3:
            continue

        # Main probabilities
        probability_outperform_1 = (
            relative >= 0.01
        ).mean()

        probability_within_1 = (
            relative.abs() <= 0.01
        ).mean()

        probability_underperform_1 = (
            relative <= -0.01
        ).mean()

        probability_positive = (
            candidate_returns > 0
        ).mean()

        # Average relative return
        mean_relative = relative.mean()

        median_relative = relative.median()

        # Historical peak
        max_gain = event_study[
            "max_gain"
        ].median()

        peak_day = event_study[
            "max_gain_day"
        ].median()

        # Stock statistics
        stock_stats = calculate_stock_statistics(
            candidate_prices,
            leader_series=leader_series
        )

        daily_returns = (
            candidate_prices
            .pct_change()
            .dropna()
        )

        forecast = calculate_event_based_forecast(
            event_study,
            daily_returns,
            horizon=TRADING_DAYS_MONTH
        )

        # A composite score for ranking.
        #
        # We prioritize:
        #   1. Probability of outperforming leader
        #   2. Positive return probability
        #   3. Historical relative return
        #   4. Lower risk
        #
        # This is a ranking score, not a probability.
        volatility_penalty = (
            stock_stats.get(
                "Annualized volatility",
                np.nan
            )
        )

        if pd.isna(volatility_penalty):
            volatility_penalty = 1

        score = (
            0.45 * probability_outperform_1
            + 0.25 * probability_positive
            + 0.20 * min(
                max(mean_relative * 5, 0),
                1
            )
            + 0.10 * probability_within_1
        )

        results.append(
            {
                "Ticker": ticker,
                "Observations": len(event_study),

                "P(Follower ≥ Leader +1%)":
                    probability_outperform_1,

                "P(Follower within ±1%)":
                    probability_within_1,

                "P(Follower ≤ Leader -1%)":
                    probability_underperform_1,

                "P(Follower positive)":
                    probability_positive,

                "Mean 21D return":
                    candidate_returns.mean(),

                "Median 21D return":
                    candidate_returns.median(),

                "Mean relative return":
                    mean_relative,

                "Median relative return":
                    median_relative,

                "Historical median peak gain":
                    max_gain,

                "Typical peak day":
                    peak_day,

                "Expected 21D return":
                    forecast.get(
                        "expected_return",
                        np.nan
                    ),

                "Forecast 95% lower":
                    forecast.get(
                        "lower_95",
                        np.nan
                    ),

                "Forecast 95% upper":
                    forecast.get(
                        "upper_95",
                        np.nan
                    ),

                "Mean log return":
                    stock_stats.get(
                        "Mean log return",
                        np.nan
                    ),

                "Daily std":
                    stock_stats.get(
                        "Daily std",
                        np.nan
                    ),

                "Annualized volatility":
                    stock_stats.get(
                        "Annualized volatility",
                        np.nan
                    ),

                "Downside volatility":
                    stock_stats.get(
                        "Downside volatility",
                        np.nan
                    ),

                "Maximum drawdown":
                    stock_stats.get(
                        "Maximum drawdown",
                        np.nan
                    ),

                "Sharpe ratio":
                    stock_stats.get(
                        "Sharpe ratio",
                        np.nan
                    ),

                "Correlation":
                    stock_stats.get(
                        "Correlation",
                        np.nan
                    ),

                "Beta":
                    stock_stats.get(
                        "Beta",
                        np.nan
                    ),

                "Ranking score":
                    score,

                "Event study":
                    event_study
            }
        )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        [
            "Ranking score",
            "P(Follower ≥ Leader +1%)"
        ],
        ascending=False
    )

    return result_df.head(top_n).reset_index(drop=True)


# ============================================================
# HISTORICAL EVENT PATH
# ============================================================

def calculate_average_event_path(
    prices,
    leader,
    candidate,
    events,
    horizon=TRADING_DAYS_MONTH
):
    """
    Calculate the average cumulative path of a candidate
    after historical leader events.
    """

    if leader not in prices.columns:
        return pd.DataFrame()

    if candidate not in prices.columns:
        return pd.DataFrame()

    common = pd.concat(
        [
            prices[leader],
            prices[candidate]
        ],
        axis=1,
        join="inner"
    ).dropna()

    common.columns = [
        "leader",
        "candidate"
    ]

    paths = []

    for _, event in events.iterrows():

        event_date = pd.Timestamp(
            event["event_date"]
        )

        positions = np.where(
            common.index >= event_date
        )[0]

        if len(positions) == 0:
            continue

        start = positions[0]

        if start + horizon >= len(common):
            continue

        leader_path = (
            common["leader"]
            .iloc[start:start + horizon + 1]
            / common["leader"].iloc[start]
            - 1
        )

        candidate_path = (
            common["candidate"]
            .iloc[start:start + horizon + 1]
            / common["candidate"].iloc[start]
            - 1
        )

        path = pd.DataFrame(
            {
                "day": range(horizon + 1),
                "leader": leader_path.values,
                "candidate": candidate_path.values
            }
        )

        paths.append(path)

    if not paths:
        return pd.DataFrame()

    all_paths = pd.concat(
        paths,
        ignore_index=True
    )

    result = (
        all_paths
        .groupby("day")
        .agg(
            leader_mean=("leader", "mean"),
            candidate_mean=("candidate", "mean"),
            candidate_median=("candidate", "median"),
            leader_median=("leader", "median")
        )
        .reset_index()
    )

    return result


# ============================================================
# CURRENT LEADER SIGNAL
# ============================================================

def calculate_current_leader_signal(
    leader_series
):
    """
    Calculate current leader momentum.
    """

    if len(leader_series) < TRADING_DAYS_MONTH + 1:
        return {}

    current_price = leader_series.iloc[-1]

    previous_price = leader_series.iloc[
        -TRADING_DAYS_MONTH - 1
    ]

    current_return = (
        current_price / previous_price
    ) - 1

    five_day_return = np.nan

    if len(leader_series) >= 6:

        five_day_return = (
            leader_series.iloc[-1]
            / leader_series.iloc[-6]
        ) - 1

    ten_day_return = np.nan

    if len(leader_series) >= 11:

        ten_day_return = (
            leader_series.iloc[-1]
            / leader_series.iloc[-11]
        ) - 1

    return {
        "current_price": current_price,
        "current_21d_return": current_return,
        "current_5d_return": five_day_return,
        "current_10d_return": ten_day_return,
        "current_date": leader_series.index[-1]
    }


# ============================================================
# FORECAST CHART
# ============================================================

def create_forecast_chart(
    result_df,
    current_prices
):
    """
    Create forecast price paths for the five followers.
    """

    fig = go.Figure()

    for _, row in result_df.iterrows():

        ticker = row["Ticker"]

        if ticker not in current_prices:
            continue

        current_price = current_prices[ticker]

        expected = row[
            "Expected 21D return"
        ]

        lower = row[
            "Forecast 95% lower"
        ]

        upper = row[
            "Forecast 95% upper"
        ]

        if pd.isna(expected):
            continue

        days = np.arange(
            0,
            TRADING_DAYS_MONTH + 1
        )

        expected_path = (
            current_price
            * (
                1
                + expected
                * days
                / TRADING_DAYS_MONTH
            )
        )

        lower_path = (
            current_price
            * (
                1
                + lower
                * days
                / TRADING_DAYS_MONTH
            )
        )

        upper_path = (
            current_price
            * (
                1
                + upper
                * days
                / TRADING_DAYS_MONTH
            )
        )

        fig.add_trace(
            go.Scatter(
                x=days,
                y=expected_path,
                mode="lines",
                name=f"{ticker} forecast",
                line=dict(
                    dash="dash"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=days,
                y=lower_path,
                mode="lines",
                name=f"{ticker} 95% lower",
                line=dict(
                    dash="dot"
                ),
                opacity=0.35,
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=days,
                y=upper_path,
                mode="lines",
                name=f"{ticker} 95% upper",
                line=dict(
                    dash="dot"
                ),
                opacity=0.35,
                showlegend=False
            )
        )

    fig.update_layout(
        title="30-Trading-Day Statistical Forecast",
        xaxis_title="Trading days",
        yaxis_title="Estimated price",
        hovermode="x unified",
        height=550,
        legend_title="Stock"
    )

    return fig


# ============================================================
# HISTORICAL EVENT CHART
# ============================================================

def create_event_chart(
    prices,
    leader,
    selected_followers,
    events
):
    """
    Plot normalized average historical event paths.
    """

    fig = go.Figure()

    leader_path = calculate_average_event_path(
        prices,
        leader,
        leader,
        events
    )

    # The function for leader-vs-itself produces 0 returns.
    # Instead calculate leader separately.
    leader_common = prices[leader].dropna()

    leader_paths = []

    for _, event in events.iterrows():

        event_date = pd.Timestamp(
            event["event_date"]
        )

        positions = np.where(
            leader_common.index >= event_date
        )[0]

        if len(positions) == 0:
            continue

        start = positions[0]

        if start + TRADING_DAYS_MONTH >= len(
            leader_common
        ):
            continue

        path = (
            leader_common.iloc[
                start:start + TRADING_DAYS_MONTH + 1
            ]
            / leader_common.iloc[start]
            - 1
        )

        leader_paths.append(path.values)

    if leader_paths:

        min_length = min(
            len(x) for x in leader_paths
        )

        leader_matrix = np.array(
            [
                x[:min_length]
                for x in leader_paths
            ]
        )

        leader_mean = leader_matrix.mean(
            axis=0
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(min_length),
                y=leader_mean * 100,
                mode="lines",
                name=leader
            )
        )

    for ticker in selected_followers:

        path = calculate_average_event_path(
            prices,
            leader,
            ticker,
            events
        )

        if path.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=path["day"],
                y=path["candidate_mean"] * 100,
                mode="lines",
                name=ticker
            )
        )

    fig.add_vline(
        x=0,
        line_dash="dot"
    )

    fig.update_layout(
        title="Average Historical Performance After Similar Leader Events",
        xaxis_title="Trading days after event",
        yaxis_title="Average cumulative return (%)",
        hovermode="x unified",
        height=550
    )

    return fig


# ============================================================
# CURRENT PRICE CHART
# ============================================================

def create_recent_movement_chart(
    prices,
    leader,
    followers
):
    """
    Last 30 trading days normalized to 100.
    """

    tickers = [
        leader
    ] + list(followers)

    available = [
        t for t in tickers
        if t in prices.columns
    ]

    recent = prices[
        available
    ].dropna(
        how="all"
    ).tail(30)

    normalized = (
        recent / recent.iloc[0]
    ) * 100

    fig = go.Figure()

    for ticker in normalized.columns:

        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker],
                mode="lines",
                name=ticker
            )
        )

    fig.update_layout(
        title="Recent 30-Trading-Day Movement",
        xaxis_title="Date",
        yaxis_title="Indexed price (start = 100)",
        hovermode="x unified",
        height=500
    )

    return fig


# ============================================================
# FORMATTING
# ============================================================

def format_percent(value):
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.2f}%"


def format_number(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.3f}"


def format_currency(value):
    if pd.isna(value):
        return "N/A"

    return f"${value:,.2f}"


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">'
        '📈 Leader → Follower Stock Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Identify S&P 500 stocks that historically followed '
        'a booming leader stock, estimate the probability '
        'of outperformance, and calculate risk-adjusted '
        '30-trading-day forecasts.'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.header("Analysis Settings")

    # Load S&P 500
    try:

        sp500 = get_sp500_constituents()

    except Exception as e:

        st.error(
            f"Unable to load S&P 500 constituents: {e}"
        )

        st.stop()

    ticker_options = sorted(
        sp500["Ticker"].tolist()
    )

    # Leader input
    default_index = (
        ticker_options.index("NVDA")
        if "NVDA" in ticker_options
        else 0
    )

    leader = st.sidebar.selectbox(
        "Current booming / leader stock",
        ticker_options,
        index=default_index,
        help=(
            "Select the stock whose current strong movement "
            "you want to investigate."
        )
    )

    leader_name_row = sp500[
        sp500["Ticker"] == leader
    ]

    if not leader_name_row.empty:
        leader_company = leader_name_row[
            "Company"
        ].iloc[0]
    else:
        leader_company = leader

    st.sidebar.caption(
        f"Selected: {leader_company}"
    )

    # Date selection
    st.sidebar.subheader(
        "Historical reference period"
    )

    start_date = st.sidebar.date_input(
        "Start date",
        value=DEFAULT_START,
        min_value=date(2010, 1, 1),
        max_value=date.today()
    )

    end_date = st.sidebar.date_input(
        "End date",
        value=DEFAULT_END,
        min_value=date(2010, 1, 2),
        max_value=date.today()
    )

    # Similarity tolerance
    st.sidebar.subheader(
        "Historical event settings"
    )

    similarity_tolerance_pct = st.sidebar.slider(
        "Leader-event similarity tolerance",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help=(
            "Example: if current 21-day leader return is "
            "+20%, a 3% tolerance searches historical events "
            "between +17% and +23%."
        )
    )

    min_events = st.sidebar.slider(
        "Minimum historical events",
        min_value=3,
        max_value=20,
        value=8
    )

    max_events = st.sidebar.slider(
        "Maximum historical events",
        min_value=10,
        max_value=60,
        value=30
    )

    st.sidebar.markdown("---")

    analyze_button = st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True
    )

    st.sidebar.markdown(
        """
        **Methodology**

        The model:

        1. Measures the leader's current 21-trading-day return.
        2. Searches the historical period for similar leader events.
        3. Examines the next 21 trading days.
        4. Measures follower performance.
        5. Ranks S&P 500 followers.
        6. Calculates statistical risk and return metrics.
        """
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if start_date >= end_date:

        st.warning(
            "Start date must be before end date."
        )

        st.stop()

    # ========================================================
    # INITIAL INFORMATION
    # ========================================================

    st.info(
        "Select a leader stock and historical period, "
        "then click **Run Analysis**."
    )

    if not analyze_button:
        st.stop()

    # ========================================================
    # DOWNLOAD CURRENT LEADER
    # ========================================================

    with st.spinner(
        f"Downloading recent {leader} data..."
    ):

        current_leader = download_current_leader(
            leader
        )

    if current_leader.empty:

        st.error(
            f"Could not download current data for {leader}."
        )

        st.stop()

    current_signal = calculate_current_leader_signal(
        current_leader
    )

    current_return = current_signal.get(
        "current_21d_return",
        np.nan
    )

    if pd.isna(current_return):

        st.error(
            "Not enough current data to calculate "
            "the leader's 21-trading-day return."
        )

        st.stop()

    # ========================================================
    # DISPLAY CURRENT LEADER
    # ========================================================

    st.header(
        f"🚨 Current Leader Signal — {leader}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current price",
            format_currency(
                current_signal["current_price"]
            )
        )

    with col2:
        st.metric(
            "5D return",
            format_percent(
                current_signal["current_5d_return"]
            )
        )

    with col3:
        st.metric(
            "10D return",
            format_percent(
                current_signal["current_10d_return"]
            )
        )

    with col4:
        st.metric(
            "21D return",
            format_percent(
                current_signal["current_21d_return"]
            )
        )

    st.caption(
        f"Latest available market date: "
        f"{current_signal['current_date'].strftime('%Y-%m-%d')}"
    )

    # ========================================================
    # DOWNLOAD HISTORICAL DATA
    # ========================================================

    # We need all current S&P constituents plus leader.
    tickers = list(
        dict.fromkeys(
            ticker_options + [leader]
        )
    )

    st.write(
        f"Downloading historical data for "
        f"approximately {len(tickers)} S&P 500 securities..."
    )

    progress = st.progress(0)

    with st.spinner(
        "Downloading historical S&P 500 data from Yahoo Finance..."
    ):

        try:

            prices = download_price_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )

            progress.progress(100)

        except Exception as e:

            st.error(
                f"Historical data download failed: {e}"
            )

            st.stop()

    # ========================================================
    # CHECK LEADER
    # ========================================================

    if leader not in prices.columns:

        st.error(
            f"Historical data for {leader} was not downloaded."
        )

        st.stop()

    leader_prices = prices[leader].dropna()

    if len(leader_prices) < MIN_HISTORY_REQUIRED:

        st.error(
            "Not enough historical data for the selected period."
        )

        st.stop()

    # ========================================================
    # IDENTIFY HISTORICAL EVENTS
    # ========================================================

    st.header(
        "1️⃣ Historical Leader Events"
    )

    tolerance = (
        similarity_tolerance_pct / 100
    )

    events = identify_leader_events(
        leader_prices=leader_prices,
        analysis_start=start_date,
        analysis_end=end_date,
        current_return=current_return,
        similarity_tolerance=tolerance,
        min_events=min_events,
        max_events=max_events
    )

    if events.empty:

        st.error(
            "No comparable historical leader events "
            "were found."
        )

        st.stop()

    # ========================================================
    # EVENT SUMMARY
    # ========================================================

    event_col1, event_col2, event_col3 = st.columns(3)

    with event_col1:
        st.metric(
            "Historical events",
            len(events)
        )

    with event_col2:
        st.metric(
            "Current leader 21D return",
            format_percent(current_return)
        )

    with event_col3:
        st.metric(
            "Average event 21D return",
            format_percent(
                events["leader_21d_return"].mean()
            )
        )

    event_display = events.copy()

    event_display["event_date"] = (
        event_display["event_date"]
        .dt.strftime("%Y-%m-%d")
    )

    event_display["leader_21d_return"] = (
        event_display["leader_21d_return"]
        .map(format_percent)
    )

    st.dataframe(
        event_display,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # RANK FOLLOWERS
    # ========================================================

    st.header(
        "2️⃣ Top 5 Historical Follower Stocks"
    )

    # Candidate tickers with sufficient data
    candidate_tickers = [
        ticker
        for ticker in ticker_options
        if ticker in prices.columns
    ]

    with st.spinner(
        "Analysing historical follower behaviour..."
    ):

        result_df = rank_followers(
            prices=prices,
            leader=leader,
            events=events,
            candidate_tickers=candidate_tickers,
            top_n=DEFAULT_TOP_CANDIDATES
        )

    if result_df.empty:

        st.error(
            "No follower stocks could be identified. "
            "Try increasing the similarity tolerance "
            "or extending the historical period."
        )

        st.stop()

    # ========================================================
    # COMPANY NAMES
    # ========================================================

    company_map = dict(
        zip(
            sp500["Ticker"],
            sp500["Company"]
        )
    )

    result_df["Company"] = (
        result_df["Ticker"]
        .map(company_map)
        .fillna("")
    )

    # ========================================================
    # TOP FIVE SUMMARY
    # ========================================================

    summary_columns = [
        "Ticker",
        "Company",
        "Observations",
        "P(Follower ≥ Leader +1%)",
        "P(Follower within ±1%)",
        "P(Follower ≤ Leader -1%)",
        "P(Follower positive)",
        "Mean 21D return",
        "Expected 21D return",
        "Annualized volatility",
        "Maximum drawdown",
        "Correlation",
        "Beta",
        "Historical median peak gain",
        "Typical peak day"
    ]

    summary = result_df[
        summary_columns
    ].copy()

    percent_columns = [
        "P(Follower ≥ Leader +1%)",
        "P(Follower within ±1%)",
        "P(Follower ≤ Leader -1%)",
        "P(Follower positive)",
        "Mean 21D return",
        "Expected 21D return",
        "Annualized volatility",
        "Maximum drawdown",
        "Historical median peak gain"
    ]

    for col in percent_columns:
        summary[col] = summary[col].map(
            format_percent
        )

    summary["Typical peak day"] = (
        summary["Typical peak day"]
        .round(1)
        .astype(str)
        + " days"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The +1% probability means the follower's "
        "21-trading-day return exceeded the leader's "
        "21-trading-day return by at least 1 percentage point."
    )

    # ========================================================
    # DETAILED CARDS
    # ========================================================

    st.header(
        "3️⃣ Detailed Risk / Return Analysis"
    )

    for rank, (_, row) in enumerate(
        result_df.iterrows(),
        start=1
    ):

        ticker = row["Ticker"]

        with st.expander(
            f"#{rank} — {ticker} — {row['Company']}",
            expanded=(rank == 1)
        ):

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "P(Follower ≥ Leader +1%)",
                    format_percent(
                        row[
                            "P(Follower ≥ Leader +1%)"
                        ]
                    )
                )

            with c2:
                st.metric(
                    "P(Follower positive)",
                    format_percent(
                        row[
                            "P(Follower positive)"
                        ]
                    )
                )

            with c3:
                st.metric(
                    "Expected 21D return",
                    format_percent(
                        row[
                            "Expected 21D return"
                        ]
                    )
                )

            with c4:
                st.metric(
                    "Annualized volatility",
                    format_percent(
                        row[
                            "Annualized volatility"
                        ]
                    )
                )

            with c5:
                st.metric(
                    "Maximum drawdown",
                    format_percent(
                        row[
                            "Maximum drawdown"
                        ]
                    )
                )

            detail_data = {
                "Metric": [
                    "Historical observations",
                    "Mean 21D return",
                    "Median 21D return",
                    "Mean relative return",
                    "Median relative return",
                    "Mean log return",
                    "Daily standard deviation",
                    "Annualized volatility",
                    "Downside volatility",
                    "Maximum drawdown",
                    "Correlation with leader",
                    "Beta to leader",
                    "Sharpe ratio",
                    "Historical median peak gain",
                    "Typical peak day",
                    "Expected 21D return",
                    "95% forecast lower",
                    "95% forecast upper"
                ],
                "Value": [
                    row["Observations"],
                    format_percent(
                        row["Mean 21D return"]
                    ),
                    format_percent(
                        row["Median 21D return"]
                    ),
                    format_percent(
                        row["Mean relative return"]
                    ),
                    format_percent(
                        row["Median relative return"]
                    ),
                    format_number(
                        row["Mean log return"]
                    ),
                    format_percent(
                        row["Daily std"]
                    ),
                    format_percent(
                        row["Annualized volatility"]
                    ),
                    format_percent(
                        row["Downside volatility"]
                    ),
                    format_percent(
                        row["Maximum drawdown"]
                    ),
                    format_number(
                        row["Correlation"]
                    ),
                    format_number(
                        row["Beta"]
                    ),
                    format_number(
                        row["Sharpe ratio"]
                    ),
                    format_percent(
                        row[
                            "Historical median peak gain"
                        ]
                    ),
                    f"{row['Typical peak day']:.1f} trading days",
                    format_percent(
                        row["Expected 21D return"]
                    ),
                    format_percent(
                        row["Forecast 95% lower"]
                    ),
                    format_percent(
                        row["Forecast 95% upper"]
                    )
                ]
            }

            detail_df = pd.DataFrame(
                detail_data
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # RECENT MOVEMENT CHART
    # ========================================================

    st.header(
        "4️⃣ Recent Leader / Follower Movement"
    )

    followers = result_df[
        "Ticker"
    ].tolist()

    recent_chart = create_recent_movement_chart(
        prices=prices,
        leader=leader,
        followers=followers
    )

    st.plotly_chart(
        recent_chart,
        use_container_width=True
    )

    # ========================================================
    # HISTORICAL EVENT CHART
    # ========================================================

    st.header(
        "5️⃣ Historical Similar-Event Performance"
    )

    event_chart = create_event_chart(
        prices=prices,
        leader=leader,
        selected_followers=followers,
        events=events
    )

    st.plotly_chart(
        event_chart,
        use_container_width=True
    )

    st.caption(
        "The chart shows the average cumulative performance "
        "after historical leader events that were similar "
        "to the current leader's 21-day movement."
    )

    # ========================================================
    # FORECAST
    # ========================================================

    st.header(
        "6️⃣ 30-Trading-Day Forecast"
    )

    st.warning(
        "Forecasts are statistical estimates based on "
        "historical conditional behaviour. They are not "
        "guaranteed future prices."
    )

    current_prices = {}

    for ticker in followers:

        if ticker in prices.columns:

            series = prices[
                ticker
            ].dropna()

            if not series.empty:

                current_prices[ticker] = (
                    series.iloc[-1]
                )

    forecast_chart = create_forecast_chart(
        result_df=result_df,
        current_prices=current_prices
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True
    )

    # ========================================================
    # FORECAST TABLE
    # ========================================================

    forecast_table = result_df[
        [
            "Ticker",
            "Expected 21D return",
            "Forecast 95% lower",
            "Forecast 95% upper",
            "Historical median peak gain",
            "Typical peak day"
        ]
    ].copy()

    forecast_table[
        "Expected 21D return"
    ] = forecast_table[
        "Expected 21D return"
    ].map(format_percent)

    forecast_table[
        "Forecast 95% lower"
    ] = forecast_table[
        "Forecast 95% lower"
    ].map(format_percent)

    forecast_table[
        "Forecast 95% upper"
    ] = forecast_table[
        "Forecast 95% upper"
    ].map(format_percent)

    forecast_table[
        "Historical median peak gain"
    ] = forecast_table[
        "Historical median peak gain"
    ].map(format_percent)

    forecast_table[
        "Typical peak day"
    ] = (
        forecast_table[
            "Typical peak day"
        ].round(1).astype(str)
        + " trading days"
    )

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # RISK RANKING
    # ========================================================

    st.header(
        "7️⃣ Risk Ranking"
    )

    risk_df = result_df[
        [
            "Ticker",
            "Annualized volatility",
            "Downside volatility",
            "Maximum drawdown",
            "Sharpe ratio",
            "Correlation",
            "Beta"
        ]
    ].copy()

    risk_df = risk_df.sort_values(
        "Annualized volatility"
    )

    risk_display = risk_df.copy()

    for col in [
        "Annualized volatility",
        "Downside volatility",
        "Maximum drawdown"
    ]:

        risk_display[col] = risk_display[
            col
        ].map(format_percent)

    for col in [
        "Sharpe ratio",
        "Correlation",
        "Beta"
    ]:

        risk_display[col] = risk_display[
            col
        ].map(format_number)

    st.dataframe(
        risk_display,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.header(
        "8️⃣ Statistical Interpretation"
    )

    best = result_df.iloc[0]

    best_ticker = best["Ticker"]

    best_probability = best[
        "P(Follower ≥ Leader +1%)"
    ]

    best_forecast = best[
        "Expected 21D return"
    ]

    best_risk = best[
        "Annualized volatility"
    ]

    best_peak_day = best[
        "Typical peak day"
    ]

    st.markdown(
        f"""
        ### Strongest historical follower: **{best_ticker}**

        Based on the selected historical period and the
        identified leader events:

        - Probability of outperforming **{leader} by at least
          1 percentage point over the next month:**
          **{format_percent(best_probability)}**
        - Statistical expected 21-trading-day return:
          **{format_percent(best_forecast)}**
        - Annualized historical volatility:
          **{format_percent(best_risk)}**
        - Historical median peak gain:
          **{format_percent(best['Historical median peak gain'])}**
        - Typical historical peak:
          approximately **{best_peak_day:.0f} trading days**
        - Correlation with leader:
          **{format_number(best['Correlation'])}**
        - Beta to leader:
          **{format_number(best['Beta'])}**
        """
    )

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.header(
        "9️⃣ Export Results"
    )

    export_df = result_df.drop(
        columns=["Event study"],
        errors="ignore"
    ).copy()

    csv = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download analysis as CSV",
        data=csv,
        file_name=(
            f"{leader}_leader_follower_analysis.csv"
        ),
        mime="text/csv"
    )

    # ========================================================
    # METHODOLOGY / DISCLAIMER
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Methodology Notes"
    )

    st.markdown(
        f"""
        **Leader event**

        The application calculates the current leader's
        **21-trading-day return** and searches the selected
        historical period for positive 21-day movements that
        fall within the selected similarity tolerance.

        **Follower probability**

        For every historical leader event, the application
        calculates the follower's subsequent 21-trading-day
        return.

        The principal probability is:

        > P(Follower 21D return ≥ Leader 21D return + 1%)

        **Forecast**

        The 30-trading-day forecast is based on the empirical
        distribution of follower returns following comparable
        historical leader events, combined with recent
        historical volatility.

        **Risk**

        Risk measures include annualized volatility, downside
        volatility, maximum drawdown, correlation and beta.

        **Important limitation**

        The current S&P 500 constituent universe is used.
        Historical index membership changes over time, so a
        historical analysis using today's constituents can
        introduce survivorship bias.

        The model also does not account for transaction costs,
        bid/ask spreads, taxes, liquidity constraints or market
        impact.

        This dashboard is a quantitative research tool and
        should not be interpreted as a guarantee of future
        investment performance.
        """
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
