import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Market Intelligence | S&P 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

TRADING_DAYS_PER_MONTH = 21
FORECAST_DAYS = 30
TRADING_DAYS_PER_YEAR = 252

DEFAULT_SIMILARITY = 5
DEFAULT_SIMULATIONS = 10000

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #111827;
    }

    .subtitle {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        color: #111827;
    }

    .research-box {
        padding: 1rem 1.2rem;
        border-radius: 0.7rem;
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }

    .method-box {
        padding: 0.9rem 1rem;
        border-radius: 0.6rem;
        background-color: #F8FAFC;
        border-left: 4px solid #64748B;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #E5E7EB;
        padding: 0.8rem;
        border-radius: 0.65rem;
        background-color: #FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_ticker(ticker):
    """
    Yahoo Finance convention:
    BRK.B -> BRK-B
    BF.B  -> BF-B
    """
    return str(ticker).strip().upper().replace(".", "-")


def signed_pct(value, decimals=1):
    if value is None or pd.isna(value):
        return "—"

    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def pct(value, decimals=1):
    if value is None or pd.isna(value):
        return "—"

    return f"{value * 100:.{decimals}f}%"


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# S&P 500 CONSTITUENTS
# ============================================================

@st.cache_data(ttl="1D", show_spinner=False)
def get_sp500_constituents():

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(
        SP500_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.StringIO(response.text)
    )

    if len(tables) == 0:
        raise ValueError(
            "Could not find S&P 500 constituent table."
        )

    df = tables[0].copy()

    required = [
        "Symbol",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}"
        )

    df["Ticker"] = (
        df["Symbol"]
        .astype(str)
        .map(normalize_ticker)
    )

    df = df[
        [
            "Ticker",
            "Security",
            "GICS Sector",
            "GICS Sub-Industry",
        ]
    ].copy()

    df = df.drop_duplicates(
        subset="Ticker"
    )

    df = df.sort_values(
        "Ticker"
    ).reset_index(drop=True)

    return df


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl="6H", show_spinner=False)
def download_price_data(
    tickers,
    years=8,
):

    tickers = [
        normalize_ticker(ticker)
        for ticker in tickers
        if ticker
    ]

    tickers = list(
        dict.fromkeys(tickers)
    )

    if not tickers:
        return pd.DataFrame()

    end_date = (
        datetime.now().date()
        + timedelta(days=1)
    )

    start_date = (
        end_date
        - timedelta(days=365 * years)
    )

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        level_0 = data.columns.get_level_values(0)

        if "Close" in level_0:
            prices = data["Close"].copy()

        else:
            raise ValueError(
                "Close price data was not returned."
            )

    else:

        if "Close" in data.columns:
            prices = data[
                ["Close"]
            ].copy()

        else:
            raise ValueError(
                "Close price data was not returned."
            )

        if len(tickers) == 1:
            prices.columns = [
                tickers[0]
            ]

    prices.columns = [
        normalize_ticker(column)
        for column in prices.columns
    ]

    prices = prices.sort_index()

    prices = prices[
        ~prices.index.duplicated(
            keep="last"
        )
    ]

    return prices


# ============================================================
# RETURNS
# ============================================================

def calculate_log_returns(prices):

    return np.log(
        prices / prices.shift(1)
    )


def calculate_rolling_return(
    price_series,
    window=TRADING_DAYS_PER_MONTH,
):

    return (
        price_series
        / price_series.shift(window)
        - 1
    )


# ============================================================
# CURRENT LEADER SIGNAL
# ============================================================

def calculate_current_leader_signal(
    prices,
    leader,
):

    if leader not in prices.columns:
        raise ValueError(
            f"{leader} is not available."
        )

    series = (
        prices[leader]
        .dropna()
    )

    if len(series) <= TRADING_DAYS_PER_MONTH:
        raise ValueError(
            "Not enough historical data."
        )

    current_price = float(
        series.iloc[-1]
    )

    prior_price = float(
        series.iloc[
            -TRADING_DAYS_PER_MONTH - 1
        ]
    )

    current_return = (
        current_price
        / prior_price
        - 1
    )

    return {
        "leader": leader,
        "price": current_price,
        "return": current_return,
        "date": series.index[-1],
    }


# ============================================================
# HISTORICAL MOMENTUM EVENTS
# ============================================================

def identify_leader_events(
    prices,
    leader,
    current_return,
    similarity_pct,
    cooldown_days=21,
):

    if current_return <= 0:
        return []

    series = (
        prices[leader]
        .dropna()
    )

    rolling_return = (
        series
        / series.shift(
            TRADING_DAYS_PER_MONTH
        )
        - 1
    ).dropna()

    tolerance = (
        similarity_pct / 100
    )

    lower_bound = (
        current_return
        - tolerance
    )

    upper_bound = (
        current_return
        + tolerance
    )

    # Only positive momentum events.
    lower_bound = max(
        lower_bound,
        0,
    )

    candidates = rolling_return[
        (
            rolling_return
            >= lower_bound
        )
        &
        (
            rolling_return
            <= upper_bound
        )
    ].index.tolist()

    if not candidates:
        return []

    selected = []

    last_selected = None

    for event_date in candidates:

        if last_selected is None:

            selected.append(
                event_date
            )

            last_selected = event_date

            continue

        days_between = (
            event_date
            - last_selected
        ).days

        if days_between >= cooldown_days:

            selected.append(
                event_date
            )

            last_selected = event_date

    # Remove very recent events
    # that overlap today's signal.

    current_date = series.index[-1]

    selected = [
        event_date
        for event_date in selected
        if (
            current_date - event_date
        ).days
        > TRADING_DAYS_PER_MONTH
    ]

    return selected


# ============================================================
# EVENT STUDY
# ============================================================

def create_event_study(
    prices,
    leader,
    event_dates,
    followers,
    forward_days=TRADING_DAYS_PER_MONTH,
):

    records = []

    if not event_dates:
        return pd.DataFrame()

    for event_date in event_dates:

        if event_date not in prices.index:
            continue

        try:
            event_position = (
                prices.index.get_loc(
                    event_date
                )
            )

        except KeyError:
            continue

        future_position = (
            event_position
            + forward_days
        )

        if (
            future_position
            >= len(prices.index)
        ):
            continue

        future_date = (
            prices.index[
                future_position
            ]
        )

        leader_start = safe_float(
            prices.loc[
                event_date,
                leader,
            ]
        )

        leader_future = safe_float(
            prices.loc[
                future_date,
                leader,
            ]
        )

        if (
            pd.isna(leader_start)
            or
            pd.isna(leader_future)
        ):
            continue

        leader_return = (
            leader_future
            / leader_start
            - 1
        )

        for ticker in followers:

            if ticker not in prices.columns:
                continue

            follower_start = safe_float(
                prices.loc[
                    event_date,
                    ticker,
                ]
            )

            follower_future = safe_float(
                prices.loc[
                    future_date,
                    ticker,
                ]
            )

            if (
                pd.isna(follower_start)
                or
                pd.isna(follower_future)
            ):
                continue

            follower_return = (
                follower_future
                / follower_start
                - 1
            )

            excess_return = (
                follower_return
                - leader_return
            )

            records.append(
                {
                    "event_date": event_date,
                    "future_date": future_date,
                    "leader": leader,
                    "leader_return": leader_return,
                    "ticker": ticker,
                    "follower_return": follower_return,
                    "excess_return": excess_return,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# CURRENT STOCK METRICS
# ============================================================

def calculate_current_stock_metrics(
    prices,
    leader,
):

    rolling = (
        prices
        / prices.shift(
            TRADING_DAYS_PER_MONTH
        )
        - 1
    )

    latest = rolling.iloc[-1]

    leader_return = safe_float(
        latest.get(leader)
    )

    records = []

    for ticker in prices.columns:

        if ticker == leader:
            continue

        stock_return = safe_float(
            latest.get(ticker)
        )

        if pd.isna(stock_return):
            continue

        series = (
            prices[ticker]
            .dropna()
        )

        if series.empty:
            continue

        current_price = float(
            series.iloc[-1]
        )

        records.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "current_21d_return": stock_return,
                "vs_leader": (
                    stock_return
                    - leader_return
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# HISTORICAL FOLLOWER STATISTICS
# ============================================================

def calculate_follower_event_statistics(
    event_study,
):

    if event_study.empty:
        return pd.DataFrame()

    records = []

    for ticker, group in (
        event_study.groupby("ticker")
    ):

        if len(group) < 2:
            continue

        follower_returns = (
            group["follower_return"]
            .dropna()
        )

        excess_returns = (
            group["excess_return"]
            .dropna()
        )

        if follower_returns.empty:
            continue

        probability_positive = (
            follower_returns > 0
        ).mean()

        probability_beats_leader = (
            excess_returns >= 0.01
        ).mean()

        probability_within_1pp = (
            excess_returns.abs() <= 0.01
        ).mean()

        probability_lags_1pp = (
            excess_returns <= -0.01
        ).mean()

        records.append(
            {
                "ticker": ticker,
                "events": len(group),
                "mean_return": follower_returns.mean(),
                "median_return": follower_returns.median(),
                "return_std": follower_returns.std(),
                "mean_excess": excess_returns.mean(),
                "median_excess": excess_returns.median(),
                "prob_positive": probability_positive,
                "prob_beats_leader_1pp": (
                    probability_beats_leader
                ),
                "prob_within_1pp": (
                    probability_within_1pp
                ),
                "prob_lags_1pp": (
                    probability_lags_1pp
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# RISK STATISTICS
# ============================================================

def calculate_risk_statistics(
    prices,
    ticker,
):

    series = (
        prices[ticker]
        .dropna()
    )

    if len(series) < 100:
        return {
            "annualized_volatility": np.nan,
            "downside_volatility": np.nan,
            "max_drawdown": np.nan,
            "annualized_return": np.nan,
            "sharpe": np.nan,
        }

    log_returns = (
        np.log(
            series / series.shift(1)
        )
        .dropna()
    )

    annualized_return = (
        log_returns.mean()
        * TRADING_DAYS_PER_YEAR
    )

    annualized_volatility = (
        log_returns.std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    downside_returns = (
        log_returns[
            log_returns < 0
        ]
    )

    downside_volatility = np.nan

    if len(downside_returns) > 1:

        downside_volatility = (
            downside_returns.std()
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )

    cumulative = np.exp(
        log_returns.cumsum()
    )

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    sharpe = np.nan

    if (
        pd.notna(annualized_volatility)
        and annualized_volatility > 0
    ):

        sharpe = (
            annualized_return
            / annualized_volatility
        )

    return {
        "annualized_volatility": annualized_volatility,
        "downside_volatility": downside_volatility,
        "max_drawdown": max_drawdown,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
    }


# ============================================================
# CORRELATION WITH LEADER
# ============================================================

def calculate_correlation(
    prices,
    leader,
    ticker,
):

    data = prices[
        [leader, ticker]
    ].dropna()

    if len(data) < 30:
        return np.nan

    returns = np.log(
        data / data.shift(1)
    ).dropna()

    if len(returns) < 30:
        return np.nan

    return returns[
        leader
    ].corr(
        returns[ticker]
    )


# ============================================================
# RESEARCH RANKING
# ============================================================

def calculate_research_score(
    row
):

    score_parts = []

    # Positive historical performance
    if pd.notna(
        row["prob_positive"]
    ):

        score_parts.append(
            0.25
            * row["prob_positive"]
        )

    # Historical probability of beating leader
    if pd.notna(
        row["prob_beats_leader_1pp"]
    ):

        score_parts.append(
            0.35
            * row[
                "prob_beats_leader_1pp"
            ]
        )

    # Historical excess return
    if pd.notna(
        row["mean_excess"]
    ):

        excess_score = np.clip(
            0.5
            + row["mean_excess"] * 5,
            0,
            1,
        )

        score_parts.append(
            0.25
            * excess_score
        )

    # Current lag
    if pd.notna(
        row["vs_leader"]
    ):

        lag_score = np.clip(
            (
                -row["vs_leader"]
                + 0.05
            ) / 0.20,
            0,
            1,
        )

        score_parts.append(
            0.15
            * lag_score
        )

    if not score_parts:
        return np.nan

    return (
        sum(score_parts)
        * 100
    )


# ============================================================
# GBM + MONTE CARLO
# ============================================================

def estimate_gbm_parameters(
    prices,
    ticker,
    event_study,
    event_dates,
):

    series = (
        prices[ticker]
        .dropna()
    )

    if len(series) < 100:
        return None

    daily_log_returns = (
        np.log(
            series / series.shift(1)
        )
        .dropna()
    )

    if daily_log_returns.empty:
        return None

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    sigma = (
        daily_log_returns.std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    # --------------------------------------------------------
    # Historical event-conditioned drift
    #
    # We use the follower's historical performance after
    # comparable leader events to estimate the annualized
    # drift used in the simulation.
    # --------------------------------------------------------

    event_returns = pd.Series(
        dtype=float
    )

    if not event_study.empty:

        selected = event_study[
            event_study["ticker"]
            == ticker
        ]

        if not selected.empty:

            event_returns = (
                selected[
                    "follower_return"
                ]
                .dropna()
            )

    if len(event_returns) >= 2:

        mean_event_return = (
            event_returns.mean()
        )

        # Approximate annualized arithmetic return
        mu = (
            mean_event_return
            * TRADING_DAYS_PER_YEAR
            / TRADING_DAYS_PER_MONTH
        )

    else:

        # Fallback to overall historical log-return drift
        mu = (
            daily_log_returns.mean()
            * TRADING_DAYS_PER_YEAR
        )

    # Prevent extreme unrealistic drift values
    mu = float(
        np.clip(
            mu,
            -1.0,
            1.0,
        )
    )

    sigma = float(
        np.clip(
            sigma,
            0.0001,
            2.0,
        )
    )

    return {
        "mu": mu,
        "sigma": sigma,
        "event_return_count": len(
            event_returns
        ),
    }


def monte_carlo_gbm(
    current_price,
    mu,
    sigma,
    days=FORECAST_DAYS,
    simulations=DEFAULT_SIMULATIONS,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )

    dt = (
        1
        / TRADING_DAYS_PER_YEAR
    )

    random_shocks = rng.normal(
        0,
        1,
        size=(
            simulations,
            days,
        ),
    )

    daily_returns = (
        (
            mu
            - 0.5
            * sigma**2
        )
        * dt
        + sigma
        * np.sqrt(dt)
        * random_shocks
    )

    cumulative_log_returns = (
        np.cumsum(
            daily_returns,
            axis=1,
        )
    )

    price_paths = (
        current_price
        * np.exp(
            cumulative_log_returns
        )
    )

    expected_path = np.mean(
        price_paths,
        axis=0,
    )

    median_path = np.median(
        price_paths,
        axis=0,
    )

    lower_path = np.percentile(
        price_paths,
        2.5,
        axis=0,
    )

    upper_path = np.percentile(
        price_paths,
        97.5,
        axis=0,
    )

    final_prices = (
        price_paths[:, -1]
    )

    final_returns = (
        final_prices
        / current_price
        - 1
    )

    probability_positive = (
        final_returns > 0
    ).mean()

    return {
        "paths": price_paths,
        "expected_path": expected_path,
        "median_path": median_path,
        "lower_path": lower_path,
        "upper_path": upper_path,
        "final_prices": final_prices,
        "final_returns": final_returns,
        "probability_positive": probability_positive,
    }


# ============================================================
# HISTORICAL PEAK TIMING
# ============================================================

def calculate_peak_statistics(
    prices,
    ticker,
    event_dates,
    horizon=FORECAST_DAYS,
):

    series = (
        prices[ticker]
        .dropna()
    )

    peak_days = []
    peak_returns = []

    for event_date in event_dates:

        if event_date not in series.index:
            continue

        position = series.index.get_loc(
            event_date
        )

        end_position = (
            position
            + horizon
        )

        if (
            end_position
            >= len(series)
        ):
            continue

        start_price = float(
            series.iloc[position]
        )

        future_prices = series.iloc[
            position + 1:
            end_position + 1
        ]

        future_returns = (
            future_prices
            / start_price
            - 1
        )

        if future_returns.empty:
            continue

        peak_idx = int(
            np.argmax(
                future_returns.values
            )
        )

        peak_day = (
            peak_idx + 1
        )

        peak_return = float(
            future_returns.iloc[
                peak_idx
            ]
        )

        peak_days.append(
            peak_day
        )

        peak_returns.append(
            peak_return
        )

    if not peak_days:
        return None

    return {
        "median_peak_day": np.median(
            peak_days
        ),
        "p25_peak_day": np.percentile(
            peak_days,
            25,
        ),
        "p75_peak_day": np.percentile(
            peak_days,
            75,
        ),
        "median_peak_return": np.median(
            peak_returns
        ),
    }


# ============================================================
# RECENT MOVEMENT CHART
# ============================================================

def create_recent_movement_chart(
    prices,
    leader,
    followers,
    days=60,
):

    selected = [
        ticker
        for ticker in [
            leader
        ] + followers
        if ticker in prices.columns
    ]

    recent = (
        prices[selected]
        .dropna(
            how="all"
        )
        .tail(days)
    )

    if recent.empty:
        return None

    normalized = (
        recent
        / recent.iloc[0]
        * 100
    )

    fig = go.Figure()

    for ticker in selected:

        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker],
                mode="lines",
                name=ticker,
            )
        )

    fig.update_layout(
        title="Recent Relative Performance",
        xaxis_title="Date",
        yaxis_title="Indexed price (start = 100)",
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ============================================================
# HISTORICAL FOLLOWER CHART
# ============================================================

def create_historical_event_chart(
    prices,
    leader,
    follower,
    event_dates,
    forward_days=21,
):

    if not event_dates:
        return None

    data = prices[
        [leader, follower]
    ].dropna()

    leader_paths = []
    follower_paths = []

    for event_date in event_dates:

        if event_date not in data.index:
            continue

        position = data.index.get_loc(
            event_date
        )

        if (
            position + forward_days
            >= len(data.index)
        ):
            continue

        leader_start = float(
            data.iloc[position][leader]
        )

        follower_start = float(
            data.iloc[position][follower]
        )

        leader_path = []
        follower_path = []

        for day in range(
            forward_days + 1
        ):

            leader_price = float(
                data.iloc[
                    position + day
                ][leader]
            )

            follower_price = float(
                data.iloc[
                    position + day
                ][follower]
            )

            leader_path.append(
                leader_price
                / leader_start
                * 100
            )

            follower_path.append(
                follower_price
                / follower_start
                * 100
            )

        leader_paths.append(
            leader_path
        )

        follower_paths.append(
            follower_path
        )

    if not leader_paths:
        return None

    leader_mean = np.mean(
        np.array(
            leader_paths
        ),
        axis=0,
    )

    follower_mean = np.mean(
        np.array(
            follower_paths
        ),
        axis=0,
    )

    x = np.arange(
        forward_days + 1
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=leader_mean,
            mode="lines",
            name=leader,
            line=dict(
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=follower_mean,
            mode="lines",
            name=follower,
            line=dict(
                width=3,
            ),
        )
    )

    fig.add_hline(
        y=100,
        line_dash="dot",
        line_width=1,
    )

    fig.update_layout(
        title=(
            "Average Historical Follow-Through"
        ),
        xaxis_title=(
            "Trading days after comparable leader event"
        ),
        yaxis_title="Indexed price",
        hovermode="x unified",
        height=500,
    )

    return fig


# ============================================================
# MONTE CARLO FORECAST CHART
# ============================================================

def create_monte_carlo_chart(
    prices,
    ticker,
    simulation,
):

    series = (
        prices[ticker]
        .dropna()
    )

    history = series.tail(60)

    current_price = float(
        series.iloc[-1]
    )

    forecast_days = len(
        simulation[
            "expected_path"
        ]
    )

    last_date = history.index[-1]

    future_dates = pd.bdate_range(
        start=(
            last_date
            + pd.Timedelta(days=1)
        ),
        periods=forecast_days,
    )

    fig = go.Figure()

    # Historical price
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical price",
            line=dict(
                width=2,
            ),
        )
    )

    # Expected Monte Carlo path
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=simulation[
                "expected_path"
            ],
            mode="lines",
            name="Expected path",
            line=dict(
                width=3,
                dash="dash",
            ),
        )
    )

    # Median
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=simulation[
                "median_path"
            ],
            mode="lines",
            name="Median path",
            line=dict(
                width=2,
                dash="dot",
            ),
        )
    )

    # Lower
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=simulation[
                "lower_path"
            ],
            mode="lines",
            name="95% lower",
            line=dict(
                width=1,
                dash="dot",
            ),
        )
    )

    # Upper
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=simulation[
                "upper_path"
            ],
            mode="lines",
            name="95% upper",
            line=dict(
                width=1,
                dash="dot",
            ),
            fill="tonexty",
            fillcolor="rgba(100,116,139,0.10)",
        )
    )

    # Current price
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_width=1,
    )

    fig.update_layout(
        title=(
            f"30-Trading-Day Monte Carlo Scenario — {ticker}"
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        """
        <div class="main-title">
            Market Intelligence Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            S&P 500 · Historical Momentum · Leader–Follower Analysis ·
            Risk · Monte Carlo Scenarios
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # LOAD S&P 500
    # ========================================================

    try:

        sp500 = get_sp500_constituents()

    except Exception as exc:

        st.error(
            "Unable to load S&P 500 constituents."
        )

        st.exception(exc)

        st.stop()

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.markdown(
        "## Research Setup"
    )

    st.sidebar.caption(
        "Select a market leader and define how similar a "
        "historical momentum event must be."
    )

    leader_options = (
        sp500["Ticker"]
        + " — "
        + sp500["Security"]
    ).tolist()

    # Try NVDA as default.
    default_index = 0

    if "NVDA — NVIDIA" in leader_options:

        default_index = (
            leader_options.index(
                "NVDA — NVIDIA"
            )
        )

    selected_label = (
        st.sidebar.selectbox(
            "Selected Leader",
            leader_options,
            index=default_index,
        )
    )

    leader = selected_label.split(
        " — "
    )[0]

    leader_company = sp500.loc[
        sp500["Ticker"] == leader,
        "Security",
    ].iloc[0]

    leader_sector = sp500.loc[
        sp500["Ticker"] == leader,
        "GICS Sector",
    ].iloc[0]

    st.sidebar.markdown("---")

    # ========================================================
    # SIMPLIFIED SIMILARITY CONTROL
    # ========================================================

    similarity_pct = (
        st.sidebar.slider(
            "Historical Momentum Similarity",
            min_value=2,
            max_value=15,
            value=5,
            step=1,
            format="%d%%",
            help=(
                "If the current leader has a 21-day return of "
                "+15%, a ±5% setting searches historical periods "
                "where the leader returned approximately +10% to +20%."
            ),
        )
    )

    st.sidebar.caption(
        f"Search range: current momentum ±{similarity_pct}%"
    )

    st.sidebar.markdown("---")

    min_events_for_ranking = (
        st.sidebar.slider(
            "Minimum comparable events",
            min_value=2,
            max_value=8,
            value=3,
            step=1,
            help=(
                "Minimum number of comparable historical events "
                "required before a stock is ranked."
            ),
        )
    )

    prefer_lagging = (
        st.sidebar.checkbox(
            "Prioritize stocks currently lagging the leader",
            value=True,
            help=(
                "Prioritize stocks whose current 21-day return "
                "is below the selected leader."
            ),
        )
    )

    top_n = (
        st.sidebar.slider(
            "Followers to display",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "### Monte Carlo"
    )

    simulations = (
        st.sidebar.selectbox(
            "Simulation paths",
            [
                5000,
                10000,
                20000,
            ],
            index=1,
        )
    )

    st.sidebar.caption(
        "Forecast horizon: 30 trading days"
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Market: S&P 500"
    )

    st.sidebar.caption(
        "Data: Yahoo Finance + current S&P 500 constituents"
    )

    if st.sidebar.button(
        "Clear cached data"
    ):

        st.cache_data.clear()

        st.rerun()

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    with st.spinner(
        "Loading S&P 500 historical market data..."
    ):

        try:

            all_tickers = (
                sp500["Ticker"]
                .tolist()
            )

            prices = download_price_data(
                all_tickers,
                years=8,
            )

        except Exception as exc:

            st.error(
                "Unable to download market data."
            )

            st.exception(exc)

            st.stop()

    if prices.empty:

        st.error(
            "No market data was returned."
        )

        st.stop()

    available_tickers = [
        ticker
        for ticker in all_tickers
        if ticker in prices.columns
    ]

    if leader not in prices.columns:

        st.error(
            f"Market data for {leader} is unavailable."
        )

        st.stop()

    # ========================================================
    # CURRENT LEADER
    # ========================================================

    try:

        leader_signal = (
            calculate_current_leader_signal(
                prices,
                leader,
            )
        )

    except Exception as exc:

        st.error(
            "Could not calculate leader momentum."
        )

        st.exception(exc)

        st.stop()

    current_return = (
        leader_signal["return"]
    )

    current_price = (
        leader_signal["price"]
    )

    signal_date = (
        leader_signal["date"]
    )

    # ========================================================
    # CURRENT SIGNAL HEADER
    # ========================================================

    st.markdown(
        '<div class="research-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        ### Current Market Signal

        **{leader_company} ({leader})** is the selected leader.

        The dashboard uses the latest **21-trading-day return**
        as the momentum signal and searches for historical periods
        where the leader displayed similar momentum.
        """
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4, metric5 = (
        st.columns(5)
    )

    with metric1:

        st.metric(
            "Leader",
            leader,
        )

    with metric2:

        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
        )

    with metric3:

        st.metric(
            "21D Momentum",
            signed_pct(
                current_return
            ),
        )

    with metric4:

        st.metric(
            "Sector",
            leader_sector,
        )

    with metric5:

        st.metric(
            "Signal Date",
            signal_date.strftime(
                "%d %b %Y"
            ),
        )

    # ========================================================
    # POSITIVE MOMENTUM CHECK
    # ========================================================

    if current_return <= 0:

        st.warning(
            f"{leader} currently has a "
            f"{signed_pct(current_return)} "
            "21-day return. This research framework is designed "
            "primarily to study positive momentum leadership. "
            "Consider selecting a currently positive leader for "
            "the intended analysis."
        )

        st.info(
            "No positive leader momentum events will be selected "
            "while the current 21-day momentum is negative."
        )

        st.stop()

    # ========================================================
    # HISTORICAL ANALOGUES
    # ========================================================

    with st.spinner(
        "Searching for comparable historical momentum events..."
    ):

        event_dates = (
            identify_leader_events(
                prices=prices,
                leader=leader,
                current_return=current_return,
                similarity_pct=similarity_pct,
            )
        )

    # ========================================================
    # EVENT SUMMARY
    # ========================================================

    event_lower = max(
        0,
        current_return
        - similarity_pct / 100,
    )

    event_upper = (
        current_return
        + similarity_pct / 100
    )

    event1, event2, event3 = (
        st.columns(3)
    )

    with event1:

        st.metric(
            "Comparable Historical Events",
            len(event_dates),
        )

    with event2:

        st.metric(
            "Historical Search Range",
            (
                f"{event_lower * 100:.1f}% "
                f"to "
                f"{event_upper * 100:.1f}%"
            ),
        )

    with event3:

        st.metric(
            "Similarity",
            f"±{similarity_pct}%",
        )

    if not event_dates:

        st.error(
            "No comparable positive momentum events were found. "
            "Try increasing Historical Momentum Similarity."
        )

        st.stop()

    if len(event_dates) < min_events_for_ranking:

        st.warning(
            f"Only {len(event_dates)} comparable historical event(s) "
            f"were found. This is below the selected minimum of "
            f"{min_events_for_ranking}. Results may therefore be less "
            "statistically robust."
        )

    # ========================================================
    # EVENT TABLE
    # ========================================================

    with st.expander(
        "View comparable historical leader events"
    ):

        event_df = pd.DataFrame(
            {
                "Historical Event": [
                    event.strftime(
                        "%d %b %Y"
                    )
                    for event in event_dates
                ]
            }
        )

        st.dataframe(
            event_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # EVENT STUDY
    # ========================================================

    followers = [
        ticker
        for ticker in available_tickers
        if ticker != leader
    ]

    with st.spinner(
        "Analyzing historical leader–follower relationships..."
    ):

        event_study = (
            create_event_study(
                prices=prices,
                leader=leader,
                event_dates=event_dates,
                followers=followers,
                forward_days=21,
            )
        )

    if event_study.empty:

        st.error(
            "No usable historical follower observations were found."
        )

        st.stop()

    # ========================================================
    # FOLLOWER STATISTICS
    # ========================================================

    follower_stats = (
        calculate_follower_event_statistics(
            event_study
        )
    )

    if follower_stats.empty:

        st.error(
            "Follower statistics could not be generated."
        )

        st.stop()

    # ========================================================
    # CURRENT STOCK METRICS
    # ========================================================

    current_metrics = (
        calculate_current_stock_metrics(
            prices,
            leader,
        )
    )

    follower_stats = follower_stats.merge(
        current_metrics,
        on="ticker",
        how="left",
    )

    # ========================================================
    # COMPANY INFORMATION
    # ========================================================

    follower_stats = follower_stats.merge(
        sp500[
            [
                "Ticker",
                "Security",
                "GICS Sector",
            ]
        ],
        left_on="ticker",
        right_on="Ticker",
        how="left",
    )

    follower_stats.drop(
        columns=["Ticker"],
        inplace=True,
        errors="ignore",
    )

    # ========================================================
    # RISK STATISTICS
    # ========================================================

    risk_records = []

    for ticker in follower_stats[
        "ticker"
    ]:

        risk = (
            calculate_risk_statistics(
                prices,
                ticker,
            )
        )

        correlation = (
            calculate_correlation(
                prices,
                leader,
                ticker,
            )
        )

        risk_records.append(
            {
                "ticker": ticker,
                "annualized_volatility": (
                    risk[
                        "annualized_volatility"
                    ]
                ),
                "downside_volatility": (
                    risk[
                        "downside_volatility"
                    ]
                ),
                "max_drawdown": (
                    risk[
                        "max_drawdown"
                    ]
                ),
                "annualized_return": (
                    risk[
                        "annualized_return"
                    ]
                ),
                "sharpe": (
                    risk[
                        "sharpe"
                    ]
                ),
                "correlation": correlation,
            }
        )

    risk_df = pd.DataFrame(
        risk_records
    )

    follower_stats = follower_stats.merge(
        risk_df,
        on="ticker",
        how="left",
    )

    # ========================================================
    # RANKING
    # ========================================================

    ranking_df = follower_stats[
        follower_stats["events"]
        >= min_events_for_ranking
    ].copy()

    # Prefer lagging stocks.
    if prefer_lagging:

        lagging = ranking_df[
            ranking_df[
                "vs_leader"
            ] < 0
        ].copy()

        if len(lagging) >= top_n:

            ranking_df = lagging

    ranking_df[
        "research_score"
    ] = ranking_df.apply(
        calculate_research_score,
        axis=1,
    )

    ranking_df = (
        ranking_df
        .sort_values(
            [
                "research_score",
                "prob_beats_leader_1pp",
                "mean_excess",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    top_followers = (
        ranking_df
        .head(top_n)
        .copy()
    )

    # ========================================================
    # FOLLOWER SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Potential Historical Followers'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Stocks are ranked according to historical follow-through, "
        "relative performance, current lagging characteristics and risk. "
        "The score is a research ranking, not an investment recommendation."
    )

    if top_followers.empty:

        st.warning(
            "No followers met the selected criteria."
        )

    else:

        display_df = top_followers[
            [
                "ticker",
                "Security",
                "GICS Sector",
                "current_21d_return",
                "vs_leader",
                "prob_positive",
                "prob_beats_leader_1pp",
                "prob_within_1pp",
                "mean_return",
                "mean_excess",
                "annualized_volatility",
                "max_drawdown",
                "correlation",
                "research_score",
            ]
        ].copy()

        display_df.columns = [
            "Ticker",
            "Company",
            "Sector",
            "Current 21D",
            "Vs Leader",
            "P(Positive)",
            "P(Beat Leader +1pp)",
            "P(Within ±1pp)",
            "Avg Follow Return",
            "Avg Excess Return",
            "Annual Volatility",
            "Max Drawdown",
            "Correlation",
            "Research Score",
        ]

        formatted = (
            display_df.copy()
        )

        percent_columns = [
            "Current 21D",
            "Vs Leader",
            "P(Positive)",
            "P(Beat Leader +1pp)",
            "P(Within ±1pp)",
            "Avg Follow Return",
            "Avg Excess Return",
            "Annual Volatility",
            "Max Drawdown",
        ]

        for column in percent_columns:

            formatted[column] = (
                formatted[column]
                .apply(pct)
            )

        formatted["Correlation"] = (
            formatted[
                "Correlation"
            ]
            .apply(
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "—"
            )
        )

        formatted["Research Score"] = (
            formatted[
                "Research Score"
            ]
            .apply(
                lambda x:
                f"{x:.1f}"
                if pd.notna(x)
                else "—"
            )
        )

        st.dataframe(
            formatted,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # DETAILED FOLLOWER ANALYSIS
    # ========================================================

    if not top_followers.empty:

        selected_follower = (
            st.selectbox(
                "Select a follower for detailed analysis",
                top_followers[
                    "ticker"
                ].tolist(),
                format_func=lambda ticker: (
                    f"{ticker} — "
                    f"{top_followers.loc["
                        top_followers["ticker"]
                        == ticker,
                        "Security"
                    ].iloc[0]}"
                ),
            )
        )

        follower_row = (
            top_followers[
                top_followers["ticker"]
                == selected_follower
            ].iloc[0]
        )

        st.markdown(
            f"""
            <div class="section-title">
                Detailed Analysis — {selected_follower}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"{follower_row['Security']} | "
            f"{follower_row['GICS Sector']}"
        )

        # ----------------------------------------------------
        # PERFORMANCE METRICS
        # ----------------------------------------------------

        detail_cols = st.columns(6)

        with detail_cols[0]:

            st.metric(
                "Current 21D",
                signed_pct(
                    follower_row[
                        "current_21d_return"
                    ]
                ),
            )

        with detail_cols[1]:

            st.metric(
                "Vs Leader",
                signed_pct(
                    follower_row[
                        "vs_leader"
                    ]
                ),
            )

        with detail_cols[2]:

            st.metric(
                "P(Positive)",
                pct(
                    follower_row[
                        "prob_positive"
                    ]
                ),
            )

        with detail_cols[3]:

            st.metric(
                "P(Beat Leader +1pp)",
                pct(
                    follower_row[
                        "prob_beats_leader_1pp"
                    ]
                ),
            )

        with detail_cols[4]:

            st.metric(
                "Avg Excess",
                signed_pct(
                    follower_row[
                        "mean_excess"
                    ]
                ),
            )

        with detail_cols[5]:

            correlation = (
                follower_row[
                    "correlation"
                ]
            )

            st.metric(
                "Correlation",
                (
                    f"{correlation:.2f}"
                    if pd.notna(correlation)
                    else "—"
                ),
            )

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Risk Profile'
            '</div>',
            unsafe_allow_html=True,
        )

        risk_cols = st.columns(5)

        with risk_cols[0]:

            st.metric(
                "Annual Volatility",
                pct(
                    follower_row[
                        "annualized_volatility"
                    ]
                ),
            )

        with risk_cols[1]:

            st.metric(
                "Downside Volatility",
                pct(
                    follower_row[
                        "downside_volatility"
                    ]
                ),
            )

        with risk_cols[2]:

            st.metric(
                "Max Drawdown",
                pct(
                    follower_row[
                        "max_drawdown"
                    ]
                ),
            )

        with risk_cols[3]:

            sharpe = (
                follower_row[
                    "sharpe"
                ]
            )

            st.metric(
                "Sharpe-like",
                (
                    f"{sharpe:.2f}"
                    if pd.notna(sharpe)
                    else "—"
                ),
            )

        with risk_cols[4]:

            st.metric(
                "Historical Events",
                int(
                    follower_row[
                        "events"
                    ]
                ),
            )

        # ----------------------------------------------------
        # HISTORICAL FOLLOW-THROUGH
        # ----------------------------------------------------

        historical_chart = (
            create_historical_event_chart(
                prices=prices,
                leader=leader,
                follower=selected_follower,
                event_dates=event_dates,
                forward_days=21,
            )
        )

        if historical_chart:

            st.plotly_chart(
                historical_chart,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # GBM + MONTE CARLO
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '30-Trading-Day Monte Carlo Scenario'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "The forecast uses a Geometric Brownian Motion model with "
            "volatility estimated from historical daily returns and "
            "drift informed by the follower's historical performance "
            "following comparable leader events."
        )

        with st.spinner(
            "Running Monte Carlo simulation..."
        ):

            gbm_params = (
                estimate_gbm_parameters(
                    prices=prices,
                    ticker=selected_follower,
                    event_study=event_study,
                    event_dates=event_dates,
                )
            )

            current_follower_price = float(
                prices[
                    selected_follower
                ]
                .dropna()
                .iloc[-1]
            )

            if gbm_params is not None:

                simulation = (
                    monte_carlo_gbm(
                        current_price=(
                            current_follower_price
                        ),
                        mu=gbm_params[
                            "mu"
                        ],
                        sigma=gbm_params[
                            "sigma"
                        ],
                        days=FORECAST_DAYS,
                        simulations=simulations,
                    )
                )

            else:

                simulation = None

        if simulation is None:

            st.warning(
                "Not enough data to construct a Monte Carlo scenario."
            )

        else:

            expected_price = float(
                simulation[
                    "expected_path"
                ][-1]
            )

            median_price = float(
                simulation[
                    "median_path"
                ][-1]
            )

            lower_price = float(
                simulation[
                    "lower_path"
                ][-1]
            )

            upper_price = float(
                simulation[
                    "upper_path"
                ][-1]
            )

            expected_return = (
                expected_price
                / current_follower_price
                - 1
            )

            median_return = (
                median_price
                / current_follower_price
                - 1
            )

            lower_return = (
                lower_price
                / current_follower_price
                - 1
            )

            upper_return = (
                upper_price
                / current_follower_price
                - 1
            )

            probability_positive = (
                simulation[
                    "probability_positive"
                ]
            )

            forecast_cols = st.columns(6)

            with forecast_cols[0]:

                st.metric(
                    "Current Price",
                    f"${current_follower_price:,.2f}",
                )

            with forecast_cols[1]:

                st.metric(
                    "Expected Price",
                    f"${expected_price:,.2f}",
                )

            with forecast_cols[2]:

                st.metric(
                    "Expected Return",
                    signed_pct(
                        expected_return
                    ),
                )

            with forecast_cols[3]:

                st.metric(
                    "Median Return",
                    signed_pct(
                        median_return
                    ),
                )

            with forecast_cols[4]:

                st.metric(
                    "P(Positive)",
                    pct(
                        probability_positive
                    ),
                )

            with forecast_cols[5]:

                st.metric(
                    "95% Price Range",
                    (
                        f"${lower_price:,.2f}"
                        f" – "
                        f"${upper_price:,.2f}"
                    ),
                )

            st.plotly_chart(
                create_monte_carlo_chart(
                    prices=prices,
                    ticker=selected_follower,
                    simulation=simulation,
                ),
                use_container_width=True,
            )

            # ------------------------------------------------
            # MODEL PARAMETERS
            # ------------------------------------------------

            param_cols = st.columns(3)

            with param_cols[0]:

                st.metric(
                    "GBM Drift (μ)",
                    f"{gbm_params['mu'] * 100:.1f}% p.a.",
                )

            with param_cols[1]:

                st.metric(
                    "Annual Volatility (σ)",
                    f"{gbm_params['sigma'] * 100:.1f}%",
                )

            with param_cols[2]:

                st.metric(
                    "Simulation Paths",
                    f"{simulations:,}",
                )

            st.info(
                "The Monte Carlo interval represents the modeled "
                "distribution of simulated prices under the selected "
                "GBM assumptions. It is a scenario range, not a "
                "guarantee or probability statement about the actual future."
            )

        # ----------------------------------------------------
        # PEAK TIMING
        # ----------------------------------------------------

        peak_stats = (
            calculate_peak_statistics(
                prices=prices,
                ticker=selected_follower,
                event_dates=event_dates,
                horizon=FORECAST_DAYS,
            )
        )

        if peak_stats:

            st.markdown(
                '<div class="section-title">'
                'Historical Peak Timing'
                '</div>',
                unsafe_allow_html=True,
            )

            peak_cols = st.columns(4)

            with peak_cols[0]:

                st.metric(
                    "Median Peak Day",
                    (
                        f"Day "
                        f"{peak_stats['median_peak_day']:.0f}"
                    ),
                )

            with peak_cols[1]:

                st.metric(
                    "Typical Peak Range",
                    (
                        f"Day "
                        f"{peak_stats['p25_peak_day']:.0f}"
                        f" – "
                        f"Day "
                        f"{peak_stats['p75_peak_day']:.0f}"
                    ),
                )

            with peak_cols[2]:

                st.metric(
                    "Median Peak Return",
                    signed_pct(
                        peak_stats[
                            "median_peak_return"
                        ]
                    ),
                )

            with peak_cols[3]:

                st.metric(
                    "Comparable Events",
                    len(event_dates),
                )

            st.caption(
                "Historical peak timing describes what happened "
                "after comparable events in the sample. It should "
                "not be interpreted as a future sell signal."
            )

    # ========================================================
    # RECENT MARKET MOVEMENT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Recent Market Movement'
        '</div>',
        unsafe_allow_html=True,
    )

    recent_followers = (
        top_followers[
            "ticker"
        ].tolist()
        if not top_followers.empty
        else []
    )

    recent_chart = (
        create_recent_movement_chart(
            prices=prices,
            leader=leader,
            followers=recent_followers,
            days=60,
        )
    )

    if recent_chart:

        st.plotly_chart(
            recent_chart,
            use_container_width=True,
        )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Research Methodology'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "How the dashboard works"
    ):

        st.markdown(
            f"""
            **1. Select a leader**

            The user selects an S&P 500 stock that is currently
            showing strong positive momentum.

            **2. Current momentum**

            The leader's latest {TRADING_DAYS_PER_MONTH}-trading-day
            return is used as the current momentum signal.

            **3. Historical analogue search**

            The system searches historical data for periods where
            the leader's 21-day return was similar to the current
            return.

            For example, if the current leader momentum is +15% and
            the similarity setting is ±5%, historical events between
            approximately +10% and +20% are considered.

            **4. Follower analysis**

            After each comparable leader event, the system examines
            the next 21 trading days of the other S&P 500 stocks.

            **5. Conditional probabilities**

            The dashboard estimates:

            - Probability of a positive follower return.
            - Probability of outperforming the leader by at least 1 percentage point.
            - Probability of staying within ±1 percentage point of the leader.
            - Probability of lagging the leader by at least 1 percentage point.

            **6. Risk analysis**

            Historical volatility, downside volatility, maximum drawdown,
            correlation and a Sharpe-like statistic are calculated.

            **7. Monte Carlo forecast**

            A Geometric Brownian Motion model is used to simulate
            {simulations:,} possible 30-trading-day price paths.

            The simulation uses:

            - Estimated drift (μ)
            - Historical volatility (σ)
            - Random Brownian-motion shocks

            **8. Historical peak timing**

            Historical comparable events are also examined to determine
            when followers typically reached their maximum return during
            the following 30 trading days.
            """
        )

    # ========================================================
    # IMPORTANT LIMITATIONS
    # ========================================================

    with st.expander(
        "Important limitations"
    ):

        st.markdown(
            """
            **Survivorship bias**

            The current analysis uses today's S&P 500 constituent list.
            Historical companies that left the index are therefore not
            represented in the analysis.

            **Correlation is not causation**

            Historical co-movement does not establish that the leader
            causes the follower to move.

            **Regime changes**

            Relationships between companies can change because of
            fundamentals, interest rates, regulation, market structure,
            valuation and other factors.

            **GBM assumptions**

            Geometric Brownian Motion is a mathematical model that
            simplifies real-world price dynamics. Real markets can show
            jumps, volatility clustering, fat tails and changing
            correlations that the standard GBM framework does not fully
            capture.

            **Monte Carlo output**

            The simulated range represents outcomes under the model's
            assumptions. It should not be interpreted as a guarantee
            that actual future prices will remain inside the interval.

            **Research purpose**

            This dashboard is a financial-market research prototype
            designed to explore historical patterns and quantitative
            relationships. It is not an automated trading system or
            personal financial advice.
            """
        )

    # ========================================================
    # EXPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Research Export'
        '</div>',
        unsafe_allow_html=True,
    )

    if not top_followers.empty:

        export_columns = [
            "ticker",
            "Security",
            "GICS Sector",
            "current_price",
            "current_21d_return",
            "vs_leader",
            "events",
            "mean_return",
            "median_return",
            "mean_excess",
            "median_excess",
            "prob_positive",
            "prob_beats_leader_1pp",
            "prob_within_1pp",
            "prob_lags_1pp",
            "annualized_return",
            "annualized_volatility",
            "downside_volatility",
            "max_drawdown",
            "sharpe",
            "correlation",
            "research_score",
        ]

        export_df = (
            top_followers[
                [
                    col
                    for col in export_columns
                    if col
                    in top_followers.columns
                ]
            ].copy()
        )

        csv = (
            export_df
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            label="Download follower research CSV",
            data=csv,
            file_name=(
                f"{leader}_follower_research.csv"
            ),
            mime="text/csv",
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    generated_at = (
        datetime.now().strftime(
            "%d %b %Y %H:%M"
        )
    )

    st.caption(
        f"S&P 500 Market Intelligence Prototype | "
        f"Generated {generated_at}"
    )

    st.caption(
        "For research and educational purposes only. "
        "Historical relationships and model scenarios do not guarantee "
        "future investment performance."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
