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
# PAGE CONFIGURATION
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

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

TRADING_MONTH = 21
FORECAST_HORIZON = 30

DEFAULT_EVENT_TOLERANCE = 0.05
DEFAULT_MIN_EVENT_RETURN = 0.05

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
        color: #111827;
    }

    .subtitle {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #111827;
    }

    .research-box {
        padding: 1rem 1.1rem;
        border-radius: 0.7rem;
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }

    .method-box {
        padding: 0.9rem 1rem;
        border-radius: 0.6rem;
        background-color: #F9FAFB;
        border-left: 4px solid #64748B;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #6B7280;
        font-size: 0.85rem;
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
# HELPER FUNCTIONS
# ============================================================

def normalize_ticker(ticker: str) -> str:
    """
    Yahoo Finance uses '-' where some index constituent lists use '.'.
    Example: BRK.B -> BRK-B
    """
    return str(ticker).strip().upper().replace(".", "-")


def pct(value, decimals=1):
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def signed_pct(value, decimals=1):
    if value is None or pd.isna(value):
        return "—"

    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def format_number(value, decimals=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


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
    """
    Retrieve current S&P 500 constituents from Wikipedia.

    Note:
    This is a current constituent list. Historical analysis therefore
    contains survivorship bias because companies that left the index
    historically are not included.
    """

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(
        WIKI_URL,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))

    if not tables:
        raise ValueError("Could not find S&P 500 constituent table.")

    df = tables[0].copy()

    required_columns = ["Symbol", "Security", "GICS Sector"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"S&P 500 table is missing columns: {missing}"
        )

    df["Ticker"] = df["Symbol"].astype(str).map(normalize_ticker)

    df = df[
        ["Ticker", "Security", "GICS Sector", "GICS Sub-Industry"]
    ].copy()

    df = df.drop_duplicates(subset="Ticker")

    return df.sort_values("Ticker").reset_index(drop=True)


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl="6H", show_spinner=False)
def download_price_data(
    tickers,
    years=8,
):
    """
    Download adjusted historical prices from Yahoo Finance.

    Uses 1-day observations and auto-adjusted prices.
    """

    tickers = [
        normalize_ticker(t)
        for t in tickers
        if t
    ]

    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return pd.DataFrame()

    end_date = datetime.now().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

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

    # MultiTicker download
    if isinstance(data.columns, pd.MultiIndex):

        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()

        elif "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()

        else:
            raise ValueError("No Close price found in downloaded data.")

    else:
        # Single ticker
        if "Close" in data.columns:
            prices = data[["Close"]].copy()

        elif "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()

        else:
            raise ValueError("No Close price found in downloaded data.")

        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    prices.columns = [
        normalize_ticker(col)
        for col in prices.columns
    ]

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    return prices


@st.cache_data(ttl="6H", show_spinner=False)
def download_single_ticker(ticker):
    """
    Fresh-ish data for the selected leader.
    """

    return download_price_data(
        [ticker],
        years=8,
    )


# ============================================================
# RETURN CALCULATIONS
# ============================================================

def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1))


def calculate_rolling_returns(prices, window=21):
    return prices / prices.shift(window) - 1


# ============================================================
# LEADER SIGNAL
# ============================================================

def calculate_current_leader_signal(
    prices,
    leader,
    window=TRADING_MONTH,
):
    if leader not in prices.columns:
        raise ValueError(
            f"{leader} was not found in the downloaded price data."
        )

    series = prices[leader].dropna()

    if len(series) <= window:
        raise ValueError(
            "Not enough historical observations to calculate "
            "the current leader signal."
        )

    current_price = float(series.iloc[-1])
    prior_price = float(series.iloc[-window - 0])

    # Use approximately 21 trading-day price change.
    current_return = current_price / prior_price - 1

    signal_date = series.index[-1]

    return {
        "leader": leader,
        "date": signal_date,
        "price": current_price,
        "prior_price": prior_price,
        "return": current_return,
    }


# ============================================================
# HISTORICAL LEADER EVENTS
# ============================================================

def identify_leader_events(
    prices,
    leader,
    target_return,
    tolerance=DEFAULT_EVENT_TOLERANCE,
    minimum_return=DEFAULT_MIN_EVENT_RETURN,
    cooldown_days=21,
):
    """
    Find historical 21-trading-day periods where the leader had
    momentum similar to today's leader momentum.

    Example:
        Current leader return = +15%
        Tolerance = +/-5%

    Historical events are approximately:
        +10% to +20%

    We also apply a cooldown so consecutive overlapping dates do not
    overwhelm the event sample.
    """

    rolling = calculate_rolling_returns(
        prices[leader],
        window=TRADING_MONTH,
    ).dropna()

    lower = max(
        minimum_return,
        target_return - tolerance,
    )

    upper = target_return + tolerance

    candidate_dates = rolling[
        (rolling >= lower)
        & (rolling <= upper)
    ].index

    if len(candidate_dates) == 0:
        return []

    selected_dates = []

    last_selected = None

    for dt in candidate_dates:
        if last_selected is None:
            selected_dates.append(dt)
            last_selected = dt
            continue

        days_since = (dt - last_selected).days

        if days_since >= cooldown_days:
            selected_dates.append(dt)
            last_selected = dt

    return selected_dates


# ============================================================
# EVENT STUDY
# ============================================================

def create_event_study(
    prices,
    leader,
    event_dates,
    followers,
    forward_days=TRADING_MONTH,
):
    """
    For each historical leader event, calculate the forward
    performance of every candidate follower.
    """

    records = []

    for event_date in event_dates:

        all_prices = prices.loc[
            prices.index >= event_date
        ]

        if len(all_prices) <= forward_days:
            continue

        future_dates = all_prices.index

        try:
            event_position = prices.index.get_loc(event_date)
        except KeyError:
            continue

        # Need enough future observations.
        if event_position + forward_days >= len(prices.index):
            continue

        future_date = prices.index[
            event_position + forward_days
        ]

        leader_event_start = prices.loc[event_date, leader]
        leader_future = prices.loc[future_date, leader]

        if pd.isna(leader_event_start) or pd.isna(leader_future):
            continue

        leader_return = (
            leader_future / leader_event_start - 1
        )

        for ticker in followers:

            if ticker not in prices.columns:
                continue

            candidate_start = prices.loc[event_date, ticker]
            candidate_future = prices.loc[future_date, ticker]

            if pd.isna(candidate_start) or pd.isna(candidate_future):
                continue

            candidate_return = (
                candidate_future / candidate_start - 1
            )

            records.append(
                {
                    "event_date": event_date,
                    "future_date": future_date,
                    "leader": leader,
                    "leader_return": leader_return,
                    "ticker": ticker,
                    "candidate_return": candidate_return,
                    "excess_return": (
                        candidate_return - leader_return
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# CURRENT LAGGING STOCK ANALYSIS
# ============================================================

def calculate_current_stock_metrics(
    prices,
    leader,
    tickers,
):
    """
    Current stock metrics relative to the leader.
    """

    rolling_21 = calculate_rolling_returns(
        prices,
        window=TRADING_MONTH,
    )

    latest_21 = rolling_21.iloc[-1]

    records = []

    leader_return = latest_21.get(leader, np.nan)

    for ticker in tickers:

        if ticker not in prices.columns:
            continue

        if ticker == leader:
            continue

        stock_return = latest_21.get(ticker, np.nan)

        price_series = prices[ticker].dropna()

        if len(price_series) < 2:
            continue

        current_price = float(price_series.iloc[-1])

        records.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "current_21d_return": stock_return,
                "vs_leader": stock_return - leader_return,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# RISK METRICS
# ============================================================

def calculate_stock_statistics(
    prices,
    ticker,
):
    series = prices[ticker].dropna()

    if len(series) < 2:
        return {
            "mean_daily_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "downside_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }

    log_returns = np.log(
        series / series.shift(1)
    ).dropna()

    mean_daily = log_returns.mean()

    annualized_return = (
        mean_daily * 252
    )

    annualized_vol = (
        log_returns.std() * np.sqrt(252)
    )

    downside = log_returns[
        log_returns < 0
    ]

    downside_vol = (
        downside.std() * np.sqrt(252)
        if len(downside) > 1
        else np.nan
    )

    cumulative = (
        1 + log_returns
    ).cumprod()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max - 1
    )

    max_drawdown = drawdown.min()

    sharpe = (
        annualized_return / annualized_vol
        if annualized_vol and annualized_vol > 0
        else np.nan
    )

    return {
        "mean_daily_return": mean_daily,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "downside_volatility": downside_vol,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


# ============================================================
# EVENT-BASED FOLLOWER STATISTICS
# ============================================================

def calculate_follower_event_statistics(
    event_study,
):
    records = []

    if event_study.empty:
        return pd.DataFrame()

    grouped = event_study.groupby("ticker")

    for ticker, group in grouped:

        if len(group) < 2:
            continue

        candidate_returns = (
            group["candidate_return"]
            .dropna()
        )

        leader_returns = (
            group["leader_return"]
            .dropna()
        )

        excess_returns = (
            group["excess_return"]
            .dropna()
        )

        if len(candidate_returns) == 0:
            continue

        probability_positive = (
            (candidate_returns > 0)
            .mean()
        )

        probability_beats_leader_1pp = (
            (excess_returns >= 0.01)
            .mean()
        )

        probability_within_1pp = (
            (excess_returns.abs() <= 0.01)
            .mean()
        )

        probability_lags_1pp = (
            (excess_returns <= -0.01)
            .mean()
        )

        mean_return = candidate_returns.mean()
        median_return = candidate_returns.median()
        std_return = candidate_returns.std()

        mean_excess = excess_returns.mean()

        leader_mean = leader_returns.mean()

        records.append(
            {
                "ticker": ticker,
                "events": len(group),
                "mean_return": mean_return,
                "median_return": median_return,
                "volatility": std_return,
                "mean_leader_return": leader_mean,
                "mean_excess": mean_excess,
                "prob_positive": probability_positive,
                "prob_beats_leader_1pp": (
                    probability_beats_leader_1pp
                ),
                "prob_within_1pp": (
                    probability_within_1pp
                ),
                "prob_lags_1pp": (
                    probability_lags_1pp
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlation(
    prices,
    leader,
    ticker,
):
    data = prices[
        [leader, ticker]
    ].dropna()

    if len(data) < 20:
        return np.nan

    returns = np.log(
        data / data.shift(1)
    ).dropna()

    if len(returns) < 20:
        return np.nan

    return returns[leader].corr(
        returns[ticker]
    )


# ============================================================
# FOLLOWER SCORE
# ============================================================

def calculate_follower_score(row):
    """
    Composite research score.

    This is not an investment recommendation.
    It combines:
        - historical probability of positive performance
        - probability of beating leader by >= 1pp
        - average excess return
        - current lag relative to leader

    Scores are designed for ranking only.
    """

    components = []

    if pd.notna(row["prob_positive"]):
        components.append(
            0.25 * row["prob_positive"]
        )

    if pd.notna(row["prob_beats_leader_1pp"]):
        components.append(
            0.35 * row["prob_beats_leader_1pp"]
        )

    if pd.notna(row["mean_excess"]):
        excess_score = np.clip(
            0.5 + row["mean_excess"] * 5,
            0,
            1,
        )

        components.append(
            0.25 * excess_score
        )

    if pd.notna(row["current_21d_return"]):
        # Prefer stocks currently lagging the leader,
        # but avoid extremely weak stocks.
        lag_score = np.clip(
            (-row["vs_leader"] + 0.05) / 0.20,
            0,
            1,
        )

        components.append(
            0.15 * lag_score
        )

    return sum(components) * 100


# ============================================================
# FORECAST MODEL
# ============================================================

def create_event_based_forecast(
    prices,
    ticker,
    event_dates,
    horizon=FORECAST_HORIZON,
):
    """
    Forecast future cumulative performance using the historical
    event-conditioned forward paths.

    For each historical event:
        Day 0 = event date price
        Day 1...Day 30 = future price relative to event date

    The forecast is the historical median/mean event path.
    Prediction intervals use historical event-path percentiles.
    """

    series = prices[ticker].dropna()

    event_paths = []

    for event_date in event_dates:

        if event_date not in series.index:
            continue

        start_position = series.index.get_loc(event_date)

        if (
            start_position + horizon
            >= len(series)
        ):
            continue

        start_price = float(
            series.iloc[start_position]
        )

        path = []

        for day in range(1, horizon + 1):

            future_price = float(
                series.iloc[start_position + day]
            )

            cumulative_return = (
                future_price / start_price - 1
            )

            path.append(
                cumulative_return
            )

        event_paths.append(path)

    if len(event_paths) < 2:
        return None

    matrix = np.array(event_paths)

    forecast_mean = np.nanmean(
        matrix,
        axis=0,
    )

    forecast_median = np.nanmedian(
        matrix,
        axis=0,
    )

    lower = np.nanpercentile(
        matrix,
        2.5,
        axis=0,
    )

    upper = np.nanpercentile(
        matrix,
        97.5,
        axis=0,
    )

    return {
        "matrix": matrix,
        "mean": forecast_mean,
        "median": forecast_median,
        "lower": lower,
        "upper": upper,
        "event_count": len(event_paths),
    }


# ============================================================
# TYPICAL PEAK ANALYSIS
# ============================================================

def calculate_peak_statistics(
    prices,
    ticker,
    event_dates,
    horizon=FORECAST_HORIZON,
):
    series = prices[ticker].dropna()

    peak_days = []
    peak_returns = []

    for event_date in event_dates:

        if event_date not in series.index:
            continue

        start_position = series.index.get_loc(event_date)

        if (
            start_position + horizon
            >= len(series)
        ):
            continue

        start_price = float(
            series.iloc[start_position]
        )

        future_prices = series.iloc[
            start_position + 1:
            start_position + horizon + 1
        ]

        returns = (
            future_prices / start_price - 1
        )

        if returns.empty:
            continue

        peak_position = int(
            np.nanargmax(
                returns.values
            )
        )

        peak_return = float(
            returns.iloc[peak_position]
        )

        peak_day = peak_position + 1

        peak_days.append(peak_day)
        peak_returns.append(peak_return)

    if not peak_days:
        return None

    return {
        "median_peak_day": float(
            np.median(peak_days)
        ),
        "mean_peak_day": float(
            np.mean(peak_days)
        ),
        "p25_peak_day": float(
            np.percentile(peak_days, 25)
        ),
        "p75_peak_day": float(
            np.percentile(peak_days, 75)
        ),
        "median_peak_return": float(
            np.median(peak_returns)
        ),
    }


# ============================================================
# CHARTS
# ============================================================

def create_recent_movement_chart(
    prices,
    leader,
    followers,
    days=60,
):
    selected = [
        ticker
        for ticker in [leader] + followers
        if ticker in prices.columns
    ]

    recent = prices[
        selected
    ].dropna(how="all").tail(days)

    normalized = recent / recent.iloc[0] * 100

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
        yaxis_title="Indexed price (start = 100)",
        xaxis_title="Date",
        hovermode="x unified",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


def create_historical_event_chart(
    prices,
    leader,
    ticker,
    event_dates,
    forward_days=TRADING_MONTH,
):
    series = prices[
        [leader, ticker]
    ].dropna()

    paths_leader = []
    paths_follower = []

    for event_date in event_dates:

        if event_date not in series.index:
            continue

        start_position = series.index.get_loc(event_date)

        if (
            start_position + forward_days
            >= len(series)
        ):
            continue

        leader_start = series.iloc[
            start_position
        ][leader]

        follower_start = series.iloc[
            start_position
        ][ticker]

        leader_path = []
        follower_path = []

        for day in range(
            forward_days + 1
        ):

            lp = series.iloc[
                start_position + day
            ][leader]

            fp = series.iloc[
                start_position + day
            ][ticker]

            leader_path.append(
                lp / leader_start * 100
            )

            follower_path.append(
                fp / follower_start * 100
            )

        paths_leader.append(
            leader_path
        )

        paths_follower.append(
            follower_path
        )

    if not paths_leader:
        return None

    leader_mean = np.mean(
        np.array(paths_leader),
        axis=0,
    )

    follower_mean = np.mean(
        np.array(paths_follower),
        axis=0,
    )

    fig = go.Figure()

    x = list(
        range(forward_days + 1)
    )

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
            name=ticker,
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
        title="Average Historical Follow-Through",
        xaxis_title="Trading days after historical leader event",
        yaxis_title="Indexed price",
        hovermode="x unified",
        height=480,
    )

    return fig


def create_forecast_chart(
    prices,
    ticker,
    forecast,
):
    series = prices[ticker].dropna()

    current_price = float(
        series.iloc[-1]
    )

    history = series.tail(60)

    future_days = np.arange(
        1,
        len(forecast["mean"]) + 1,
    )

    last_date = history.index[-1]

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(future_days),
    )

    mean_price = (
        current_price
        * (1 + forecast["mean"])
    )

    median_price = (
        current_price
        * (1 + forecast["median"])
    )

    lower_price = (
        current_price
        * (1 + forecast["lower"])
    )

    upper_price = (
        current_price
        * (1 + forecast["upper"])
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

    # Forecast central path
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=mean_price,
            mode="lines",
            name="Expected path",
            line=dict(
                width=3,
                dash="dash",
            ),
        )
    )

    # Median historical path
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=median_price,
            mode="lines",
            name="Median historical path",
            line=dict(
                width=2,
                dash="dot",
            ),
        )
    )

    # 95% lower
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower_price,
            mode="lines",
            name="95% lower bound",
            line=dict(
                width=1,
                dash="dot",
            ),
        )
    )

    # 95% upper
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper_price,
            mode="lines",
            name="95% upper bound",
            line=dict(
                width=1,
                dash="dot",
            ),
            fill="tonexty",
            fillcolor="rgba(100,116,139,0.10)",
        )
    )

    fig.update_layout(
        title=f"30-Trading-Day Scenario: {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=520,
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

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        'Market Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'S&P 500 | Historical momentum, leader–follower relationships, '
        'risk and forward-looking scenarios'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # LOAD S&P 500
    # --------------------------------------------------------

    try:
        sp500 = get_sp500_constituents()

    except Exception as exc:

        st.error(
            "Unable to load the S&P 500 constituent list."
        )

        st.exception(exc)

        st.stop()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.markdown(
        "## Research Setup"
    )

    st.sidebar.caption(
        "Select a market leader and define the historical "
        "momentum conditions to investigate."
    )

    leader_options = (
        sp500["Ticker"]
        + " — "
        + sp500["Security"]
    ).tolist()

    selected_label = st.sidebar.selectbox(
        "Leading / booming stock",
        leader_options,
        index=(
            leader_options.index(
                "NVDA — NVIDIA"
            )
            if "NVDA — NVIDIA" in leader_options
            else 0
        ),
    )

    leader = selected_label.split(" — ")[0]

    leader_company = sp500.loc[
        sp500["Ticker"] == leader,
        "Security",
    ].iloc[0]

    leader_sector = sp500.loc[
        sp500["Ticker"] == leader,
        "GICS Sector",
    ].iloc[0]

    st.sidebar.markdown(
        "---"
    )

 tolerance_pct = st.sidebar.slider(
    "Historical event tolerance",
    min_value=2,
    max_value=15,
    value=5,
    step=1,
    format="%d%%",
    help=(
        "Historical leader events are selected when the "
        "21-trading-day return falls within this percentage-point "
        "range of the current leader's momentum."
    ),
)

minimum_event_return_pct = st.sidebar.slider(
    "Minimum historical leader return",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    format="%d%%",
)

    min_events = st.sidebar.slider(
        "Minimum historical events",
        min_value=2,
        max_value=8,
        value=3,
        step=1,
    )

    lag_only = st.sidebar.checkbox(
        "Prefer stocks currently lagging the leader",
        value=True,
        help=(
            "The ranking will prioritize stocks whose current "
            "21-day return is below that of the selected leader."
        ),
    )

    top_n = st.sidebar.slider(
        "Number of followers to display",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Data source: Yahoo Finance / S&P 500 constituent data"
    )

    if st.sidebar.button(
        "Clear cached data"
    ):
        st.cache_data.clear()
        st.rerun()

    # --------------------------------------------------------
    # DOWNLOAD PRICES
    # --------------------------------------------------------

    progress = st.empty()

    with st.spinner(
        "Loading S&P 500 historical market data..."
    ):
        try:

            all_tickers = sp500[
                "Ticker"
            ].tolist()

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

    progress.empty()

    if prices.empty:
        st.error(
            "No market data was returned."
        )
        st.stop()

    # Keep only tickers with enough data.
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

    # --------------------------------------------------------
    # CURRENT LEADER SIGNAL
    # --------------------------------------------------------

    try:

        leader_signal = calculate_current_leader_signal(
            prices,
            leader,
        )

    except Exception as exc:

        st.error(
            "Unable to calculate the current leader signal."
        )

        st.exception(exc)

        st.stop()

    current_leader_return = leader_signal[
        "return"
    ]

    signal_date = leader_signal[
        "date"
    ]

    current_price = leader_signal[
        "price"
    ]

    # --------------------------------------------------------
    # HEADER SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="research-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        ### Current Market Signal

        **{leader_company} ({leader})** is the selected market leader.

        The model uses the leader's latest **21-trading-day return**
        as the momentum signal and searches the historical record for
        periods with similar performance.
        """
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:
        st.metric(
            "Leader",
            leader,
        )

    with metric2:
        st.metric(
            "Current price",
            f"${current_price:,.2f}",
        )

    with metric3:
        st.metric(
            "21D return",
            signed_pct(
                current_leader_return
            ),
        )

    with metric4:
        st.metric(
            "Sector",
            leader_sector,
        )

    with metric5:
        st.metric(
            "Signal date",
            signal_date.strftime(
                "%d %b %Y"
            ),
        )

    # --------------------------------------------------------
    # WARNING FOR WEAK CURRENT SIGNAL
    # --------------------------------------------------------

    if current_leader_return <= 0:

        st.warning(
            f"The selected leader currently has a "
            f"{signed_pct(current_leader_return)} "
            f"21-day return. The historical follower model is "
            "designed primarily for positive momentum regimes."
        )

    # --------------------------------------------------------
    # HISTORICAL EVENTS
    # --------------------------------------------------------

    with st.spinner(
        "Searching for historical periods with similar leader momentum..."
    ):

        event_dates = identify_leader_events(
            prices=prices,
            leader=leader,
            target_return=current_leader_return,
            tolerance=tolerance,
            minimum_return=minimum_event_return,
            cooldown_days=TRADING_MONTH,
        )

    # Do not include the most recent event if it is effectively today.
    if len(event_dates) > 0:

        event_dates = [
            dt
            for dt in event_dates
            if (
                signal_date - dt
            ).days > TRADING_MONTH
        ]

    if len(event_dates) < min_events:

        st.warning(
            f"Only {len(event_dates)} comparable historical event(s) "
            f"were found. Consider widening the event tolerance or "
            f"reducing the minimum leader return."
        )

    # --------------------------------------------------------
    # SHOW EVENT INFORMATION
    # --------------------------------------------------------

    event_col1, event_col2, event_col3 = st.columns(3)

    with event_col1:
        st.metric(
            "Historical analogues",
            len(event_dates),
        )

    with event_col2:
        st.metric(
            "Target leader return",
            signed_pct(
                current_leader_return
            ),
        )

    with event_col3:
        st.metric(
            "Event tolerance",
            f"±{tolerance * 100:.0f} pp",
        )

    if event_dates:

        event_table = pd.DataFrame(
            {
                "Historical event date": [
                    dt.strftime("%d %b %Y")
                    for dt in event_dates
                ]
            }
        )

        with st.expander(
            "View historical leader events"
        ):
            st.dataframe(
                event_table,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # EVENT STUDY
    # --------------------------------------------------------

    followers = [
        ticker
        for ticker in available_tickers
        if ticker != leader
    ]

    with st.spinner(
        "Analyzing historical follower relationships..."
    ):

        event_study = create_event_study(
            prices=prices,
            leader=leader,
            event_dates=event_dates,
            followers=followers,
            forward_days=TRADING_MONTH,
        )

    if event_study.empty:

        st.error(
            "No usable follower observations were found "
            "for the selected historical events."
        )

        st.stop()

    follower_stats = (
        calculate_follower_event_statistics(
            event_study
        )
    )

    if follower_stats.empty:

        st.error(
            "Unable to generate follower statistics."
        )

        st.stop()

    # --------------------------------------------------------
    # CURRENT STOCK METRICS
    # --------------------------------------------------------

    current_metrics = (
        calculate_current_stock_metrics(
            prices,
            leader,
            available_tickers,
        )
    )

    follower_stats = follower_stats.merge(
        current_metrics,
        on="ticker",
        how="left",
    )

    # Company / sector details
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

    follower_stats = follower_stats.drop(
        columns=["Ticker"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # RISK + CORRELATION
    # --------------------------------------------------------

    risk_records = []

    for ticker in follower_stats["ticker"]:

        stats = calculate_stock_statistics(
            prices,
            ticker,
        )

        corr = calculate_correlation(
            prices,
            leader,
            ticker,
        )

        risk_records.append(
            {
                "ticker": ticker,
                "annualized_volatility": stats[
                    "annualized_volatility"
                ],
                "downside_volatility": stats[
                    "downside_volatility"
                ],
                "max_drawdown": stats[
                    "max_drawdown"
                ],
                "sharpe": stats[
                    "sharpe"
                ],
                "annualized_return": stats[
                    "annualized_return"
                ],
                "correlation": corr,
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

    # --------------------------------------------------------
    # RANKING FILTER
    # --------------------------------------------------------

    ranking_df = follower_stats.copy()

    ranking_df = ranking_df[
        ranking_df["events"] >= min_events
    ].copy()

    # Prefer current laggers.
    if lag_only:

        lagging_df = ranking_df[
            ranking_df["vs_leader"] < 0
        ].copy()

        # Fall back to full universe if too few laggers survive.
        if len(lagging_df) >= top_n:
            ranking_df = lagging_df

    ranking_df["research_score"] = ranking_df.apply(
        calculate_follower_score,
        axis=1,
    )

    ranking_df = ranking_df.sort_values(
        [
            "research_score",
            "prob_beats_leader_1pp",
            "mean_excess",
        ],
        ascending=False,
    ).reset_index(drop=True)

    top_followers = ranking_df.head(
        top_n
    ).copy()

    # --------------------------------------------------------
    # LEADER vs FOLLOWER SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Potential Historical Followers'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Ranking combines historical follow-through, relative "
        "performance and current lagging characteristics. "
        "It is a research ranking, not an investment recommendation."
    )

    if top_followers.empty:

        st.warning(
            "No stocks met the current ranking criteria."
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

        # Formatting copy
        formatted_df = display_df.copy()

        percentage_columns = [
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

        for col in percentage_columns:
            formatted_df[col] = (
                formatted_df[col]
                .apply(lambda x: pct(x))
            )

        formatted_df["Correlation"] = (
            formatted_df["Correlation"]
            .apply(
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "—"
                )
            )
        )

        formatted_df["Research Score"] = (
            formatted_df["Research Score"]
            .apply(
                lambda x: (
                    f"{x:.1f}"
                    if pd.notna(x)
                    else "—"
                )
            )
        )

        st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # TOP FOLLOWER DETAILS
    # --------------------------------------------------------

    if not top_followers.empty:

        selected_follower = st.selectbox(
            "Select a follower for detailed analysis",
            top_followers["ticker"].tolist(),
            format_func=lambda x: (
                f"{x} — "
                f"{top_followers.loc[top_followers['ticker'] == x, 'Security'].iloc[0]}"
            ),
        )

        follower_row = top_followers[
            top_followers["ticker"]
            == selected_follower
        ].iloc[0]

        company_name = follower_row[
            "Security"
        ]

        follower_sector = follower_row[
            "GICS Sector"
        ]

        # ----------------------------------------------------
        # FOLLOWER HEADER
        # ----------------------------------------------------

        st.markdown(
            f'<div class="section-title">'
            f'Detailed View — {selected_follower}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"{company_name} | {follower_sector}"
        )

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
                "P(Beat +1pp)",
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
            st.metric(
                "Correlation",
                (
                    f"{follower_row['correlation']:.2f}"
                    if pd.notna(
                        follower_row["correlation"]
                    )
                    else "—"
                ),
            )

        # ----------------------------------------------------
        # RISK PROFILE
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
                "Annualized Volatility",
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
            sharpe_value = follower_row[
                "sharpe"
            ]

            st.metric(
                "Sharpe-like Ratio",
                (
                    f"{sharpe_value:.2f}"
                    if pd.notna(sharpe_value)
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
        # HISTORICAL EVENT CHART
        # ----------------------------------------------------

        event_chart = create_historical_event_chart(
            prices=prices,
            leader=leader,
            ticker=selected_follower,
            event_dates=event_dates,
            forward_days=TRADING_MONTH,
        )

        if event_chart is not None:

            st.plotly_chart(
                event_chart,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '30-Trading-Day Historical Scenario'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "The forward scenario is based on the historical "
            "paths observed after comparable leader events. "
            "The shaded range represents the 2.5th–97.5th "
            "percentile of historical event outcomes."
        )

        forecast = create_event_based_forecast(
            prices=prices,
            ticker=selected_follower,
            event_dates=event_dates,
            horizon=FORECAST_HORIZON,
        )

        if forecast is None:

            st.warning(
                "Not enough historical events are available "
                "to construct a 30-day scenario."
            )

        else:

            current_follower_price = float(
                prices[
                    selected_follower
                ].dropna().iloc[-1]
            )

            predicted_return_30d = (
                forecast["mean"][-1]
            )

            median_return_30d = (
                forecast["median"][-1]
            )

            lower_return_30d = (
                forecast["lower"][-1]
            )

            upper_return_30d = (
                forecast["upper"][-1]
            )

            predicted_price = (
                current_follower_price
                * (
                    1
                    + predicted_return_30d
                )
            )

            median_price = (
                current_follower_price
                * (
                    1
                    + median_return_30d
                )
            )

            lower_price = (
                current_follower_price
                * (
                    1
                    + lower_return_30d
                )
            )

            upper_price = (
                current_follower_price
                * (
                    1
                    + upper_return_30d
                )
            )

            forecast_cols = st.columns(5)

            with forecast_cols[0]:
                st.metric(
                    "Current Price",
                    f"${current_follower_price:,.2f}",
                )

            with forecast_cols[1]:
                st.metric(
                    "30D Expected Return",
                    signed_pct(
                        predicted_return_30d
                    ),
                )

            with forecast_cols[2]:
                st.metric(
                    "30D Median Return",
                    signed_pct(
                        median_return_30d
                    ),
                )

            with forecast_cols[3]:
                st.metric(
                    "95% Lower",
                    signed_pct(
                        lower_return_30d
                    ),
                )

            with forecast_cols[4]:
                st.metric(
                    "95% Upper",
                    signed_pct(
                        upper_return_30d
                    ),
                )

            st.plotly_chart(
                create_forecast_chart(
                    prices=prices,
                    ticker=selected_follower,
                    forecast=forecast,
                ),
                use_container_width=True,
            )

            # ------------------------------------------------
            # PEAK TIMING
            # ------------------------------------------------

            peak_stats = calculate_peak_statistics(
                prices=prices,
                ticker=selected_follower,
                event_dates=event_dates,
                horizon=FORECAST_HORIZON,
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
                        "Median peak day",
                        f"Day {peak_stats['median_peak_day']:.0f}",
                    )

                with peak_cols[1]:
                    st.metric(
                        "Typical range",
                        (
                            f"Day {peak_stats['p25_peak_day']:.0f}"
                            f"–"
                            f"Day {peak_stats['p75_peak_day']:.0f}"
                        ),
                    )

                with peak_cols[2]:
                    st.metric(
                        "Median peak return",
                        signed_pct(
                            peak_stats[
                                "median_peak_return"
                            ]
                        ),
                    )

                with peak_cols[3]:
                    st.metric(
                        "Events used",
                        len(event_dates),
                    )

                st.info(
                    "Historical peak timing indicates when similar "
                    "events tended to reach their maximum observed "
                    "forward return. It should not be interpreted "
                    "as a reliable future sell signal."
                )

    # --------------------------------------------------------
    # RECENT MARKET MOVEMENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Recent Market Movement'
        '</div>',
        unsafe_allow_html=True,
    )

    recent_followers = (
        top_followers["ticker"].tolist()
        if not top_followers.empty
        else []
    )

    st.plotly_chart(
        create_recent_movement_chart(
            prices=prices,
            leader=leader,
            followers=recent_followers,
            days=60,
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # RESEARCH METHODOLOGY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Methodology'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "How the analysis works"
    ):

        st.markdown(
            f"""
            **1. Leader signal**

            The selected S&P 500 stock is treated as the current
            market leader. Its latest **{TRADING_MONTH}-trading-day
            return** is used as the current momentum signal.

            **2. Historical analogues**

            The application searches the historical price series for
            periods where the leader's {TRADING_MONTH}-day return was
            similar to today's return.

            **3. Follower analysis**

            For each other S&P 500 constituent, the application measures
            its subsequent {TRADING_MONTH}-day return after those
            historical leader events.

            **4. Conditional probabilities**

            The dashboard calculates:

            - Probability the follower produced a positive return.
            - Probability the follower outperformed the leader by at least
              1 percentage point.
            - Probability the follower stayed within ±1 percentage point
              of the leader.
            - Probability the follower lagged the leader by at least
              1 percentage point.

            **5. Risk**

            Historical volatility, downside volatility, maximum drawdown,
            correlation and a simple Sharpe-like statistic are calculated.

            **6. Forward scenario**

            The 30-day scenario is based on the distribution of historical
            follower price paths following comparable leader events.

            The shaded range is a historical prediction interval, not a
            guarantee of future performance.

            **7. Peak timing**

            Historical event paths are examined to identify the typical
            trading day on which the follower reached its maximum observed
            return during the subsequent 30-trading-day window.
            """
        )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    with st.expander(
        "Important research limitations"
    ):

        st.markdown(
            """
            **Survivorship bias**

            The application currently uses today's S&P 500 constituents
            for the historical analysis. Companies that left the index in
            earlier periods are therefore not represented.

            **No causality**

            A historical leader–follower relationship represents statistical
            association. It does not demonstrate that movement in the leader
            causes movement in the follower.

            **Historical relationships can change**

            Correlations, momentum relationships and market regimes can
            change over time.

            **Prediction interval**

            The 95% range is based on the historical distribution of comparable
            events. It should not be interpreted as a 95% probability that
            the future price will remain within the range.

            **Research use**

            This dashboard is designed as an analytical research prototype
            and should not be treated as personal financial advice or as an
            automated trading system.
            """
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

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
            "current_21d_return",
            "vs_leader",
            "events",
            "mean_return",
            "median_return",
            "volatility",
            "mean_excess",
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
                    if col in top_followers.columns
                ]
            ]
            .copy()
        )

        csv = export_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download follower research CSV",
            data=csv,
            file_name=(
                f"{leader}_follower_analysis.csv"
            ),
            mime="text/csv",
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("---")

    data_timestamp = datetime.now().strftime(
        "%d %b %Y %H:%M"
    )

    st.caption(
        f"Research prototype | S&P 500 | "
        f"Analysis generated {data_timestamp}"
    )

    st.caption(
        "For educational and research purposes only. "
        "Historical analysis does not guarantee future performance."
    )


if __name__ == "__main__":
    main()
