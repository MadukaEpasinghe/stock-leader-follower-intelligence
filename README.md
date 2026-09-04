# Stock Leader–Follower Intelligence Dashboard

A data-driven investment research dashboard built with Python and Streamlit to identify potential stock opportunities based on historical leader–follower relationships.

The application allows users to select a booming/leading stock and analyze historical periods when similar momentum occurred. It then identifies stocks that historically performed alongside or after the leader and ranks potential followers based on historical performance, conditional probabilities, risk, and expected future returns.

## Key Features

- 🌍 Support for multiple global stock market indices
- 📈 Identify and analyze stock momentum
- 🔎 Detect historical periods similar to the current leader's performance
- 🔗 Identify stocks that historically followed the selected leader
- 📊 Calculate conditional follower probabilities
- 📉 Analyze volatility, drawdowns, correlation, and other risk measures
- 🔮 Forecast potential performance over the next 30 trading days
- 📐 Generate 95% prediction intervals
- ⏱️ Analyze typical timing of historical peaks
- 📊 Visualize recent, historical, and forecast price movements
- 📥 Export analysis results to CSV
- 🌐 Interactive Streamlit web dashboard

## Methodology

The analysis uses historical daily stock prices and a 21-trading-day period as an approximation of one month.

For a selected leader stock, the application searches historical data for periods where the stock experienced similar positive momentum. It then evaluates how other stocks performed following those historical leader events.

Follower stocks are ranked using measures such as:

- Probability of positive 21-day return
- Probability of outperforming the leader
- Probability of achieving at least 1 percentage point higher growth than the leader
- Historical average return
- Volatility
- Downside risk
- Maximum drawdown
- Correlation with the leader
- Historical peak timing

The forecasting component uses historical event-conditioned performance to estimate the expected return over the next 30 trading days together with an uncertainty range.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- yfinance
- Plotly
- SciPy
- Requests

## Disclaimer

### Check with a live demo
https://stock-leader-follower-intelligence-zsrhrbcal5ycuehv344jht.streamlit.app/ 
This project is intended for educational, research, and investment-analysis purposes. Historical relationships and statistical patterns do not guarantee future performance. Forecasts and probabilities should not be interpreted as financial advice or guaranteed investment outcomes.
