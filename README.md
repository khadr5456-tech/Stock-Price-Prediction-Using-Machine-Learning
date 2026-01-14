📊 Apple Stock Analysis & Prediction Dashboard

This project is a full-featured Streamlit web application designed to analyze and predict Apple (AAPL) stock prices. It integrates historical data analysis, interactive visualizations, and machine learning predictions into a single educational platform.

🔹 Features

Historical Data Analysis

Fetches stock data from Yahoo Finance using yfinance.

Displays raw historical data with sorting, searching, and filtering options.

Calculates key statistics like average closing price, highest/lowest price, trading volume, and volatility.

Exploratory Data Analysis (EDA)

Interactive line charts for closing prices and moving averages (7, 14, 30 days).

Distribution plots and boxplots for deeper insight into price behavior.

Volume analysis with bar charts and overlay of closing price trends.

Correlation insights and basic sentiment analysis using TextBlob.

Machine Learning Prediction

Random Forest Regressor predicts future stock prices based on historical data.

Supports three complexity levels:

Simple: basic features and fewer estimators.

Medium: additional technical features like daily return.

Complex: includes moving averages, volatility, and volume ratio.

Generates future predictions for a specified number of days.

Calculates metrics: RMSE, R², MAPE, and expected price change.

KPIs & Dashboard

Displays key indicators: current price, average predicted price, prediction accuracy, and number of days analyzed.

Highlights daily trend direction: Upward 🟢, Downward 🔴, Stable ⚪.

Interactive dashboard layout with tabs, metrics cards, and customizable options.

Downloadable Reports

Export historical data and future predictions as CSV files.

Interactive charts powered by Plotly for better visualization.

Customizable Interface

Styled with custom CSS for buttons, sidebar, tabs, and cards.

Users can select:

Date range for analysis.

Whether to display raw data, EDA, and predictions.

Prediction days and model complexity.

Educational Purpose

Designed for learning financial data analysis, machine learning, and Python visualization tools.

Includes guidance for interpreting trends, model accuracy, and investment recommendations (not financial advice).

🔹 How It Works

Data Loading

Uses yfinance to fetch historical Apple stock prices.

Data columns: Open, High, Low, Close, Volume.

Handles missing data and multi-level columns.

Feature Engineering

Creates moving averages (MA7, MA14, MA30) and daily returns.

Calculates price changes, volatility, volume ratio, and price ranges.

Drops NaN values after feature computation.

Training & Prediction

Selects features based on chosen complexity.

Splits data into training (80%) and testing (20%) sets.

Trains a RandomForestRegressor model.

Predicts both testing set prices and future stock prices.

Updates feature values iteratively for multi-day prediction.

Visualization

Interactive line charts for historical and predicted prices.

Histogram and box plots for price distribution.

Bar chart with secondary y-axis for volume and closing prices.

Future predictions include confidence intervals ±5%.

Dashboard Flow

Sidebar: Date input, analysis options, prediction settings.

Main page: Welcome page, KPIs, raw data, analysis, prediction results.

Tabs: Organized charts and insights for easier exploration.

🔹 Technologies Used

Python 3.x

Streamlit: For interactive web dashboard.

YFinance: To fetch stock data.

Pandas & NumPy: Data manipulation and calculations.

Matplotlib & Seaborn: Basic static visualization.

Plotly: Interactive charts.

TextBlob: Basic sentiment analysis.

Scikit-learn: Random Forest Regression and metrics.

🔹 Installation
git clone <repo_url>
cd apple-stock-dashboard
pip install -r requirements.txt
streamlit run app.py


Requirements include: streamlit, yfinance, pandas, numpy, matplotlib, seaborn, textblob, scikit-learn, plotly.

🔹 Usage

Open the sidebar to select date range.

Enable options: raw data, exploratory analysis, and/or price prediction.

Select prediction days and model complexity.

Click “Start Analysis” to run the dashboard.

Explore interactive charts, KPIs, and trend analysis.

Download data and predictions as CSV.

🔹 Notes

For educational purposes only – not financial advice.

Predictions are based on historical data and machine learning; markets can be volatile.

Daily updates recommended for more accurate predictions.
