# =========================================================
# STOCK PRICE PREDICTION PROJECT (APPLE INC. - AAPL)
# Steps 1 → 3 (including Sentiment and all 5 output files)
# =========================================================

import yfinance as yf
import pandas as pd
from textblob import TextBlob
from GoogleNews import GoogleNews
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# STEP 1: Collect Stock Data
# =========================================================
print("📥 Step 1: Collecting Apple stock data...")
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
data.reset_index(inplace=True)

# ✅ Fix MultiIndex issue (important)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.to_csv("stock_data.csv", index=False)
print("✅ Step 1 Completed: stock_data.csv saved successfully.\n")

# =========================================================
# STEP 2: Clean Data + Collect News Sentiment
# =========================================================
print("🧹 Step 2: Cleaning data and collecting news sentiment...")

# Fill missing values
data.fillna(method="ffill", inplace=True)

# ✅ Collect Apple-related news sentiment
print("📰 Collecting and analyzing Apple-related news sentiment...")
googlenews = GoogleNews(lang='en')
googlenews.search('Apple Stock')
news = googlenews.result()

news_df = pd.DataFrame(news)
if not news_df.empty:
    news_df["Date"] = pd.to_datetime(news_df["date"], errors="coerce")
    news_df["Sentiment"] = news_df["title"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    daily_sentiment = news_df.groupby(news_df["Date"].dt.date)["Sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"Date": "Date", "Sentiment": "Sentiment"}, inplace=True)
else:
    # fallback in case GoogleNews gives no data
    daily_sentiment = pd.DataFrame({"Date": data["Date"], "Sentiment": 0})

# Convert Date column formats for merging
data["Date"] = pd.to_datetime(data["Date"]).dt.date
daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"]).dt.date

# ✅ Merge stock and sentiment data
merged = pd.merge(data, daily_sentiment, on="Date", how="left")
merged["Sentiment"].fillna(0, inplace=True)
merged.to_csv("clean_data.csv", index=False)
print("✅ Step 2 Completed: clean_data.csv (with Sentiment) saved.\n")

# =========================================================
# STEP 3: Exploratory Data Analysis (EDA)
# =========================================================
print("📊 Step 3: Performing Exploratory Data Analysis...")

# 1️⃣ Price over time
plt.figure(figsize=(10, 5))
plt.plot(merged["Date"], merged["Close"], color='blue')
plt.title("Apple Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.tight_layout()
plt.savefig("price_plot.png")
plt.close()

# 2️⃣ Price distribution
plt.figure(figsize=(7, 4))
sns.histplot(merged["Close"], bins=30, kde=True)
plt.title("Distribution of Apple Stock Prices")
plt.xlabel("Close Price ($)")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

# 3️⃣ Sentiment vs Price
plt.figure(figsize=(10, 5))
plt.scatter(merged["Sentiment"], merged["Close"], color='green', alpha=0.6)
plt.title("Relationship Between News Sentiment and Stock Price")
plt.xlabel("Sentiment Score")
plt.ylabel("Close Price ($)")
plt.tight_layout()
plt.savefig("sentiment_plot.png")
plt.close()

# 4️⃣ Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(merged[["Open", "High", "Low", "Close", "Volume", "Sentiment"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("✅ Step 3 Completed Successfully!")
print("\n📂 Files generated:")
print("📄 stock_data.csv")
print("📄 clean_data.csv")
print("📊 price_plot.png")
print("📈 price_distribution.png")
print("💬 sentiment_plot.png")
print("🔥 correlation_heatmap.png")


