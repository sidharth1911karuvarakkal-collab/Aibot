import ccxt
import pandas as pd
import ta
import time
import requests
from flask import Flask
import threading
from datetime import datetime
import pytz
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import os
import subprocess

# ==============================
# 🔧 AUTO TEXTBLOB SETUP (RUNS ONCE)
# ==============================
if not os.path.exists("textblob_data_installed"):
    try:
        subprocess.run(["python", "-m", "textblob.download_corpora"])
        open("textblob_data_installed", "w").close()
        print("✅ TextBlob data installed")
    except Exception as e:
        print("TextBlob setup error:", e)

# ==============================
# 🔑 SETTINGS
# ==============================
TOKEN = "8249716998:AAEM8PmCb9fia4UgagdbOClMwNOD_TBdqz4"
CHAT_ID = "6094849602"

def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg})
    except:
        pass

# ==============================
# 📊 EXCHANGE
# ==============================
exchange = ccxt.okx()
symbol = 'BTC/USDT'

# ==============================
# 🤖 MODELS
# ==============================
model_xgb = None
model_rf = None

def load_models():
    global model_xgb, model_rf
    try:
        model_xgb = joblib.load("model_xgb.pkl")
        model_rf = joblib.load("model_rf.pkl")
        print("✅ Models Loaded")
    except:
        print("⚠️ No models yet (will train automatically)")

# ==============================
# 📈 DATA FUNCTION
# ==============================
def get_data(tf):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=120),
                      columns=['time','open','high','low','close','volume'])

    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['ema'] = df['close'].ewm(span=20).mean()

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    return df

# ==============================
# 🤖 FEATURES (MULTI TF)
# ==============================
def get_features(df1, df5, df15):
    l1, l5, l15 = df1.iloc[-1], df5.iloc[-1], df15.iloc[-1]

    return [[
        l1['rsi'], l5['rsi'], l15['rsi'],
        l1['macd'], l5['macd'], l15['macd'],
        l1['ema'], l5['ema'], l15['ema'],
        l1['atr'], l1['adx']
    ]]

# ==============================
# 🤖 AI DECISION (ENSEMBLE)
# ==============================
def ai_decision(features):
    p1 = model_xgb.predict_proba(features)[0]
    p2 = model_rf.predict_proba(features)[0]

    buy_prob = (p1[1] + p2[1]) / 2
    sell_prob = (p1[0] + p2[0]) / 2

    return buy_prob, sell_prob

# ==============================
# 📰 FREE CRYPTO NEWS SENTIMENT
# ==============================
last_news_time = 0
cached_sentiment = 0

def get_news_sentiment():
    global last_news_time, cached_sentiment

    try:
        # Cache for 10 minutes
        if time.time() - last_news_time < 600:
            return cached_sentiment

        url = "https://cryptocurrency.cv/api/news"
        data = requests.get(url).json()

        articles = data[:10]

        score = 0
        count = 0

        for a in articles:
            title = a.get("title", "")
            polarity = TextBlob(title).sentiment.polarity
            score += polarity
            count += 1

        sentiment = score / count if count > 0 else 0

        cached_sentiment = sentiment
        last_news_time = time.time()

        return sentiment

    except Exception as e:
        print("News Error:", e)
        return 0

# ==============================
# 🤖 SIGNAL LOGIC
# ==============================
def check_signal(df1, df5, df15):
    if model_xgb is None or model_rf is None:
        return None

    features = get_features(df1, df5, df15)
    buy_p, sell_p = ai_decision(features)

    sentiment = get_news_sentiment()

    if sentiment > 0.1:
        news_bias = "BUY"
    elif sentiment < -0.1:
        news_bias = "SELL"
    else:
        news_bias = "NEUTRAL"

    signal = None

    if buy_p > 0.7:
        signal = "BUY"
    elif sell_p > 0.7:
        signal = "SELL"

    # News filter
    if signal == "BUY" and news_bias == "SELL":
        signal = None
    if signal == "SELL" and news_bias == "BUY":
        signal = None

    # Momentum filter
    rsi = df1.iloc[-1]['rsi']
    if signal == "BUY" and rsi < 55:
        signal = None
    if signal == "SELL" and rsi > 45:
        signal = None

    price = df1.iloc[-1]['close']
    atr = df1.iloc[-1]['atr']

    if signal == "BUY":
        tp = price + 2 * atr
        sl = price - 1.5 * atr
    elif signal == "SELL":
        tp = price - 2 * atr
        sl = price + 1.5 * atr
    else:
        return None

    return signal, price, tp, sl, max(buy_p, sell_p), sentiment

# ==============================
# 🤖 TRAIN AI
# ==============================
def train_ai():
    global model_xgb, model_rf

    while True:
        try:
            print("🤖 Training AI...")

            df = get_data('1m')
            df['future'] = df['close'].shift(-3)
            df['target'] = (df['future'] > df['close']).astype(int)
            df = df.dropna()

            X, y = [], []

            for i in range(len(df)-1):
                X.append([
                    df.iloc[i]['rsi'],
                    df.iloc[i]['macd'],
                    df.iloc[i]['ema'],
                    df.iloc[i]['atr'],
                    df.iloc[i]['adx']
                ])
                y.append(df.iloc[i]['target'])

            model_xgb = XGBClassifier(n_estimators=50, max_depth=3, verbosity=0)
            model_rf = RandomForestClassifier(n_estimators=100)

            model_xgb.fit(X, y)
            model_rf.fit(X, y)

            joblib.dump(model_xgb, "model_xgb.pkl")
            joblib.dump(model_rf, "model_rf.pkl")

            send_telegram("🤖 AI Updated (Auto + News + Ensemble)")

        except Exception as e:
            print("AI Error:", e)

        time.sleep(86400)

# ==============================
# 🤖 BOT LOOP
# ==============================
def run_bot():
    send_telegram("🚀 AI Bot LIVE (Auto Setup Enabled)")

    last_signal = ""

    while True:
        try:
            df1 = get_data('1m')
            df5 = get_data('5m')
            df15 = get_data('15m')

            result = check_signal(df1, df5, df15)

            if result:
                side, price, tp, sl, conf, sentiment = result

                if side != last_signal:
                    send_telegram(f"""
{side} SIGNAL

💰 Price: {price}
🎯 TP: {tp}
🛑 SL: {sl}

🤖 AI Confidence: {round(conf*100,2)}%
📰 Sentiment: {round(sentiment,3)}
""")
                    last_signal = side

            time.sleep(30)

        except Exception as e:
            print("Bot Error:", e)
            time.sleep(10)

# ==============================
# 🌐 FLASK
# ==============================
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Bot Running (Auto Setup Active)"

# ==============================
# ▶️ START
# ==============================
if __name__ == "__main__":
    load_models()
    threading.Thread(target=run_bot).start()
    threading.Thread(target=train_ai).start()
    app.run(host="0.0.0.0", port=10000)
