import ccxt
import pandas as pd
import ta
import time
import requests
from flask import Flask, send_file
import threading
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import os
import subprocess
import tarfile

# ==============================
# TEXTBLOB SETUP
# ==============================
if not os.path.exists("textblob_data_installed"):
    try:
        subprocess.run(["python", "-m", "textblob.download_corpora"])
        open("textblob_data_installed", "w").close()
        print("✅ TextBlob data installed")
    except Exception as e:
        print("TextBlob error:", e)

# ==============================
# TELEGRAM SETTINGS
# ==============================
TOKEN = "8249716998:AAEM8PmCb9fia4UgagdbOClMwNOD_TBdqz4"
CHAT_ID = "6094849602"

def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg}
        )
    except Exception as e:
        print("Telegram Error:", e)

# ==============================
# EXCHANGE
# ==============================
exchange = ccxt.okx()
symbol = 'BTC/USDT'

# ==============================
# MODELS
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
        print("⚠️ No models yet (training will start)")

# ==============================
# DATA
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
# FEATURES (FIXED - 5 FEATURES)
# ==============================
def get_features(df1):
    l1 = df1.iloc[-1]
    return [[
        l1['rsi'],
        l1['macd'],
        l1['ema'],
        l1['atr'],
        l1['adx']
    ]]

# ==============================
# AI DECISION
# ==============================
def ai_decision(features):
    p1 = model_xgb.predict_proba(features)[0]
    p2 = model_rf.predict_proba(features)[0]
    buy_prob = (p1[1] + p2[1]) / 2
    sell_prob = (p1[0] + p2[0]) / 2
    return buy_prob, sell_prob

# ==============================
# SIGNAL LOGIC
# ==============================
def check_signal(df1):
    if model_xgb is None or model_rf is None:
        print("❌ Models not loaded")
        return None

    features = get_features(df1)
    buy_p, sell_p = ai_decision(features)

    price = df1.iloc[-1]['close']
    atr = df1.iloc[-1]['atr']

    print(f"[DEBUG] Buy: {buy_p:.2f} | Sell: {sell_p:.2f} | Price: {price}")

    signal = None

    # LOWERED THRESHOLD (TEST MODE)
    if buy_p > 0.5:
        signal = "BUY"
    elif sell_p > 0.5:
        signal = "SELL"

    if not signal:
        return None

    if signal == "BUY":
        tp = price + 2 * atr
        sl = price - 1.5 * atr
    else:
        tp = price - 2 * atr
        sl = price + 1.5 * atr

    return signal, price, tp, sl, max(buy_p, sell_p)

# ==============================
# TRAIN AI
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

            with tarfile.open("model.pkg", "w:gz") as tar:
                tar.add("model_xgb.pkl")
                tar.add("model_rf.pkl")

            load_models()

            send_telegram("🤖 AI Model Updated")

        except Exception as e:
            print("Training Error:", e)

        time.sleep(86400)

# ==============================
# BOT LOOP
# ==============================
def run_bot():
    send_telegram("🚀 BOT LIVE (Final Fixed)")
    last_signal = ""

    while True:
        try:
            df1 = get_data('1m')

            result = check_signal(df1)

            if result:
                side, price, tp, sl, conf = result

                if side != last_signal:
                    send_telegram(f"""
{side} SIGNAL

💰 Price: {price}
🎯 TP: {tp}
🛑 SL: {sl}

🤖 Confidence: {round(conf*100,2)}%
""")
                    last_signal = side

            time.sleep(30)

        except Exception as e:
            print("Bot Error:", e)
            time.sleep(10)

# ==============================
# FLASK
# ==============================
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Bot Running"

@app.route('/download_model_pkg')
def download():
    if os.path.exists("model.pkg"):
        return send_file("model.pkg")
    return "No model yet"

# ==============================
# START
# ==============================
if __name__ == "__main__":
    load_models()

    print("🚀 Starting Threads...")

    threading.Thread(target=run_bot, daemon=True).start()
    threading.Thread(target=train_ai, daemon=True).start()

    print("✅ Bot Started Successfully")

    app.run(host="0.0.0.0", port=10000, use_reloader=False)
