import ccxt
import pandas as pd
import ta
import time
import requests
from flask import Flask
import threading
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import tarfile

# ==============================
# TELEGRAM SETTINGS
# ==============================
TOKEN = "8249716998:AAEM8PmCb9fia4UgagdbOClMwNOD_TBdqz4"
CHAT_ID = "6094849602"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": CHAT_ID,
        "text": msg
    })

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
        print("⚠️ No models found, training will start")

# ==============================
# DATA
# ==============================
def get_data(tf):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=120),
                      columns=['time','open','high','low','close','volume'])

    macd = ta.trend.MACD(df['close'])

    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema'] = df['close'].ewm(span=20).mean()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    return df

# ==============================
# FEATURES
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
        return None

    features = get_features(df1)
    buy_p, sell_p = ai_decision(features)

    last = df1.iloc[-1]

    rsi = last['rsi']
    macd = last['macd']
    macd_signal = last['macd_signal']

    price = last['close']
    atr = last['atr']

    print(f"[DEBUG] Buy:{buy_p:.2f} Sell:{sell_p:.2f} RSI:{rsi:.2f} MACD:{macd:.2f}")

    signal = None

    # ✅ HIGH ACCURACY FILTER
    if buy_p > 0.6 and rsi > 55 and macd > macd_signal:
        signal = "BUY"

    elif sell_p > 0.6 and rsi < 45 and macd < macd_signal:
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

            model_xgb = XGBClassifier(n_estimators=80, max_depth=4, verbosity=0)
            model_rf = RandomForestClassifier(n_estimators=150)

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
    send_telegram("🚀 BOT LIVE (NO WEBHOOK MODE)")

    last_signal = ""
    last_signal_time = 0
    cooldown = 180  # 3 minutes

    while True:
        try:
            df1 = get_data('1m')
            result = check_signal(df1)

            if result:
                side, price, tp, sl, conf = result

                if side != last_signal and time.time() - last_signal_time > cooldown:

                    msg = (
                        f"📊 {side} SIGNAL\n\n"
                        f"💰 Price: {round(price,2)}\n"
                        f"🎯 TP: {round(tp,2)}\n"
                        f"🛑 SL: {round(sl,2)}\n\n"
                        f"🤖 Confidence: {round(conf*100,2)}%\n"
                        f"⏱ Time: {time.strftime('%H:%M:%S')}"
                    )

                    send_telegram(msg)

                    last_signal = side
                    last_signal_time = time.time()

            time.sleep(30)

        except Exception as e:
            print("Bot Error:", e)
            time.sleep(10)

# ==============================
# FLASK (FOR UPTIME ROBOT)
# ==============================
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Bot Running"

# ==============================
# START
# ==============================
if __name__ == "__main__":
    load_models()

    print("🚀 Starting Bot...")

    threading.Thread(target=run_bot, daemon=True).start()
    threading.Thread(target=train_ai, daemon=True).start()

    app.run(host="0.0.0.0", port=10000)
