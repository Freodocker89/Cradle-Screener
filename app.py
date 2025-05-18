import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import time
import datetime

st.set_page_config(layout="wide")

BITGET = ccxt.bitget()
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']

st.title("📊 Cradle Screener")
selected_timeframes = st.multiselect("Select Timeframes to Scan", TIMEFRAMES, default=['1h', '4h', '1d'])

# Auto-run toggle with unique key to avoid duplicate error
auto_run = st.checkbox("⏱️ Auto Run on Candle Close", key="auto_run_checkbox")

st.write("This screener shows valid Cradle setups detected on the last fully closed candle only.")

result_placeholder = st.container()
placeholder = st.empty()

# Auto-run logic based on selected timeframes

if 'last_run_minute' not in st.session_state:
    st.session_state.last_run_minute = -1

def should_auto_run():
    now = datetime.datetime.utcnow()
    total_minutes = now.hour * 60 + now.minute
    for tf in selected_timeframes:
        unit = tf[-1]
        value = int(tf[:-1])
        if unit == 'm':
            tf_minutes = value
        elif unit == 'h':
            tf_minutes = value * 60
        elif unit == 'd':
            tf_minutes = value * 60 * 24
        elif unit == 'w':
            tf_minutes = value * 60 * 24 * 7
        elif unit == 'M':
            continue
        else:
            continue

        if total_minutes % tf_minutes == 0 and now.minute != st.session_state.last_run_minute:
            st.session_state.last_run_minute = now.minute
            return True
    return False

# Track scan state to prevent refresh interruptions
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'last_run_minute' not in st.session_state:
    st.session_state.last_run_minute = -1

run_scan = False
manual_triggered = st.button("Run Screener", key="manual_run_button")

def should_auto_run():
    now = datetime.datetime.utcnow()
    total_minutes = now.hour * 60 + now.minute
    for tf in selected_timeframes:
        unit = tf[-1]
        value = int(tf[:-1])
        if unit == 'm': tf_minutes = value
        elif unit == 'h': tf_minutes = value * 60
        elif unit == 'd': tf_minutes = value * 60 * 24
        elif unit == 'w': tf_minutes = value * 60 * 24 * 7
        else: continue
        if total_minutes % tf_minutes == 0 and now.minute != st.session_state.last_run_minute:
            st.session_state.last_run_minute = now.minute
            return True
    return False

def should_trigger_scan():
    if manual_triggered:
        return True
    if auto_run and should_auto_run():
        return True
    return False

if should_trigger_scan():
    run_scan = True
    st.session_state.is_scanning = True
manual_triggered = st.button("Run Screener", key="manual_run_button")

if auto_run and not st.session_state.is_scanning and not run_scan:
    st_autorefresh(interval=5000, limit=None, key="auto_cradle_refresh")

if manual_triggered or (auto_run and should_auto_run()):
    run_scan = True

def highlight_cradle(row):
    color = 'background-color: #003300' if row['Setup'] == 'Bullish' else 'background-color: #330000'
    return [color] * len(row)

def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = BITGET.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception:
        return None

def check_cradle_setup(df, index):
    ema10 = df['close'].ewm(span=10).mean()
    ema20 = df['close'].ewm(span=20).mean()

    if index < 2 or index >= len(df):
        return None

    prev = df.iloc[index - 1]
    curr = df.iloc[index]

    # Bullish Cradle
    if (
        prev['close'] < prev['open'] and
        ema10.iloc[index - 1] > ema20.iloc[index - 1] and
        prev['low'] <= ema10.iloc[index - 1] and
        curr['close'] > curr['open']
    ):
        return 'Bullish'

    # Bearish Cradle
    if (
        prev['close'] > prev['open'] and
        ema10.iloc[index - 1] < ema20.iloc[index - 1] and
        prev['high'] >= ema10.iloc[index - 1] and
        curr['close'] < curr['open']
    ):
        return 'Bearish'

    return None

def analyze_cradle_setups(symbols, timeframes):
    result_containers = {tf: st.container() for tf in timeframes}

    for tf in timeframes:
        previous_setups = []
        status_line = st.empty()
        progress_bar = st.progress(0)
        eta_placeholder = st.empty()
        time_taken_placeholder = st.empty()
        total = len(symbols)
        start_time = time.time()

        for idx, symbol in enumerate(symbols):
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining_time = avg_time * (total - (idx + 1))
            mins, secs = divmod(int(remaining_time), 60)

            status_line.info(f"🔍 Scanning: {symbol} on {tf} ({idx+1}/{total})")
            progress_bar.progress((idx + 1) / total)
            eta_placeholder.markdown(f"⏳ Estimated time remaining: {mins}m {secs}s")

            df = fetch_ohlcv(symbol, tf)
            if df is None or len(df) < 5:
                continue

            setup_previous = check_cradle_setup(df, len(df) - 2)

            if setup_previous:
                previous_setups.append({
                    'Symbol': symbol,
                    'Timeframe': tf,
                    'Setup': setup_previous,
                    'Detected On': 'Previous Candle'
                })

            time.sleep(0.3)

        result_containers[tf].empty()
        if previous_setups:
            temp_df = pd.DataFrame(previous_setups).style.apply(highlight_cradle, axis=1)
            result_containers[tf].markdown(f"### 📈 Cradle Setups – {tf} (Last Closed Candle)", unsafe_allow_html=True)
            result_containers[tf].dataframe(temp_df, use_container_width=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        tmin, tsec = divmod(int(elapsed_time), 60)
        time_taken_placeholder.success(f"✅ Finished scanning {tf} in {tmin}m {tsec}s")

if run_scan:
    st.session_state.is_scanning = True
    placeholder.info("Starting scan...")
    with st.spinner("Scanning Bitget markets... Please wait..."):
        markets = BITGET.load_markets()
        symbols = [s for s in markets if '/USDT:USDT' in s and markets[s]['type'] == 'swap']
        analyze_cradle_setups(symbols, selected_timeframes)

    result_placeholder.success("Scan complete!")
    st.session_state.is_scanning = False

