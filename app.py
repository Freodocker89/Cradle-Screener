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

# Track scan state to prevent refresh interruptions
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'last_run_timestamp' not in st.session_state:
    st.session_state.last_run_timestamp = 0

run_scan = False
manual_triggered = st.button("Run Screener", key="manual_run_button")

# Auto-run logic using timestamp window
def should_auto_run():
    now = datetime.datetime.utcnow()
    now_ts = int(now.timestamp())

    for tf in selected_timeframes:
        unit = tf[-1]
        value = int(tf[:-1])
        if unit == 'm':
            tf_seconds = value * 60
        elif unit == 'h':
            tf_seconds = value * 60 * 60
        elif unit == 'd':
            tf_seconds = value * 60 * 60 * 24
        elif unit == 'w':
            tf_seconds = value * 60 * 60 * 24 * 7
        else:
            continue

        if (now_ts % tf_seconds) < 30 and (now_ts - st.session_state.last_run_timestamp) > tf_seconds - 30:
            st.session_state.last_run_timestamp = now_ts
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

if auto_run and not st.session_state.is_scanning and not run_scan:
    st_autorefresh(interval=15000, limit=None, key="auto_cradle_refresh")

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

    if index < 3 or index >= len(df):
        return None

    curr = df.iloc[index]
    prev = df.iloc[index - 1]
    prev2 = df.iloc[index - 2]

    # Define the cradle zone for each pullback candle
    lower_cradle_prev2 = min(ema10.iloc[index - 2], ema20.iloc[index - 2])
    upper_cradle_prev2 = max(ema10.iloc[index - 2], ema20.iloc[index - 2])
    lower_cradle_prev1 = min(ema10.iloc[index - 1], ema20.iloc[index - 1])
    upper_cradle_prev1 = max(ema10.iloc[index - 1], ema20.iloc[index - 1])

    # Bullish Cradle Setup
    if (
        prev2['close'] < prev2['open'] and
        prev['close'] < prev['open'] and
        ema10.iloc[index - 2] > ema20.iloc[index - 2] and
        lower_cradle_prev2 <= prev2['low'] <= upper_cradle_prev2 and
        lower_cradle_prev1 <= prev['low'] <= upper_cradle_prev1 and
        curr['close'] > curr['open']
    ):
        return 'Bullish'

    # Bearish Cradle Setup
    if (
        prev2['close'] > prev2['open'] and
        prev['close'] > prev['open'] and
        ema10.iloc[index - 2] < ema20.iloc[index - 2] and
        lower_cradle_prev2 <= prev2['high'] <= upper_cradle_prev2 and
        lower_cradle_prev1 <= prev['high'] <= upper_cradle_prev1 and
        curr['close'] < curr['open']
    ):
        return 'Bearish'

    return None

    curr = df.iloc[index]
    prev = df.iloc[index - 1]
    prev2 = df.iloc[index - 2]

    # Cradle zone bounds (between EMAs)
    lower_cradle_prev2 = min(ema10.iloc[index - 2], ema20.iloc[index - 2])
    upper_cradle_prev2 = max(ema10.iloc[index - 2], ema20.iloc[index - 2])
    lower_cradle_prev1 = min(ema10.iloc[index - 1], ema20.iloc[index - 1])
    upper_cradle_prev1 = max(ema10.iloc[index - 1], ema20.iloc[index - 1])

    # Bullish Cradle
    if (
        prev2['close'] < prev2['open'] and
        prev['close'] < prev['open'] and
        ema10.iloc[index - 2] > ema20.iloc[index - 2] and
        lower_cradle_prev2 <= prev2['low'] <= upper_cradle_prev2 and
        lower_cradle_prev1 <= prev['low'] <= upper_cradle_prev1 and
        curr['close'] > curr['open']
    ):
        return 'Bullish'

    # Bearish Cradle
    if (
        prev2['close'] > prev2['open'] and
        prev['close'] > prev['open'] and
        ema10.iloc[index - 2] < ema20.iloc[index - 2] and
        lower_cradle_prev2 <= prev2['high'] <= upper_cradle_prev2 and
        lower_cradle_prev1 <= prev['high'] <= upper_cradle_prev1 and
        curr['close'] < curr['open']
    ):
        return 'Bearish'

    return None

    # Check last two candles
    curr = df.iloc[index]
    prev = df.iloc[index - 1]
    prev2 = df.iloc[index - 2]

    # Bullish Cradle
    if (
        prev2['close'] < prev2['open'] and  # candle 1 bearish
        prev['close'] < prev['open'] and    # candle 2 bearish
        ema10.iloc[index - 2] > ema20.iloc[index - 2] and  # correct bullish EMA trend
        prev2['low'] <= ema10.iloc[index - 2] and
        prev['low'] <= ema10.iloc[index - 1] and
        curr['close'] > curr['open']       # current candle bullish
    ):
        return 'Bullish'

    # Bearish Cradle
    if (
        prev2['close'] > prev2['open'] and  # candle 1 bullish
        prev['close'] > prev['open'] and    # candle 2 bullish
        ema10.iloc[index - 2] < ema20.iloc[index - 2] and  # correct bearish EMA trend
        prev2['high'] >= ema10.iloc[index - 2] and
        prev['high'] >= ema10.iloc[index - 1] and
        curr['close'] < curr['open']       # current candle bearish
    ):
        return 'Bearish'

    return None

    # Check last two candles
    curr = df.iloc[index]
    prev = df.iloc[index - 1]
    prev2 = df.iloc[index - 2]

    # Bullish Cradle
    if (
        prev2['close'] < prev2['open'] and  # candle 1 bearish
        prev['close'] < prev['open'] and    # candle 2 bearish
        ema10.iloc[index - 2] > ema20.iloc[index - 2] and  # trend up
        (prev2['low'] <= ema10.iloc[index - 2] or prev2['low'] <= ema20.iloc[index - 2]) and
        (prev['low'] <= ema10.iloc[index - 1] or prev['low'] <= ema20.iloc[index - 1]) and
        curr['close'] > curr['open']       # current candle bullish
    ):
        return 'Bullish'

    # Bearish Cradle
    if (
        prev2['close'] > prev2['open'] and  # candle 1 bullish
        prev['close'] > prev['open'] and    # candle 2 bullish
        ema10.iloc[index - 2] < ema20.iloc[index - 2] and  # trend down
        (prev2['high'] >= ema10.iloc[index - 2] or prev2['high'] >= ema20.iloc[index - 2]) and
        (prev['high'] >= ema10.iloc[index - 1] or prev['high'] >= ema20.iloc[index - 1]) and
        curr['close'] < curr['open']       # current candle bearish
    ):
        return 'Bearish'

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
                symbol_clean = symbol.replace('/USDT:USDT', 'USDT')
                previous_setups.append({
                'Symbol': symbol,
                    'Timeframe': tf,
                    'Setup': setup_previous,
                    'Detected On': 'Previous Candle'
                })

            time.sleep(0.3)

        result_containers[tf].empty()
        if previous_setups:
            df_result = pd.DataFrame(previous_setups)
            result_containers[tf].markdown(f"### 📈 Cradle Setups – {tf} (Last Closed Candle)", unsafe_allow_html=True)
            styled_df = df_result.style.apply(highlight_cradle, axis=1)
            result_containers[tf].dataframe(styled_df, use_container_width=True)

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
