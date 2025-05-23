import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import time
import datetime

st.set_page_config(layout="wide")

BITGET = ccxt.bitget()
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']

st.title("📊 Cradle + BoS Screener")
selected_timeframes = st.multiselect("Select Timeframes to Scan", TIMEFRAMES, default=['1h', '4h', '1d'])

# Auto-run toggle with unique key to avoid duplicate error
auto_run = st.checkbox("⏱️ Auto Run on Candle Close", key="auto_run_checkbox")

st.write("This screener shows valid Cradle and BoS setups detected on the last fully closed candle only.")

result_placeholder = st.container()
placeholder = st.empty()

if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'last_run_timestamp' not in st.session_state:
    st.session_state.last_run_timestamp = 0

run_scan = False
manual_triggered = st.button("Run Screener", key="manual_run_button")

def should_auto_run():
    now = datetime.datetime.utcnow()
    now_ts = int(now.timestamp())

    for tf in selected_timeframes:
        unit = tf[-1]
        value = int(tf[:-1])
        if unit == 'm': tf_seconds = value * 60
        elif unit == 'h': tf_seconds = value * 60 * 60
        elif unit == 'd': tf_seconds = value * 86400
        elif unit == 'w': tf_seconds = value * 604800
        else: continue

        if (now_ts % tf_seconds) < 30 and (now_ts - st.session_state.last_run_timestamp) > tf_seconds - 30:
            st.session_state.last_run_timestamp = now_ts
            return True
    return False

def should_trigger_scan():
    return manual_triggered or (auto_run and should_auto_run())

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

    lower_cradle_prev2 = min(ema10.iloc[index - 2], ema20.iloc[index - 2])
    upper_cradle_prev2 = max(ema10.iloc[index - 2], ema20.iloc[index - 2])
    lower_cradle_prev1 = min(ema10.iloc[index - 1], ema20.iloc[index - 1])
    upper_cradle_prev1 = max(ema10.iloc[index - 1], ema20.iloc[index - 1])

    if (
        prev2['close'] < prev2['open'] and
        prev['close'] < prev['open'] and
        ema10.iloc[index - 2] > ema20.iloc[index - 2] and
        lower_cradle_prev2 <= prev2['low'] <= upper_cradle_prev2 and
        lower_cradle_prev1 <= prev['low'] <= upper_cradle_prev1 and
        curr['close'] > curr['open']
    ):
        return 'Bullish'

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

def check_bos_retest(df, index):
    if index < 3 or index >= len(df):
        return None

    curr = df.iloc[index]
    prev = df.iloc[index - 1]
    prev2 = df.iloc[index - 2]
    prev3 = df.iloc[index - 3]

    if (
        prev2['high'] > prev3['high'] and
        prev['low'] < prev2['high'] and
        curr['close'] > curr['open']
    ):
        return 'Bullish BoS'

    if (
        prev2['low'] < prev3['low'] and
        prev['high'] > prev2['low'] and
        curr['close'] < curr['open']
    ):
        return 'Bearish BoS'

    return None

def analyze_bos_setups(df, symbol, tf, index):
    setup = check_bos_retest(df, index)
    if setup:
        return {
            'Symbol': symbol,
            'Timeframe': tf,
            'Setup': setup,
            'Detected On': 'Previous Candle'
        }
    return None

def analyze_combined_setups(df, symbol, tf, index, cradle_func):
    cradle_result = cradle_func(df, index)
    bos_result = analyze_bos_setups(df, symbol, tf, index)

    cradle_entry = None
    bos_entry = None

    if cradle_result:
        cradle_entry = {
            'Symbol': symbol,
            'Timeframe': tf,
            'Setup': cradle_result,
            'Detected On': 'Previous Candle'
        }

    if bos_result:
        bos_entry = {
            'Symbol': symbol,
            'Timeframe': tf,
            'Setup': bos_result['Setup'],
            'Detected On': bos_result['Detected On']
        }

    return cradle_entry, bos_entry

def analyze_cradle_setups(symbols, timeframes):
    result_containers = {tf: st.container() for tf in timeframes}

    for tf in timeframes:
        previous_setups = []
        bos_setups = []
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

            cradle_entry, bos_entry = analyze_combined_setups(df, symbol, tf, len(df) - 2, check_cradle_setup)

            if cradle_entry:
                previous_setups.append(cradle_entry)
            if bos_entry:
                bos_setups.append(bos_entry)

            time.sleep(0.3)

        result_containers[tf].empty()
        if previous_setups:
            df_result = pd.DataFrame(previous_setups)
            result_containers[tf].markdown(f"### 📈 Cradle Setups – {tf} (Last Closed Candle)", unsafe_allow_html=True)
            styled_df = df_result.style.apply(highlight_cradle, axis=1)
            result_containers[tf].dataframe(styled_df, use_container_width=True)

        if bos_setups:
            df_bos = pd.DataFrame(bos_setups)
            result_containers[tf].markdown(f"### ⚡ BoS + Retest Setups – {tf}", unsafe_allow_html=True)
            result_containers[tf].dataframe(df_bos, use_container_width=True)

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
