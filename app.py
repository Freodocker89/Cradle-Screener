import sys
import io
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(layout="wide")

BITGET = ccxt.bitget()
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']

# === Theme Toggle ===
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def switch_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

st.button("Toggle Theme", on_click=switch_theme)

# Apply theme styles
if st.session_state.theme == 'dark':
    background_color = '#111'
    text_color = '#fff'
    border_color = '#444'
else:
    background_color = '#fff'
    text_color = '#000'
    border_color = '#ccc'

st.markdown(f"""
    <style>
    body {{
        background-color: {background_color} !important;
        color: {text_color} !important;
    }}
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div > input,
    .stMultiSelect > div > div > div > div,
    .stButton > button {{
        color: {text_color};
        background-color: transparent;
    }}
    </style>
""", unsafe_allow_html=True)

table_styles = {
    'background-color': background_color,
    'color': text_color,
    'border': f'1px solid {border_color}'
}

st.title("Cradle Screener")
selected_timeframes = st.multiselect("Select Timeframes to Scan", TIMEFRAMES, default=['1h', '4h', '1d'])

auto_run = st.checkbox("⏱️ Auto Run on Candle Close", key="auto_run_checkbox")
st.write("This screener shows valid Cradle setups detected on the last fully closed candle only.")

placeholder = st.empty()

if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'last_run_timestamp' not in st.session_state:
    st.session_state.last_run_timestamp = 0
if 'results' not in st.session_state:
    st.session_state.results = {}

run_scan = False
manual_triggered = st.button("Run Screener", key="manual_run_button")

def should_auto_run():
    now = datetime.datetime.utcnow()
    now_ts = int(now.timestamp())
    for tf in selected_timeframes:
        unit = tf[-1]
        value = int(tf[:-1])
        if unit == 'm': tf_seconds = value * 60
        elif unit == 'h': tf_seconds = value * 3600
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

def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = BITGET.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception:
        return None

def check_cradle_setup(df):
    ema10 = df['close'].ewm(span=10).mean()
    ema20 = df['close'].ewm(span=20).mean()

    if len(df) < 13:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    e10_c1, e20_c1 = ema10.iloc[-3], ema20.iloc[-3]
    cradle_top = max(e10_c1, e20_c1)
    cradle_bot = min(e10_c1, e20_c1)

    c1_body = abs(c1['close'] - c1['open'])
    c2_range = c2['high'] - c2['low']

    last_10_ranges = df.iloc[-13:-3].apply(lambda row: row['high'] - row['low'], axis=1)
    avg_range_10 = last_10_ranges.mean()

    if (
        e10_c1 > e20_c1 and
        c1['close'] < c1['open'] and
        cradle_bot <= c1['close'] <= cradle_top and
        c2['close'] > c2['open'] and
        c2_range < avg_range_10
    ):
        return 'Bullish'

    if (
        e10_c1 < e20_c1 and
        c1['close'] > c1['open'] and
        cradle_bot <= c1['close'] <= cradle_top and
        c2['close'] < c2['open'] and
        c2_range < avg_range_10
    ):
        return 'Bearish'

    return None

def analyze_cradle_setups(symbols, timeframes):
    results = {}
    for tf in timeframes:
        tf_results = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(fetch_ohlcv, symbol, tf): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                df = future.result()
                if df is None or len(df) < 13:
                    continue
                signal = check_cradle_setup(df)
                if signal:
                    tf_results.append({
                        'Symbol': symbol,
                        'Setup': signal,
                        'Timeframe': tf
                    })
        results[tf] = tf_results
    return results

if run_scan:
    st.session_state.is_scanning = True
    placeholder.info("Starting scan...")
    with st.spinner("Scanning Bitget markets... Please wait..."):
        markets = BITGET.load_markets()
        symbols = [s for s in markets if '/USDT:USDT' in s and markets[s]['type'] == 'swap']
        st.success(f"Scanning {len(symbols)} symbols across: {', '.join(selected_timeframes)}")
        st.session_state.results = analyze_cradle_setups(symbols, selected_timeframes)
    placeholder.success("Scan complete!")
    st.session_state.is_scanning = False

    for tf in selected_timeframes:
        st.subheader(f"Results for {tf}")
        results = st.session_state.results.get(tf, [])
        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("No valid setups found.")

