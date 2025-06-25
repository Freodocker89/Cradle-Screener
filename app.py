import sys
import io
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import time
import datetime
import requests
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
    .css-1lcbmhc .e1fqkh3o10 {{
        overflow: visible !important;
        max-height: none !important;
    }}
    .dataframe td:has(div:contains('Bullish')) {{
        color: green;
    }}
    .dataframe td:has(div:contains('Bearish')) {{
        color: red;
    }}
    .dataframe td:has(div:contains('%')) {{
        text-align: right;
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

small_candle_ratio = st.selectbox("Candle 2 max size (% of 25-bar avg range)", [25, 33, 50, 66, 75, 100], index=2) / 100
sort_option = st.selectbox("Sort Results By", ["Symbol", "Setup", "MarketCap", "MarketCapRank"], index=0)

placeholder = st.empty()

if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'last_run_timestamp' not in st.session_state:
    st.session_state.last_run_timestamp = 0
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'cached_market_caps' not in st.session_state:
    st.session_state.cached_market_caps = None
if 'market_caps_timestamp' not in st.session_state:
    st.session_state.market_caps_timestamp = 0

run_scan = False
manual_triggered = st.button("Run Screener", key="manual_run_button")

if manual_triggered:
    run_scan = True
    st.session_state.is_scanning = True

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

    candle1 = df.iloc[index - 2]
    candle2 = df.iloc[index - 1]

    cradle_top_prev = max(ema10.iloc[index - 2], ema20.iloc[index - 2])
    cradle_bot_prev = min(ema10.iloc[index - 2], ema20.iloc[index - 2])

    avg_range = (df['high'] - df['low']).rolling(25).mean()
    c2_range = candle2['high'] - candle2['low']

    is_small_candle = c2_range <= (avg_range.iloc[index - 1] * small_candle_ratio)

    if (
        ema10.iloc[index - 2] > ema20.iloc[index - 2] and
        candle1['close'] < candle1['open'] and
        cradle_bot_prev <= candle1['close'] <= cradle_top_prev and
        candle2['close'] > candle2['open'] and
        is_small_candle
    ):
        return 'Bullish'

    if (
        ema10.iloc[index - 2] < ema20.iloc[index - 2] and
        candle1['close'] > candle1['open'] and
        cradle_bot_prev <= candle1['close'] <= cradle_top_prev and
        candle2['close'] < candle2['open'] and
        is_small_candle
    ):
        return 'Bearish'

    return None

def process_symbol_tf(symbol, tf):
    df = fetch_ohlcv(symbol, tf)
    if df is None or len(df) < 30:
        return tf, None
    setup = check_cradle_setup(df, len(df) - 1)
    if setup:
        return tf, {
            'Symbol': symbol,
            'Setup': setup
        }
    return tf, None

def analyze_cradle_setups(symbols, timeframes):
    start_time = time.time()
    results = {tf: [] for tf in timeframes}
    futures = []

    with ThreadPoolExecutor(max_workers=25) as executor:
        for tf in timeframes:
            for symbol in symbols:
                futures.append(
                    executor.submit(process_symbol_tf, symbol, tf)
                )

        for idx, future in enumerate(as_completed(futures)):
            tf, result = future.result()
            if result:
                results[tf].append(result)

            elapsed = int(time.time() - start_time)
            placeholder.markdown(f"⏱️ Scanning {idx+1}/{len(futures)} — Elapsed: {elapsed}s")

    st.session_state.results = results

def display_results():
    for tf, results in st.session_state.results.items():
        if not results:
            st.markdown(f"### {tf} — No Setups Found")
            continue

        df = pd.DataFrame(results)

        df['MarketCap'] = 0
        df['MarketCapRank'] = 0
        df['Volume (24h)'] = 0
        df['Liquidity'] = ''
        df['% Change 1h'] = 0
        df['% Change 24h'] = 0
        df['% Change 7d'] = 0

        df['% Change 1h'] = df['% Change 1h'].map(lambda x: f"{x:.2f}%")
        df['% Change 24h'] = df['% Change 24h'].map(lambda x: f"{x:.2f}%")
        df['% Change 7d'] = df['% Change 7d'].map(lambda x: f"{x:.2f}%")

        if sort_option in df.columns:
            df = df.sort_values(by=sort_option)

        st.markdown(f"## {tf}")
        df = df.drop(columns=['Timeframe'], errors='ignore')
        st.dataframe(df.style.set_properties(**table_styles), use_container_width=True)

if run_scan:
    st.session_state.is_scanning = True
    placeholder.info("Starting scan...")
    with st.spinner("Scanning Bitget markets... Please wait..."):
        markets = BITGET.load_markets()
        symbols = [s for s in markets if '/USDT:USDT' in s and markets[s]['type'] == 'swap']
        analyze_cradle_setups(symbols, selected_timeframes)
    placeholder.success("Scan complete!")
    display_results()
    st.session_state.is_scanning = False

