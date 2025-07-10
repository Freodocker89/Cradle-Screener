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
    .dataframe td:has(div:contains('+')) {{
        color: green;
    }}
    .dataframe td:has(div:contains('-')) {{
        color: red;
    }}
    .dataframe td:has(div:contains('%')) {{
        text-align: right;
    }}
    .stDataFrameContainer {{
        overflow: visible !important;
    }}
    </style>
""", unsafe_allow_html=True)

# === Controls ===
st.title("Cradle Screener")
selected_timeframes = st.multiselect("Select Timeframes to Scan", TIMEFRAMES, default=['1h', '4h', '12h'])
small_candle_ratio = st.selectbox("Candle 2 max size (% of 25-bar avg range)", [25, 33, 50, 66, 75, 100], index=2) / 100
swing_strength = st.slider("Swing Strength (for pivot detection)", min_value=1, max_value=5, value=2, step=1)
sort_option = st.selectbox("Sort Results By", ["Rank", "Symbol", "Trend", "MarketCap"], index=0)
show_momentum = st.checkbox("Show Momentum Candidates Table", value=True)
manual_triggered = st.button("Run Screener")

# === Session State Init ===
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'momentum' not in st.session_state:
    st.session_state.momentum = {}
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'cached_market_caps' not in st.session_state:
    st.session_state.cached_market_caps = None
if 'market_caps_timestamp' not in st.session_state:
    st.session_state.market_caps_timestamp = 0

# === Market Cap Fetching ===
def fetch_market_caps():
    now = time.time()
    if st.session_state.cached_market_caps and now - st.session_state.market_caps_timestamp < 86400:
        return st.session_state.cached_market_caps

    market_caps = {}
    headers = {"X-CMC_PRO_API_KEY": st.secrets["CMC_API_KEY"]}
    for start in range(1, 2001, 100):
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        params = {"start": start, "limit": 100, "convert": "USD"}
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, params=params)
                data = response.json()
                if 'data' in data:
                    for item in data['data']:
                        symbol = item['symbol'].upper()
                        quote = item['quote']['USD']
                        market_caps[symbol] = (
                            quote.get('market_cap'),
                            item.get('cmc_rank'),
                            quote.get('volume_24h'),
                            quote.get('percent_change_1h'),
                            quote.get('percent_change_24h'),
                            quote.get('percent_change_7d')
                        )
                    break
            except:
                time.sleep(1.5)

    st.session_state.cached_market_caps = market_caps
    st.session_state.market_caps_timestamp = now
    return market_caps

# === Format Helpers ===
def format_market_cap(val):
    if val is None: return None
    return f"${val/1e9:.2f}B" if val >= 1e9 else f"${val/1e6:.2f}M" if val >= 1e6 else f"${val/1e3:.2f}K"

def format_volume(val):
    if val is None: return None
    return f"${val/1e9:.2f}B" if val >= 1e9 else f"${val/1e6:.2f}M" if val >= 1e6 else f"${val/1e3:.2f}K"

def format_percent(p):
    return f"{p:+.2f}%" if p is not None else None

def classify_liquidity(vol):
    if vol is None: return "Unknown"
    elif vol > 100_000_000: return "High"
    elif vol > 10_000_000: return "Medium"
    else: return "Low"

# === OHLCV Fetch ===
def fetch_ohlcv(symbol, tf):
    try:
        ohlcv = BITGET.fetch_ohlcv(symbol, tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except:
        return None

# === MACD Calculation ===
def calculate_macd(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    return macd_line

# === Cradle Setup Detection ===
def check_cradle_setup(df):
    ema10 = df['close'].ewm(span=10).mean()
    ema20 = df['close'].ewm(span=20).mean()
    macd = calculate_macd(df)

    if len(df) < 28: return None, None

    c1, c2 = df.iloc[-3], df.iloc[-2]
    cradle_top, cradle_bot = max(ema10.iloc[-3], ema20.iloc[-3]), min(ema10.iloc[-3], ema20.iloc[-3])
    avg_range = df.iloc[-28:-3].apply(lambda r: r['high'] - r['low'], axis=1).mean()
    c2_range = c2['high'] - c2['low']

    trend = None
    if ema10.iloc[-3] > ema20.iloc[-3] and c1['close'] < c1['open'] and cradle_bot <= c1['close'] <= cradle_top and c2['close'] > c2['open'] and c2_range < small_candle_ratio * avg_range:
        trend = 'Bullish'
    if ema10.iloc[-3] < ema20.iloc[-3] and c1['close'] > c1['open'] and cradle_bot <= c1['close'] <= cradle_top and c2['close'] < c2['open'] and c2_range < small_candle_ratio * avg_range:
        trend = 'Bearish'

    if not trend:
        return None, None

    def find_swings(df, kind, strength):
        cond = pd.Series([True] * len(df))
        for i in range(1, strength + 1):
            if kind == 'high':
                cond &= (df['high'] > df['high'].shift(i)) & (df['high'] > df['high'].shift(-i))
            else:
                cond &= (df['low'] < df['low'].shift(i)) & (df['low'] < df['low'].shift(-i))
        return df[cond]

    swings = find_swings(df[-30:].reset_index(drop=True), 'high' if trend == 'Bullish' else 'low', swing_strength)
    if len(swings) < 2:
        return trend, None

    last, prev = swings.iloc[-1], swings.iloc[-2]
    if trend == 'Bullish' and macd[last.name] > macd[prev.name]:
        return trend, True
    elif trend == 'Bearish' and macd[last.name] < macd[prev.name]:
        return trend, True
    else:
        return trend, False

# === Swing + Momentum Helpers ===
def wick_noise_score(df, n=20):
    recent = df[-n:]
    ratios = (recent['high'] - recent['low']) / (recent['close'] - recent['open']).abs().replace(0, 1e-6)
    avg_ratio = ratios.mean()
    return min(avg_ratio, 5)  # cap to avoid extreme influence

def trend_quality_score(df, trend, strength=2):
    def find_swings(df, kind):
        cond = pd.Series([True] * len(df))
        for i in range(1, strength + 1):
            if kind == 'high':
                cond &= (df['high'] > df['high'].shift(i)) & (df['high'] > df['high'].shift(-i))
            else:
                cond &= (df['low'] < df['low'].shift(i)) & (df['low'] < df['low'].shift(-i))
        swings = df[cond].copy().reset_index()
        return swings

    window = df[-50:].reset_index(drop=True)
    if trend == 'Bullish':
        highs = find_swings(window, 'high')
        lows = find_swings(window, 'low')
        if len(highs) >= 2 and len(lows) >= 2:
        if highs.index[-1] > highs.index[-2] and highs['high'].iloc[-1] > highs['high'].iloc[-2] and \
           lows.index[-1] > lows.index[-2] and lows['low'].iloc[-1] > lows['low'].iloc[-2]:
            return 4
        elif highs['high'].iloc[-1] > highs['high'].iloc[-2] or lows['low'].iloc[-1] > lows['low'].iloc[-2]:
            return 3
            return 2
    elif trend == 'Bearish':
        lows = find_swings(window, 'low')
        highs = find_swings(window, 'high')
        if len(lows) >= 2 and len(highs) >= 2:
        if lows.index[-1] > lows.index[-2] and lows['low'].iloc[-1] < lows['low'].iloc[-2] and \
           highs.index[-1] > highs.index[-2] and highs['high'].iloc[-1] < highs['high'].iloc[-2]:
            return 4
        elif lows['low'].iloc[-1] < lows['low'].iloc[-2] or highs['high'].iloc[-1] < highs['high'].iloc[-2]:
            return 3
            return 2
    return 0

# === Momentum Detection ===
def detect_momentum(df):
    ema10 = df['close'].ewm(span=10).mean()
    ema20 = df['close'].ewm(span=20).mean()
    macd = calculate_macd(df)

    if ema10.iloc[-1] > ema20.iloc[-1] and macd.iloc[-1] > macd.iloc[-2]:
        trend = 'Bullish'
    elif ema10.iloc[-1] < ema20.iloc[-1] and macd.iloc[-1] < macd.iloc[-2]:
        trend = 'Bearish'
    else:
        return None

    def find_swings(df, kind, strength):
        cond = pd.Series([True] * len(df))
        for i in range(1, strength + 1):
            if kind == 'high':
                cond &= (df['high'] > df['high'].shift(i)) & (df['high'] > df['high'].shift(-i))
            else:
                cond &= (df['low'] < df['low'].shift(i)) & (df['low'] < df['low'].shift(-i))
        return df[cond]

    window = df[-30:].reset_index(drop=True)
    if trend == 'Bullish':
        highs = find_swings(window, 'high', swing_strength)
        if len(highs) >= 2 and highs['high'].iloc[-1] > highs['high'].iloc[-2]:
            return 'Bullish'
    elif trend == 'Bearish':
        lows = find_swings(window, 'low', swing_strength)
        if len(lows) >= 2 and lows['low'].iloc[-1] < lows['low'].iloc[-2]:
            return 'Bearish'

    return None

    def find_swings(df, kind, strength):
        cond = pd.Series([True] * len(df))
        for i in range(1, strength + 1):
            if kind == 'high':
                cond &= (df['high'] > df['high'].shift(i)) & (df['high'] > df['high'].shift(-i))
            else:
                cond &= (df['low'] < df['low'].shift(i)) & (df['low'] < df['low'].shift(-i))
        return df[cond]

    window = df[-30:].reset_index(drop=True)
    if trend == 'Bullish':
        highs = find_swings(window, 'high', swing_strength)
        if len(highs) >= 2 and highs['high'].iloc[-1] > highs['high'].iloc[-2]:
            return 'Bullish'
    elif trend == 'Bearish':
        lows = find_swings(window, 'low', swing_strength)
        if len(lows) >= 2 and lows['low'].iloc[-1] < lows['low'].iloc[-2]:
            return 'Bearish'

    return None

# === Run Scanner ===
def run_scan():
    st.session_state.is_scanning = True
    st.info("🔄 Scanning... please wait")
    start_time = time.time()

    markets = BITGET.load_markets()
    symbols = [s for s in markets if '/USDT:USDT' in s and markets[s]['type'] == 'swap']
    market_caps = fetch_market_caps()

    results = {}
    momentum_results = {}

    for tf in selected_timeframes:
        tf_results = []
        tf_momentum = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(fetch_ohlcv, s, tf): s for s in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                df = future.result()
                if df is None or len(df) < 35: continue

                trend, convergent = check_cradle_setup(df)
                momentum_trend = detect_momentum(df)
                sym_clean = sym.split('/')[0].replace(':USDT','')
                cap = market_caps.get(sym_clean, [None]*6)

                if trend:
                    tf_results.append({
                        'Rank': cap[1],
                        'Symbol': sym,
                        'Trend': trend,
                        'MACD Convergent': convergent,
                        'MarketCap': format_market_cap(cap[0]),
                        'Volume (24h)': format_volume(cap[2]),
                        'Liquidity': classify_liquidity(cap[2]),
                        '% Change 1h': format_percent(cap[3]),
                        '% Change 24h': format_percent(cap[4]),
                        '% Change 7d': format_percent(cap[5])
                    })

                if momentum_trend:
                    structure = trend_quality_score(df, momentum_trend, swing_strength)
                    macd_slope = (calculate_macd(df).iloc[-1] - calculate_macd(df).iloc[-5])
                    macd_score = min(max(macd_slope * 100, 0), 3)
                    wick_penalty = min(wick_noise_score(df), 3)
                    momentum_rank = round(structure + macd_score - wick_penalty, 2)
                    tf_momentum.append({
                        'Momentum Rank': momentum_rank,
                        'Symbol': sym,
                        'Momentum': momentum_trend,
                        'MarketCap': format_market_cap(cap[0]),
                        'Volume (24h)': format_volume(cap[2]),
                        'Liquidity': classify_liquidity(cap[2]),
                        '% Change 1h': format_percent(cap[3]),
                        '% Change 24h': format_percent(cap[4]),
                        '% Change 7d': format_percent(cap[5])
                    })

        results[tf] = tf_results
        momentum_results[tf] = tf_momentum

    st.session_state.results = results
    st.session_state.momentum = momentum_results
    elapsed = time.time() - start_time
    st.success(f"✅ Scan complete in {elapsed:.1f} seconds.")
    st.session_state.is_scanning = False

# === Trigger Scan ===
if manual_triggered:
    run_scan()

# === Display Results ===
for tf, res in st.session_state.results.items():
    st.subheader(f"Results for {tf}")
    if res:
        df = pd.DataFrame(res)
        df["MACD Convergent"] = df["MACD Convergent"].map(lambda x: "✅" if x is True else "❌" if x is False else "N/A")
        if sort_option in df.columns:
            df = df.sort_values(by=sort_option, na_position='last')
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No setups found for this timeframe.")

if show_momentum:
    for tf, res in st.session_state.momentum.items():
        st.subheader(f"Momentum Candidates for {tf}")
        if res:
            df = pd.DataFrame(res)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No momentum candidates found for this timeframe.")

