import streamlit as st
import pandas as pd
import ccxt
import time

# === CONFIG ===
TIMEFRAMES = ['1m', '3m', '5m', '10m', '15m', '20m', '30m', '1h', '2h', '4h', '6h', '8h', '10h', '12h', '16h', '1d', '1w']
BITGET = ccxt.bitget()

# === HELPERS ===
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

    ema_zone_low = min(ema10.iloc[index - 1], ema20.iloc[index - 1])
    ema_zone_high = max(ema10.iloc[index - 1], ema20.iloc[index - 1])

    if prev['close'] < prev['open'] and ema_zone_low <= prev['close'] <= ema_zone_high and curr['close'] > curr['open']:
        return 'Bullish'
    if prev['close'] > prev['open'] and ema_zone_low <= prev['close'] <= ema_zone_high and curr['close'] < curr['open']:
        return 'Bearish'
    return None

def analyze_cradle_setups(symbols, timeframes):
    all_previous_setups = []
    result_container = result_placeholder

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

        end_time = time.time()
        elapsed_time = end_time - start_time
        tmin, tsec = divmod(int(elapsed_time), 60)
        time_taken_placeholder.success(f"✅ Finished scanning {tf} in {tmin}m {tsec}s")

        if previous_setups:
            result_container.markdown(f"### 📈 Cradle Setups – {tf} (Last Closed Candle)")
            styled_previous = pd.DataFrame(previous_setups).style.apply(highlight_cradle, axis=1)
            result_container.dataframe(styled_previous, use_container_width=True)

        all_previous_setups.extend(previous_setups)

    return pd.DataFrame(all_previous_setups)

# === STREAMLIT UI ===
st.set_page_config(layout="wide")
st.title("📊 Cradle Screener")

selected_timeframes = st.multiselect("Select Timeframes to Scan", TIMEFRAMES, default=['1h', '4h', '1d'])
st.write("This screener shows valid Cradle setups detected on the last fully closed candle only.")

result_placeholder = st.empty()
placeholder = st.empty()

from datetime import datetime, timedelta

def get_shortest_timeframe(selected):
    timeframe_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '10m': 10, '15m': 15, '20m': 20, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '10h': 600,
        '12h': 720, '16h': 960, '1d': 1440, '1w': 10080
    }
    return min([timeframe_minutes[tf] for tf in selected])

def seconds_until_next_close(minutes):
    now = datetime.utcnow()
    total_minutes = now.hour * 60 + now.minute
    next_close = ((total_minutes // minutes) + 1) * minutes
    delta_minutes = next_close - total_minutes
    next_time = now + timedelta(minutes=delta_minutes)
    seconds_left = int((next_time - now).total_seconds())
    return seconds_left

auto_refresh = st.checkbox("🔁 Auto-run at next candle close", value=False)

if auto_refresh:
    mins = get_shortest_timeframe(selected_timeframes)
    wait_seconds = seconds_until_next_close(mins)
    st.markdown(f"🕒 Waiting for next {mins}-minute candle close: refreshing in {wait_seconds} seconds")
    st.experimental_rerun() if wait_seconds <= 1 else st_autorefresh(interval=60000, limit=None, key="auto_refresh")

if st.button("Run Screener"):
    placeholder.info("Starting scan...")
    with st.spinner("Scanning Bitget markets... Please wait..."):
        markets = BITGET.load_markets()
        symbols = [s for s in markets if '/USDT:USDT' in s and markets[s]['type'] == 'swap']
        previous_df = analyze_cradle_setups(symbols, selected_timeframes)

    if not previous_df.empty:
        result_placeholder.markdown("### 📈 Cradle Setups (Last Closed Candle)")
        styled_previous = previous_df.style.apply(highlight_cradle, axis=1)
        result_placeholder.dataframe(styled_previous, use_container_width=True)

    if previous_df.empty:
        result_placeholder.warning("No valid Cradle setups found.")

    result_placeholder.success("Scan complete!")

