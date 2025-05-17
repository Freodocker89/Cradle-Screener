st.title("📊 Cradle Screener")

selected_timeframes = st.multiselect("Select Timeframes to Scan", TIMEFRAMES, default=['1h', '4h', '1d'])
st.write("This screener shows valid Cradle setups detected on the last fully closed candle only.")

result_placeholder = st.container()
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
        analyze_cradle_setups(symbols, selected_timeframes)

    result_placeholder.success("Scan complete!")

