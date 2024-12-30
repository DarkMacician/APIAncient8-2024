import requests
from datetime import datetime

from DataFetching.Database import gen


def fetch_historical_1m_data(symbol, start_time, end_time):
    """
    Fetch historical data for a given symbol at 1-minute intervals.
    """
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if the request fails
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1  # Move to the next interval

    return all_data


def save_to_mongodb(data, symbol):
    """
    Save historical data to MongoDB.
    """
    formatted_data = [
        {
            "timestamp": datetime.fromtimestamp(entry[0] / 1000).strftime('%Y-%m-%d %H:%M'),  # Convert ms to datetime
            "open_price": float(entry[1]),
            "high_price": float(entry[2]),
            "low_price": float(entry[3]),
            "close_price": float(entry[4]),
            "volume": float(entry[5]),
            "symbol": symbol
        }
        for entry in data
    ]

    gen.insert_many(formatted_data)
    print(f"Data for {symbol} successfully saved to MongoDB.")


def fetch_all_symbols():
    """
    Fetch all available symbols from Binance.
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return [symbol['symbol'] for symbol in data['symbols']]


def fetch_btc_data(symbol="BTCUSDT", start_time="2024-12-29 00:00", end_time="2024-12-30 14:50"):
    """
    Fetch and save historical data for a specific symbol or all symbols.
    """
    start_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M").timestamp() * 1000)
    end_time = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M").timestamp() * 1000)

    if symbol:
        print(f"Fetching data for {symbol}...")
        data = fetch_historical_1m_data(symbol, start_time, end_time)
        save_to_mongodb(data, symbol)
    else:
        symbols = fetch_all_symbols()
        for sym in symbols:
            print(f"Fetching data for {sym}...")
            data = fetch_historical_1m_data(sym, start_time, end_time)
            save_to_mongodb(data, sym)


if __name__ == "__main__":
    fetch_btc_data()
