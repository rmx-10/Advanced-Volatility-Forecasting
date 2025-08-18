import yfinance as yf

def load_data(config):
    data = yf.download(config["data"]["ticker"], start=config["data"]["start_date"], end=config["data"]["end_date"])
    data["returns"] = data["Adj Close"].pct_change().dropna()
    return data.dropna()
