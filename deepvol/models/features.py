import numpy as np

def make_features(df, config):
    window = config["data"]["window"]
    X, y = [], []

    realized_vol = df["returns"].rolling(window).std().dropna()
    returns = df["returns"].dropna()

    for i in range(window, len(returns)):
        X.append(returns.values[i-window:i].reshape(-1,1))
        y.append(realized_vol.values[i])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]
