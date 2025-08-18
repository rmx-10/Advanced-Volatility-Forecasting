from arch import arch_model
import numpy as np

def fit_garch(df):
    returns = df["returns"].dropna() * 100
    model = arch_model(returns, vol="Garch", p=1, q=1)
    res = model.fit(disp="off")
    return np.sqrt(res.conditional_volatility / 100)
