import pandas as pd

def load_data(file_path, date_col, price_col):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    df = df[[date_col, price_col]].dropna()
    df.sort_values(by=date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
