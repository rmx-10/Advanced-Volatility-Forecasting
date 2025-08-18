import pandas as pd
from deepvol.data import load_data
from deepvol.features import make_features
from deepvol.models.lstm_model import LSTMVolatility
from deepvol.models.garch_model import fit_garch

def evaluate_models(config):
    df = load_data(config)
    X_train, y_train, X_test, y_test = make_features(df, config)

    # Load trained LSTM
    lstm = LSTMVolatility.load("experiments/lstm_model.pth")
    y_pred_lstm = lstm.predict(X_test)

    # Fit GARCH
    y_pred_garch = fit_garch(df)

    return {
        "y_test": y_test,
        "y_pred_lstm": y_pred_lstm,
        "y_pred_garch": y_pred_garch
    }
