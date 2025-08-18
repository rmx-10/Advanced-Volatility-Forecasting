import pandas as pd
from deepvol.data import load_data
from deepvol.features import make_features
from deepvol.models.lstm_model import LSTMVolatility

def train_lstm(config):
    df = load_data(config)
    X_train, y_train, X_test, y_test = make_features(df, config)

    model = LSTMVolatility(
        input_size=X_train.shape[2],
        hidden_size=config["model"]["lstm_hidden"],
        num_layers=config["model"]["lstm_layers"],
        lr=config["model"]["learning_rate"]
    )
    model.train_model(X_train, y_train, epochs=config["model"]["epochs"], batch_size=config["model"]["batch_size"])
    model.save("experiments/lstm_model.pth")
    return model
