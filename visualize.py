import matplotlib.pyplot as plt

def plot_results(results, config):
    y_test = results["y_test"]
    y_pred_lstm = results["y_pred_lstm"]
    y_pred_garch = results["y_pred_garch"]

    plt.figure(figsize=(12,6))
    plt.plot(y_test, label="Realized Volatility", color="black")
    plt.plot(y_pred_lstm, label="LSTM Forecast", color="blue")
    plt.plot(y_pred_garch, label="GARCH Forecast", color="red")
    plt.legend()
    plt.title("Volatility Forecast Comparison")
    plt.savefig(config["paths"]["figures_dir"] + "vol_forecast.png")
    plt.show()
