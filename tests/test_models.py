from deepvol.models.lstm_model import LSTMVolatility

def test_lstm_init():
    model = LSTMVolatility(input_size=1, hidden_size=32, num_layers=1, lr=0.001)
    assert model is not None
