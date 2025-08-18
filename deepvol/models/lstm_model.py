import torch
import torch.nn as nn
import torch.optim as optim

class LSTMVolatility(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, X, y, epochs, batch_size):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    def predict(self, X):
        with torch.no_grad():
            return self(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path):
        model = LSTMVolatility(input_size=1, hidden_size=64, num_layers=2, lr=0.001)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
