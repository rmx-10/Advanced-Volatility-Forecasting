import yaml
from deepvol import data_loader, feature_engineering, train, evaluation

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data
df = data_loader.load_data(config["data"]["file_path"],
                           config["data"]["date_column"],
                           config["data"]["price_column"])

# Feature engineering
X, y = feature_engineering.create_features(df, config["features"]["window"])

# Train model
model, history = train.train_model(X, y, config["model"])

# Evaluate model
evaluation.plot_results(history, "experiments/results")

