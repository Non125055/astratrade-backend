import torch
import torch.nn as nn
import numpy as np

# -------------------------------
# TRANSFORMER MODEL (same as Colab)
# -------------------------------
class TransformerAI(nn.Module):
    def __init__(self, input_dim=12, model_dim=64, layers=2, heads=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)

        enc = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x[:, -1])


# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
model = None
print("MODEL LOADING DISABLED FOR DEBUG")



# -------------------------------
# PREDICT FUNCTION
# -------------------------------
def predict_next_price(window_data):
    """
    window_data = numpy array of shape (40, 12)
    (40 timesteps, 12 features)
    """
    x = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # make batch=1
    with torch.no_grad():
        pred = model(x).item()
    return float(pred)

