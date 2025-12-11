import torch
import torch.nn as nn
import numpy as np

# -------------------------------------------------
# MODEL INPUT SIZE (from embed.weight shape)
# Your model uses 11 input features
# -------------------------------------------------
INPUT_DIM = 11


# -------------------------------------------------
# Transformer AI Model (same architecture as training)
# -------------------------------------------------
class TransformerAI(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, model_dim=64, layers=2, heads=2):
        super().__init__()

        # Input embedding layer
        self.embed = nn.Linear(input_dim, model_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Output layer
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        x shape = [batch, seq_len, input_dim]
        """
        x = self.embed(x)                       # -> [batch, seq_len, model_dim]
        x = self.encoder(x)                     # -> transformer features
        x = x[:, -1]                            # last timestep
        return self.fc(x)                       # -> price movement prediction


# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------
model = None
try:
    model = TransformerAI(input_dim=INPUT_DIM)

    # Load model.pth from disk
    state = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state)

    model.eval()

    print("MODEL LOADED SUCCESSFULLY WITH INPUT_DIM =", INPUT_DIM)

except Exception as e:
    # Keep model = None but print error so logs show what's wrong
    print("ERROR LOADING MODEL:", repr(e))
    model = None


# -------------------------------------------------
# PREDICTION HELPER (used by main.py)
# -------------------------------------------------
def predict_next_price(window_data):
    """
    Expects window_data as:
      - numpy array of shape (seq_len, input_dim)  OR
      - list or nested list with same shape

    Returns:
      - float prediction (model output)
    """

    if model is None:
        raise RuntimeError("Model not loaded")

    # Convert to numpy (if list)
    arr = np.array(window_data, dtype=np.float32)

    # Ensure shape is (seq_len, input_dim)
    if arr.ndim == 1:
        # single feature vector given — reshape to (1, input_dim)
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        # expected (seq_len, input_dim)
        pass
    else:
        raise ValueError(f"Unexpected input array shape {arr.shape}")

    # If input_dim mismatch, try to handle gracefully
    if arr.shape[1] != INPUT_DIM:
        raise ValueError(f"Window data feature-size {arr.shape[1]} != expected {INPUT_DIM}")

    # Convert to tensor with batch dim
    x = torch.from_numpy(arr).unsqueeze(0)  # shape -> (1, seq_len, input_dim)
    x = x.to(dtype=torch.float32)

    with torch.no_grad():
        pred = model(x)            # shape (1,1) or (1,)
        # Handle output shape
        if isinstance(pred, torch.Tensor):
            val = pred.squeeze().cpu().numpy().item()
        else:
            val = float(pred)

    return float(val)
