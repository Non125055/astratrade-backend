import torch
import torch.nn as nn

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
try:
    model = TransformerAI(input_dim=INPUT_DIM)

    # Load model.pth from disk
    state = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state)

    model.eval()

    print("MODEL LOADED SUCCESSFULLY WITH INPUT_DIM =", INPUT_DIM)

except Exception as e:
    print("ERROR LOADING MODEL:", e)
    model = None

