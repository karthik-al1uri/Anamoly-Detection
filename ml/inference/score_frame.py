import torch
from torch import nn

from ml.models.autoencoder import ConvAutoencoder


def compute_reconstruction_error(frame_tensor: torch.Tensor, model: ConvAutoencoder | None = None) -> float:
    autoencoder = model or ConvAutoencoder()
    criterion = nn.MSELoss()
    with torch.no_grad():
        reconstructed = autoencoder(frame_tensor)
        return float(criterion(reconstructed, frame_tensor).item())
