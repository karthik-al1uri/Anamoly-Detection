import torch
from torch import nn, optim

from ml.models.autoencoder import ConvAutoencoder


def train_step(batch: torch.Tensor, model: ConvAutoencoder, optimizer: optim.Optimizer) -> float:
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    reconstructed = model(batch)
    loss = criterion(reconstructed, batch)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> None:
    model = ConvAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dummy_batch = torch.rand(8, 3, 64, 64)
    loss = train_step(dummy_batch, model, optimizer)
    print({"message": "training scaffold ready", "loss": loss})


if __name__ == "__main__":
    main()
