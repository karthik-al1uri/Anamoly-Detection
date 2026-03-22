import torch
from PIL import Image
from torch import nn

from ml.data.mvtec import build_image_transform
from ml.models.autoencoder import ConvAutoencoder


def _ensure_batch(frame_tensor: torch.Tensor) -> torch.Tensor:
    if frame_tensor.dim() == 3:
        return frame_tensor.unsqueeze(0)
    return frame_tensor


def compute_batch_reconstruction_errors(
    frame_tensor: torch.Tensor,
    model: ConvAutoencoder,
    device: str | torch.device = "cpu",
) -> list[float]:
    criterion = nn.MSELoss(reduction="none")
    batched_frames = _ensure_batch(frame_tensor).to(device)
    autoencoder = model.to(device)
    autoencoder.eval()

    with torch.no_grad():
        reconstructed = autoencoder(batched_frames)
        loss_map = criterion(reconstructed, batched_frames)
        return loss_map.view(loss_map.size(0), -1).mean(dim=1).cpu().tolist()


def compute_reconstruction_error(
    frame_tensor: torch.Tensor,
    model: ConvAutoencoder | None = None,
    device: str | torch.device = "cpu",
) -> float:
    autoencoder = model or ConvAutoencoder()
    return float(compute_batch_reconstruction_errors(frame_tensor, autoencoder, device=device)[0])


def prepare_image_tensor(image_path: str, image_size: int = 256) -> torch.Tensor:
    transform = build_image_transform(image_size)
    with Image.open(image_path) as image:
        return transform(image.convert("RGB"))


def score_image_path(
    image_path: str,
    model: ConvAutoencoder,
    image_size: int = 256,
    device: str | torch.device = "cpu",
) -> float:
    frame_tensor = prepare_image_tensor(image_path, image_size=image_size)
    return compute_reconstruction_error(frame_tensor, model=model, device=device)


def load_model_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[ConvAutoencoder, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "category": checkpoint.get("category"),
        "image_size": checkpoint.get("image_size", 256),
        "threshold": checkpoint.get("threshold"),
        "epochs": checkpoint.get("epochs"),
        "threshold_strategy": checkpoint.get("threshold_strategy"),
        "threshold_details": checkpoint.get("threshold_details"),
    }
    return model, metadata
