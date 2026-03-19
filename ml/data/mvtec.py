from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_image_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def discover_categories(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def list_image_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class NormalImageDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: list[Path], image_size: int = 256) -> None:
        self.image_paths = image_paths
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB"))


class LabeledImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int = 256) -> None:
        self.samples = samples
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(image_path)


def _category_root(dataset_root: str | Path, category: str) -> Path:
    return Path(dataset_root) / category


def create_training_dataset(dataset_root: str | Path, category: str, image_size: int = 256) -> NormalImageDataset:
    train_root = _category_root(dataset_root, category) / "train" / "good"
    return NormalImageDataset(list_image_files(train_root), image_size=image_size)



def create_evaluation_dataset(dataset_root: str | Path, category: str, image_size: int = 256) -> LabeledImageDataset:
    test_root = _category_root(dataset_root, category) / "test"
    samples: list[tuple[Path, int]] = []

    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()) if test_root.exists() else []:
        label = 0 if defect_dir.name == "good" else 1
        samples.extend((image_path, label) for image_path in list_image_files(defect_dir))

    return LabeledImageDataset(samples, image_size=image_size)
