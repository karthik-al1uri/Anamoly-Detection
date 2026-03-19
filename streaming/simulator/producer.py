from pathlib import Path
import time


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iterate_frames(dataset_path: str, fps: int = 30):
    frame_paths = sorted(
        path
        for path in Path(dataset_path).glob("**/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    delay = 1 / max(fps, 1)

    for frame_path in frame_paths:
        yield frame_path
        time.sleep(delay)


def stream_frames(dataset_path: str, fps: int = 30, limit: int | None = None) -> None:
    for index, frame_path in enumerate(iterate_frames(dataset_path, fps=fps), start=1):
        print({"frame": str(frame_path), "status": "sent"})
        if limit is not None and index >= limit:
            break


def main() -> None:
    stream_frames(dataset_path="data/raw", fps=30)


if __name__ == '__main__':
    main()
