from pathlib import Path
import time


def stream_frames(dataset_path: str, fps: int = 30) -> None:
    frame_paths = sorted(Path(dataset_path).glob('**/*'))
    frame_paths = [path for path in frame_paths if path.is_file()]
    delay = 1 / fps

    for frame_path in frame_paths:
        print({"frame": str(frame_path), "status": "sent"})
        time.sleep(delay)


def main() -> None:
    stream_frames(dataset_path='data/raw')


if __name__ == '__main__':
    main()
