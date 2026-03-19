from pathlib import Path


def build_index(source_dir: str) -> dict[str, int]:
    documents = [path for path in Path(source_dir).glob('**/*') if path.is_file()]
    return {'documents_indexed': len(documents)}


if __name__ == '__main__':
    print(build_index('docs'))
