from typing import List
from pathlib import Path

def load_text_files(folder: str) -> List[str]:
    p = Path(folder)
    texts = []
    for f in p.glob('**/*.txt'):
        texts.append(f.read_text(encoding='utf-8'))
    return texts


def load_text_files_with_sources(folder: str):
    """Return a list of (relpath, text) tuples for all .txt files under folder."""
    p = Path(folder)
    out = []
    for f in p.glob('**/*.txt'):
        rel = str(f.relative_to(p))
        out.append((rel, f.read_text(encoding='utf-8')))
    return out
