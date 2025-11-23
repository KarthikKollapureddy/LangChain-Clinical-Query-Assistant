from typing import List
from pathlib import Path

def load_text_files(folder: str) -> List[str]:
    p = Path(folder)
    texts = []
    for f in p.glob('**/*.txt'):
        texts.append(f.read_text(encoding='utf-8'))
    return texts
