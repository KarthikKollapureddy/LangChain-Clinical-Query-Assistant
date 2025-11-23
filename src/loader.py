from typing import List, Tuple
from pathlib import Path
import re


def load_text_files(folder: str) -> List[str]:
    p = Path(folder)
    texts = []
    for f in p.glob('**/*.txt'):
        texts.append(f.read_text(encoding='utf-8'))
    return texts


def _extract_source_title(text: str) -> str:
    """Try to extract a leading 'Source: <title>' header from the file text.
    Returns empty string if not found."""
    if not text:
        return ''
    # Consider the first 2 lines for a Source: header
    first_chunk = '\n'.join(text.splitlines()[:3])
    m = re.search(r"^Source\s*:\s*(.+)$", first_chunk, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ''


def load_text_files_with_sources(folder: str) -> List[Tuple[str, str, str]]:
    """Return a list of (relpath, text, source_title) tuples for all .txt files under folder."""
    p = Path(folder)
    out = []
    for f in p.glob('**/*.txt'):
        rel = str(f.relative_to(p))
        txt = f.read_text(encoding='utf-8')
        src_title = _extract_source_title(txt)
        out.append((rel, txt, src_title))
    return out
