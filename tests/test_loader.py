import os
import sys
# ensure repo root is importable for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.loader import load_text_files


def test_load_text_files_reads_data():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    texts = load_text_files(data_dir)
    assert isinstance(texts, list)
    assert len(texts) > 0
