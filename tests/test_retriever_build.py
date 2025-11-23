import os
import sys
import shutil
import tempfile
# ensure repo root on path for pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retriever import build_vectorstore, load_vectorstore


def test_build_vectorstore_dev_fake():
    # Force dev fake embeddings
    os.environ['DEV_FAKE_EMBS'] = '1'
    tmpdir = tempfile.mkdtemp()
    try:
        coll = build_vectorstore(data_dir='data', persist_dir=tmpdir)
        assert coll is not None
        # load back
        coll2 = load_vectorstore(persist_dir=tmpdir)
        assert coll2 is not None
    finally:
        shutil.rmtree(tmpdir)
        os.environ.pop('DEV_FAKE_EMBS', None)
