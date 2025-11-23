import os
import sys


def test_static_files_exist():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    web_dir = os.path.join(repo_root, 'web')
    assert os.path.isdir(web_dir)
    for name in ('index.html', 'styles.css', 'app.js'):
        assert os.path.exists(os.path.join(web_dir, name)), f"{name} missing"
