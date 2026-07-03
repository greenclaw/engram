import subprocess

import pytest


@pytest.fixture
def git_repo():
    """Init a git repo at `path` (created if needed) with a committer identity; returns the path."""
    def _init(path):
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
        return path

    return _init
