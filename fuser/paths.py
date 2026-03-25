from platformdirs import user_cache_dir
from pathlib import Path


def get_cache_dir():
    return Path(user_cache_dir(appname="fuser", appauthor=False, ensure_exists=True))
