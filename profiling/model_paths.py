import json
import os
from pathlib import Path


def resolve_model_name(env_key, default, config_path=None):
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]

    candidates = []
    if config_path is not None:
        candidates.append(Path(config_path))
    else:
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / ".model_env.json",
                cwd / "graph_merging" / ".model_env.json",
                Path(__file__).resolve().parent / ".model_env.json",
            ]
        )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            payload = {}
        value = payload.get(env_key)
        if value:
            return value

    return default

