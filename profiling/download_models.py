import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def _sanitize_dirname(model_id):
    return model_id.replace("/", "__")


def _download(model_id, models_dir, token=None):
    target_dir = models_dir / _sanitize_dirname(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    return target_dir


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download required HF models locally and emit env paths."
    )
    parser.add_argument(
        "--models-dir",
        default=str(base_dir / "models"),
        help="Directory to store downloaded models.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model id for text embeddings.",
    )
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id for theme extraction.",
    )
    parser.add_argument(
        "--graph-model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HF model id for question graph extraction.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token (or set HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--env-out",
        default=str(base_dir / ".model_env.json"),
        help="Write a JSON file with local model paths.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = _download(args.embedding_model, models_dir, token=args.token)
    llm_path = _download(args.llm_model, models_dir, token=args.token)
    graph_path = _download(args.graph_model, models_dir, token=args.token)

    env_payload = {
        "EMBEDDING_MODEL_NAME": str(embedding_path),
        "LLM_MODEL_NAME": str(llm_path),
        "GRAPH_LLM_MODEL": str(graph_path),
    }

    env_out = Path(args.env_out)
    env_out.write_text(json.dumps(env_payload, indent=2))

    print("Downloaded models to:", models_dir)
    print("Model paths written to:", env_out)
    print("Set env vars (example):")
    for key, value in env_payload.items():
        print(f"  export {key}='{value}'")


if __name__ == "__main__":
    main()
