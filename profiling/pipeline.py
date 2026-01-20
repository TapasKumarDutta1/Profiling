import json
import os
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from embedding_utils import embed_texts
from retrieval_utils import retrieve_similar_indices
from theme_extractor import build_theme_items
from healing_utils import build_healed_graph
from model_paths import resolve_model_name

_GRAPH_LLM = None
_QUESTION_GRAPH_CACHE = {}

GRAPH_MODEL_NAME = resolve_model_name("GRAPH_LLM_MODEL", "Qwen/Qwen2-1.5B-Instruct")
QUESTION_GRAPH_CHUNK_SIZE = 400
QUESTION_GRAPH_CHUNK_OVERLAP = 0
MAX_THEMES_PER_QUERY = 4
NODE_TH1 = 0.85
NODE_TH2 = 0.7
NODE_MAX_PAIRS = 500
NODE_RELATION_QUESTIONS = 2


def ensure_models_downloaded(base_dir=None, auto_download=True):
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    fallback_paths = [base_dir / ".model_env.json"]
    if base_dir.name != "graph_merging":
        fallback_paths.append(base_dir / "graph_merging" / ".model_env.json")

    if any(path.exists() for path in fallback_paths):
        return

    print("Missing .model_env.json; run download_models.py to fetch models.")
    print(f"Example: python {base_dir / 'download_models.py'}")

    if not auto_download:
        raise FileNotFoundError("Missing .model_env.json")

    subprocess.check_call([sys.executable, str(base_dir / "download_models.py")])


def load_search_history(path="cleaned_search_history.csv"):
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(
            {
                "title": [
                    "how to train a cat to use a leash",
                    "best cat harness sizes",
                    "cat leash training troubleshooting",
                ],
                "time": [
                    "2024-06-23 22:21:50.431000+00:00",
                    "2024-06-24 10:05:12.000000+00:00",
                    "2024-06-25 09:15:40.000000+00:00",
                ],
            }
        )

    cols = {c.lower(): c for c in df.columns}
    if "title" not in cols:
        for candidate in ["query", "search", "text", "question"]:
            if candidate in cols:
                df = df.rename(columns={cols[candidate]: "title"})
                break
    if "time" not in cols:
        for candidate in ["timestamp", "created_at", "date", "datetime"]:
            if candidate in cols:
                df = df.rename(columns={cols[candidate]: "time"})
                break

    if "id" in cols and cols["id"] != "id":
        df = df.rename(columns={cols["id"]: "id"})

    if "title" not in df.columns or "time" not in df.columns:
        raise ValueError("Expected columns 'title' and 'time'.")

    df = df[["title", "time", "id"]].copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["title", "time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _get_graph_llm(model_name=None):
    global _GRAPH_LLM
    if _GRAPH_LLM is not None:
        return _GRAPH_LLM

    model_name = model_name or os.environ.get(
        "GRAPH_LLM_MODEL", "Qwen/Qwen2-1.5B-Instruct"
    )
    from raw_txt_to_graph import LLMHandler

    _GRAPH_LLM = LLMHandler(model_name=model_name)
    return _GRAPH_LLM


def extract_question_graph(
    question,
    model_name=None,
    chunk_size=400,
    chunk_overlap=0,
):
    cache_key = (question, model_name, chunk_size, chunk_overlap)
    if cache_key in _QUESTION_GRAPH_CACHE:
        return _QUESTION_GRAPH_CACHE[cache_key]

    from raw_txt_to_graph import (
        build_knowledge_graph,
        chunk_text,
        extract_entities_relationships,
        serialize_graph,
    )

    llm = _get_graph_llm(model_name=model_name)
    chunks = chunk_text(question, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    summaries = extract_entities_relationships(chunks, llm)
    graph = build_knowledge_graph(summaries)
    payload = serialize_graph(graph)

    _QUESTION_GRAPH_CACHE[cache_key] = payload
    return payload


def build_profile_for_index(
    i,
    df,
    embeddings,
    n_sim=None,
    window_days=7,
    extract_question_graphs=True,
    graph_model_name=None,
    graph_chunk_size=400,
    graph_chunk_overlap=0,
    max_themes=4,
):
    retrieved_idx = retrieve_similar_indices(
        i, df, embeddings, n_sim=n_sim, window_days=window_days
    )
    base_question = df.loc[i, "title"]
    retrieved_questions = [df.loc[idx, "title"] for idx in retrieved_idx]

    theme_source = retrieved_questions or [base_question]
    theme_items, theme_json = build_theme_items(
        theme_source, theme_prefix=f"{i}", max_themes=max_themes
    )

    row_id = df.loc[i, "id"] if "id" in df.columns else i
    if pd.isna(row_id):
        row_id = i
    base_q_id = f"q::{row_id}"
    question_graphs = {}
    if extract_question_graphs:
        try:
            question_graphs[base_q_id] = extract_question_graph(
                base_question,
                model_name=graph_model_name,
                chunk_size=graph_chunk_size,
                chunk_overlap=graph_chunk_overlap,
            )
        except Exception as exc:
            warnings.warn(f"Question graph extraction failed: {exc}")

    return {
        "index": int(i),
        "question_id": base_q_id,
        "time": df.loc[i, "time"],
        "question": base_question,
        "retrieved_indices": retrieved_idx,
        "retrieved_questions": retrieved_questions,
        "theme_source_questions": theme_source,
        "themes": theme_items,
        "theme_json": theme_json,
        "question_graphs": question_graphs,
    }


def build_graph(profiles):
    nodes = {}
    edges = []

    for profile in profiles:
        themes = profile.get("themes", [])
        for theme in themes:
            theme_id = theme["id"]
            nodes[theme_id] = {
                "id": theme_id,
                "label": theme["text"],
                "type": "theme",
                "vector_text": theme["vector_text"],
                "question_id": profile.get("question_id", f"q::{profile['index']}"),
            }

        question_graphs = profile.get("question_graphs", {})
        for q_id, q_graph in question_graphs.items():
            if not q_graph:
                continue
            entity_nodes = q_graph.get("nodes", [])
            rel_edges = q_graph.get("edges", [])
            rel_labels = q_graph.get("relations", [])
            if len(rel_labels) < len(rel_edges):
                rel_labels = rel_labels + [""] * (len(rel_edges) - len(rel_labels))

            for entity in entity_nodes:
                entity_id = f"{q_id}::{entity}"
                nodes[entity_id] = {
                    "id": entity_id,
                    "label": entity,
                    "type": "entity",
                    "question_id": q_id,
                }

            for (source, target), relation in zip(rel_edges, rel_labels):
                source_id = f"{q_id}::{source}"
                target_id = f"{q_id}::{target}"
                nodes.setdefault(
                    source_id,
                    {
                        "id": source_id,
                        "label": source,
                        "type": "entity",
                        "question_id": q_id,
                    },
                )
                nodes.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "label": target,
                        "type": "entity",
                        "question_id": q_id,
                    },
                )
                edges.append(
                    {
                        "source": source_id,
                        "relation": relation,
                        "target": target_id,
                        "kind": "entity_relation",
                    }
                )

    nodes_df = pd.DataFrame(nodes.values())
    if edges:
        edges_df = pd.DataFrame(edges)
        edges_df["triple"] = (
            edges_df["source"]
            + " > "
            + edges_df["relation"]
            + " > "
            + edges_df["target"]
        )
    else:
        edges_df = pd.DataFrame(columns=["source", "relation", "target", "kind", "triple"])

    return nodes_df, edges_df


def plot_merged_graph(nodes_df, edges_df):
    if nodes_df.empty:
        print("No nodes to plot.")
        return

    graph = nx.Graph()
    for _, row in nodes_df.iterrows():
        graph.add_node(row["id"], label=row.get("label", ""), type=row.get("type", ""))

    for _, row in edges_df.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            label=row.get("relation", ""),
            kind=row.get("kind", ""),
        )

    plt.figure(figsize=(12, 8))
    layout = nx.spring_layout(graph, seed=42)

    node_colors = []
    for node in graph.nodes():
        ntype = graph.nodes[node].get("type", "")
        if ntype == "theme":
            node_colors.append("#ffd166")
        elif ntype == "entity":
            node_colors.append("#c3e88d")
        else:
            node_colors.append("#dddddd")

    nx.draw_networkx_nodes(graph, layout, node_size=700, node_color=node_colors)
    nx.draw_networkx_edges(graph, layout, width=1.2, edge_color="#555")

    labels = {node: data.get("label", node) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, layout, labels=labels, font_size=8)

    edge_labels = nx.get_edge_attributes(graph, "label")
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    plt.show()


def reduce_nodes_df(nodes_df):
    if nodes_df.empty:
        return nodes_df

    if "label" in nodes_df.columns:
        label_texts = nodes_df["label"].fillna(nodes_df["id"]).astype(str).tolist()
    else:
        label_texts = nodes_df["id"].astype(str).tolist()

    embeddings = embed_texts(label_texts)
    question_ids = (
        nodes_df["question_id"].tolist() if "question_id" in nodes_df.columns else [None] * len(nodes_df)
    )
    reduced = pd.DataFrame(
        {
            "id": nodes_df["id"].tolist(),
            "question_id": question_ids,
            "embedding": [emb.tolist() for emb in embeddings],
        }
    )
    return reduced


def cleanup_fn(nodes_df, edges_df, question_time_map, asof=None, smallest_percent=0.01):
    if nodes_df.empty:
        return nodes_df, edges_df

    G = nx.Graph()
    G.add_nodes_from(nodes_df["id"].tolist())
    if not edges_df.empty:
        for _, row in edges_df.iterrows():
            G.add_edge(row["source"], row["target"])

    components = [list(c) for c in nx.connected_components(G)]
    if not components:
        return nodes_df, edges_df

    if asof is None:
        times = pd.to_datetime(list(question_time_map.values()), errors="coerce")
        asof = times.max()
        if pd.isna(asof):
            asof = pd.Timestamp.utcnow()

    id_to_time = {}
    if "question_id" in nodes_df.columns:
        for _, row in nodes_df.iterrows():
            qid = row.get("question_id")
            if isinstance(qid, list):
                qids = [item for item in qid if item is not None]
            else:
                qids = [qid] if qid is not None else []
            for q in qids:
                t = question_time_map.get(q)
                if t is not None:
                    id_to_time[row["id"]] = pd.to_datetime(t, errors="coerce", utc=True)
                    break

    components.sort(key=len)
    cutoff = max(1, math.ceil(len(components) * smallest_percent))
    target_components = components[:cutoff]

    remove_ids = set()
    for comp in target_components:
        times = [id_to_time.get(n) for n in comp]
        times = [t for t in times if pd.notna(t)]
        if not times:
            continue

        buckets = [f"{t.year:04d}-{t.month:02d}" for t in times]
        mode_bucket = Counter(buckets).most_common(1)[0][0]
        mode_time = pd.to_datetime(mode_bucket + "-01", utc=True)


        if (asof - mode_time) >= pd.Timedelta(days=365):
            remove_ids.update(comp)

    cleaned_nodes_df = nodes_df[~nodes_df["id"].isin(remove_ids)].copy()
    if edges_df.empty:
        cleaned_edges_df = edges_df
    else:
        cleaned_edges_df = edges_df[
            edges_df["source"].isin(cleaned_nodes_df["id"])
            & edges_df["target"].isin(cleaned_nodes_df["id"])
        ].copy()

    return cleaned_nodes_df, cleaned_edges_df


def run_iterative_build(
    sample_size,
    df,
    title_embeddings,
    n_sim=None,
    window_days=7,
    extract_question_graphs=True,
    graph_model_name=GRAPH_MODEL_NAME,
    graph_chunk_size=QUESTION_GRAPH_CHUNK_SIZE,
    graph_chunk_overlap=QUESTION_GRAPH_CHUNK_OVERLAP,
    max_themes=MAX_THEMES_PER_QUERY,
    node_th1=NODE_TH1,
    node_th2=NODE_TH2,
    node_max_pairs=NODE_MAX_PAIRS,
    node_relation_questions=NODE_RELATION_QUESTIONS,
):
    profiles = []
    final_nodes_df = pd.DataFrame()
    final_edges_df = pd.DataFrame()
    last_cleanup_year = None
    last_build_time = None

    for i in range(sample_size):
        profile = build_profile_for_index(
            i,
            df,
            title_embeddings,
            n_sim=n_sim,
            window_days=window_days,
            extract_question_graphs=extract_question_graphs,
            graph_model_name=graph_model_name,
            graph_chunk_size=graph_chunk_size,
            graph_chunk_overlap=graph_chunk_overlap,
            max_themes=max_themes,
        )
        profiles.append(profile)

        current_time = pd.to_datetime(profile["time"], errors="coerce")

        if last_build_time is None or (
            pd.notna(current_time)
            and current_time - last_build_time >= pd.Timedelta(days=14)
        ):
            final_nodes_df, final_edges_df = build_healed_graph(
                profiles,
                build_graph,
                node_th1=node_th1,
                node_th2=node_th2,
                node_max_pairs=node_max_pairs,
                node_relation_questions=node_relation_questions,
                new_question_id=profile.get("question_id", f"q::{i}"),
            )

            save_graph_json(final_nodes_df, final_edges_df, i)
            # plot_merged_graph(final_nodes_df, final_edges_df)
            final_nodes_df = reduce_nodes_df(final_nodes_df)

            if pd.notna(current_time):
                last_build_time = current_time

        if pd.notna(current_time):
            year = int(current_time.year)
            if last_cleanup_year is None:
                last_cleanup_year = year
            elif year != last_cleanup_year:
                question_time_map = {
                    p.get("question_id", f"q::{p['index']}"): p.get("time")
                    for p in profiles
                }
                final_nodes_df, final_edges_df = cleanup_fn(
                    final_nodes_df, final_edges_df, question_time_map, asof=current_time
                )
                last_cleanup_year = year

    return profiles, final_nodes_df, final_edges_df

def save_graph_json(nodes_df, edges_df, step, out_dir="snapshot_dir"):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "nodes": nodes_df.to_dict(orient="records"),
        "edges": edges_df.to_dict(orient="records"),
    }
    path = os.path.join(out_dir, f"healed_graph_{step:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path
