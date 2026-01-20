import warnings

import numpy as np
import pandas as pd

from embedding_utils import embed_texts
from theme_extractor import llm_text

NODE_TH1 = 0.85
NODE_TH2 = 0.7
NODE_MAX_PAIRS = 500
NODE_RELATION_QUESTIONS = 2


def _node_label_texts(nodes_df):
    if "label" in nodes_df.columns:
        return nodes_df["label"].fillna(nodes_df["id"]).astype(str).tolist()
    return nodes_df["id"].astype(str).tolist()


def _build_context(node_id, label_map, edges_df, max_edges=4):
    if edges_df.empty:
        return ""

    matches = edges_df[
        (edges_df["source"] == node_id) | (edges_df["target"] == node_id)
    ]
    lines = []
    for _, row in matches.head(max_edges).iterrows():
        source = label_map.get(row["source"], row["source"])
        target = label_map.get(row["target"], row["target"])
        relation = row.get("relation", "")
        if relation:
            lines.append(f"{source} -[{relation}]-> {target}")
        else:
            lines.append(f"{source} -> {target}")
    return "\n".join(lines)


def _collect_question_texts(node_id, nodes_df, edges_df, question_map, max_questions=2):
    if not question_map or edges_df.empty:
        return []

    connected = edges_df[
        (edges_df["source"] == node_id) | (edges_df["target"] == node_id)
    ]
    if connected.empty:
        return []

    node_ids = pd.unique(connected[["source", "target"]].values.ravel("K")).tolist()
    if "question_id" in nodes_df.columns:
        qids = nodes_df.set_index("id").reindex(node_ids)["question_id"].dropna().tolist()
    else:
        qids = []

    questions = []
    seen = set()
    for qid in qids:
        if qid in seen:
            continue
        text = question_map.get(qid)
        if not text:
            continue
        questions.append(text)
        seen.add(qid)
        if len(questions) >= max_questions:
            break
    return questions


def _format_questions(questions):
    if not questions:
        return "None"
    return "\n".join(f"- {q}" for q in questions)


def _flatten_question_ids(value):
    if value is None:
        return []
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_flatten_question_ids(item))
        return flattened
    return [value]


def extract_node_relation(label_a, label_b, context_a, context_b, questions_a, questions_b):
    system = "You are a concise analyst."
    prompt = (
        "Using only the provided context, describe the relationship between Node A "
        "and Node B in a short phrase (max 8 words). If unclear, return 'related'.\n\n"
        f"Node A: {label_a}\n"
        f"Context A:\n{context_a or 'None'}\n\n"
        f"Questions A:\n{_format_questions(questions_a)}\n\n"
        f"Node B: {label_b}\n"
        f"Context B:\n{context_b or 'None'}\n\n"
        f"Questions B:\n{_format_questions(questions_b)}"
    )
    text = llm_text(prompt, system, fallback=lambda: "related")
    return text.strip() or "related"


def extract_merge_label(label_a, label_b, context_a, context_b, questions_a, questions_b):
    system = "You output short labels only."
    prompt = (
        "Using only the provided context, produce a single merged label for Node A "
        "and Node B (1-3 words). No punctuation.\n\n"
        f"Node A: {label_a}\n"
        f"Context A:\n{context_a or 'None'}\n\n"
        f"Questions A:\n{_format_questions(questions_a)}\n\n"
        f"Node B: {label_b}\n"
        f"Context B:\n{context_b or 'None'}\n\n"
        f"Questions B:\n{_format_questions(questions_b)}"
    )
    fallback = lambda: label_a or label_b or "merged node"
    text = llm_text(prompt, system, fallback=fallback)
    return (text.strip() or fallback()).replace(".", "")

def build_node_links(
    nodes_df,
    edges_df,
    question_map,
    th1=NODE_TH1,
    th2=NODE_TH2,
    max_nodes_for_pairwise=NODE_MAX_PAIRS,
    max_relation_questions=NODE_RELATION_QUESTIONS,
    new_node_ids=None,

):
    if nodes_df.empty:
        return []

    node_ids = nodes_df["id"].tolist()
    label_texts = _node_label_texts(nodes_df)

    if len(node_ids) > max_nodes_for_pairwise:
        warnings.warn(
            "Skipping embedding-based healing due to size; increase NODE_MAX_PAIRS to override."
        )
        return []
    new_set = set(new_node_ids) if new_node_ids else None

    embeddings = embed_texts(label_texts)
    label_map = {node_id: label for node_id, label in zip(node_ids, label_texts)}

    links = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if new_set:
                a_new = node_ids[i] in new_set
                b_new = node_ids[j] in new_set
                if a_new == b_new:
                    continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            
            if sim >= th1:
                links.append(
                    {
                        "node_id_a": node_ids[i],
                        "node_id_b": node_ids[j],
                        "relation": "same",
                        "similarity": sim,
                        "note": "merge",
                    }
                )
            elif sim > th2:
                context_a = _build_context(node_ids[i], label_map, edges_df)
                context_b = _build_context(node_ids[j], label_map, edges_df)
                questions_a = _collect_question_texts(
                    node_ids[i],
                    nodes_df,
                    edges_df,
                    question_map,
                    max_questions=max_relation_questions,
                )
                questions_b = _collect_question_texts(
                    node_ids[j],
                    nodes_df,
                    edges_df,
                    question_map,
                    max_questions=max_relation_questions,
                )
                relation = extract_node_relation(
                    label_texts[i],
                    label_texts[j],
                    context_a,
                    context_b,
                    questions_a,
                    questions_b,
                )
                links.append(
                    {
                        "node_id_a": node_ids[i],
                        "node_id_b": node_ids[j],
                        "relation": "related",
                        "similarity": sim,
                        "note": relation,
                    }
                )

    return links


def merge_nodes(node_links, node_ids):
    parent = {nid: nid for nid in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for link in node_links:
        if link["relation"] == "same":
            union(link["node_id_a"], link["node_id_b"])

    groups = {}
    for nid in node_ids:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    canonical = {}
    for root, members in groups.items():
        canonical_id = sorted(members)[0]
        for nid in members:
            canonical[nid] = canonical_id

    return canonical, groups


def apply_node_healing(
    nodes_df,
    edges_df,
    node_links,
    node_mapping,
    node_groups,
    question_map,
    max_relation_questions,
):
    def map_node(node_id):
        return node_mapping.get(node_id, node_id)

    healed_edges = edges_df.copy()
    if not healed_edges.empty:
        healed_edges["source"] = healed_edges["source"].map(map_node)
        healed_edges["target"] = healed_edges["target"].map(map_node)
        healed_edges = healed_edges[healed_edges["source"] != healed_edges["target"]]
        healed_edges["triple"] = (
            healed_edges["source"]
            + " > "
            + healed_edges["relation"]
            + " > "
            + healed_edges["target"]
        )

    label_texts = _node_label_texts(nodes_df)
    label_map = {node_id: label for node_id, label in zip(nodes_df["id"], label_texts)}
    merged_overrides = {}
    for root, members in node_groups.items():
        if len(members) < 2:
            continue
        canonical_id = node_mapping.get(members[0], members[0])
        n1, n2 = members[0], members[1]
        context_a = _build_context(n1, label_map, edges_df)
        context_b = _build_context(n2, label_map, edges_df)
        questions_a = _collect_question_texts(
            n1,
            nodes_df,
            edges_df,
            question_map,
            max_questions=max_relation_questions,
        )
        questions_b = _collect_question_texts(
            n2,
            nodes_df,
            edges_df,
            question_map,
            max_questions=max_relation_questions,
        )
        merged_label = extract_merge_label(
            label_map.get(n1, n1),
            label_map.get(n2, n2),
            context_a,
            context_b,
            questions_a,
            questions_b,
        )
        question_ids = []
        if "question_id" in nodes_df.columns:
            for node_id in members:
                qid = nodes_df.loc[nodes_df["id"] == node_id, "question_id"]
                if not qid.empty:
                    question_ids.extend(_flatten_question_ids(qid.iloc[0]))
        question_ids = list(dict.fromkeys(q for q in question_ids if q))
        merged_embedding = embed_texts([merged_label])[0].tolist()
        merged_overrides[canonical_id] = {
            "label": merged_label,
            "question_id": question_ids,
            "embedding": merged_embedding,
        }

    healed_nodes = {}
    for _, row in nodes_df.iterrows():
        node_id = row["id"]
        mapped_id = node_mapping.get(node_id, node_id)
        if mapped_id not in healed_nodes:
            new_row = row.to_dict()
            new_row["id"] = mapped_id
            if mapped_id in merged_overrides:
                new_row.update(merged_overrides[mapped_id])
            healed_nodes[mapped_id] = new_row

    related_edges = []
    for link in node_links:
        if link["relation"] != "related":
            continue
        src = node_mapping.get(link["node_id_a"], link["node_id_a"])
        tgt = node_mapping.get(link["node_id_b"], link["node_id_b"])
        if src == tgt:
            continue
        related_edges.append(
            {
                "source": src,
                "relation": link["note"] or "related",
                "target": tgt,
                "kind": "node_relation",
            }
        )

    if related_edges:
        rel_df = pd.DataFrame(related_edges)
        rel_df["triple"] = (
            rel_df["source"] + " > " + rel_df["relation"] + " > " + rel_df["target"]
        )
        healed_edges = pd.concat([healed_edges, rel_df], ignore_index=True)

    healed_nodes_df = pd.DataFrame(healed_nodes.values())
    return healed_nodes_df, healed_edges


def build_healed_graph(
    profiles,
    build_graph_fn,
    node_th1=NODE_TH1,
    node_th2=NODE_TH2,
    node_max_pairs=NODE_MAX_PAIRS,
    node_relation_questions=NODE_RELATION_QUESTIONS,
    new_question_id=None):
    if not profiles:
        empty_nodes = pd.DataFrame(
            columns=["id", "label", "type", "vector_text", "question_id"]
        )
        empty_edges = pd.DataFrame(columns=["source", "relation", "target", "kind", "triple"])
        return empty_nodes, empty_edges

    if build_graph_fn is None:
        raise ValueError("build_graph_fn is required to build the base graph.")

    base_nodes_df, base_edges_df = build_graph_fn(profiles)
    new_node_ids = None
    if new_question_id:
        new_node_ids = base_nodes_df.loc[
            base_nodes_df["question_id"] == new_question_id, "id"
        ].tolist()
    question_map = {f"q::{p['index']}": p.get("question", "") for p in profiles}
    node_links = build_node_links(
        base_nodes_df,
        base_edges_df,
        question_map,
        th1=node_th1,
        th2=node_th2,
        max_nodes_for_pairwise=node_max_pairs,
        max_relation_questions=node_relation_questions,
        new_node_ids=new_node_ids
    )
    if not node_links:
        return base_nodes_df, base_edges_df

    node_ids = base_nodes_df["id"].tolist()
    node_mapping, node_groups = merge_nodes(node_links, node_ids)
    final_nodes_df, final_edges_df = apply_node_healing(
        base_nodes_df,
        base_edges_df,
        node_links,
        node_mapping,
        node_groups,
        question_map,
        node_relation_questions,
    )
    return final_nodes_df, final_edges_df


