import argparse
import json
import logging
import os
import re
from typing import Dict, List, Tuple

import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ENTITY_EXTRACTION_PROMPT = """
Extract all entities and their relationships from the text.
Respond in the following format, and only this format:

Entities:
- <Entity 1>
- <Entity 2>
...

Relationships:
- <Entity 1> -> <Relationship> -> <Entity 2>
- <Entity 3> -> <Relationship> -> <Entity 4>
...
"""

REL_PATTERN = re.compile(
    r"^\s*-\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+?)\s*$", re.MULTILINE
)


class LLMHandler:
    """Initialize and interact with the LLM, matching the repo behavior."""

    def __init__(self, model_name: str):
        logging.info("Initializing LLMHandler with model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.device = self.model.device
        logging.info("Model loaded on device: %s", self.device)

    def get_response(self, prompt: str, content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for text analysis. "
                    "Follow the user's instructions precisely."
                ),
            },
            {"role": "user", "content": f"{content}\n\n---\n\n{prompt}"},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks similar to the indexing pipeline."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def extract_entities_relationships(chunks: List[str], llm: LLMHandler) -> List[str]:
    """Use the LLM to extract entities and relationships from each chunk."""
    elements = []
    for index, chunk in enumerate(chunks):
        logging.info("Processing chunk %s/%s...", index + 1, len(chunks))
        response = llm.get_response(ENTITY_EXTRACTION_PROMPT, chunk)
        elements.append(response)
    return elements


def build_knowledge_graph(element_summaries: List[str]) -> nx.Graph:
    """Parse the LLM output and build a NetworkX graph."""
    graph = nx.Graph()
    for i, summary in enumerate(element_summaries):
        if "Relationships:" not in summary:
            logging.warning("No 'Relationships:' block found in summary %s.", i)
            continue

        for match in REL_PATTERN.finditer(summary):
            source = match.group(1).strip()
            relation = match.group(2).strip()
            target = match.group(3).strip()

            if not source or not relation or not target:
                logging.warning("Skipping malformed relationship in summary %s.", i)
                continue

            graph.add_node(source)
            graph.add_node(target)
            graph.add_edge(source, target, label=relation)

    logging.info(
        "Knowledge graph built: %s nodes, %s edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def serialize_graph(graph: nx.Graph) -> Dict[str, List]:
    """Return nodes, edges, and relations in JSON-friendly form."""
    nodes = sorted(graph.nodes())
    edges = [(u, v) for u, v in graph.edges()]
    relations = [data.get("label", "") for _, _, data in graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "relations": relations}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw text to a knowledge graph using an LLM."
    )
    parser.add_argument("--input", required=True, help="Path to a text file.")
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model to use.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2000,
        help="Chunk size for LLM extraction.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Chunk overlap for LLM extraction.",
    )
    parser.add_argument(
        "--gml_out",
        default="knowledge_graph.gml",
        help="Path to write the NetworkX GML graph.",
    )
    parser.add_argument(
        "--json_out",
        default="graph.json",
        help="Path to write the JSON summary (nodes/edges/relations).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    llm = LLMHandler(model_name=args.model_name)
    element_summaries = extract_entities_relationships(chunks, llm)
    graph = build_knowledge_graph(element_summaries)

    nx.write_gml(graph, args.gml_out)
    logging.info("Wrote graph to: %s", args.gml_out)

    payload = serialize_graph(graph)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote JSON summary to: %s", args.json_out)


if __name__ == "__main__":
    main()
