import json
import re
import warnings

from embedding_utils import embed_texts
from model_paths import resolve_model_name

_LLM_CACHE = {}
_LLM_MODEL = None
_LLM_TOKENIZER = None

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "how",
    "what",
    "when",
    "where",
    "why",
    "are",
    "can",
    "best",
    "use",
    "using",
    "into",
    "about",
    "your",
    "their",
    "they",
    "you",
    "vs",
}


def _keywords(text, k=8):
    tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if not tokens:
        return []
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:k]]


def _fallback_theme_vector(theme_text, questions):
    kw = _keywords(" ".join(questions))
    if not kw:
        kw = _keywords(theme_text)
    if not kw:
        kw = ["general", "inquiry"]
    return ", ".join(kw[:8])


def _normalize_vector_text(text):
    tokens = [t.strip().lower() for t in re.split(r"[,;\n]+", str(text))]
    tokens = [t for t in tokens if t]
    return ", ".join(tokens[:10])


def _fallback_themes(questions, max_themes=4):
    kw = _keywords(" ".join(questions), k=max_themes)
    if not kw:
        kw = ["general"]
    return {f"theme{i + 1}": term for i, term in enumerate(kw[:max_themes])}


def _parse_theme_json(text, max_themes=4):
    try:
        payload = json.loads(text)
    except Exception:
        return None

    items = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            label = str(value).strip() if value is not None else ""
            if label:
                items.append((str(key), label))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            label = str(value).strip() if value is not None else ""
            if label:
                items.append((f"theme{idx + 1}", label))
    else:
        return None

    if not items:
        return None

    if any(re.search(r"\d+$", key) for key, _ in items):
        def _sort_key(item):
            match = re.search(r"(\d+)$", item[0])
            return int(match.group(1)) if match else 10**9
        items = sorted(items, key=_sort_key)

    items = items[:max_themes]
    return {key: value for key, value in items}


def _get_llama(model_name=None):
    global _LLM_MODEL, _LLM_TOKENIZER
    if _LLM_MODEL is not None and _LLM_TOKENIZER is not None:
        return _LLM_TOKENIZER, _LLM_MODEL

    model_name = model_name or resolve_model_name(
        "LLM_MODEL_NAME", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        _LLM_TOKENIZER = tokenizer
        _LLM_MODEL = model
        return tokenizer, model
    except Exception as exc:
        warnings.warn(f"Llama load failed: {exc}")
        return None, None


def llm_text(prompt, system, model_name=None, temperature=0.2, max_new_tokens=128, fallback=None):
    cache_key = (system, prompt, model_name, temperature, max_new_tokens)
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    tokenizer, model = _get_llama(model_name=model_name)
    text = None
    if tokenizer and model:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
        do_sample = temperature > 0
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        output_ids = generated[0][len(model_inputs.input_ids[0]) :].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if not text:
        if fallback is not None:
            text = fallback()
        else:
            text = "Fallback summary: " + (" ".join(_keywords(prompt)) or "general")

    _LLM_CACHE[cache_key] = text
    return text


def extract_themes(questions, max_themes=4):
    system = "You output JSON only."
    prompt = (
        "Return a JSON object with keys theme1..themeN. "
        "Values must be short theme labels (1-3 words). "
        f"Return 1 to {max_themes} themes based on the questions. "
        "No extra text.\n\n"
        + "\n".join(f"- {q}" for q in questions)
    )
    fallback_payload = lambda: json.dumps(_fallback_themes(questions, max_themes=max_themes))
    text = llm_text(prompt, system, fallback=fallback_payload)
    parsed = _parse_theme_json(text, max_themes=max_themes)
    if not parsed:
        parsed = _fallback_themes(questions, max_themes=max_themes)
    return parsed


def extract_theme(questions):
    themes = extract_themes(questions, max_themes=1)
    return next(iter(themes.values()))


def extract_theme_vector(theme_text, questions):
    system = "You output keyword vectors only."
    prompt = (
        "Turn the theme into a compact concept vector. "
        "Return 6-10 comma-separated keywords or short phrases; no prose.\n\n"
        f"Theme: {theme_text}\n"
        + "\n".join(f"- {q}" for q in questions)
    )
    text = llm_text(
        prompt,
        system,
        fallback=lambda: _fallback_theme_vector(theme_text, questions),
    )
    normalized = _normalize_vector_text(text)
    return normalized or _fallback_theme_vector(theme_text, questions)


def extract_theme_relation(theme_a, theme_b):
    system = "You are a concise analyst."
    prompt = (
        "Describe the relationship between the two themes in one short sentence.\n\n"
        f"Theme A: {theme_a}\n"
        f"Theme B: {theme_b}"
    )
    return llm_text(prompt, system)


def build_theme_items(theme_source, theme_prefix, max_themes=4):
    theme_json = extract_themes(theme_source, max_themes=max_themes)
    theme_items = []
    for idx, (key, theme_text) in enumerate(theme_json.items()):
        theme_vector_text = extract_theme_vector(theme_text, theme_source)
        theme_embedding = embed_texts([theme_vector_text])[0]
        theme_items.append(
            {
                "id": f"{theme_prefix}::{idx + 1}",
                "key": key,
                "text": theme_text,
                "vector_text": theme_vector_text,
                "embedding": theme_embedding,
            }
        )
    return theme_items, theme_json
