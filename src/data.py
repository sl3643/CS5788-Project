from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from src.utils import save_jsonl, resolve_path


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the original text content
    """
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    """
    Count whitespace-separated words in a text string
    """
    return len(text.split())


def is_good_text(text: str, min_words: int, max_words: int) -> bool:
    """
    Return whether a text is usable for sentence-level representation analysis
    """
    if not text:
        return False

    word_count = count_words(text)
    if word_count < min_words or word_count > max_words:
        return False

    # WikiText contains heading-like lines such as '= Title ='. These are not
    # useful as natural-language examples
    if text.startswith("=") and text.endswith("="):
        return False

    # Remove examples that are mostly punctuation or formatting artifacts
    num_alpha = sum(ch.isalpha() for ch in text)
    if num_alpha < 0.5 * max(1, len(text.replace(" ", ""))):
        return False

    return True


def load_and_filter_texts(data_cfg: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """
    Load, filter, and sample texts according to the data config
    """
    dataset_name = data_cfg["dataset_name"]
    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "test")
    text_field = data_cfg.get("text_field", "text")
    num_texts = int(data_cfg.get("num_texts", 300))
    min_words = int(data_cfg.get("min_words", 8))
    max_words = int(data_cfg.get("max_words", 80))

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    candidates: list[str] = []
    seen: set[str] = set()

    for row in dataset:
        raw_text = row.get(text_field, "")
        text = normalize_text(str(raw_text))

        if not is_good_text(text, min_words=min_words, max_words=max_words):
            continue

        if text in seen:
            continue

        seen.add(text)
        candidates.append(text)

    if len(candidates) < num_texts:
        raise ValueError(
            f"Only found {len(candidates)} usable texts, but config asks for {num_texts}. "
            "Lower num_texts or relax min_words/max_words."
        )

    rng = random.Random(seed)
    sampled = rng.sample(candidates, num_texts)

    records = [
        {
            "id": idx,
            "text": text,
            "num_words": count_words(text),
        }
        for idx, text in enumerate(sampled)
    ]

    return records


def prepare_dataset(config: Dict[str, Any]) -> Path:
    """
    Prepare and save the processed text dataset
    Returns:
        Path to the saved JSONL file
    """
    seed = int(config.get("project", {}).get("seed", 42))
    data_cfg = config["data"]
    output_path = resolve_path(data_cfg["processed_path"])

    records = load_and_filter_texts(data_cfg, seed=seed)
    save_jsonl(records, output_path)

    print(f"Saved {len(records)} texts to {output_path}")
    print("First example:")
    print(f"  {records[0]['text'][:200]}")

    return output_path