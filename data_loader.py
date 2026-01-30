"""JacRED data loading, document selection, and constraint table construction."""

import json
from collections import defaultdict


def load_jacred(base_path: str = "/tmp/JacRED/") -> dict:
    """Load all JacRED splits and metadata."""
    data = {}
    for split in ["train", "dev", "test"]:
        with open(f"{base_path}{split}.json", encoding="utf-8") as f:
            data[split] = json.load(f)

    with open(f"{base_path}meta/rel2id.json", encoding="utf-8") as f:
        data["rel2id"] = json.load(f)
    with open(f"{base_path}meta/ent2id.json", encoding="utf-8") as f:
        data["ent2id"] = json.load(f)
    with open(f"{base_path}meta/rel_info.json", encoding="utf-8") as f:
        data["rel_info"] = json.load(f)

    return data


def doc_to_text(doc: dict) -> str:
    """Convert tokenized sentences to plain text."""
    sentences = []
    for sent in doc["sents"]:
        sentences.append("".join(sent))
    return "".join(sentences)


def char_count(doc: dict) -> int:
    """Count total characters in a document."""
    return sum(len(tok) for sent in doc["sents"] for tok in sent)


def select_dev_docs(dev_data: list, n: int = 10) -> list[dict]:
    """Select n docs stratified by document size."""
    sorted_docs = sorted(dev_data, key=char_count)
    total = len(sorted_docs)
    indices = [int(total * (i + 0.5) / n) for i in range(n)]
    selected = []
    for idx in indices:
        doc = sorted_docs[idx].copy()
        doc["doc_text"] = doc_to_text(doc)
        selected.append(doc)
    return selected


def select_few_shot(train_data: list) -> dict:
    """Select a short, clear document as few-shot example."""
    candidates = []
    for doc in train_data:
        n_ents = len(doc["vertexSet"])
        n_labels = len(doc.get("labels", []))
        chars = char_count(doc)
        if 150 <= chars <= 250 and 5 <= n_ents <= 12 and 3 <= n_labels <= 15:
            candidates.append((chars, doc))

    candidates.sort(key=lambda x: x[0])
    if not candidates:
        # Fallback: pick shortest doc with at least some labels
        sorted_train = sorted(train_data, key=char_count)
        for doc in sorted_train:
            if len(doc.get("labels", [])) >= 3:
                doc = doc.copy()
                doc["doc_text"] = doc_to_text(doc)
                return doc

    doc = candidates[0][1].copy()
    doc["doc_text"] = doc_to_text(doc)
    return doc


def format_few_shot_output(doc: dict) -> dict:
    """Convert a JacRED doc into the expected extraction output format for few-shot."""
    entities = []
    for i, vs in enumerate(doc["vertexSet"]):
        entities.append({
            "id": f"e{i}",
            "name": vs[0]["name"],
            "type": vs[0]["type"],
        })

    relations = []
    for label in doc.get("labels", []):
        h_name = doc["vertexSet"][label["h"]][0]["name"]
        t_name = doc["vertexSet"][label["t"]][0]["name"]
        evidence_sents = []
        for sid in label.get("evidence", []):
            if sid < len(doc["sents"]):
                evidence_sents.append("".join(doc["sents"][sid]))
        relations.append({
            "head": f"e{label['h']}",
            "relation": label["r"],
            "tail": f"e{label['t']}",
            "evidence": "".join(evidence_sents),
        })

    return {"entities": entities, "relations": relations}


def build_constraint_table(train_data: list) -> dict[str, set[tuple[str, str]]]:
    """Build domain/range type constraint table from training data."""
    table = defaultdict(set)
    for doc in train_data:
        for label in doc.get("labels", []):
            h_type = doc["vertexSet"][label["h"]][0]["type"]
            t_type = doc["vertexSet"][label["t"]][0]["type"]
            table[label["r"]].add((h_type, t_type))
    return dict(table)
