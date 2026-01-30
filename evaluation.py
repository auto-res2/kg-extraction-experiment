"""Entity alignment and P/R/F1 evaluation against JacRED gold labels."""

import unicodedata
from extraction import Triple


def _normalize(s: str) -> str:
    """Normalize text for matching: NFKC + lowercase + strip."""
    return unicodedata.normalize("NFKC", s).strip().lower()


def align_entities(
    predicted_entities: list[dict],
    gold_vertex_set: list[list[dict]],
) -> dict[str, int]:
    """Map predicted entity names to gold vertexSet indices.

    Returns {predicted_entity_id: gold_vertex_index}.
    Uses 3-pass matching: exact -> normalized -> substring.
    """
    alignment = {}
    used_gold = set()

    # Build lookup: gold index -> list of mention names
    gold_mentions = {}
    for idx, mentions in enumerate(gold_vertex_set):
        gold_mentions[idx] = [m["name"] for m in mentions]

    pred_list = [(e["id"], e["name"]) for e in predicted_entities]

    # Pass 1: Exact match
    for pred_id, pred_name in pred_list:
        if pred_id in alignment:
            continue
        for gold_idx, names in gold_mentions.items():
            if gold_idx in used_gold:
                continue
            if pred_name in names:
                alignment[pred_id] = gold_idx
                used_gold.add(gold_idx)
                break

    # Pass 2: Normalized match
    for pred_id, pred_name in pred_list:
        if pred_id in alignment:
            continue
        pred_norm = _normalize(pred_name)
        for gold_idx, names in gold_mentions.items():
            if gold_idx in used_gold:
                continue
            for gname in names:
                if _normalize(gname) == pred_norm:
                    alignment[pred_id] = gold_idx
                    used_gold.add(gold_idx)
                    break
            if pred_id in alignment:
                break

    # Pass 3: Substring match (prefer longest overlap)
    for pred_id, pred_name in pred_list:
        if pred_id in alignment:
            continue
        pred_norm = _normalize(pred_name)
        best_gold = None
        best_overlap = 0
        for gold_idx, names in gold_mentions.items():
            if gold_idx in used_gold:
                continue
            for gname in names:
                gnorm = _normalize(gname)
                if pred_norm in gnorm or gnorm in pred_norm:
                    overlap = min(len(pred_norm), len(gnorm))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_gold = gold_idx
        if best_gold is not None and best_overlap >= 2:
            alignment[pred_id] = best_gold
            used_gold.add(best_gold)

    return alignment


def evaluate_relations(
    predicted_triples: list[Triple],
    gold_labels: list[dict],
    entity_alignment: dict[str, int],
) -> dict:
    """Compute P, R, F1 by comparing predicted triples against gold labels.

    Returns dict with precision, recall, f1, tp, fp, fn, and detail lists.
    """
    # Build gold set: {(head_idx, tail_idx, relation)}
    gold_set = set()
    for label in gold_labels:
        gold_set.add((label["h"], label["t"], label["r"]))

    # Evaluate predictions
    tp = 0
    fp = 0
    fp_details = []
    matched_gold = set()

    for triple in predicted_triples:
        h_idx = entity_alignment.get(triple.head)
        t_idx = entity_alignment.get(triple.tail)

        if h_idx is None or t_idx is None:
            fp += 1
            fp_details.append({
                "head": triple.head_name,
                "relation": triple.relation,
                "tail": triple.tail_name,
                "reason": "entity_not_aligned",
            })
            continue

        key = (h_idx, t_idx, triple.relation)
        if key in gold_set:
            tp += 1
            matched_gold.add(key)
        else:
            fp += 1
            fp_details.append({
                "head": triple.head_name,
                "relation": triple.relation,
                "tail": triple.tail_name,
                "reason": "wrong_relation",
            })

    fn = len(gold_set) - len(matched_gold)
    fn_details = []
    for h, t, r in gold_set:
        if (h, t, r) not in matched_gold:
            fn_details.append({"head_idx": h, "tail_idx": t, "relation": r})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_details": fp_details,
        "fn_details": fn_details,
    }


def aggregate_results(per_doc: list[dict]) -> dict:
    """Micro-average P/R/F1 across documents."""
    total_tp = sum(d["tp"] for d in per_doc)
    total_fp = sum(d["fp"] for d in per_doc)
    total_fn = sum(d["fn"] for d in per_doc)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
