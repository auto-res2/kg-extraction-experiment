"""Extraction logic for Baseline and Proposed (Two-Stage) conditions."""

from dataclasses import dataclass, field

from google import genai

from schemas import EXTRACTION_SCHEMA, VERIFICATION_SCHEMA
from prompts import build_system_prompt, build_extraction_prompt, build_verification_prompt
from llm_client import call_gemini
from data_loader import format_few_shot_output


@dataclass
class Triple:
    head: str        # entity id (e.g. "e0")
    head_name: str
    head_type: str
    relation: str    # P-code
    tail: str        # entity id
    tail_name: str
    tail_type: str
    evidence: str


def _parse_extraction_result(result: dict) -> tuple[list[dict], list[Triple]]:
    """Parse LLM extraction output into entities and triples."""
    entities = result.get("entities", [])
    id_to_entity = {e["id"]: e for e in entities}

    triples = []
    for rel in result.get("relations", []):
        head_ent = id_to_entity.get(rel["head"], {})
        tail_ent = id_to_entity.get(rel["tail"], {})
        if not head_ent or not tail_ent:
            continue
        triples.append(Triple(
            head=rel["head"],
            head_name=head_ent.get("name", ""),
            head_type=head_ent.get("type", ""),
            relation=rel["relation"],
            tail=rel["tail"],
            tail_name=tail_ent.get("name", ""),
            tail_type=tail_ent.get("type", ""),
            evidence=rel.get("evidence", ""),
        ))
    return entities, triples


def filter_invalid_labels(triples: list[Triple], valid_relations: set[str]) -> list[Triple]:
    """Remove triples with unknown relation P-codes."""
    return [t for t in triples if t.relation in valid_relations]


def filter_invalid_entity_types(triples: list[Triple], valid_types: set[str]) -> list[Triple]:
    """Remove triples with unknown entity types."""
    return [t for t in triples if t.head_type in valid_types and t.tail_type in valid_types]


def apply_domain_range_constraints(
    triples: list[Triple],
    constraint_table: dict[str, set[tuple[str, str]]],
) -> list[Triple]:
    """Remove triples where (head_type, tail_type) is not observed in training data."""
    filtered = []
    for t in triples:
        allowed = constraint_table.get(t.relation)
        if allowed is None:
            # Unknown relation, keep (already handled by filter_invalid_labels)
            filtered.append(t)
        elif (t.head_type, t.tail_type) in allowed:
            filtered.append(t)
    return filtered


def run_baseline(
    doc: dict,
    few_shot: dict,
    client: genai.Client,
    schema_info: dict,
) -> tuple[list[dict], list[Triple]]:
    """Condition 1: Single LLM call extraction."""
    system_prompt = build_system_prompt(schema_info["rel_info"])
    few_shot_output = format_few_shot_output(few_shot)
    user_prompt = build_extraction_prompt(
        doc["doc_text"], few_shot["doc_text"], few_shot_output, mode="baseline"
    )

    result = call_gemini(client, system_prompt, user_prompt, EXTRACTION_SCHEMA)
    entities, triples = _parse_extraction_result(result)

    valid_rels = set(schema_info["rel_info"].keys())
    valid_types = {"PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"}
    triples = filter_invalid_labels(triples, valid_rels)
    triples = filter_invalid_entity_types(triples, valid_types)

    return entities, triples


def run_proposed(
    doc: dict,
    few_shot: dict,
    client: genai.Client,
    schema_info: dict,
    constraint_table: dict,
) -> tuple[list[dict], list[Triple], dict]:
    """Condition 2: Two-stage Generate + Verify."""
    # Stage 1: Recall-oriented extraction
    system_prompt = build_system_prompt(schema_info["rel_info"])
    few_shot_output = format_few_shot_output(few_shot)
    user_prompt = build_extraction_prompt(
        doc["doc_text"], few_shot["doc_text"], few_shot_output, mode="recall"
    )

    result = call_gemini(client, system_prompt, user_prompt, EXTRACTION_SCHEMA)
    entities, candidates = _parse_extraction_result(result)

    valid_rels = set(schema_info["rel_info"].keys())
    valid_types = {"PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"}
    candidates = filter_invalid_labels(candidates, valid_rels)
    candidates = filter_invalid_entity_types(candidates, valid_types)

    stage1_count = len(candidates)

    # Stage 2: Verification in batches
    entity_id_to_name = {e["id"]: e["name"] for e in entities}
    verified = _verify_candidates(
        doc, candidates, entity_id_to_name, client, schema_info, batch_size=10
    )

    stage2_count = len(verified)

    # Post-processing: domain/range constraints
    final = apply_domain_range_constraints(verified, constraint_table)
    final_count = len(final)

    stats = {
        "stage1_candidates": stage1_count,
        "stage2_kept": stage2_count,
        "after_constraints": final_count,
    }
    return entities, final, stats


def _verify_candidates(
    doc: dict,
    candidates: list[Triple],
    entity_id_to_name: dict,
    client: genai.Client,
    schema_info: dict,
    batch_size: int = 10,
) -> list[Triple]:
    """Stage 2: Batch-verify candidates."""
    if not candidates:
        return []

    verified = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        batch_dicts = [
            {
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "evidence": t.evidence,
            }
            for t in batch
        ]

        verify_prompt = build_verification_prompt(
            doc["doc_text"], batch_dicts, entity_id_to_name, schema_info["rel_info"]
        )

        system_prompt = (
            "あなたは関係抽出の検証者です。"
            "提示された関係候補が文書の内容に基づいて正しいかどうかを判定してください。"
        )

        result = call_gemini(client, system_prompt, verify_prompt, VERIFICATION_SCHEMA)

        decisions = {d["candidate_index"]: d["keep"] for d in result.get("decisions", [])}
        for j, triple in enumerate(batch):
            if decisions.get(j, True):  # Default to keep if missing
                verified.append(triple)

    return verified
