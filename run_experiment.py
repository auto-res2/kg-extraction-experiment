"""Main experiment script: Baseline vs Proposed on JacRED dev subset."""

import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_jacred, select_dev_docs, select_few_shot, build_constraint_table
from llm_client import load_api_key, create_client
from extraction import run_baseline, run_proposed
from evaluation import align_entities, evaluate_relations, aggregate_results

ENV_PATH = os.path.expanduser(
    "~/Library/CloudStorage/Dropbox/secrets/.env"
)
NUM_DOCS = 10


def run_condition(name, docs, few_shot, client, schema_info, constraint_table=None):
    """Run one experimental condition on all docs."""
    print(f"\n--- {name} ---")
    per_doc_results = []

    for i, doc in enumerate(docs):
        title = doc["title"]

        if constraint_table is not None:
            entities, triples, stats = run_proposed(
                doc, few_shot, client, schema_info, constraint_table
            )
            stats_str = (
                f" S1={stats['stage1_candidates']}"
                f" S2={stats['stage2_kept']}"
                f" Final={stats['after_constraints']}"
            )
        else:
            entities, triples = run_baseline(doc, few_shot, client, schema_info)
            stats = None
            stats_str = ""

        alignment = align_entities(entities, doc["vertexSet"])
        metrics = evaluate_relations(triples, doc.get("labels", []), alignment)

        print(
            f"  [{i+1}/{len(docs)}] {title}: "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
            f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}){stats_str}"
        )

        doc_result = {
            "title": title,
            "num_gold_entities": len(doc["vertexSet"]),
            "num_gold_labels": len(doc.get("labels", [])),
            "num_predicted": len(triples),
            "num_entities_aligned": len(alignment),
            **metrics,
        }
        if stats:
            doc_result["stage_stats"] = stats
        per_doc_results.append(doc_result)

    agg = aggregate_results(per_doc_results)
    print(
        f"  Aggregate: P={agg['precision']:.2f} R={agg['recall']:.2f} F1={agg['f1']:.2f} "
        f"(TP={agg['tp']} FP={agg['fp']} FN={agg['fn']})"
    )
    return {"per_doc": per_doc_results, "aggregate": agg}


def main():
    print("=== JacRED KG Extraction Experiment ===")
    print(f"Model: gemini-3-flash-preview")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    print("\nLoading data...")
    data = load_jacred()
    dev_docs = select_dev_docs(data["dev"], n=NUM_DOCS)
    few_shot = select_few_shot(data["train"])
    constraint_table = build_constraint_table(data["train"])

    print(f"Dev docs: {NUM_DOCS} (stratified by size)")
    print(f"Few-shot: {few_shot['title']}")
    print(f"Constraint table: {len(constraint_table)} relation types")
    for doc in dev_docs:
        n_ents = len(doc["vertexSet"])
        n_rels = len(doc.get("labels", []))
        print(f"  - {doc['title']} (ents={n_ents}, rels={n_rels})")

    # Initialize LLM
    api_key = load_api_key(ENV_PATH)
    client = create_client(api_key)

    schema_info = {
        "rel_info": data["rel_info"],
        "ent2id": data["ent2id"],
        "rel2id": data["rel2id"],
    }

    # Run conditions
    baseline_results = run_condition(
        "Condition 1: Baseline (One-shot)", dev_docs, few_shot, client, schema_info
    )
    proposed_results = run_condition(
        "Condition 2: Proposed (Generate+Verify)",
        dev_docs, few_shot, client, schema_info, constraint_table
    )

    # Comparison
    b = baseline_results["aggregate"]
    p = proposed_results["aggregate"]
    print("\n=== Comparison ===")
    print(f"{'':>12} {'Precision':>10} {'Recall':>8} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"{'Baseline':>12} {b['precision']:>10.2f} {b['recall']:>8.2f} {b['f1']:>6.2f} {b['tp']:>5} {b['fp']:>5} {b['fn']:>5}")
    print(f"{'Proposed':>12} {p['precision']:>10.2f} {p['recall']:>8.2f} {p['f1']:>6.2f} {p['tp']:>5} {p['fp']:>5} {p['fn']:>5}")

    # Save results
    output = {
        "experiment": {
            "model": "gemini-3-flash-preview",
            "num_docs": NUM_DOCS,
            "few_shot_doc": few_shot["title"],
            "timestamp": datetime.now().isoformat(),
        },
        "conditions": {
            "baseline": baseline_results,
            "proposed": proposed_results,
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
