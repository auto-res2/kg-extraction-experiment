"""JSON schemas for Gemini Structured Outputs."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"],
                    },
                },
                "required": ["id", "name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["head", "relation", "tail", "evidence"],
            },
        },
    },
    "required": ["entities", "relations"],
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {"type": "integer"},
                    "keep": {"type": "boolean"},
                },
                "required": ["candidate_index", "keep"],
            },
        },
    },
    "required": ["decisions"],
}
