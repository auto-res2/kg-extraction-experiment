"""Prompt templates for entity/relation extraction and verification."""

import json


RELATION_JAPANESE = {
    "P1376": "首都（〜の首都である）",
    "P131": "行政区画（〜に位置する行政区画）",
    "P276": "所在地（〜に所在する）",
    "P937": "活動場所（〜で活動した）",
    "P27": "国籍（〜の国籍を持つ）",
    "P569": "生年月日",
    "P570": "没年月日",
    "P19": "出生地",
    "P20": "死没地",
    "P155": "前任・前作（〜の後に続く）",
    "P40": "子（〜の子である）",
    "P3373": "兄弟姉妹",
    "P26": "配偶者",
    "P1344": "参加イベント（〜に参加した）",
    "P463": "所属（〜に所属する）",
    "P361": "上位概念（〜の一部である）",
    "P6": "首長（〜の首長である）",
    "P127": "所有者（〜に所有される）",
    "P112": "設立者（〜が設立した）",
    "P108": "雇用主（〜に雇用される）",
    "P137": "運営者（〜が運営する）",
    "P69": "出身校（〜で教育を受けた）",
    "P166": "受賞（〜を受賞した）",
    "P170": "制作者（〜が制作した）",
    "P175": "出演者・パフォーマー",
    "P123": "出版社（〜が出版した）",
    "P1441": "登場作品（〜に登場する）",
    "P400": "プラットフォーム",
    "P36": "首都（〜が首都である）",
    "P156": "後任・次作（〜の前にある）",
    "P710": "参加者（〜が参加した）",
    "P527": "構成要素（〜を含む）",
    "P1830": "所有物（〜を所有する）",
    "P121": "運営対象（〜を運営する）",
    "P674": "登場人物（作品の登場人物）",
}

ENTITY_TYPES_JAPANESE = {
    "PER": "人物",
    "ORG": "組織",
    "LOC": "場所・地名",
    "ART": "作品・人工物・賞",
    "DAT": "日付",
    "TIM": "時間",
    "MON": "金額",
    "%": "パーセンテージ・数値",
}


def build_system_prompt(rel_info: dict) -> str:
    """Build system prompt with entity types and relation definitions."""
    entity_lines = []
    for etype, desc in ENTITY_TYPES_JAPANESE.items():
        entity_lines.append(f"  - {etype}: {desc}")

    relation_lines = []
    for pcode, eng_name in rel_info.items():
        ja_desc = RELATION_JAPANESE.get(pcode, "")
        relation_lines.append(f"  - {pcode} ({eng_name}): {ja_desc}")

    return f"""あなたは日本語文書から知識グラフ（エンティティと関係）を抽出する専門家です。

## タスク
与えられた日本語文書から、エンティティ（固有表現）とエンティティ間の関係を抽出してください。

## エンティティタイプ（8種類）
{chr(10).join(entity_lines)}

## 関係タイプ（35種類、Pコードで指定）
{chr(10).join(relation_lines)}

## ルール
- エンティティには上記のタイプのみ使用してください。
- 関係には上記のPコード（P131, P27等）のみ使用してください。自由記述は禁止です。
- 各関係には、根拠となる文書中のテキストをevidenceとして付与してください。
- headとtailにはentitiesのidを指定してください。"""


def build_extraction_prompt(
    doc_text: str,
    few_shot_text: str,
    few_shot_output: dict,
    mode: str = "baseline",
) -> str:
    """Build user prompt for extraction."""
    few_shot_json = json.dumps(few_shot_output, ensure_ascii=False, indent=2)

    mode_instruction = ""
    if mode == "recall":
        mode_instruction = """
重要: できるだけ多くの関係を漏れなく抽出してください。確信度が低い場合でも、可能性がある関係は候補として含めてください。
後の検証ステップで精度を高めるため、この段階では再現率（recall）を優先してください。"""

    return f"""## 例
入力文書:
{few_shot_text}

出力:
{few_shot_json}

## 対象文書
{doc_text}

上記の文書からエンティティと関係を抽出してください。{mode_instruction}"""


def build_verification_prompt(
    doc_text: str,
    candidates: list[dict],
    entity_map: dict,
    rel_info: dict,
) -> str:
    """Build verification prompt for Stage 2."""
    candidate_lines = []
    for i, c in enumerate(candidates):
        head_name = entity_map.get(c["head"], c["head"])
        tail_name = entity_map.get(c["tail"], c["tail"])
        rel_code = c["relation"]
        rel_name = rel_info.get(rel_code, "不明")
        ja_desc = RELATION_JAPANESE.get(rel_code, "")
        candidate_lines.append(
            f"候補{i}: {head_name} --[{rel_code}: {rel_name}]--> {tail_name}\n"
            f"  根拠: {c.get('evidence', '(なし)')}\n"
            f"  関係の定義: {ja_desc}"
        )

    return f"""以下の文書と、そこから抽出された関係候補を検証してください。

## 文書
{doc_text}

## 関係候補
{chr(10).join(candidate_lines)}

各候補について、文書の内容がこの関係を支持しているかどうかを判定してください。
判定基準:
- 文書中に明確な根拠があるか
- エンティティの型が関係と整合しているか
- 関係の方向（head→tail）が正しいか
根拠が不十分な候補はkeep=falseとしてください。"""
