# JacRED KG Extraction Experiment

日本語文書レベル関係抽出データセット **JacRED** を用いた、低コストLLMによる知識グラフ抽出の実験。

## 概要

日本語の文書群からentity（エンティティ）とrelation（関係）を抽出し、知識グラフを構築する手法の初期実験。Gemini Flash系モデルのStructured Outputsを活用し、Baseline（one-shot抽出）とProposed（Two-Stage: 候補生成→検証）を比較する。

## 実験設定

- **データ**: JacRED dev set から10文書（サイズ別に層化サンプリング）
- **モデル**: Gemini 2.0 Flash / 2.5 Flash / 3 Flash Preview
- **評価**: P / R / F1（文書レベル関係抽出）

### Baseline: One-shot抽出
1回のLLM呼び出しでエンティティと関係を同時に抽出。

### Proposed: Two-Stage（候補生成→検証）
- **Stage 1**: Recall重視でエンティティ・関係候補を多めに生成
- **Stage 2**: 各候補をバッチで検証（keep/drop判定）
- **後処理**: domain/range型制約テーブル（trainデータから自動構築）で不正な候補を除外

## 結果

| モデル | thinking | 条件 | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 2.0-flash | なし | Baseline | 0.20 | 0.15 | 0.17 | 22 | 86 | 126 |
| 2.0-flash | なし | Proposed | 0.35 | 0.19 | 0.25 | 28 | 51 | 121 |
| 2.5-flash | OFF | Baseline | 0.17 | 0.16 | 0.17 | 24 | 115 | 124 |
| 2.5-flash | OFF | Proposed | 0.30 | 0.12 | 0.17 | 18 | 42 | 130 |
| 2.5-flash | 2048 | Baseline | 0.18 | 0.17 | 0.17 | 25 | 115 | 123 |
| 2.5-flash | 2048 | Proposed | 0.36 | 0.21 | **0.27** | 31 | 54 | 117 |
| 3-flash-preview | OFF | Baseline | 0.26 | 0.16 | 0.20 | 24 | 70 | 124 |
| 3-flash-preview | OFF | Proposed | 0.36 | 0.22 | **0.27** | 32 | 56 | 116 |
| 3-flash-preview | 2048 | Baseline | 0.31 | 0.22 | 0.26 | 33 | 74 | 115 |
| 3-flash-preview | 2048 | Proposed | 0.37 | 0.20 | 0.26 | 30 | 52 | 118 |

### 主な知見

1. **Proposed手法はPrecisionを大幅改善**: 全モデルでFPが40-60%削減
2. **gemini-3-flash-preview (no thinking)** がコスパ最良: F1=0.27を約2分で達成
3. **thinkingの効果**: Baselineの改善には有効（3-flash Baseline: 0.20→0.26）だが、Proposedでは検証が厳しくなりすぎてTPを落とす傾向
4. **Recallが全体的に低い** (0.12-0.22): 主な課題。プロンプト改善やfew-shot例の追加が必要

## ファイル構成

```
run_experiment.py   # メインスクリプト（全体オーケストレーション）
data_loader.py      # JacREDデータ読込、文書選択、制約テーブル構築
llm_client.py       # Gemini API呼び出し（Structured Outputs対応）
prompts.py          # 全プロンプトテンプレート
extraction.py       # Baseline / Proposed の抽出ロジック
evaluation.py       # エンティティ照合 + P/R/F1算出
schemas.py          # Gemini Structured Output用JSONスキーマ
results.json        # 最新の実験結果
```

## 使い方

```bash
# JacREDデータセットを取得
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# 実行
export GEMINI_API_KEY="your-key-here"
python3 run_experiment.py
```

## 参考

- **JacRED**: [YoumiMa/JacRED](https://github.com/YoumiMa/JacRED) - Japanese Document-level Relation Extraction Dataset
- **論文**: Ma et al., "Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer", LREC-COLING 2024
