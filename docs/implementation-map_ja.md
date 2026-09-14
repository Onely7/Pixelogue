# 実装対応表

このページは、処理上の約束をコード、保存物、コマンド、テストへ対応付けます。quality gate（品質の合否判定）や visual group（見た目が同一・近似した画像のまとまり）が初出なら、先に各手順書を読んでください。

| 約束 | 主な実装 | 保存・出力する根拠 | 入口 |
|---|---|---|---|
| 厳密な設定、正確な件数配分、モデル役割の固定 | `config.py`, `planner.py` | 解決済み設定、件数表、model revision | `compile` |
| 利用条件、画像正規化、重複 group、split 分離 | `images.py`, `operations.py`, `sscd.py` | 画像台帳、失敗理由、split、manifest hash | `ingest` |
| 固定した Open Images V7 validation 標本 | `open_images.py`, `validation/open_images_v7_manifest.jsonl` | source・rights JSONL、非公開取得 metadata | `prepare` |
| stage ごとの入力制限を持つローカル構造化推論 | `prompts.py`, `serving.py` | hash 付き request・response・token 数・model lock | `synthesize`, `rate-existing` |
| 画像単位の有界並列化とモデル呼び出し時間の記録 | `pipeline.py`, `store.py`, `profiling.py` | 入力順の対話とstage別時間集計 | `synthesize`, `profile` |
| 質問・回答より前の指示候補選択 | `planner.py`, `pipeline.py` | 候補集合と選択応答 | `synthesize` |
| 質問適合、公開要求、主張、計算、集合、修復後の全再評価 | `pipeline.py`, `ledger.py`, `evaluation.py`, `rules.py` | turn 評価と変更不能な試行 artifact | `synthesize`, `rate-existing` |
| ローカル実行状態、再生、バックアップ、復元 | `store.py` | SQLite 台帳と hash 付き artifact tree | `replay`, `backup`, `restore` |
| 固定 pool の CP-SAT 選抜と独立再計数 | `selection.py`, `operations.py` | pool hash、selection manifest、audit hash | `freeze-pool`, `select`, `audit` |
| 検証用出典を除外した公開文だけの学習出力 | `export.py` | training・rating・provenance・selection の分離ファイル | `export` |
| 正例と単一誤りの手続き生成 probe | `fixtures.py`, `capabilities.py` | split・stratum 別 capability report | `make-fixtures`, `evaluate-capabilities` |

`compile` は設定、画像、生成、評価、選抜、決定的ルールの公開契約を JSON Schema として出力します。Pydantic は未知 field と暗黙の型変換を拒否します。JSON・YAML reader は、通常の field 検証だけでは扱えない重複 key と非有限数も拒否します。

controller の自動テストでは scripted client（決めた応答を返すテスト専用実装）を使います。この応答をstandard データとして出力することはできません。実際の BF16 model 起動と少数の自然画像 pilot は、十分な空き GPU と固定 revision の重みが必要なため運用者が実行します。pilot の結果を 30,000 対話の認定の代わりにはしません。
