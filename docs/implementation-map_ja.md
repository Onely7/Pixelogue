# 実装対応表

このページは、処理上の約束をコード、保存物、コマンド、テストへ対応付けます。quality gate（品質の合否判定）や visual group（見た目が同一・近似した画像のまとまり）が初出なら、先に各手順書を読んでください。

| 約束 | 主な実装 | 保存・出力する根拠 | 入口 |
|---|---|---|---|
| 78タスクのカタログの版管理、生成するタスク一覧、実行時の受付判定、必須検証器 | `task_catalog.py`, `catalog.py`, `task_docs.py`, `task_registry.py`, `task_runtime.py`, `task_verification.py`, `knowledge_verifiers.py`, `box_verifier.py`, `premise_verifiers.py` | カタログ識別、`task_admission`、操作検証、操作集計 | `compile`, `synthesize` |
| 厳密な設定、正確な件数配分、モデル役割の固定 | `config.py`, `planner.py` | 解決済み設定、件数表、model revision | `compile` |
| 利用条件、画像正規化、重複 group、split 分離 | `images.py`, `operations.py`, `sscd.py` | 画像台帳、失敗理由、split、manifest hash | `ingest` |
| 固定した Open Images V7 validation 標本 | `open_images.py`, `validation/open_images_v7_manifest.jsonl` | source・rights JSONL、非公開取得 metadata | `prepare` |
| stage ごとの入力制限、出力量に応じたタイムアウト、反復停止を持つローカル構造化推論 | `prompts.py`, `serving.py` | hash 付き request・response・token 数・model lock | `synthesize`, `rate-existing` |
| 画像単位の有界並列化、評価呼び出しの並行化、モデル呼び出し時間の記録 | `pipeline.py`, `store.py`, `profiling.py` | 入力順の対話とstage別時間集計 | `synthesize`, `profile` |
| 画像ごとに1回のルーター分析と、起草前のrun全体での系統割り当て | `routing.py`, `pipeline.py`, `store.py` | ルーターの応答、`turn_route` の行、`image-profile-abstentions` | `synthesize` |
| 質問案の直接起草と、評価器を呼ぶ前の決定的な受付 | `drafting.py`, `pipeline.py`, `evaluation.py`, `prompts.py` | `question-drafts`、`draft-rejections`、`draft-abstentions`、`public-text-rejections` | `synthesize` |
| 2評価器の統合質問ゲート、総合評価、画像全体での再評価、1回の回答修復 | `gates.py`, `pipeline.py`, `focused_views.py`, `evaluation.py` | `question-gate-decisions`、`rating-decisions`、`answer-attempts`、`focus-views` | `synthesize`, `rate-existing` |
| 盲検の読み手を対で使う操作検証器 | `task_verification.py`, `rules.py`, 各検証器モジュール | `operation-checks` | `synthesize`, `rate-existing` |
| 合格済み部分の保持と、再開可能なターン | `pipeline.py`, `store.py` | `conversation-stops`、`conversation-stop-reasons`、`turn_commit`、`conversation_commit` | `synthesize` |
| ローカル実行状態、再生、バックアップ、復元 | `store.py` | SQLite 台帳と hash 付き artifact tree | `replay`, `backup`, `restore` |
| 固定 pool の CP-SAT 選抜と独立再計数 | `selection.py`, `operations.py` | pool hash、selection manifest、audit hash | `freeze-pool`, `select`, `audit` |
| 検証用出典を除外した公開文だけの学習出力 | `export.py` | training・rating・provenance・selection の分離ファイル | `export` |
| 正例と単一誤りの手続き生成 probe | `fixtures.py`, `capabilities.py` | split・stratum 別 capability report | `make-fixtures`, `evaluate-capabilities` |

`compile` は設定、画像、生成、評価、選抜、決定的ルールの公開契約を JSON Schema として出力します。画像の分析結果、質問案の一覧、質問ゲートの票も含みます。Pydantic は未知 field と暗黙の型変換を拒否します。JSON・YAML reader は、通常の field 検証だけでは扱えない重複 key と非有限数も拒否します。

controller の自動テストでは scripted client（決めた応答を返すテスト専用実装）を使います。この応答をstandard データとして出力することはできません。実際の BF16 model 起動と少数の自然画像 pilot は、十分な空き GPU と固定 revision の重みが必要なため運用者が実行します。pilot の結果を 30,000 対話の認定の代わりにはしません。
