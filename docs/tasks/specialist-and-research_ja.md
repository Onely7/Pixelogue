# 専門タスクの検証と研究実験

[English](specialist-and-research.md) · [タスクカタログ](README_ja.md)

## 候補選択と実行環境

標準65タスクには通常検証経路があります。専門拡張7タスクには初版の検証器を実装しました。`configs/specialist-pilot.yaml` は7タスクすべてを指定しますが、指定だけでは選択対象になりません。通常選択には、**実際に使用する生成モデルとprocessorの組**、検証器版、対応領域に一致する校正証明、動作する実行環境、画像ごとの根拠が必要です。候補上限8件も維持します。`compile`、`doctor`、`task-status` でタスク別の理由を確認できます。

```bash
uv sync --locked --extra cpu --group dev
uv sync --locked --directory runtime/validators
uv run --locked pixelogue compile --config configs/specialist-pilot.yaml --output artifacts/specialist-compile.json
uv run --locked pixelogue task-status --config configs/specialist-pilot.yaml --output-stem artifacts/task-status
```

CPUの全テスト後は `--junit artifacts/junit.xml`、特定のGPU runには `--conversations artifacts/<run>/conversations.jsonl` を指定します。タスク別レポートに、記録された回答ターンと生成モデルが入ります。共有CPU fixtureの合格だけでは、各タスクの正例・誤答・根拠不足の境界を網羅したことになりません。タスク固有の境界は実測まで `not_recorded` とします。候補まで到達して回答生成しなかったGPU試行は回答ターンに数えません。

専門環境はSymPy、music21、RDKit、Playwrightを本体や `runtime/vllm/` と別に固定します。初版は、明示された有理数の幾何条件、単声部の完全小節、立体表記を含まない原子・結合グラフ、接続が確定した二端子回路、画面上の単一UI操作、静的SVG・限定TikZ、静的HTML/CSSを扱います。未対応記法、根拠不足、2つの独立抽出の不一致は `UNKNOWN` です。構文が正しいだけでは合格しません。

描画workerは静的な許可構文だけを受け付け、JavaScriptとService Workerを無効化し、ブラウザ通信を遮断します。OS側は `bwrap` によってネットワークなし・読み取り専用mountへ隔離し、時間・メモリ・プロセス数を制限します。元の画像viewと隔離描画のRGB差・前景の重なりを照合します。OS隔離やChromiumが使えなければ図・画面コード復元は環境不足です。隔離を外して候補コードを実行しないでください。UI操作は指定内容のみを検査し、操作自体は実行しません。

## 評価専用経路と校正

未校正の専門タスクは `evaluate-specialist` でのみ評価します。JSONL入力は `compile` が出す `SpecialistEvaluationCase` Schemaに従い、評価専用の `ImageArtifact`、公開の質問と回答、確認用画像なら独立に得た正解ラベルの出典を含めます。正解ラベルはモデルへ渡しません。2評価者の画像抽出、専門検証の結果、根拠hashを非公開で保存し、通常の学習exportへ入れません。

```bash
nvidia-smi
uv run --locked pixelogue doctor --config configs/specialist-pilot.yaml --output artifacts/doctor.json
uv run --locked pixelogue evaluate-specialist --config configs/specialist-pilot.yaml --cases validation/local-specialist-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/specialist-evaluation --run-id specialist-eval-v1
uv run --locked pixelogue calibration-build --results-dir artifacts/specialist-evaluation/results --output artifacts/specialist-calibration.json --report artifacts/specialist-calibration-report.json
```

モデル呼び出し前には空きGPUを確認し、使う番号を明示してください。長時間処理は `tmux` で実行し、ログ・GPU状態・出力件数を監視します。1台だけでもpilotのモデルを順次呼べます。4台の確保は不要です。初期の累積4 GPU時間には読み込みと全割当GPUを含めます。一時的な1 GPU pilotは同じ `Qwen/Qwen3.5-9B` endpointへの別々のblind callです。標準のQwen3.8/Gemma組や評価モデルの多様性を示すものではありません。BF16設定を無断で量子化しません。

`calibration-build` は、重複しない確認用画像groupを使い、Clopper–Pearson法で片側95%の区間を計算します。誤受理率の上限5%以下、正例受理率の下限80%以上の両方が必要です。`UNKNOWN` は正例受理に数えません。通信失敗、不正出力、調整用画像、正例・負例が揃わないgroupは保留として報告します。証明書を手編集せず、適格なmanifestが得られたらコピーした設定の `tasks.calibration_manifest` に指定し、再度 `compile` してください。pilotモデルの校正を標準モデルへ流用できません。

`evaluate-specialist`、`research-history`、`research-ablation` は完了済み試行を再利用して再開します。失敗記録は残し、再試行には `--retry-failed` を指定します。専門タスク評価はモデル呼び出しごとにも保存するため、同じ入力・設定識別を保持し、再試行を挟んで評価モデルのendpointを順次起動できます。

## 研究専用CLI

`research-exposure` は画像・履歴・質問・操作・もっともらしい回答・明確な誤答と試行順序を呼び出し前に固定します。回答非提示、もっともらしい回答提示、誤答提示を、根拠・操作一致・回答可能性・非重複性・自然さの5項目で別々に評価します。条件・反復ごとに別の呼び出しを行い、完了済み試行は再開時に再利用します。失敗・未実行・費用不明を区別し、対応のある提示差と、人手で無効と確認した質問に限定したAIASをJSON・CSV・Markdownへ出力します。GPUが1台しか使えなければ評価者AとBを別の出力ディレクトリで順次実行します。

```bash
uv run --locked pixelogue research-exposure --cases validation/local-exposure-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-exposure-a --evaluator generator_a --plan-only
uv run --locked pixelogue research-exposure --cases validation/local-exposure-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-exposure-a --evaluator generator_a
uv run --locked pixelogue research-exposure-report --output-dir artifacts/research-exposure-a
```

`audit-pack` は accepted・rejected・abstained をそれぞれ抽出し、抽出母集団と実際の抽出率を保存します。`questions.html` に回答を含めず、回答品質は `answers.html` で別に見ます。手法名、モデル名、自動判定、誤り注入種別は閲覧資料に出しません。JSONLテンプレートを独立評価者が記入し、`audit-resolve` が既定3名の元評価を保持して一致・不一致・不明・未評価・裁定を区別します。自動判定を人手の正解ラベルへ置き換えません。

```bash
uv run --locked pixelogue audit-pack --cases validation/local-audit-cases.jsonl --output-dir artifacts/audit --rate 0.1
uv run --locked pixelogue audit-resolve --pack artifacts/audit/pack.json --question-votes artifacts/audit/question-votes.jsonl --answer-votes artifacts/audit/answer-votes.jsonl --output artifacts/audit/resolution.json
```

`research-history` は確定済み公開メッセージの正確な引用範囲を結び付けます。指示対象の引き継ぎと公開条件の変更では、回答を見る前に妥当な別履歴と条件変化を調べ、回答後に同じ回答が元履歴では成立し別履歴では成立しないかを2評価者で調べます。独立ターンにwitnessを付けません。往復番号だけでは履歴依存を認定しません。

`research-ablation` は質問を固定し、ゲートの回答前・回答後・省略、評価者A・B・両方、検証済み履歴のみ・生成済み履歴を保持の18条件を比較します。初回は修復を使わず、回答後はholisticとタスク固有検証を維持します。誤った往復を含む生成済み履歴は研究専用traceへ保存し、通常exportへ混ぜません。報告には各深さの開始件数と到達件数を含めます。初版は質問を固定してゲート時点を比べるもので、質問生成自体の変化は測定しません。

比較実験の `depth.csv` は、人手ラベルがある場合の誤受理・誤拒否と、両評価者の誤りの重なりを記録します。独立した人手ラベルがない場合、この件数は0です。履歴実験と人手監査の裁定結果は、それぞれJSON・CSV・Markdownで保存します。

```bash
uv run --locked pixelogue research-history --cases validation/local-history-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-history
uv run --locked pixelogue research-ablation --cases validation/local-ablation-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-ablation --plan-only
uv run --locked pixelogue research-ablation --cases validation/local-ablation-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-ablation
uv run --locked pixelogue research-ablation-report --output-dir artifacts/research-ablation
```

各CLIは手元で準備した画像と正解ラベルを必要とします。リポジトリへ画像bytesは含めません。CPU fixtureはcontrollerの動作と棄権を確認しますが、自然画像の抽出精度は保証しません。単価表がなければ費用は `null` です。大規模コーパス、複数studentのSFT、学習曲線、全面的なcontamination検査は次段階です。
