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

CPUの全テスト後は `--junit artifacts/junit.xml`、特定のGPU runには `--conversations artifacts/<run>/conversations.jsonl` を指定します。タスク別レポートでは、試行した回答ターンと確定ターンを分け、各ターンの状態と生成モデルを記録します。共有CPU fixtureの合格だけでは、各タスクの正例・誤答・根拠不足の境界を網羅したことになりません。タスク固有の境界は実測まで `not_recorded` とします。候補まで到達して回答生成しなかったGPU試行は回答ターンに数えません。

専門環境はSymPy、music21、RDKit、Playwrightを本体や `runtime/vllm/` と別に固定します。初版は、明示された有理数の幾何条件、単声部の完全小節、立体表記を含まない原子・結合グラフ、接続が確定した二端子回路、画面上の単一UI操作、静的SVG・限定TikZ、静的HTML/CSSを扱います。未対応記法、根拠不足、2つの独立抽出の不一致は `UNKNOWN` です。構文が正しいだけでは合格しません。
通常選択では、専門検証器ごとに登録した版と校正証明の版を照合します。抽出指示や検証契約を変えたときは登録版を上げ、以前の証明で改訂後のタスクを有効化しないようにします。
幾何では、2つの抽出が型付き前提ごとに一致し、対応する画像領域のIoUが0.1以上であることを求めます。説明文の違いや前提の記載順は同じ証明事実を変えません。重複した前提や範囲外の根拠は拒否します。
幾何契約の版3では、解く前に登録規則ごとの変数と定数の個数を検査します。変数IDはsymbolic workerが受理する記法に一致させ、完全な抽出では対象を前提に結び付けます。説明に印刷された数値を引用していても、`given`の定数欄が欠けていれば拒否し、最大1回のSchema修正を試みます。
単声部の完全小節では、音符・休符の記譜上の長さの合計が公開された拍子と正確に一致する必要があります。全音符のbaseは`1`、二分音符は`2`、四分音符は`4`、付点は長さの`3/2`倍です。不整合なモデル出力は再試行し、解消しなければ未検証とします。
化学構造の抽出を再試行するときは、範囲外に出た原子または結合の最初の領域を示します。修正後も実際に見える分子の範囲に収まらなければ失敗です。
芳香族結合が閉じた環に属さない場合もグラフ照合前に拒否し、回答を見せずに具体的な再試行理由を返します。見た目に根拠のない芳香族の鎖を完全な分子として認定しません。

描画workerは静的な許可構文だけを受け付け、JavaScriptとService Workerを無効化し、ブラウザ通信とChromiumのGPU描画を遮断します。ユーザー名前空間が使える場合は `bwrap`、使えない場合は上限付きのユーザー `systemd` サービスとLinux Landlockで隔離します。後者はIPv4/IPv6の禁止とプロジェクトのファイル読み書き禁止を実際に検査してから有効化します。システムライブラリ・フォント・`/proc` の読み取りを許し、`/tmp` 内の専用ディレクトリだけを読み書き可能にして、他の `/tmp` ファイルの内容は読めないようにします。サービスはメモリ・タスク数・実行時間も制限します。元の画像viewと隔離描画のRGB差・前景の重なりを照合します。OS隔離やChromiumが使えなければ図・画面コード復元は環境不足です。隔離を外して候補コードを実行しないでください。UI操作は指定内容のみを検査し、操作自体は実行しません。
回答を見ない描画用の抽出では、画像に見えるラベルとその可視範囲だけを判断します。隔離環境の利用可否や校正はcontrollerが別に検査するため、画像の依頼文にその説明がないことを視覚根拠不足として扱いません。

UI操作では、両評価者が公開された対象文字列をcontrol IDとしてそのまま使い、有効状態が一致することを求めます。操作対象領域のIoUはクリックで0.5以上、フォーカスと入力で0.7以上とします。クリックでは、両者が同じ有効な対象を特定していれば、部品の種類名が異なっても認めます。フォーカスと入力では、両者とも入力欄と判定する必要があります。無関係な操作部品の抽出差は判定に使いません。回答の点は両方の対象領域内にあり、配信画像viewのピクセル座標変換と一致する必要があります。対象が未解決または不一致なら棄権します。
公開された対象名は、画面に表示された文字の引用ではなく、機能の説明である場合があります。画像から一意に特定できる場合だけ対応付け、隠れた操作や別の機能を推測しません。

## 評価専用経路と校正

未校正の専門タスクは `evaluate-specialist` でのみ評価します。JSONL入力は `compile` が出す `SpecialistEvaluationCase` Schemaに従い、評価専用の `ImageArtifact`、公開の質問と回答、確認用画像なら独立に得た正解ラベルの出典を含めます。正解ラベルはモデルへ渡しません。2評価者の画像抽出、専門検証の結果、根拠hashを非公開で保存し、通常の学習exportへ入れません。

```bash
nvidia-smi
uv run --locked pixelogue doctor --config configs/specialist-pilot.yaml --output artifacts/doctor.json
uv run --locked pixelogue evaluate-specialist --config configs/specialist-pilot.yaml --cases validation/local-specialist-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/specialist-evaluation --run-id specialist-eval-v1
uv run --locked pixelogue calibration-build --results-dir artifacts/specialist-evaluation/results --output artifacts/specialist-calibration.json --report artifacts/specialist-calibration-report.json
```

モデル呼び出し前には空きGPUを確認し、使う番号を明示してください。長時間処理は `tmux` で実行し、ログ・GPU状態・出力件数を監視します。1台だけでもpilotのモデルを順次呼べます。4台の確保は不要です。初期の累積4 GPU時間には読み込みと全割当GPUを含めます。一時的な1 GPU pilotは同じ `Qwen/Qwen3.5-9B` endpointへの別々のblind callです。標準のQwen3.8/Gemma組や評価モデルの多様性を示すものではありません。BF16設定を無断で量子化しません。

`calibration-build` は結果ディレクトリの隣にある固定済み `input.json` を必須とし、予定した全ケースIDを照合します。結果ファイルが1件でも欠ければ、その入力の証明書をすべて保留し、欠落IDを報告します。入力が揃ってから、重複しない確認用画像groupを使い、Clopper–Pearson法で片側95%の区間を計算します。誤受理率の上限5%以下、正例受理率の下限80%以上の両方が必要です。`UNKNOWN` は正例受理に数えません。確認用ケースが1件でも失敗・独立正解ラベル欠落なら、同じタスク・領域・モデル・検証器版のgroup全体を保留し、成功済みケースだけでは証明書を作りません。通信失敗、不正出力、調整用画像、正例・負例が揃わないgroupは保留として報告します。証明書を手編集せず、適格なmanifestが得られたらコピーした設定の `tasks.calibration_manifest` に指定し、再度 `compile` してください。pilotモデルの校正を標準モデルへ流用できません。

`evaluate-specialist`、`research-history`、`research-ablation` は完了済み試行を再利用して再開します。失敗記録は残し、再試行には `--retry-failed` を指定します。専門タスク評価はモデル呼び出しごとにも保存するため、同じ入力・設定識別を保持し、再試行を挟んで評価モデルのendpointを順次起動できます。
共有GPUで専門構造の抽出が遅い場合は、評価用にコピーした設定の `runtime.request_timeout_seconds` を最大600秒まで指定できます。既定値は180秒のままです。コピーした設定をrunと共に記録し、時間切れを検証器の判定として扱わないでください。
専門抽出が構造化出力の検証に失敗した場合は、具体的なSchema違反と領域・回路接続の条件を伝えてblind callを1回だけ追加します。不正な試行は記録に残し、再度失敗すれば `FAILED` のままで校正件数には入れません。
専門抽出がトークン上限で完全なJSONになる前に止まった場合も、打ち切りを伝えてblind callを1回だけ追加します。Schema違反と長さ上限の再試行は別々に集計します。化学構造の抽出に限り、制約付きデコードで観測した空白反復を避けるため、この再試行ではJSONオブジェクト形式を使います。返答は従来の厳格な `ChemicalSource` Schemaと2評価者のグラフ照合をそのまま通す必要があります。元の不完全出力を残し、2回目も不正なら `FAILED` のままです。
専門抽出のSchemaでは、公開済みの校正領域、scope ID、配信した画像view IDを固定し、楽譜では対象小節範囲も固定します。抽出には `tasks.evidence_max_tokens` を使い、途中で切れた分子グラフを完全な結果として扱いません。

## 研究専用CLI

`research-exposure` は画像・履歴・質問・操作・もっともらしい回答・明確な誤答と試行順序を呼び出し前に固定します。回答非提示、もっともらしい回答提示、誤答提示を、根拠・操作一致・回答可能性・非重複性・自然さの5項目で別々に評価します。条件・反復ごとに別の呼び出しを行い、完了済み試行は再開時に再利用します。失敗・未実行・費用不明を区別し、対応のある提示差と、人手で無効と確認した質問に限定したAIASをJSON・CSV・Markdownへ出力します。GPUが1台しか使えなければ評価者AとBを別の出力ディレクトリで順次実行します。

```bash
uv run --locked pixelogue research-exposure --cases validation/local-exposure-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-exposure-a --evaluator generator_a --plan-only
uv run --locked pixelogue research-exposure --cases validation/local-exposure-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-exposure-a --evaluator generator_a
uv run --locked pixelogue research-exposure-report --output-dir artifacts/research-exposure-a
```

`audit-pack` は accepted・rejected・abstained をそれぞれ抽出し、抽出母集団と実際の抽出率を保存します。`questions.html` に回答を含めず、回答品質は `answers.html` で別に見ます。手法名、モデル名、自動判定、誤り注入種別は閲覧資料に出しません。JSONLテンプレートを独立評価者が記入し、`audit-resolve` が既定3名の元評価を保持して一致・不一致・不明・未評価・裁定を区別します。自動判定を人手の正解ラベルへ置き換えません。

自動確定ターンを全件確認し、停止・棄却例を標本監査するときは `--accepted-rate 1 --rejected-rate 0.25 --abstained-rate 0.25` を指定します。指定しない区分には `--rate` を適用します。区分ごとの指定率と実際の抽出率、母集団を保存するため、監査件数と抽出母数を区別できます。

`audit-cases` は保存済みの生成結果から母集団を作ります。確定済み往復は accepted のターン例とし、次の公開質問・回答がないまま会話が止まった場合は別の停止例を残します。停止例も抽出母集団に含め、存在しない質問票・回答票は対象外とします。実行エラーは別件数で報告します。個別ターンの合格と品質候補の完成を混同しません。

```bash
uv run --locked pixelogue audit-cases --conversations artifacts/pilot/conversations.jsonl --artifact-root artifacts/prepared --output artifacts/audit-cases.jsonl
uv run --locked pixelogue audit-pack --cases validation/local-audit-cases.jsonl --output-dir artifacts/audit --rate 0.1
uv run --locked pixelogue audit-resolve --pack artifacts/audit/pack.json --question-votes artifacts/audit/question-votes.jsonl --answer-votes artifacts/audit/answer-votes.jsonl --output artifacts/audit/resolution.json
```

`research-history` は確定済み公開メッセージの正確な引用範囲を結び付けます。指示対象の引き継ぎと公開条件の変更では、回答を見る前に妥当な別履歴と条件変化を調べ、回答後に同じ回答が元履歴では成立し別履歴では成立しないかを2評価者で調べます。独立ターンにwitnessを付けません。往復番号だけでは履歴依存を認定しません。

`research-ablation` は質問を固定し、ゲートの回答前・回答後・省略、評価者A・B・両方、検証済み履歴のみ・生成済み履歴を保持の18条件を比較します。初回は修復を使わず、回答後はholisticとタスク固有検証を維持します。誤った往復を含む生成済み履歴は研究専用traceへ保存し、通常exportへ混ぜません。報告には各深さの開始件数と到達件数を含めます。初版は質問を固定してゲート時点を比べるもので、質問生成自体の変化は測定しません。
[2画像の開発用case](../../validation/confirmed_two_image_ablation.jsonl)には、今回のpilot用に画像ごとに異なる可視属性・関係の質問を2つずつ固定しています。質問の人手適格ラベルは未設定です。計画される36条件は、独立監査が済むまでは挙動の観測として扱います。

比較実験の `depth.csv` は、人手ラベルがある場合の誤受理・誤拒否と、両評価者の誤りの重なりを記録します。独立した人手ラベルがない場合、件数は未測定（JSONでは `null`、CSVでは空欄）とし、0はラベル付き例を確認して誤りがなかった場合に限ります。回答提示実験も同じ区別を使います。履歴実験と人手監査の裁定結果は、それぞれJSON・CSV・Markdownで保存します。

```bash
uv run --locked pixelogue research-history --cases validation/local-history-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-history
uv run --locked pixelogue research-ablation --cases validation/local-ablation-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-ablation --plan-only
uv run --locked pixelogue research-ablation --cases validation/local-ablation-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-ablation
uv run --locked pixelogue research-ablation-report --output-dir artifacts/research-ablation
```

各CLIは手元で準備した画像と正解ラベルを必要とします。リポジトリへ画像bytesは含めません。CPU fixtureはcontrollerの動作と棄権を確認しますが、自然画像の抽出精度は保証しません。単価表がなければ費用は `null` です。大規模コーパス、複数studentのSFT、学習曲線、全面的なcontamination検査は次段階です。
