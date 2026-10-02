# 復旧と CI

各 run は、ローカルの SQLite WAL データベース 1 個と、内容 hash で保存した変更不能なファイルを使います。既定の保存先は `/var/tmp/pixelogue` です。SQLite 公式資料は[WAL が network filesystem では動作しない](https://sqlite.org/wal.html)と説明しているため、既知のNFS、CIFS、SMB、SSHFS は拒否します。

## 保存済み run を検査する

```sh
uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id open-images-pilot
```

SQLite の整合性、全 artifact の hash、完了済み model call に応答 artifact があることを確認します。不一致は監査エラーです。同じ hash 名で別 run のファイルを補ってはいけません。

同じ run を開ける coordinator は 1 process だけです。2 つ目は `RUN_ALREADY_ACTIVE` で停止します。request 数と最大出力 token の予約は推論前に保存します。process が異常終了しても予約は消えないため、再開後に予算を暗黙に超えません。

## 整合したバックアップを作る

```sh
uv run --locked pixelogue backup \
  --config configs/pilot.yaml \
  --run-id open-images-pilot \
  --destination /shared/pixelogue-backups/open-images-pilot-001
```

SQLite の backup API でローカル DB の整合した snapshot を作り、変更不能 artifact をコピーします。実行中の `.sqlite3`、`-wal`、`-shm` を別々にコピーしないでください。

新しい空のローカルディレクトリへ復元します。

```sh
uv run --locked pixelogue restore \
  --database /shared/pixelogue-backups/open-images-pilot-001/open-images-pilot.sqlite3 \
  --artifacts /shared/pixelogue-backups/open-images-pilot-001/open-images-pilot-artifacts \
  --destination /var/tmp/pixelogue/open-images-pilot-restored
```

継続前に復元 run へ `replay` を実行します。設定 hash が変わっていれば再開を拒否するため、意図した設定変更には新しい run ID を使います。

設定hashにはアプリのコード、プロンプト、Schema、タスク資源、専門workerとlock、モデル・processorの設定、seed、固定したOpen Images入力を含めます。`synthesize` は準備時の `manifest.json` も必須とし、元のsource・権利manifestの識別を含めます。指定した `images.jsonl` は準備済みmanifestと完全一致する必要があります。契約や入力が変わったら新しいrun IDを使ってください。研究実験の固定planと完了済み試行は `artifacts/` 内に別保存し、通常の対話commitにはしません。

## ローカル検査と GitHub Actions

```sh
uv sync --locked --extra cpu --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
uv build
uv run --locked pre-commit run --all-files
```

`.github/workflows/quality.yml` も同じ lock 済み環境を使います。ty は GitHub 用の標準出力形式でworkflow annotation を作り、独自 SARIF 変換は行いません。gitleaks は pre-commit に残しています。

`.github/workflows/codeql.yml` は GPU 不要の独立した CodeQL advanced setup です。GitHub の**Code security → Code scanning** で CodeQL default setup を無効にし、workflow setup だけを有効にしてください。両方を有効にすると解析設定が重複します。workflow は[CodeQL Action の現行案内](https://github.com/github/codeql-action)に従い v4 を使います。

## 終了済みのholistic会話

holisticの合成は、最終出力を `conversation_commit` に記録し、変更不能な
`conversations` artifactへ結び付けます。同じrunを再開すると、保存済みの最終出力と
合格済み部分を再利用し、再び延長しません。最終記録の前に停止した場合は、
順序付きの `turn_commit` から再開します。既存DBは開く時に新しいtableを追加します。
設定hashは、detailedとholisticの結果を同じrunへ混在させることを防ぎます。

2ターン以上の合格後に品質理由で停止した場合、合格済み部分を
`QUALITY_CANDIDATE` として保持できます。元の停止会話は非公開の
`conversation-stops` artifactへ残し、棄却ターンをexportへ含めません。
実行エラーはERRORのままです。予定の長さの完走を求める場合は
`evaluation.retain_accepted_prefix: false` を指定し、設定変更後は新しいrun IDを使います。
`rate-existing` は保存Q/Aだけを再評価し、未生成ターンを補いません。
空の入力や1ターンの入力は品質候補になりません。

再評価は、推論失敗を該当会話へ記録して残りの入力を続行します。
不正出力の再試行を使い切った場合はABSTAINED、通信・サーバーの失敗はERRORです。
元の質問と回答を停止ターンに保持し、会話ID・ターン番号・理由を非公開artifactへ保存します。
失敗したターンをcommitしたり、合格済み部分に含めたりしません。

## 大きな合成run

合成は完了した会話をJSONLへ順次書き出し、ファイル全体を毎回書き換えません。
概要は最初の1件、100件ごと、終了時に更新します。同じ変更不能な入力manifestと
設定で再開すると、保存された会話・ターンcommitから出力を復元します。

新しいrequest artifactは、繰り返し現れる画像data URLを内容hashで管理する
`request-images` へ分離します（`archive_format: image-refs-v1`）。HTTP payloadと
request hashは変えません。監査には
`pixelogue.serving.read_request_artifact(store, artifact_hash)` を使い、正確な要求を復元します。
旧形式の埋め込み画像にも対応します。バックアップには参照先の画像artifactも含めてください。


## 各推論試行の費用と再開

`runtime.refill_completed_images` の既定値は `false` です。`true` では、遅い先行画像を待っている間も完了した画像の枠に次の画像を投入し、結果は入力順で返します。実行中の画像数は設定上限内、投入済み・待機中の結果は指定並列数の2倍までです。各会話の公開履歴とDB処理の直列化を維持します。実行方式の比較には新しいrun識別を使います。

実際のHTTP試行ごとに `model_call_attempt` を保存します。受信応答は費用を確定する前に永続化し、形式不正でも有効なusageを計上します。usageの欠測・不正値は不明として保持し、その出力上限分の予算予約を残します。中断したrunを開くと保存応答の計上を一度だけ確定します。同じ要求・同じ `trial_id` の再開は不正応答も再利用し、別の実験試行には別IDを付けます。キャッシュ利用、通信再試行、中断、形式不正は別記録です。

`profile` は実際の試行件数、状態、キャッシュ件数、判明分のトークン小計を表示します。欠測を含む総数はnullです。並列呼び出し時間の合計と推論の実時間は異なり、モデル読み込み・GPU割当時間は別の運用記録として扱います。過去の訂正集計は別資料へ保存し、原本の台帳を書き換えません。

盲検の評価者には、それぞれ異なる固定の試行IDを付けます。pilotで同じendpointを使う場合も別々に呼び出し、再開時には各評価者の保存応答を再利用します。試行IDは非公開の運用情報で、モデルに見せるpayloadには含めません。
