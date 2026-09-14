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
