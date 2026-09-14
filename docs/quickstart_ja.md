# CPU クイックスタート

モデル重みを取得せずに、導入と基本契約を確認する手順です。すべてリポジトリのルートで実行します。

## 1. ロック済み環境を作る

Pixelogue は Python 3.12 を使用します。uv は `.python-version` を読み、`.venv` を自動で作成します。

```sh
uv sync --locked --extra cpu --group dev
uv run --locked pixelogue --help
```

`--locked` は `pyproject.toml` と `uv.lock` が食い違うと処理を止めます。実行するたびに別の依存バージョンが入ることを防ぐための指定です。

## 2. pilot 設定をコンパイルする

```sh
uv run --locked pixelogue compile \
  --config configs/pilot.yaml \
  --output artifacts/compiled-plan.json
```

標準出力には `output` と `compiled_hash` を持つ JSON が 1 行表示されます。出力ファイルには、有効な設定、言語別件数、タスク・評価カタログ、JSON Schema が入ります。ここで分かるのは設定が矛盾していないことです。GPU やモデルサーバーの準備完了を意味しません。

## 3. 小さな正解付き fixture を作る

```sh
uv run --locked pixelogue make-fixtures \
  --destination data/fixtures \
  --pairs-per-stratum 2
```

8 種類の検査領域について、正しい回答と誤りを 1 つだけ含む回答を作ります。development とconfirmation は別の分割です。`2` は接続確認用の少数件であり、能力認定用の十分な件数ではありません。

## 4. CPU 検査を実行する

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
```

`uv sync --locked` がロックの不一致を報告した場合、依存変更を確認したうえで `uv lock` を明示的に実行し、`pyproject.toml` と `uv.lock` を一緒にコミットします。
