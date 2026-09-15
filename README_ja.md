# Pixelogue

Pixelogue は、画像に基づく複数往復の質問と回答を作る Python 3.12 製のパイプラインです。画像の利用条件を検査し、専用モデルで自然な指示を選び、2〜6 往復の対話を生成します。その後、互いの判定を見せない2回の評価、多様性を考慮した選抜、学習用出力までを扱います。

このリポジトリに含まれるのはパイプライン本体と小規模検証用の仕組みです。30,000 対話の完成データや追加学習済みの重みは含みません。

## 処理中に守られる境界

- 指示選択器は設定で 1 つだけ有効にします。失敗時に別の選択器へ自動切り替えしません。
- 質問が2回の独立した画像適合検査を通るまで、回答を生成しません。
- 回答を見る前に両評価器で公開要求を固定し、有効な要求を 1 件ずつ評価します。
- 評価器AとBは、互いの判定や生成側の役割を知らずに別々に評価します。standardプロファイルでは、Qwen3.8-27BとGemma 4 31Bという異なるモデル系列を使います。
- Open Images V7 の validation 画像とその近似画像グループは検証専用です。学習用には出力できません。
- モデルの要求、応答、revision、processor revision、トークン数を保存します。
- SQLite の WAL はローカルファイルシステムに置き、整合したバックアップだけを共有領域へコピーします。
- BF16 を固定し、量子化指定は設定エラーとして拒否します。

## CPU だけで始める

[uv](https://docs.astral.sh/uv/) をインストールし、リポジトリのルートで実行します。

```sh
uv sync --locked --extra cpu --group dev
uv run --locked pixelogue compile \
  --config configs/pilot.yaml \
  --output artifacts/compiled-plan.json
uv run --locked pixelogue make-fixtures \
  --destination data/fixtures \
  --pairs-per-stratum 2
uv run --locked pytest
```

`compile` は設定全体を検査し、24 種類のタスク、28 個の評価基準、言語ごとの正確な件数、JSON Schema を出力します。ここまではモデル重みも GPU も不要です。

## 読む順番

1. [CPU クイックスタート](docs/quickstart_ja.md)
2. [詳しいパイプラインガイド](docs/pipeline/README_ja.md)
3. [画像と Open Images V7](docs/data_ja.md)
4. [モデルと GPU の確認](docs/models-and-gpu_ja.md)
5. [生成・選抜・出力](docs/workflow_ja.md)
6. [復旧と CI](docs/recovery-and-ci_ja.md)
7. [Qwen3.5-9B pilot の実測結果](docs/validation/qwen35-9b-pilot_ja.md)
8. [実装対応表](docs/implementation-map_ja.md)

## モデルの役割

| 役割 | 既定のリポジトリ |
|---|---|
| 指示選択器 | `Qwen/Qwen3.5-2B` |
| 設定でのみ切り替える選択器 | `Qwen/Qwen3.6-35B-A3B` |
| 生成器・評価器 A | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 生成器・評価器 B | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| 独立した学習側画像プロセッサー | `Qwen/Qwen3-VL-8B-Instruct` |

`configs/pilot.yaml` は、1 GPUで動作を検証するための一時的なプロファイルです。生成器と評価器の2つの論理的な役割を、1つの `Qwen/Qwen3.5-9B` エンドポイントへ割り当てています。この構成ではパイプラインの動作を確認できますが、異なるモデルによる評価の多様性は確認できません。`configs/standard.yaml` で使用する本来のモデルは変更していません。

各モデルと画像プロセッサーのrevisionは設定例で固定しています。意図して変更した場合は、必ず `doctor` を再実行してください。

## 開発時の検査

```sh
uv sync --locked --extra cpu --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
uv build
uv run --locked pre-commit run --all-files
```

取得画像、モデル重み、実行中のデータベース、`_references/`、開発者だけが使う `_docs/` はGit に含めません。
