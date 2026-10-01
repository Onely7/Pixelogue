# Pixelogue

Pixelogue は、画像に基づく複数往復の質問と回答を作る Python 3.12 製のパイプラインです。画像の利用条件を検査し、専用モデルで自然な指示を選び、2〜6 往復の対話を生成します。その後、互いの判定を見せない2回の評価、多様性を考慮した選抜、学習用出力までを扱います。

このリポジトリに含まれるのはパイプライン本体と小規模検証用の仕組みです。30,000 対話の完成データや追加学習済みの重みは含みません。

## 処理中に守られる境界

- 指示選択器は設定で 1 つだけ有効にします。失敗時に別の選択器へ自動切り替えしません。
- 既定の `evaluation.mode: holistic` は、質問と回答を生成した後、画像・履歴と合わせて各モデルが1回ずつ総合評価します。両者が `MET` の場合だけ合格し、不一致・判断不能は保留にします。
- 総合評価に事実の正確さ・要求充足・履歴・言語・明示された形式・安全性を含めます。v7では回答前に両評価器が操作・公開条件・領域の適合性を確認し、操作ごとの追加検証も必須にします。detailed用の要求・主張抽出、項目別採点、回答修復は行いません。
- 空の文章、質問の完全な繰り返し、内部プロンプトの転載はコードで除外します。依頼された再分類に新しい事実は必須ではありません。
- `evaluation.retain_accepted_prefix: true` では、品質判定などで途中終了しても、2ターン以上の合格済み部分を採用候補にします。失敗した後半は非公開の `conversation-stops` に保存し、学習出力には含めません。実行例外は採用に変換しません。
- `evaluation.mode: detailed` で従来の分解評価を比較用に利用できます。この方式では予定ターン数の完走が必要です。評価設定を変更した場合は新しいrun IDを使用してください。
- 評価器AとBは、互いの判定や生成側の役割を知らずに別々に評価します。standardプロファイルでは、Qwen3.8-27B-FP8とGemma 4 31B(W4A16 compressed-tensors)という異なるモデル系列を使います。
- Open Images V7 の validation 画像とその近似画像グループは検証専用です。学習用には出力できません。
- モデルの要求、応答、revision、processor revision、トークン数を保存します。
- SQLite の WAL はローカルファイルシステムに置き、整合したバックアップだけを共有領域へコピーします。
- モデルのdtypeはBF16に固定します。量子化は各生成器に固定されたチェックイン済みの方式(Qwen3.8-27BはFP8、Gemma 4 31BはW4A16 compressed-tensors)だけを許可し、それ以外の値は設定エラーとして拒否します。

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

`compile` は設定全体を検査し、65の標準候補＋7の条件付き拡張、合計72タスク、旧方式用の28個の評価基準、有効な評価設定、言語ごとの正確な件数、JSON Schema を出力します。ここまではモデル重みも GPU も不要です。

領域ごとの根拠、検証器と環境に基づく候補判定、旧24タスクからの移行は[タスクカタログv7](docs/tasks/README_ja.md)にまとめています。標準65タスクにはCPU検証経路を実装済みです。専門拡張7タスクは、対象モデルでの校正と実行環境が揃った場合に通常選択へ入ります。

## 読む順番

1. [CPU クイックスタート](docs/quickstart_ja.md)
2. [詳しいパイプラインガイド](docs/pipeline/README_ja.md)
3. [画像と Open Images V7](docs/data_ja.md)
4. [モデルと GPU の確認](docs/models-and-gpu_ja.md)
5. [生成・選抜・出力](docs/workflow_ja.md)
6. [復旧と CI](docs/recovery-and-ci_ja.md)
7. [Qwen3.5-9B pilot の実測結果](docs/validation/qwen35-9b-pilot_ja.md)
8. [Qwen3.5-9B 処理速度の検証結果](docs/validation/qwen35-9b-throughput_ja.md)
9. [品質・速度改善の実測と運用](docs/validation/quality-and-performance_ja.md)
10. [実装対応表](docs/implementation-map_ja.md)
11. [Open Images合成の確認とタスク別実例](docs/validation/synthesis-campaign_ja.md)

## モデルの役割

| 役割 | 既定のリポジトリ |
|---|---|
| 指示選択器 | `Qwen/Qwen3.5-2B` |
| 設定でのみ切り替える選択器 | `Qwen/Qwen3.6-35B-A3B` |
| 生成器・評価器 A | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| 生成器・評価器 B | [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) |
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
