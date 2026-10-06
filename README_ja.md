# Pixelogue

Pixelogue は、画像に基づく複数往復の質問と回答を作る Python 3.12 製のパイプラインです。画像の利用条件を検査し、タスク系統の割り当てのために各画像を分析してから、2〜6 往復の対話を起草します。各往復は、互いの判定を見せない2つの評価器が回答の前後で確認します。その後、多様性を考慮した選抜、学習用出力までを扱います。

このリポジトリに含まれるのはパイプライン本体と小規模検証用の仕組みです。30,000 対話の完成データや追加学習済みの重みは含みません。

## 処理中に守られる境界

- ルーター（`models.router`）は生成器 A の `Qwen/Qwen3.8-27B` サーバーそのもので、タスク系統の割り当てのために各画像を分析するだけです。対話は書かず、失敗時に別のモデルへ切り替えることもありません。一時的な Qwen3.5-9B パイロットだけは、別の `Qwen/Qwen3.5-2B` ルーターを使います。
- 各往復の質問案は、そのターンに割り当てた操作について、会話を担当する生成器が直接書きます。回答を作る前に決定的な検査と両評価器による統合質問ゲートを行い、最初に通過した質問だけに回答します。
- 総合評価に事実の正確さ・要求充足・履歴・言語・明示された形式・安全性を含めます。両評価器が `MET` の場合だけ合格し、不一致・判断不能は保留にします。例外は画像全体での再評価1回と、両評価器が評価し直す回答修復1回だけです。該当する操作検証器もすべて通過する必要があります。
- 空の文章、質問や事実の繰り返し、内部プロンプトの転載はコードで除外します。依頼された再分類に新しい事実は必須ではありません。
- `evaluation.retain_accepted_prefix: true` では、品質判定などで途中終了しても、2ターン以上の合格済み部分を採用候補にします。失敗した後半は非公開の `conversation-stops` に保存し、学習出力には含めません。実行例外は採用に変換しません。評価設定を変更した場合は新しいrun IDを使用してください。
- 評価器AとBは、互いの判定や生成側の役割を知らずに別々に評価します。standardプロファイルでは、Qwen3.8-27BとGemma 4 31Bという異なるモデル系列を使います。
- Open Images V7 の validation 画像とその近似画像グループは検証専用です。学習用には出力できません。
- モデルの要求、応答、revision、processor revision、トークン数を保存します。
- SQLite の WAL はローカルファイルシステムに置き、整合したバックアップだけを共有領域へコピーします。
- モデルのdtypeはBF16に固定し、量子化したモデルは使いません。設定で受け付けるのは、非量子化の `Qwen/Qwen3.8-27B` と `google/gemma-4-31B-it` の組か、一時的な Qwen3.5-9B パイロットだけです。`quantization` の設定は未知の項目として拒否します。

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

`compile` は設定全体を検査し、65の標準候補＋7の条件付き拡張、合計72タスクのカタログ、タスクごとの実行可否、有効な設定、言語ごとの正確な件数、JSON Schema を出力します。ここまではモデル重みも GPU も不要です。

系統の割り当て、質問案の受付、検証器と環境に基づく実行可否、旧24タスクからの移行は[タスクカタログv7](docs/tasks/README_ja.md)にまとめています。標準65タスクにはCPU検証経路を実装済みで、合成で質問案を作るのもこれらだけです。専門拡張7タスクは `evaluate-specialist` で検証し、対象モデルでの校正と実行環境が必要です。

## 読む順番

1. [CPU クイックスタート](docs/quickstart_ja.md)
2. [パイプラインガイド](docs/pipeline/README_ja.md)
3. [画像と Open Images V7](docs/data_ja.md)
4. [モデルと GPU の確認](docs/models-and-gpu_ja.md)
5. [生成・選抜・出力](docs/workflow_ja.md)
6. [復旧と CI](docs/recovery-and-ci_ja.md)
7. [Qwen3.5-9B pilot の実測結果](docs/validation/qwen35-9b-pilot_ja.md)
8. [Qwen3.5-9B 処理速度の検証結果](docs/validation/qwen35-9b-throughput_ja.md)
9. [品質・速度改善の実測と運用](docs/validation/quality-and-performance_ja.md)
10. [実装対応表](docs/implementation-map_ja.md)
11. [Open Images合成の確認とタスク別実例](docs/validation/synthesis-campaign_ja.md)

[採用率と合成速度の比較手順](docs/validation/yield-and-speed_ja.md)に、比較条件・計測・監査・実例HTMLの解釈をまとめています。`docs/validation/` の報告は以前のパイプラインで測定した記録であり、現在の直接起草の経路を説明するものではありません。

公開出力仕様、独立した根拠抽出、質問条件の固定と少数画像での比較は、
[検証契約の改訂2](docs/tasks/verification-contract-v2_ja.md)を参照してください。

## モデルの役割

| 役割 | 既定のリポジトリ |
|---|---|
| 画像のルーター | `Qwen/Qwen3.8-27B`（生成器 A のサーバー） |
| 生成器・評価器 A | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 生成器・評価器 B | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| 独立した学習側画像プロセッサー | `Qwen/Qwen3-VL-8B-Instruct` |

`configs/pilot.yaml` は、1 GPUで動作を検証するための一時的なプロファイルです。生成器と評価器の2つの論理的な役割を、1つの `Qwen/Qwen3.5-9B` エンドポイントへ割り当てています。この構成ではパイプラインの動作を確認できますが、異なるモデルによる評価の多様性は確認できません。画像の割り当ては別の `Qwen/Qwen3.5-2B` サーバーが行います。`configs/standard.yaml` で使用する本来のモデルは変更していません。

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
