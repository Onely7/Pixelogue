# Open Images合成の確認とタスク別実例

Open Images V7のvalidation splitから選んだ2,000画像は、合成パイプラインの評価用です。
既存20画像を除き、seed `20261001` によるImageIDのSHA-256順で選択しました。
公式画像の取得・既存ingestに失敗した画像と、正規化画素の完全重複だけを置き換えました。
画像ラベル、説明、bounding boxは選択器・生成器・評価者へ渡しません。
近似画像の重複検査は行っていません。

[固定した2,000画像の出典・ハッシュ](../../validation/open_images_v7_eval_2000_manifest.jsonl)
を使って復元できます。画像bytesと合成結果はGitへ含めません。

実際の評価は、実行中のユーザー指示により1,400画像へ縮小して完了しました。
凍結済みの入力順の先頭1,400枚を使い、元のモデル割当とrun識別を保持しています。
以下の2,000画像の取得・設定は元の画像群の手順です。対象変更と最終結果は
`artifacts/open-images-1400/` に保存し、取り込み済みmanifestや学習利用の可否は変更していません。

```sh
uv run --locked python validation/fetch_open_images_eval_2000.py
uv run --locked pixelogue ingest \
  --config configs/split-pilot.yaml \
  --sources data/open-images-v7-eval-2000/sources.jsonl \
  --rights data/open-images-v7-eval-2000/rights.jsonl \
  --image-root data/open-images-v7-eval-2000 \
  --artifact-root artifacts/prepared-open-images-v7-eval-2000 \
  --workers 8
```

## 実行条件

記録したキャンペーンは、現在は廃止した FP8・W4A16 の組と Qwen3.5-2B 選択器を、zao01 の
96 GiB GPU 1枚用の旧設定で実行しました。再実行では標準の BF16 の組、Qwen3.8-27B ルーター、
分割配置を使うため、結果は直接比較できません。
`configs/split-pilot.yaml` のコピーで `profile: standard`、
`data.pilot: false`、`data.target_dialogues: 2000`、`seed: 20261001` とします。
`data.open_images.enabled: false`、`image_ids_manifest: null` とし、
20画像用の `prepare` 設定から独立した取り込み済みmanifestを渡します。
候補8、画像並列2、最低2ターン、両評価者、品質ゲートを維持します。
専門7タスクは校正未達なので通常選択へ加えません。

```sh
uv run --locked pixelogue synthesize \
  --config artifacts/open-images-2000/config.yaml \
  --images artifacts/prepared-open-images-v7-eval-2000/images.jsonl \
  --artifact-root artifacts/prepared-open-images-v7-eval-2000 \
  --run-id open-images-2000-20261001 --workers 2 \
  --output artifacts/open-images-2000/conversations.jsonl
```

実際の設定、コード、モデル、画像順序、生成モデル割当、予定turn数を実行前に固定します。
同じ試行の再開では同じrun IDとmanifestを使います。契約を変更した場合は別runへ分けます。
SQLite WALはzao01の `/var/tmp/pixelogue` に保存し、完全snapshotを共有領域へバックアップします。
GPU状態・endpoint・出力進捗を10分間隔で保存し、終了直後に予約watcherを再開します。

## エラー・採用率・多様性の点検

```sh
uv run --locked pixelogue run-diagnostics \
  --config artifacts/open-images-2000/config.yaml \
  --run-id open-images-2000-20261001 \
  --conversations artifacts/open-images-2000/conversations.jsonl \
  --output-stem artifacts/open-images-2000/diagnostics
uv run --locked python -m pixelogue.synthesis_campaign_report \
  --images artifacts/prepared-open-images-v7-eval-2000/images.jsonl \
  --conversations artifacts/open-images-2000/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images-v7-eval-2000 \
  --output-dir artifacts/open-images-2000/report
```

採用候補会話は `QUALITY_CANDIDATE` の2〜6確定turnだけです。
未採用会話に残る確定prefix、失敗turn、質問に到達する前の停止を別に数えます。
`ERROR=0` でも、不正なモデル出力による棄権がないとは限りません。
全2,000画像の処理終了と、各段階の形式不備・再試行・停止理由を併せて確認します。

集計は候補会話のタスク/分野別件数、最多タスク占有率、質問の完全重複、会話深さ、
モデル別・言語別成績を含みます。意味的な質問の重複と人手品質は独立票なしでは未測定です。
写真主体の画像群だけで表・チャート・楽譜・回路などのカタログ全体を網羅できるとは扱いません。

## 画像・指示・回答の閲覧

生成される `report/examples.html` は全72タスクを一覧化し、タスク・指示・回答と分野で検索できます。
`examples.md`、`examples.jsonl`、`task-coverage.csv`、`report.json` も保存します。
各タスクで最大3件の実例を、元の質問・回答・先行する公開履歴・会話IDとともに提示します。
画像は正規化した実入力の縮小JPEGです。HTMLには画像をdata URLとして埋め込むため、
`examples.html` だけを移動しても表示できます。Markdown版では別の `thumbnails/` ディレクトリを使います。

自動品質候補と、未採用会話の診断用prefixを明示します。
実例がないタスクは「実例なし」と記載し、カタログの例文や人工的な回答で補いません。
この閲覧資料は人手監査の正解票ではなく、評価画像を学習用にexportするものでもありません。

## 1,400画像の実測結果

標準Qwen/Gemma対で自動品質候補265件（18.9%）を得ました。2ターン204件、3ターン54件、
4ターン7件です。残る1,135画像は棄却・棄権でした。会話の実行エラーは0件ですが、
モデルの不正呼び出し352件、再試行593件が残っています。候補は5タスク・2分野に限られ、
候補ターンの48.8%が物体識別でした。実行と保存済み出力の再利用を確認できた一方、
採用率と指示の多様性には課題が残ります。独立した人手品質は未測定です。

ローカルの `artifacts/open-images-1400/report/examples.html` は全72タスクを一覧化し、
実例の有無を明記します。`final-results.md`、`diagnostics.md`、`evaluation-public/` に
実測結果・停止理由・評価用の公開会話を保存しました。対象範囲の再開では元の全manifestと
モデル割当を使い、処理を変更後の先頭範囲に限定し、推論を禁止しました。
呼び出し・費用・確定結果・公開会話のbytesが同一であることを確認しています。
生成時のコードは `86746c6` です。このrunの再開には凍結済みrevisionを使います。
後から追加したHTML生成の変更は、今後の合成runでは新しいコード識別となります。

[English guide](synthesis-campaign.md)
