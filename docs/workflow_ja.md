# 生成・選抜・出力

画像の取り込みが終わり、必要なモデルのエンドポイントが `doctor --check-servers` を通過した後の手順です。

各判定点の理由と生成物の実例は、[詳しいパイプラインガイド](pipeline/README_ja.md)で説明しています。

## 1. pilot 対話を生成する

```sh
uv run --locked pixelogue synthesize \
  --config configs/pilot.yaml \
  --images artifacts/prepared-open-images/images.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-pilot \
  --output artifacts/open-images-pilot/conversations.jsonl
```

同時に処理する独立画像の上限は `runtime.max_concurrent_images` で指定します。standardと通常pilotは4件、1 GPUのモデル対pilotは2件を上限とします。直列実行には `--workers 1`、測定時には設定上限以下の `--workers N` を指定します。より高い並列数を比較するときは、別の設定で上限を明示します。1つの対話内の往復は順番どおりに処理し、最終的な対話レコードも入力順で保存します。

言語と2つの生成モデルは、小さなbatchでも設定比率どおりに割り当てます。1対話の質問と回答は同じ生成モデルが担当します。既定の `evaluation.mode: holistic` は次の順序です。

1. 画像の領域ごとに、capabilityと根拠を観察する
2. 65の標準候補と、有効化・環境・対象モデルでの校正が揃った専門拡張から、画像ごとの根拠に合う操作を絞る
3. 公開パラメータと実行条件を具体化し、回答予算内に収まる候補を選択器へ渡す
4. 質問を生成し、重複・内部プロンプト転載を除外した後、選択タスクを見せずに操作を分類し、両評価器が操作と領域への適合性を確認する
5. 回答を生成し、操作条件を含めて2モデルが独立して総合評価する
6. 操作ごとの追加検証もすべて通過したターンだけを確定する

言語・形式・安全性も総合評価に含めます。実行可能な操作と未実装検証器は[タスクカタログv7](tasks/README_ja.md)で確認できます。detailed用の要求・主張抽出、項目別採点、回答修復はholisticでは行いません。総合評価の項目IDは `Q_HOLISTIC` です。

`NO_SUITABLE_CANDIDATE` や不合格・保留で計画を終了した場合も、`evaluation.retain_accepted_prefix: true` なら2ターン以上の合格済み部分を採用候補に残します。失敗したターンは非公開の `conversation-stops` に保存し、学習出力には含めません。通信・schema障害は実行エラーとして扱います。

従来の分解評価は `evaluation.mode: detailed` で使用できます。この方式では回答前の質問適合性・要求抽出、回答後の主張抽出・項目別採点・集合照合を行い、明確な不合格には1回の修復を試み、予定ターン数の完走を要求します。方式を変更する場合は新しいrun IDを使ってください。

`conversations.jsonl`、`conversations.summary.json`、確定操作と最後の確定操作を数える `conversations.operations.json` が作られます。summary はデータセット別なので、Open Images の結果を他の画像との平均だけで隠しません。検証用対話は全経路を通りますが、学習用には昇格できません。

実行中または完了後にモデル呼び出しの所要時間とトークン数を確認するには、ローカルDBを `profile` へ渡します。

```sh
uv run --locked pixelogue profile \
  --database /var/tmp/pixelogue/open-images-pilot/run.sqlite3 \
  --output artifacts/open-images-pilot/inference-profile.json
```

新しいrunでは、HTTP要求ごとの所要時間を直接記録します。古いDBでは、保存された応答同士の時間差から直列実行時の所要時間を推定し、レポートの測定方法を `completion_gap_estimate` と表示します。

現在の台帳は、モデルの固定情報、処理段階、要求ハッシュが同じモデル呼び出しを1行にまとめます。そのため、同じ要求の再試行は `profile` の1行を共有することがあります。実際のHTTP試行数が必要な場合は `budget.request_count` または応答アーティファクト数を使い、所要時間と処理段階ごとの傾向には `profile` を使います。両方を使った例は[Qwen3.5-9B処理速度の検証結果](validation/qwen35-9b-throughput_ja.md)で確認できます。

## 2. 保存済み対話を変更せず再評価する

```sh
uv run --locked pixelogue rate-existing \
  --config configs/pilot.yaml \
  --conversations artifacts/open-images-pilot/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-rerate \
  --output artifacts/open-images-pilot/rated-conversations.jsonl
```

保存した質問と回答を fresh な二重評価にかけます。公開文は書き換えません。結果と出典別 summary はsidecar として保存します。

評価者の不正出力が再試行後も続く会話は `ABSTAINED`、通信・サーバー障害は `ERROR` と記録し、残りの会話を処理します。停止したターンの質問・回答と非公開の失敗理由を保存し、そのターンは確定・採用しません。

## 3. 学習候補を固定して選抜する

全 turn を通過し、学習利用が許可された対話だけが pool 候補になります。

```sh
uv run --locked pixelogue freeze-pool \
  --conversations artifacts/conversations.jsonl \
  --output artifacts/frozen-pool.json
uv run --locked pixelogue select \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --output artifacts/selection.json
uv run --locked pixelogue audit \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --selection artifacts/selection.json \
  --output artifacts/audit.json
```

CP-SAT は総数と言語数を厳密に合わせ、task family の下限、画像・visual group・semantic family ごとの上限を同時に満たします。`INFEASIBLE` は固定 pool では条件を満たせない状態、`UNKNOWN` は時間内に結論できなかった状態です。独立監査は solver 変数を使わずに再計数し、全条件を満たした場合だけ監査 hash を `selection.json` に結び付けます。

## 4. 標準学習 bundle を出力する

pilot profile は必ず拒否されます。学習利用条件を確認済みの standard run だけを指定します。

```sh
uv run --locked pixelogue export \
  --config configs/standard.yaml \
  --conversations artifacts/conversations.jsonl \
  --selection artifacts/selection.json \
  --destination artifacts/export
```

| ファイル | 内容 |
|---|---|
| `training.jsonl` | 公開 user/assistant 文。画像参照は最初の user message に 1 回だけ |
| `ratings.jsonl` | turn ごとの評価項目と集約結果 |
| `provenance.jsonl` | 出典、visual group、生成モデル、独立した student processor lock |
| `selection.json` | 固定 pool、選抜 ID、solver status、監査 hash |

候補 ID、選択理由、評価理由、画像タイトル、運用情報は `training.jsonl` に入りません。学習側のQwen3-VL-8B processor lock は指示選択器と独立しており、provenance に記録します。

### 生成中の進捗表示

`synthesize` は既定で標準エラー出力へtqdmの進捗バーを表示します。開始時、画像ごとの保存後、
および待機中の10秒ごとに、保存済み件数／総数、worker数、品質候補・棄却・評価不能・
エラーの件数、経過時間、処理速度、推定残り時間を表示します。総数は `data.target_dialogues` の上限を反映し、
処理済み件数は合格件数とは異なります。件数は入力順の保存に合わせて更新するため、
後続workerが完了していても、先行画像の保存までは加算されません。待機表示は
コントローラーが待っていることを示し、モデルサーバーの正常性を保証するものではありません。
中断時は `finished` ではなく `interrupted` と表示します。`--quiet` で非表示、
`2> progress.log` で進捗の保存、`> report.json` で完了時の標準出力JSONの保存ができます。
