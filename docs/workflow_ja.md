# 生成・選抜・出力

画像の取り込みが終わり、必要な model endpoint が `doctor --check-servers` を通過した後の手順です。

## 1. pilot 対話を生成する

```sh
uv run --locked pixelogue synthesize \
  --config configs/pilot.yaml \
  --images artifacts/prepared-open-images/images.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-pilot \
  --output artifacts/open-images-pilot/conversations.jsonl
```

言語と 2 つの生成モデルは、小さな batch でも設定比率どおりの正確な件数に割り当てます。1 対話の
質問、回答、1 回だけ許される回答修復は同じ生成モデルが担当します。各 turn は次の順序です。

1. 画像から範囲を限定した capability を得る
2. 24 タスクから対応する指示候補を作る
3. 画像と確定済み履歴だけを渡し、設定した選択器で候補を 1 つ選ぶ
4. 具体的な質問文を生成する
5. Qwen3.8 と Gemma の両方で質問の画像適合性を確認する
6. 回答を見る前に、両評価器が有効な公開要求を独立して洗い出す
7. 質問適合性と要求一覧が一致した場合だけ回答を生成する
8. 回答の全主張を抽出し、該当する評価項目を両モデルで検査する
9. 一致した型付き計算は浮動小数点数を使わず再計算し、網羅回答の集合も照合する
10. 明確な不合格を修復する場合は同じ生成モデルで 1 回だけ直し、全評価をやり直す
11. 必須 gate をすべて通過した turn だけ確定する

`NO_SUITABLE_CANDIDATE` ならその計画を終了します。代替選択器へ自動的には切り替えません。
評価不一致、根拠不足、通信・schema 障害は明確な不合格と区別します。
要求は公開 user message の正確な文字範囲を参照しなければなりません。両者の要求一覧が
一致しなければ回答生成前に保留するため、回答を見てから要求を弱めることはできません。

`conversations.jsonl` と `conversations.summary.json` が作られます。summary はデータセット別なので、
Open Images の結果を他の画像との平均だけで隠しません。検証用対話は全経路を通りますが、学習用
には昇格できません。

## 2. 保存済み対話を変更せず再評価する

```sh
uv run --locked pixelogue rate-existing \
  --config configs/pilot.yaml \
  --conversations artifacts/open-images-pilot/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-rerate \
  --output artifacts/open-images-pilot/rated-conversations.jsonl
```

保存した質問と回答を fresh な二重評価にかけます。公開文は書き換えません。結果と出典別 summary は
sidecar として保存します。

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

CP-SAT は総数と言語数を厳密に合わせ、task family の下限、画像・visual group・semantic family ごとの
上限を同時に満たします。`INFEASIBLE` は固定 pool では条件を満たせない状態、`UNKNOWN` は時間内に
結論できなかった状態です。独立監査は solver 変数を使わずに再計数し、全条件を満たした場合だけ
監査 hash を `selection.json` に結び付けます。

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

候補 ID、選択理由、評価理由、画像タイトル、運用情報は `training.jsonl` に入りません。学習側の
Qwen3-VL-8B processor lock は指示選択器と独立しており、provenance に記録します。
