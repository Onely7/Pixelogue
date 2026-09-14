# 対話を選抜して出力する

各往復の評価をすべて通過した対話でも、最終データセットに適しているとは限りません。同じ画像の対話が多すぎたり、特定のタスクへ偏ったり、検証専用画像から作られていたりすることがあります。対話単体ではなく、データセット全体の条件を扱うのが選抜処理です。

[前へ：対話を生成して評価する](generation-and-evaluation_ja.md) · [目次へ戻る](README_ja.md) · [次へ：実際の生成物を見る](artifact-examples_ja.md)

## 1. 必要に応じて保存済み対話を再評価する

`rate-existing` は、保存済みの質問と回答を変更せず、新しい評価に通します。

```sh
uv run --locked pixelogue rate-existing \
  --config configs/pilot.yaml \
  --conversations artifacts/open-images-pilot-001/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-rerate-001 \
  --output artifacts/open-images-pilot-001/rated-conversations.jsonl
```

元のrunとは別のrun IDを使います。評価器を変更した後の確認には役立ちますが、確定した公開回答をその場で書き換える機能ではありません。結果と出典別summaryは、元の結果に添える別ファイルとして保存します。

## 2. 選抜候補を固定する

```sh
uv run --locked pixelogue freeze-pool \
  --conversations artifacts/conversations.jsonl \
  --output artifacts/frozen-pool.json
```

`QUALITY_CANDIDATE` の対話だけが固定poolへ入ります。詳しい対話レコードから、選抜に必要な項目だけを取り出します。具体的には、conversation ID、言語、image ID、visual group ID、task family、semantic family、プロファイルの並び、画像の用途です。並び順も含めたpool全体をハッシュで固定します。

先に固定するのは、選抜中や監査中に入力候補が変わると、判定がどの候補集合に対するものか分からなくなるためです。

## 3. 偏りを抑えて選抜する

```sh
uv run --locked pixelogue select \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --output artifacts/selection.json
```

CP-SATは、整数の条件を同時に満たしながら、各候補を「含める・含めない」のどちらかへ決めるソルバーです。Pixelogueでは、次の条件を指定できます。

- 全体の件数を正確に合わせる。
- 言語ごとの件数を正確に合わせる。
- task familyごとの最低件数を満たす。
- 同じ画像、visual group、semantic familyから選ぶ上限を守る。
- 検証用画像を除外する。

再現性を保つため、並列探索に使うworker数は1とし、乱数の初期値であるseedも固定します。`OPTIMAL` または `FEASIBLE` なら、選ばれたIDが保存されます。`INFEASIBLE` は固定poolでは条件を満たせない状態です。`UNKNOWN` は制限時間内に解を確定できなかった状態です。前者では候補の追加や配分の見直し、後者では時間制限やソルバー動作の調査が必要になります。

## 4. ソルバーとは別の方法で監査する

```sh
uv run --locked pixelogue audit \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --selection artifacts/selection.json \
  --output artifacts/audit.json
```

監査処理はソルバーの内部変数を使わず、選ばれたIDからすべての条件を整数で数え直します。poolハッシュ、存在しないID、IDの重複、検証用画像の混入も確認します。合格すると、監査ハッシュが `selection.json` へ結び付けられます。

監査結果を結び付けていないselectionは `export` が拒否します。重要な条件をソルバーと出力境界の異なる実装で確認する仕組みです。

## 5. 4つのファイルへ分けて出力する

pilotプロファイルから学習用データは出力できません。確認済みのstandard設定と、学習利用できる画像を使います。

```sh
uv run --locked pixelogue export \
  --config configs/standard.yaml \
  --conversations artifacts/conversations.jsonl \
  --selection artifacts/selection.json \
  --destination artifacts/export
```

| ファイル | 内容 |
|---|---|
| `training.jsonl` | 公開する質問と回答。画像参照は最初のuser messageに1回だけ入る。 |
| `ratings.jsonl` | 往復ごとの評価項目と集約結果。 |
| `provenance.jsonl` | 取得元、画像、類似画像グループ、生成モデル、学習側画像プロセッサーの固定情報。 |
| `selection.json` | 固定pool、選抜ID、ソルバーの結果、監査との結び付き。 |

candidate ID、選択理由、評価理由、画像タイトル、実行情報は `training.jsonl` に入りません。前段の設定に誤りがあっても、検証用画像はexport時にもう一度拒否されます。

## 6. 検証、バックアップ、復元

run終了後に、データベースとすべての確定済みアーティファクトを検証します。

```sh
uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id open-images-pilot-001
```

SQLiteのbackup APIを使い、整合した状態で保存します。

```sh
uv run --locked pixelogue backup \
  --config configs/pilot.yaml \
  --run-id open-images-pilot-001 \
  --destination /shared/pixelogue-backups/open-images-pilot-001
```

使用中の `.sqlite3`、`-wal`、`-shm` を別々にコピーしてはいけません。新しい空のローカルディレクトリへ復元し、再開前に `replay` を実行します。完全な手順と失敗時の扱いは[復旧ガイド](../recovery-and-ci_ja.md)を参照してください。

公開対話、評価、来歴、監査済みselectionは、同じ変更不能な入力を参照する必要があります。
