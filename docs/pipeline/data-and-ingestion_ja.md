# 画像を準備する

画像ファイルが1枚あれば、すぐに生成を始められるわけではありません。Pixelogueは画素を展開する前に、画像の取得元を表す安定した識別情報と、利用条件の確認結果を求めます。この順序にすることで、確認していない画像が生成処理へ紛れ込むことを防ぎます。

[目次へ戻る](README_ja.md) · [次へ：対話を生成して評価する](generation-and-evaluation_ja.md)

## 1. 設定を検査する

モデルや画像を取得する前に `compile` を実行します。

```sh
uv run --locked pixelogue compile \
  --config configs/pilot.yaml \
  --output artifacts/compiled-plan.json
```

設定、言語ごとの正確な件数、タスクと評価基準のカタログ、JSON Schemaが解決され、結果全体の`compiled_hash` が保存されます。GPU推論の途中で見つけるより、この時点でSchemaの誤りを直す方が短時間で済みます。

pilotプロファイルは小規模な動作確認用です。生成件数と4 GPU時間の上限があり、学習用データとしてexportできません。standardプロファイルは、内容を確認した本番実行に使います。

## 2. 入力画像を用意する

画像を用意する方法は、大きく分けて2つあります。

### 固定済みのOpen Images V7 validation画像

```sh
uv run --locked pixelogue prepare \
  --config configs/pilot.yaml \
  --destination data/open-images-v7
```

Gitに含まれるマニフェストには、20件のvalidation画像ID、取得元情報、期待するハッシュが固定されています。`prepare` が取得するのは、この20件だけです。用途は常に検証用となります。

```text
data/open-images-v7/
├── images/                              # 取得した画像。Git対象外
├── sources.jsonl                        # モデル入力に使える画像台帳
├── rights.jsonl                         # 利用条件の確認結果
├── open_images_private_metadata.jsonl   # URLなど運用者向けの情報
└── open_images_failures.jsonl           # 再現可能な取得失敗
```

ラベル、矩形、タイトル、説明文は `sources.jsonl` へ移さず、どのモデルにも渡しません。生成した質問の正解として使うこともありません。

### ローカル画像または別に取得した画像

`sources.jsonl` と `rights.jsonl` に、1行につき1個のJSON objectを書きます。2つの台帳は`rights_record_id` で対応付けます。

```json
{"source_id":"local:blue-sign-001","image_path":"images/blue-sign.png","source_group_ids":[],"rights_record_id":"rights:blue-sign-001","purpose":"training","dataset":"local-reviewed","dataset_split":"source"}
```

```json
{"rights_record_id":"rights:blue-sign-001","license_uri":"https://example.org/licence","attribution":"Example author","processing_allowed":true,"qa_redistribution_allowed":true,"image_redistribution_allowed":false,"training_allowed":true,"valid_from":"2026-09-15T00:00:00Z"}
```

これは形式を示す例であり、実在する画像の利用許諾を表すものではありません。実際の利用条件は運用者が確認します。画像パスは `--image-root` からの相対パスです。絶対パスと `..` によるディレクトリ外への移動は拒否されます。

学習用画像では、確認時点で `processing_allowed`、`qa_redistribution_allowed`、`training_allowed` がすべてtrueである必要があります。検証用画像は処理できても、学習用exportからは除外されます。

## 3. 動作確認用の画像を作る

```sh
uv run --locked pixelogue make-fixtures \
  --destination data/fixtures \
  --pairs-per-stratum 2
```

fixtureは、期待どおりの例と、1種類だけ誤りを含む例を組にしたプログラム生成画像です。自然画像に対する正答率を測るものではなく、接続や評価処理を確かめるために使います。

`evaluate-capabilities` を使うと、正解付きfixtureを2つの評価器へ渡して確認できます。期待する採否はコントローラー側だけで保持し、モデルへのプロンプトには含めません。

## 4. 画像を正規化してグループ化する

```sh
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources data/open-images-v7/sources.jsonl \
  --rights data/open-images-v7/rights.jsonl \
  --image-root data/open-images-v7 \
  --artifact-root artifacts/prepared-open-images
```

`ingest` は、次の順序で処理します。

1. 重複したsource IDとrights ID、対応する利用条件がない参照を拒否する。
2. 利用条件の有効期間と、学習用・検証用の用途を確認する。
3. `--image-root` の内側にあるパスを解決し、安全にファイルを読む。
4. 未対応形式、アニメーション、20 MiB超のファイル、展開後に4,000万画素を超える画像、短辺が128 px未満の画像を拒否する。
5. EXIF orientationとOpen Imagesに記録された回転を適用する。
6. ICCプロファイルがあればsRGBへ変換し、透明部分を白に重ね、決定的なRGB PNGを保存する。
7. 保存したファイルと正規化後の画素を、それぞれハッシュ化する。
8. 同一画素と、`source_group_ids` で宣言した画像をまとめる。必要ならSSCDによる近似画像検査も加える。
9. 類似画像グループに検証画像が1件でもあれば、グループ全体を検証用にする。
10. 学習用画像は、類似画像グループと設定seedから再現可能なsplitへ割り当てる。

個々の不適格画像は `failures.jsonl` に残り、正常な画像まで失われることはありません。一方、IDの重複やrights recordへの不正な参照は、再開時の意味を曖昧にするため、マニフェスト全体を拒否します。

## 5. 準備結果を確認する

```text
artifacts/prepared-open-images/
├── images/
│   └── ab/abcdef...png
├── images.jsonl
├── failures.jsonl
└── manifest.json
```

`images.jsonl` は `synthesize` へ直接渡すファイルです。各行には取得元、用途、データセット、元データと画素のハッシュ、類似画像グループ、画像サイズ、media type、正規化済み画像の相対パスが入ります。

GPUを使う前に件数を確認します。

```sh
wc -l artifacts/prepared-open-images/images.jsonl
wc -l artifacts/prepared-open-images/failures.jsonl
uv run --locked pixelogue doctor \
  --config configs/pilot.yaml \
  --check-servers
```

`failures.jsonl` に行があれば、理由をすべて確認します。固定した20画像を完全に再現できた場合は空になりますが、任意の画像群で常に空でなければならないわけではありません。

## 6. 実行中のデータベースはローカルへ置く

リポジトリがNFS上にあっても、SQLite WALはローカルファイルシステムを必要とします。現在の設定は、実行中のrunを `/var/tmp/pixelogue` に保存します。準備済み画像は別の場所へ置けますが、使用中のデータベースとWALファイルはローカルに置いてください。

`images.jsonl` を読み取れ、利用条件とグループ化の検査を通り、必要なモデルサーバーが正常なら、画像の準備は完了です。
