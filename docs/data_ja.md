# 画像と Open Images V7

Pixelogue は、画像台帳と対応する利用条件台帳が揃ってから画像を読みます。画像台帳は取得元と学習用・検証用の区分を表し、利用条件台帳は処理や再配布が許されるかを表します。両者は`rights_record_id` で結び付けます。

## 固定した Open Images validation 画像を取得する

[固定マニフェスト](../validation/open_images_v7_manifest.jsonl)には 20 件の画像 ID、帰属、ライセンス URL、掲載ページ、取得バイトの SHA-256 を記録しています。画像そのものは各画像の再配布条件に従い、Git には含めません。

```sh
uv run --locked pixelogue prepare \
  --config configs/pilot.yaml \
  --destination data/open-images-v7
```

公式 validation メタデータを読み、公式 Open Images S3 バケットから画像を取得します。固定した出典情報とバイトのハッシュが一致しない画像は採用しません。主な出力は次のとおりです。

- `sources.jsonl`: モデルへ渡してもよい画像台帳
- `rights.jsonl`: 検証専用の利用条件台帳
- `open_images_private_metadata.jsonl`: URL、タイトルなど運用用の情報
- `open_images_failures.jsonl`: 再現可能な失敗理由。20 件揃えば空
- `images/`: Git 対象外の画像ファイル

タイトル、ラベル、矩形、Localized Narratives などのアノテーションは、選択器・生成器・評価器の入力に入りません。また、既存アノテーションを自由形式の質問全体に対する正解とは扱いません。

## 画像を検査して正規化する

```sh
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources data/open-images-v7/sources.jsonl \
  --rights data/open-images-v7/rights.jsonl \
  --image-root data/open-images-v7 \
  --artifact-root artifacts/prepared-open-images
```

ディレクトリ外へのパス、利用条件不足、未対応形式、アニメーション、20 MiB を超えるファイル、4,000 万画素を超える画像、短辺 128 px 未満の画像を拒否します。EXIF と Open Images の反時計回り回転を適用し、ICC profile があれば sRGB へ変換します。透明部分は白で合成し、決定的なRGB PNG を推論用 view として保存します。

同一画素と宣言済み source group は常にまとめます。SSCD による近似画像の検査も行う場合は、visual 依存とハッシュを固定したローカル TorchScript モデルを指定します。

```sh
uv sync --locked --extra cpu --extra visual --group dev
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources path/to/sources.jsonl \
  --rights path/to/rights.jsonl \
  --image-root path/to/images \
  --artifact-root artifacts/prepared \
  --sscd-model models/sscd.torchscript.pt \
  --sscd-sha256 64文字のSHA256
```

同じ visual group に検証専用画像が 1 件でもあれば、グループ全体を検証専用にします。さらに`export` でも検証画像を拒否するため、学習出力への混入を 2 段階で防ぎます。

公式資料には [V7 validation と取得方法](https://storage.googleapis.com/openimages/web/download_v7.html)および[回転値が反時計回りの角度であること](https://storage.googleapis.com/openimages/web/2018-05-17-rotation-information.html)が説明されています。
