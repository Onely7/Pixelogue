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

## 多様なWeb画像による検証用サンプル

[固定したCommons manifest](../validation/diverse_web_eval_manifest.jsonl) に、60の大分類から外観を確認した画像を各1枚記録しています。文書、表、チャート、図、医用画像、画面、写真などを含みます。画像本体はGit管理外の `data/` に置き、取得元ページ、サムネイルのハッシュ、寸法、帰属情報だけをcommitします。用途は**検証専用**です。独立した正解ラベルは付いていないため、モデルが生成した質問・回答を正解として扱いません。

```sh
uv run --locked python validation/fetch_diverse_eval.py
uv run --locked pixelogue ingest \
  --config configs/gpu-watch-diverse.yaml \
  --sources data/diverse-web-eval/sources.jsonl \
  --rights data/diverse-web-eval/rights.jsonl \
  --image-root data/diverse-web-eval \
  --artifact-root artifacts/prepared-diverse-web-eval
```

取得処理は固定したCommonsページID、サムネイルSHA-256、記録済みライセンスURIを照合します。`ingest` でも通常どおり画像と利用条件を検査します。`failures.jsonl` を確認し、合成前に60枚すべてが受理されたことを確認してください。分類40は複数フレームを並べた**静止コンタクトシート**です。アニメーション画像そのものは取り込み対象外です。分類名とCommonsの説明文はモデル入力に渡しません。検証器の精度を測る校正には、別途、独立した人手ラベルが必要です。

多様画像レポートのタスク一覧と件数には、`COMMITTED` のターンだけを含めます。最後に棄却・棄権された試行は診断用の停止記録に残し、確定タスク件数には含めません。

## 独立した6画像の保留評価セット

[保留評価manifest](../validation/holdout_web_eval_manifest.jsonl)には、写真・文書・表・チャート・フローチャート・アプリ画面の別のCommonsページ6件を固定しています。ページID、画素ハッシュ、視覚groupは、調整用60枚およびOpen Imagesの固定サンプルと重なりません。取得処理は検証専用の出典・権利記録を復元し、画像本体はGitに含めません。

```sh
uv run --locked python validation/fetch_holdout_eval.py
uv run --locked pixelogue ingest \
  --config configs/gpu-watch-diverse.yaml \
  --sources data/holdout-web-eval/sources.jsonl \
  --rights data/holdout-web-eval/rights.jsonl \
  --image-root data/holdout-web-eval \
  --artifact-root artifacts/prepared-holdout-web-eval
```

合成前に6枚すべての受理を確認します。少数の画像で形式別の動作を点検するためのセットであり、一般的な成功率や人手の回答正解ラベルは示しません。後で質問・回答を監査する際も、調整用画像とは分けて扱います。

楽譜・幾何・回路の追加画像3枚は[専門評価用manifest](../validation/specialist_web_eval_manifest.jsonl)に固定しています。`uv run --locked python validation/fetch_specialist_eval.py` で画像と出典・権利記録を復元し、`data/specialist-web-eval` と `data/specialist-extra-eval` をそれぞれ検証専用として取り込めます。取得時にバイトハッシュと寸法を照合します。同じ画像に別の候補回答を与えても、独立した校正標本には数えません。

## PubChem構造式の検証用サンプル

[固定したPubChem manifest](../validation/pubchem_2d_eval_manifest.jsonl)には、C・N・O・F・Cl・Brの3～5重原子からなる中性・非環式の2D画像89件を記録しています。開発用10件と、異なる分子構造の確認用79件を分けています。元データのSMILESをcontroller側の非公開正解ラベルとして使います。この範囲は整ったデジタル描画に限られ、手書き構造式や任意の化学画像の精度を示すものではありません。画像本体と回答ラベルはGit管理外です。

```sh
uv sync --locked --directory runtime/validators
runtime/validators/.venv/bin/python validation/fetch_pubchem_chemical_eval.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/pubchem-2d-simple-eval/sources.jsonl \
  --rights data/pubchem-2d-simple-eval/rights.jsonl \
  --image-root data/pubchem-2d-simple-eval \
  --artifact-root artifacts/prepared-pubchem-2d-simple-eval
uv run --locked python validation/build_pubchem_chemical_cases.py
```

取得時にCID、画像SHA-256、寸法、画像群、SMILESハッシュを固定manifestと照合します。`private_labels.jsonl` は `data/` にのみ保存します。case生成では確認用の正例20件と誤答59件を固定し、モデルにはCID・元のSMILES・期待判定を渡しません。通常選択に入れるには、実測結果が校正の信頼区間基準を満たす必要があります。

## ScreenSpotのUIクリック検証用サンプル

[固定したScreenSpot manifest](../validation/screenspot_ui_eval_manifest.jsonl)には、人手で文字対象の領域が注釈された異なるスクリーンショット89件を記録します。開発用10件と確認用79件を分けます。[ScreenSpotのtest split](https://huggingface.co/datasets/bevaya/ScreenSpot)を出典とし、対象領域はローカルの非公開ラベルファイルにだけ保存してモデルには渡しません。対象範囲は画面上の文字対象に対する単一クリック指定であり、任意のUI操作や実操作は含みません。

```sh
uv run --locked python validation/fetch_screenspot_ui_eval.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/screenspot-ui-eval/sources.jsonl \
  --rights data/screenspot-ui-eval/rights.jsonl \
  --image-root data/screenspot-ui-eval \
  --artifact-root artifacts/prepared-screenspot-ui-eval
uv run --locked python validation/build_screenspot_ui_cases.py
```

取得時に画像と注釈のハッシュを固定manifestと照合します。抽出時には64ビットのdifference hashで距離5以下の近似画面も除きますが、近似コピーの完全な監査ではありません。確認用の異なる画面について、対象領域の中心を示す正例20件と領域外を示す誤答59件を固定します。画像は検証専用でGit管理外です。通常選択には、実際の抽出と判定が校正基準を満たす必要があります。

最初の確認結果を開発時に調べたため、別の[ScreenSpot保留セットmanifest](../validation/screenspot_ui_holdout_manifest.jsonl)に追加の確認用画面150件を固定します。両セットで画素ハッシュが一致するものとdifference hashの距離5以下を除きます。最初の固定時だけ `--freeze --holdout` を指定し、その後の復元は `--holdout` だけを指定します。非公開のケース作成処理は、評価前に正例50件と領域外クリックの誤答100件を固定します。

```sh
uv run --locked python validation/fetch_screenspot_ui_eval.py --holdout
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/screenspot-ui-holdout/sources.jsonl \
  --rights data/screenspot-ui-holdout/rights.jsonl \
  --image-root data/screenspot-ui-holdout \
  --artifact-root artifacts/prepared-screenspot-ui-holdout
uv run --locked python validation/build_screenspot_ui_cases.py --holdout
```

## PrIMuSの楽譜検証用サンプル

[固定したPrIMuS manifest](../validation/primus_music_eval_manifest.jsonl)には、[公式PrIMuSアーカイブ](https://grfia.dlsi.ua.es/primus/)から異なる楽曲IDの譜例89件を記録します。印刷楽譜と独立したMEIラベルが対応します。初版の校正領域は完成した第1小節、単声部、ト音・ヘ音記号、調号なし、対応する音符・休符の音価に限り、臨時記号・タイ・連符は含めません。細長い楽譜画像を取り込み前に正確に2倍へ拡大します。元アーカイブ、画像、MEIラベルはGit管理外で、モデル入力へラベルを渡しません。

```sh
uv sync --locked --directory runtime/validators
uv run --locked python validation/fetch_primus_music_eval.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/primus-music-eval/sources.jsonl \
  --rights data/primus-music-eval/rights.jsonl \
  --image-root data/primus-music-eval \
  --artifact-root artifacts/prepared-primus-music-eval
uv run --locked python validation/build_primus_music_cases.py
```

取得器は元アーカイブ、MEI、元PNG、変換後PNGと導出したラベルのハッシュを照合します。確認用の異なる楽曲について正しい転写20件と隣接音への誤答59件を固定します。ケース生成時のラベル整合検査は画像読取精度を測りません。通常選択は、モデルの独立抽出と校正基準の実測結果に従います。権利レコードは検証用途に限る利用者の指示に基づき出典ページを示すもので、楽譜の再利用ライセンスを主張しません。

## 取得済みのCVDF train画像を使う

`prepare-local-train` は、未加工のCVDF画像 `<ImageID>.jpg` が並ぶディレクトリと
公式画像メタデータCSVを入力にするCPU専用コマンドです。画像の再取得は行いません。
従来の `prepare` は、引き続き固定validation画像の検証専用です。

```sh
uv run --locked pixelogue prepare-local-train \
  --metadata /work/datasets/openimages_v7/metadata/images.csv \
  --image-root /work/datasets/openimages_v7/images/cvdf/train \
  --destination /work/outputs/pixelogue/oi-train-input \
  --count 32 \
  --workers 4 \
  --seed 20260916 \
  --evaluation-images artifacts/prepared-open-images/images.jsonl
```

リポジトリのルートで実行します。`--evaluation-images` は省略可能で、複数指定できます。
手元の検証画像の取り込み済み台帳をすべて渡してください。既定の
`--validation-manifest validation/open_images_v7_manifest.jsonl` に記録された検証画像の
IDと元画像ハッシュは常に除外します。追加台帳は全件が検証用途であることを検査し、
正規化画素のハッシュも照合するため、別IDの完全一致コピーも除外できます。
**近似画像の判定は実施しません。** この制限は `manifest.json` に記録します。
本番の学習用出力前には、上記の固定SSCDモデルを用いて検証画像と合同でグループ化するか、
別途検証した除外リストで入力CSVを絞ってください。別々の `ingest` 実行間では
近似画像グループを自動的に照合できません。

CSVを2回逐次走査し、候補の保持メモリを `--count` に比例する範囲に抑えます。
対象となるローカル画像から `seed:ImageID` のSHA-256順に抽出するため、CSVの行順には
依存しません。2回目の走査で入力の変更と選択IDの重複を検査します。
候補に入り得る順位の画像だけ存在確認するため、欠損ファイルの件数は全画像の監査件数では
ありません。既存の出力ディレクトリへの上書きは拒否します。

対象は `Subset=train`、16桁の16進ID、明示されたCC BY 2.0/2.5/3.0/4.0のURL、
作者名とHTTPSの掲載ページ、既知の回転値が揃う画像です。それ以外の利用条件や
回転情報が欠けた画像は除外します。`90` と `90.0` の両表記に対応しています。
CVDF画像は画素を回転せずEXIFを除去しているため、正規化時にCSVの回転を一度適用します。
既に回転補正した画像には使用しないでください。利用条件台帳は帰属を保持した処理・学習・
QA再配布を許可し、画像自体の再配布は許可しない設定です。タイトルなどの元メタデータは
非公開記録に保存し、推論入力には渡しません。

出力は `sources.jsonl`、`rights.jsonl`、正規化した `images/`、推論用 `images.jsonl`、
`failures.jsonl`、`selected-ids.json`、`private-metadata.jsonl`、`manifest.json` です。
manifestには入力ハッシュ、seed、件数、split割り当て、除外検査の範囲を残します。
準備時刻を記録するため、再実行で同じになるのは選択IDや画像ハッシュであり、
利用条件台帳のバイト列全体ではありません。`--count` は画像検査前の候補数です。
画像検査で落ちた候補は理由を保存し、自動補充しません。生成前に採用件数を確認してください。
**正規化まで行うため、追加の `ingest` は不要です。**

サーバー確認後、`configs/standard.yaml` と、出力先を `synthesize --artifact-root`、
その中の `images.jsonl` を `--images` に指定して生成できます。
`data.target_dialogues` は処理する画像数の上限で、合格対話の保証件数ではありません。
不足する場合は対象IDが重複しない追加バッチと新しいrun IDを用意します。
現在のstandard設定の生成言語は英語です。pilotからの学習用出力は禁止したままです。
`export` は画像参照を書き出すだけで画像をコピーしないため、正規化画像と非公開の帰属記録も
bundleとともに保持してください。

`--workers 4` で最大4画像を並列処理します（既定値1、上限64）。並列化するのは
画像の読み込み・デコード・回転・色変換・PNG保存です。CSVの抽出と再検査、画像の
グループ化は直列のままです。出力順序・グループ・画像ハッシュは直列処理と一致します。
進捗は拒否画像を含む完了件数を数え、先行画像の処理中でも後続画像の完了を反映します。
並列数を増やすとメモリとストレージ帯域を多く使い、速度は画像と保存先に依存します。

準備・生成ともに標準エラー出力へtqdmのバーを表示します。準備は段階ごとの件数、
経過時間、処理速度、総数が分かる場合の推定残り時間を表示します。CSVの1回目の走査は
総数不明、2回目は1回目の行数を総数に使います。行数確認の追加走査はありません。
完了JSONは標準出力のままです。`> report.json` で結果、`2> progress.log` で進捗を
保存できます。`--quiet` でバーを非表示にできます。

sRGB変換後の正規化PNGにはEXIFやICCメタデータを引き継がず、生成されたプロファイルの
時刻情報を除去します。以前に準備した画像は符号化ハッシュが変わる場合があるため、
新しい準備ディレクトリとrun IDを使用してください。実画像31件のCPU処理を各1回測定した
結果は1 workerで6.65秒、4 workersで1.87秒で、manifestは一致しました。
この測定はCSV走査を含まず、反復測定による性能保証ではありません。
