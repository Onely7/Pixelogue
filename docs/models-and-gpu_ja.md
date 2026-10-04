# モデルとGPUの確認

Pixelogue は、指示選択と対話生成を別の役割に分けます。standard プロファイルでは、生成器と評価器に異なる2つのモデル系列を使います。一時的な1 GPU用の pilot では、パイプライン全体の動作を確認するため、小さい代替モデルを使います。

| 役割 | モデル | 既定のエンドポイント |
|---|---|---|
| 指示選択器 | `Qwen/Qwen3.5-2B` | `http://127.0.0.1:8000/v1` |
| 明示的に切り替える選択器 | `Qwen/Qwen3.6-35B-A3B` | `http://127.0.0.1:8001/v1` |
| 生成器・評価器A | `Qwen/Qwen3.8-27B-FP8` | `http://127.0.0.1:8002/v1` |
| 生成器・評価器B | `google/gemma-4-31B-it-qat-w4a16-ct` | `http://127.0.0.1:8003/v1` |

`configs/standard.yaml` では、Qwen3.8-27B-FP8 と Gemma 4 31B が対話を同数ずつ生成します。割り当てたモデルは、1回だけ許される修復を含め、対話が終わるまで変えません。さらに、すべての対話を両方のモデルが別々に評価します。互いの判定は入力へ含めません。

`configs/pilot.yaml` は、一時的な検証用の設定です。2つの論理的な役割を、port 8002で動く1つの `Qwen/Qwen3.5-9B` サーバーへ割り当てます。評価要求は別々に送りますが、同じ重みを使うため、確認できるのはパイプラインの接続です。異なるモデルによる評価の多様性は確認できません。

`configs/paired-one-gpu-pilot.yaml` は標準の Qwen3.8／Gemma 4 の組と Qwen3.5-2B 選択器を維持し、空いている 96 GiB 級 GPU 1 台へ3サーバーを配置する pilot 設定です。対応する `runtime/vllm/*-onegpu.yaml` はポート 18102・18103・18100、メモリ比率 0.44・0.41・0.10 を使います。同じ GPU を明示指定して順番に起動し、`doctor --config configs/paired-one-gpu-pilot.yaml --check-servers` で確認します。標準設定のモデル名や量子化は変更しません。RTX PRO 6000 Blackwell 1 台で3サーバーの起動、画像要求、合成、replayを確認しました。他の GPU では容量を再確認してください。
`configs/paired-one-gpu-diverse.yaml` は同じendpointとモデル固定値を使い、Commonsの60分類を検証専用で処理するため、対象数を60にします。学習exportには含めません。

## 1. GPUを使う直前に調べる

モデルを起動する直前に実行します。

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

`doctor` が空きとみなすのは、使用率が0%で、使用メモリが1 GiB未満のGPUです。`nvidia-smi` のプロセス一覧も確認し、他の処理が使っているGPUは選びません。

静的検査の `ready: true` は、固定したBF16モデルサーバーを現在の空きGPUへ割り当てられるという意味です。同じリポジトリ、revision、エンドポイントを共有する生成器の役割は、1つのサーバーとして数えます。この検査だけでは、実際の推論成功を確認できません。
固定したFP8とW4A16の標準チェックポイントについては、常駐重みの保守的な概算値とシャードごとに8 GiBのキャッシュ余裕を使います。チェックポイントのrevisionやハードウェアを変えた場合は再測定します。

起動後の `doctor --check-servers` は、設定した配信モデル名（served model name）を確認します。この時点では、選んだGPUが使用中に見えるのが正常です。既に正常に動いているサーバーへ、別のGPUを割り当て直す必要はありません。

## 2. 推論用のuv環境を分けて導入する

GPU用パッケージがCPU開発環境を暗黙に変えないよう、vLLMは別のlock fileで管理します。

```sh
uv sync --project runtime/vllm --locked
```

vLLMは0.29.0に固定しています。サーバー設定には、モデルのrevision、dtype(BF16)、context長、tensor parallel数、GPUメモリ使用率、生成時の既定値を記録しています。標準生成モデルは同時系列数を64に制限します。1GPU起動試験ではvLLMのより大きい既定値がQwenのMambaキャッシュ容量を超えました。量子化は生成器ごとに固定されており、`generator-a.yaml`はQwen3.8-27B-FP8向けに`quantization: fp8`を、`generator-b.yaml`はW4A16版Gemma向けに`quantization: compressed-tensors`を指定します。`configs/standard.yaml`も同じ値を反映しており、`ModelConfig.validate_roles`はこれら以外の量子化指定を拒否します。

## 3. 長時間の GPU 処理を tmux 内で動かす

standard 構成では、次のサーバー設定を使います。

```text
runtime/vllm/generator-a.yaml        -> Qwen3.8-27B-FP8、port 8002、tensor parallel size 2
runtime/vllm/generator-b.yaml        -> Gemma 4 31B、port 8003、tensor parallel size 2
runtime/vllm/selector-default.yaml   -> Qwen3.5-2B、port 8000、tensor parallel size 1
```

起動直前に `doctor --config configs/standard.yaml` を実行し、空いていると判定された GPU だけを割り当てます。各サーバーは、別々の `tmux` ウィンドウで起動します。次の例では、GPU番号を実際に空いていた番号へ置き換えてください。

```sh
CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a.yaml

CUDA_VISIBLE_DEVICES=2,3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-b.yaml

CUDA_VISIBLE_DEVICES=4 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/selector-default.yaml
```

ここで示したGPU番号は例です。必要なメモリは、ハードウェアと推論環境によって変わります。`doctor` による静的検査と、実際の起動確認の両方を行ってください。

### 一時的な1 GPU用 pilot

生成器、選択器、pilot、監視用にウィンドウを分けます。

```sh
tmux new-session -d -s pixelogue-qwen35-pilot -n generator
tmux new-window -t pixelogue-qwen35-pilot -n control
tmux new-window -t pixelogue-qwen35-pilot -n selector
tmux new-window -t pixelogue-qwen35-pilot -n pilot
```

直前に確認した空き GPU を明示します。次のGPU番号は例です。このコマンドで起動するQwen3.5-9Bは一時的な代替モデルであり、standard構成の生成器ではありません。

先に大きいモデルを起動します。

```sh
CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-qwen35-9b.yaml
```

生成モデルの一覧を取得できるまで待ち、同じGPUで選択器を起動します。

```sh
curl -fsS http://127.0.0.1:8002/v1/models

CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/selector-default.yaml
```

Gitに含まれるGPUメモリ使用率の上限は、9Bサーバーが0.68、2Bの選択器が0.20です。48 GiB級のメモリを持つRTX 6000 Adaで確認した値なので、別のGPUへそのまま適用せず、容量を再確認してください。

各ウィンドウの出力は、一意な名前のローカルログへ保存します。`pane_dead=0` だけでは正常と判断できません。モデル一覧、ログ、GPU使用状況、出力行数、model callの状態も確認します。

### 上限付きの空きGPU監視

`gpu_watch.py` は `nvidia-smi` でローカルの各GPUを2回調べ、計算プロセスも確認します。1 GPUのpilot設定について `doctor` が実行可能と判定した場合だけ予約します。メモリ予約には添付されたCUDA割り当てスクリプトの方式を使い、連続した行列積の負荷は省きます。モデル起動中は予約用プロセスがGPUメモリの25%を保持したまま生成モデルを起動します。生成モデルの `/v1/models` 応答後に予約を解放し、引き継ぎを確認してから選択器を起動します。予約のない隙間を作りません。pilot終了後は90%を確保するプロセスで空きGPUを再予約し、停止指示または累積予算の上限まで保持します。Slurmのノード表示が空きでも、他の計算プロセスが載るGPUは使いません。

通常の選択器portは別サービスが使用しているため、この監視経路では18002と18000を使います。固定した非量子化Qwen3.5-9BとQwen3.5-2Bを使います。1回のjobは画像1枚の確認、ユーザーが異なる視覚質問に答えられると確認した2枚での意図的な中断・再開、Open Imagesの20画像、Webから用意した60大分類の評価画像を順に処理します。完了runの再利用・replay、モデル名、`doctor --check-servers`、出力行数、実行エラーも確認します。確認済み2枚には独立した回答ラベルはありません。サーバーとrunのログを分けて保存し、標準モデル組や専門タスクの校正を合格扱いにはしません。

```sh
mkdir -p artifacts/gpu-watch
tmux new-session -d -s pixelogue-gpu-watch -c "$PWD" -n monitor \
  'uv run --locked python src/pixelogue/gpu_watch.py watch >> artifacts/gpu-watch/monitor.log 2>&1'
tail -f artifacts/gpu-watch/monitor.log
```

`artifacts/gpu-watch/state.json` は最後の確認時刻、使用中のPID、GPU、段階、累積GPU秒数、確定した各割当区間を記録します。各runの `phase-events.jsonl` にはモデル読み込み、各推論バッチ、停止処理の境界時刻を残します。予約プロセスとモデルが重なる時間も、同じGPUの割当として1回だけ計上します。ログにも定期的に生存状況を出します。停止して予約を解放するにはtmuxのpaneへ `Ctrl-C` を送ります。累積上限に達する前に自動停止し、再起動しても記録済み時間を引き継ぎます。実行結果は `artifacts/gpu-watch/<run-id>/` に保存します。このホストのGPU 0〜5は見えているSlurmのGPU区画外なので、監視は実際のローカルGPUを直接確認します。

ホームディレクトリを共有する複数ホストでは、`--state-file PATH` でホストと起動ごとに異なる親ディレクトリを指定します。台帳ロックと `doctor.json` もそのディレクトリを使い、`wait-release` にも同じ指定を渡します。他ホストの予約や以前の起動のPIDを再利用せず、古い起動の記録は費用集計のため保存します。数値の測定結果が得られないGPUは予約対象から外し、正常なGPUの検査を続けます。`doctor` は不明な使用率を `null` として報告し、そのGPUを空きと判定しません。

監視は選択した物理GPUの `nvidia-smi` UUIDを取得し、予約とpilotの
`CUDA_VISIBLE_DEVICES` に指定します。不調なGPUがある場合、CUDAの番号と物理GPUの
番号が異なることがあります。選択したUUID上の予約PIDとメモリを確認してください。
プロセスの生存だけでは予約を確認できません。このようなホストで手動起動する場合も、
検査済みのUUIDを指定します。

同じホストで2台を予約する場合は、台帳のディレクトリを分けた予約専用の監視を2つ起動し、共通のローカル `--admission-lock PATH` を指定します。ロック内で空き状態を再確認し、予約プロセスのメモリ確保が確認できるまで保持します。各監視は異なるGPUを1台確保して割当時間を記録するため、2つの台帳のGPU時間を合計します。そのホストの全監視へ同じ確保用ロックを渡してください。

`tmux` はSSH切断後も続きますが、ホストの再起動後には自動復旧しません。無人運用ではホスト側に起動時の開始処理と定期的な復旧確認を設定し、ログアウト後もユーザーサービスが維持されることを確認します。予約状態はPIDに加えてGPU上のプロセスとメモリでも確認します。意図したGPU引き継ぎの前に復旧確認を無効にし、ジョブ終了後は成否にかかわらず再び有効にします。

pilotが成功・失敗のどちらで終了しても、直後に次のGPU予約要求を `state.json` に記録し、最初の空きGPUスキャンから確保を試みます。直前に使ったGPUを優先し、空きがなければ5秒間隔で再確認します。予約状態とpilotの終了コードは `state.json` に保存します。累積GPU時間の上限は守り、残り予算内に予約を開始できない場合は `budget_exhausted` と記録します。

別のtmux jobへ確保済みGPUを引き継ぐ前に、予約を停止して `uv run --locked python src/pixelogue/gpu_watch.py wait-release --gpu-index 4 --timeout-seconds 180` を実行します。GPUが2回続けて空きと判定され、台帳の使用中予約が消え、台帳ロックが解放されるまで待ちます。GPUメモリだけが先に解放されることがあるため、このコマンドの成功前に次のモデルjobを起動しません。

現在のpilotは、固定Open Images比較の後に[60分類の検証用画像](data_ja.md#多様なweb画像による検証用サンプル)も処理します。結果、診断、replay、分類別のJSON/CSV/Markdown集計を、run内の `diverse/` と `diverse-*` に分けて保存します。このpilotは一時的な1GPUモデル設定を使い、独立した正解ラベルを持ちません。受理件数だけでは検証器の精度は確認できません。

固定したABBA比較が完了したら、`python3 validation/compare_paired_runs.py artifacts/plan-followup/EXPERIMENT_DIR` で集計します。計画した画像順、生成モデルと対象言語の割当が全実行で同一であること、各段階の完了を確認し、試行・確定ターン数、完成会話数、反復時の重なり、モデル呼び出し、再試行、画像ごとの所要時間の中央値・最大値、段階別GPU時間をJSON・CSV・Markdownに保存します。確定ターンのタスク別件数は保存済み公開ターンと照合します。2ターン以上確定した画像数と、確定ターンがない画像の停止段階も記録します。独立した人手評価票が確定するまでは、人手確認済み会話数/GPU時間と人手基準の誤り率をnullにします。自動判定の品質候補を正解ラベルとして扱いません。

中断した比較を再開する場合、GPUの割当区間を個別に記録し、中断した段階の試行も残します。実験GPU時間にはモデル読み込み、中断前の処理、停止処理を含め、モデル起動間の空白時間を除きます。予約時間は累積台帳に残します。経過時間とモデル起動回数は別々に報告し、複数の起動をまたいだ比較を同じ起動内での速度比較として扱いません。

後から明示的に許可された検証では、`watch` コマンドへ `--campaign-id recheck-YYYYMMDD --additional-gpu-hours 2` を加えます。既存の累積台帳にこの時間をキャンペーンIDごとに1回だけ加算します。同じIDで再起動しても予算を再加算せず、完了済みのpilotも再実行しません。新しいキャンペーンはpilotを1回実行し、job後に空きGPUを再予約して延長予算の上限まで保持します。
継続的な検証を利用者が許可した場合、名前付きの延長1件につき最大24 GPU時間を指定できます。モデル読み込み、推論、予約の時間を同じ台帳へ算入します。延長してもモデル名、量子化、空きGPUの判定条件は変わりません。

失敗したpilotを調査する間は `--reserve-only` を加えます。空きGPUを保持してpilotを起動しません。修正後にwatcherを停止し、同じキャンペーンIDでこの指定を外して再起動します。両方の段階を同じ累積GPU台帳に算入します。

pilotが失敗し、予算が残っている場合は `--retry-failed-pilot --campaign-id ID` で、そのキャンペーンの試行済みフラグだけを戻せます。GPU時間は追加しません。元の失敗ログを保存し、修正コードには新しいrun IDを使います。

診断pilotが正常終了した後も、同じキャンペーンに未使用のGPU時間が10分以上あれば `--rerun-completed-pilot --campaign-id ID` で現行コードを再検証できます。予算は増やしません。実行前にジョブ後の予約プロセスを正常停止してください。再実行には新しいrun IDを使い、ジョブ終了後は再度予約します。

## 4. 実際に応答できるまで待つ

vLLMは重みを読み込んだ後も、compile、CUDA graphの準備、画像入力のウォームアップを行います。PIDが存在し、GPUメモリを確保していても、起動完了とは限りません。

```sh
curl -fsS http://127.0.0.1:8002/v1/models
curl -fsS http://127.0.0.1:8003/v1/models
curl -fsS http://127.0.0.1:8000/v1/models
uv run --locked pixelogue doctor \
  --config configs/standard.yaml \
  --check-servers
```

一時的な1 GPU用プロファイルを検証するときは `configs/pilot.yaml` を使い、port 8003の確認を省きます。

50画像の処理を始める前に、本番と同じ構造化出力を使って1画像だけ試します。モデル一覧を返せても、guided decodingが特定のJSON Schemaを受理できない場合があるためです。

トークン上限で終了した応答に必須項目が欠けている場合は、単なるSchema不一致ではなく未完了として記録します。証拠抽出は出力上限を最大8192トークンまで増やして再試行します。再試行でも全Schemaと領域制約を満たす必要があり、欠けた観測をMETとして補いません。

トークン上限で終わったJSONの末尾に512文字以上の空白が続き、内容が無効な場合は `MODEL_WHITESPACE_RUNAWAY` と記録します。再試行には終了方法を具体的に伝え、出力上限は元の値に据え置きます。末尾に空白があってもSchemaを満たす完全なJSONなら受理します。不正な応答は非公開artifactに保持し、欠けた項目や視覚的事実を推測で補いません。

Qwenには、サーバーの既定値と各要求の両方で `enable_thinking=false` を指定します。thinkingの文章や内部の制御情報を公開対話へ含めてはいけません。

## 5. 代替選択器は別に確認する

`Qwen/Qwen3.6-35B-A3B` は、設定のコピーで次の値を明示した場合だけ使います。

```yaml
models:
  active_selector: alternative
```

メモリ計画では、総パラメータ数35Bのモデルとして扱います。十分な空き容量があるときに、port 8001で別runとして確認します。既定の選択器と同時には動かさず、失敗時の自動切り替え先にも使いません。

## 6. GPUの全割当時間を計上する

GPU時間は「経過時間 × 使用したGPU枚数」です。1枚を45分使うと0.75 GPU時間、2枚なら1.5 GPU時間です。モデルの読み込み、動作確認、予約プロセスを含めます。初期pilotは4 GPU時間でしたが、その後に許可された検証は名前付きの延長として同じ台帳へ記録します。開始・終了時刻をGit対象外の `_docs/` に記録します。

Pixelogueは要求とトークン使用量を記録しますが、外部vLLMプロセスの起動時間は観測できません。その分は運用者が加算します。サーバーを用意できなかった検査は、合格ではなく未実行です。

## 7. 正解付きの能力検査を行う

必要なサーバーがすべて正常になってから実行します。

```sh
uv run --locked pixelogue evaluate-capabilities \
  --config configs/pilot.yaml \
  --fixtures data/fixtures/fixtures.jsonl \
  --fixture-root data/fixtures \
  --output artifacts/capability-report.json
```

期待する採否はコントローラー内だけで使い、評価呼び出しへ渡しません。developmentとconfirmationの件数も分けます。少数fixtureの結果で確認できるのは接続であり、自然画像に対する正答率ではありません。

各サーバーが生成と評価のどこで使われるかは、[詳しいパイプラインガイド](pipeline/README_ja.md)で説明しています。

表・チャート・図の抽出と回答解析では、デコーダーの制約と同じ出力Schemaをモデルが読む本文にも渡します。回答解析はnullを許す結果欄も明示し、`MET` は文面を解析できたことを表します。正誤は別の画像抽出結果との計算で判定します。画像抽出には回答を、回答解析には画像を渡しません。不正出力は上限付き再試行後も拒否します。

## 評価回帰テスト

GPUと `doctor --check-servers` を確認し、標準モデルのサーバーが起動している間だけ、次の外部モデルテストを実行します。通常のCPUテストではスキップされます。

```sh
PIXELOGUE_LIVE_RUBRIC=1 PIXELOGUE_LIVE_CONFIG=configs/paired-one-gpu-pilot.yaml \
  uv run --locked pytest -q -s tests/test_rubric_semantics.py
```

1 GPU用設定ではport 18102と18103へ接続します。モデルの識別と量子化は標準構成のままです。


## サーバー条件を実行識別に固定する

実験用endpointの `serving_runtime` には `vllm_version`、`structured_output_backend`、`disable_any_whitespace`、`server_manifest_sha256` を指定できます。manifestには実際の起動設定・固定したruntime lock・起動コマンド・版確認を保存します。これらを変更するとrunと応答キャッシュの識別が変わります。宣言だけでは実サーバーの設定を確認したことにならないため、制御された起動とログで照合します。固定vLLMで空白抑制を使うには `xgrammar` または `guidance` を明示し、backendも変えた条件は設定一式の比較として扱います。

## 根拠の出力形式を比較する

比較用設定で `tasks.evidence_format: array` を指定すると、能力名を列挙型で制限した観測配列を使います。比較結果を確認するまで既定は `keyed` です。両形式は説明文・領域・能力上限・内部検証を共通にし、欠測はUNKNOWNのまま保持します。能力や根拠IDの重複、異なるview、親領域外の観測は拒否します。

## JSONの空白暴走を抑える

実験用の `runtime/vllm/*-whitespace.yaml` 生成器設定は `xgrammar` と
`disable_any_whitespace: true` を指定します。JSONトークン間の任意の空白を
制限し、文字列内の通常の空白は維持します。固定したvLLM 0.29.0では、
サーバー起動時に設定する必要があります。リクエストに同名のフィールドを
送っただけでは有効化を確認できません。既存サーバーを再起動し、起動manifestを
記録したうえで比較してください。
標準サーバー設定は従来のデコード設定を維持します。高速化した実験条件は、
根拠が支持する候補数とタスク範囲も維持できた場合に採用します。
独立した反復停止は `configs/repetition-detection-pilot.yaml` で
固定した標準モデルのペアを使って評価できます。

評価用に、エンジンの反復停止を別途指定できます。

```yaml
runtime:
  repetition_detection:
    min_pattern_size: 1
    max_pattern_size: 4
    min_count: 64
    recovery: retry
    stages: [evidence_extraction, candidate_binding]
```

既定は無効です。出力の確率を変えず、連続するトークン列の反復を検知して停止します。
この閾値は実験用で、正しい長い反復も停止する可能性があります。
適用可能なのは根拠抽出と候補具体化のみです。
質問・回答生成には適用しません。

`finish_reason: repetition` は `MODEL_OUTPUT_REPETITION` として記録し、
生の応答と使用トークンを保存し、構文が正しいJSONでも途中停止した応答は拒否します。
指定した抽出段階では `recovery: retry` により、`structured_output_max_attempts`
の範囲で修正指示を付けて再試行します。元の出力上限を維持し、反復を理由に
上限を増やしたり途中出力を採用したりしません。比較用の `recovery: abstain`
は即時棄権を維持します。
設定変更後は異なるrun IDを使います。この機能は固定したvLLMの反復検知APIに
対応するendpointが必要です。

[vLLM 0.29.0の起動設定](https://docs.vllm.ai/en/v0.29.0/cli/serve/)、
[SamplingParams](https://docs.vllm.ai/en/v0.29.0/api/vllm/sampling_params/)、
[構造化出力](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/)で
利用可能な設定を確認できます。導入済み版の全起動設定は
`uv run --project runtime/vllm --locked vllm serve --help=all` で確認してください。
設定項目の存在だけで、全モデル・バックエンドでの動作を保証するものではありません。

### 整形を保ったままJSONの連続空白を制限する

任意の `runtime.json_whitespace_max_chars: 32` は根拠抽出・候補具体化の
Schemaだけに非公開の拡張を付けます。JSON要素間の空白数を制限し、通常の改行と
字下げは許します。文字列の値に含まれる空白は内容として維持します。
停止した応答の観測・適格性判定・公開パラメータをcontrollerが補完することはありません。

アプリとは別のvLLM環境へ、次の版固定パッチを適用します。

```bash
runtime/vllm/.venv/bin/python runtime/vllm/whitespace_patch.py
runtime/vllm/.venv/bin/python runtime/vllm/whitespace_patch.py --check
```

適用・復元の前にモデルサーバーを停止し、処理後に再起動します。環境の
`uv sync` 後も検査・再適用してください。vLLM 0.29.0、XGrammar 0.2.6と
元ファイル全体のSHAを検査し、検証済みbackupを保存します。別の変更があれば
上書きを拒否します。`--restore` は元のバイト列を復元します。
返された `patched_sha256` を生成endpointの
`serving_runtime.xgrammar_whitespace_patch_sha256` に設定し、実際のサーバー
manifestを記録します。`structured_output_backend: xgrammar` と
`disable_any_whitespace: false` が必要です。未記録・非対応の実行識別は拒否します。
上限値はSchemaに含まれ、呼出・文法キャッシュの識別へ反映されます。変更後は
新しいrun IDを使います。拡張のない基準呼出は従来の文法生成を保ちます。
質問・回答と専門sourceのSchemaには、この拡張を渡しません。

パッチは既存の `max_whitespace_cnt` を渡すだけで、重み・量子化を変更しません。
[版固定XGrammarの実装](https://github.com/mlc-ai/xgrammar/blob/v0.2.6/python/xgrammar/compiler.py)と
[版固定vLLMのbackend](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/structured_output/backend_xgrammar.py)を参照できます。
生成可能な経路は変わるため、新しいモデル・画像分野に採用する前に支持数と
会話全体の品質を確認します。
