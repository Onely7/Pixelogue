# モデルとGPUの確認

Pixelogue は、画像の割り当てと、対話の生成・評価を別の役割に分けます。standard プロファイルでは、生成器と評価器に異なる2つのモデル系列を使い、どちらも量子化しません。一時的な1 GPU用の pilot では、パイプライン全体の動作を確認するため、小さい代替モデルを使います。

| 役割 | モデル | 既定のエンドポイント |
|---|---|---|
| 画像のルーター | `Qwen/Qwen3.8-27B`（生成器Aのサーバー） | `http://127.0.0.1:8002/v1` |
| 生成器・評価器A | `Qwen/Qwen3.8-27B` | `http://127.0.0.1:8002/v1` |
| 生成器・評価器B | `google/gemma-4-31B-it` | `http://127.0.0.1:8003/v1` |

`configs/standard.yaml` では、Qwen3.8-27B と Gemma 4 31B が対話を同数ずつ生成します。割り当てたモデルが、その対話の質問案、回答、1回だけ許される回答修復を最後まで担当します。さらに、すべての往復を両方のモデルが別々の盲検呼び出しで評価します。回答前の質問ゲート、往復全体の総合評価、各操作検証器の2回の読み取りを、それぞれのモデルが1回ずつ担当します。ゲートと総合評価は、2つの `MET` が揃った場合だけ合格です。互いの判定は入力へ含めません。ルーター（`models.router`）は、タスク系統の割り当てのために各画像を分析するだけです。1回の要求で受け取る画像は1枚で、対話は書きません。ルーターは生成器Aのサーバーそのものです。`models.router` には `models.generator_a` と完全に同じ内容を書く必要があり（チェックイン済みの設定は YAML のアンカーを使います）、割り当て用のサーバーやメモリを追加しません。

量子化したモデルは使わず、dtype は BF16 のままです。設定で受け付けるのは `Qwen/Qwen3.8-27B` と `google/gemma-4-31B-it` の組か、一時的な Qwen3.5-9B パイロットの組だけで、`quantization` の設定は未知の項目として拒否します。以前の FP8・W4A16 のチェックポイントと、別の Qwen3.5-2B ルーターは、標準の組では使えなくなりました。

`configs/pilot.yaml` は、一時的な検証用の設定です。2つの論理的な役割を、port 8002で動く1つの `Qwen/Qwen3.5-9B` サーバーへ割り当て、画像の割り当ては port 8000 の別の `Qwen/Qwen3.5-2B` サーバーが行います。評価要求は別々に送りますが、同じ重みを使うため、確認できるのはパイプラインの接続です。異なるモデルによる評価の多様性は確認できません。

`configs/split-pilot.yaml` は標準のモデルを維持したまま、同じホストにあるとは限らない3台の GPU へ配置します。Gemma 4 31B は 48 GiB の GPU 2台（`runtime/vllm/generator-b-split.yaml`、port 18703）、画像の割り当ても行う Qwen3.8-27B は 96 GiB の GPU 1台（`runtime/vllm/generator-a-split.yaml`、port 18702、メモリ比率 0.88）です。Gemma を RTX 6000 Ada 2台、Qwen を別ホストの RTX PRO 6000 Blackwell 1台に置き、SSH のポート転送でつないで動作を確認しました（[3節](#ホストをまたぐ分割配置)）。他の GPU では容量を再確認してください。これは pilot 用の配置で、standard プロファイルは変更しません。
`configs/split-diverse.yaml` は同じendpointとモデル固定値を使い、Commonsの60分類を検証専用で処理するため、対象数を60にします。学習exportには含めません。
`configs/split-diverse-269.yaml` は同じendpointでstandardプロファイルを使い、[269枚の多様な検証用セット](data_ja.md#269枚の多様な検証用セット)をseed 20261004で処理します。固定したmanifestに、各画像の生成器と予定ターン数を記録しています。

## 1. GPUを使う直前に調べる

モデルを起動する直前に実行します。

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

`doctor` が空きとみなすのは、使用率が0%で、使用メモリが1 GiB未満のGPUです。`nvidia-smi` のプロセス一覧も確認し、他の処理が使っているGPUは選びません。

静的検査の `ready: true` は、固定したBF16モデルサーバーを現在の空きGPUへ割り当てられるという意味です。ルーターと生成器Aのように、同じリポジトリ、revision、エンドポイントを共有する役割は、1つのサーバーとして数えます。この検査だけでは、実際の推論成功を確認できません。
`doctor` は、公開されているチェックポイントの大きさに10%を加えて、BF16 の常駐重みを見積もります。Qwen3.8-27B は約57 GiB、Gemma 4 31B は約64 GiB で、tensor parallel のシャードに均等に分けます。そのため1モデルあたり、48 GiB の GPU 2台か、96 GiB の GPU 1台が必要です。チェックポイントのrevisionやハードウェアを変えた場合は再測定します。

起動後の `doctor --check-servers` は、設定した配信モデル名（served model name）を確認します。この時点では、選んだGPUが使用中に見えるのが正常です。既に正常に動いているサーバーへ、別のGPUを割り当て直す必要はありません。

## 2. 推論用のuv環境を分けて導入する

GPU用パッケージがCPU開発環境を暗黙に変えないよう、vLLMは別のlock fileで管理します。

```sh
uv sync --project runtime/vllm --locked
```

vLLMは0.29.0に固定しています。サーバー設定には、dtype(BF16)、context長、tensor parallel数、GPUメモリ使用率、構造化出力のbackend、生成時の既定値を記録しています。どちらの生成モデルも同時系列数を、計測時と同じ32に制限します。以前の1GPU起動試験では、vLLMのより大きい既定値がQwenのMambaキャッシュ容量を超えました。構造化出力は、どちらも `xgrammar` と `disable_any_whitespace: false` です。量子化の方式を指定するサーバー設定はありません。

評価器には画像全体と、結び付けた領域の切り出し画像を1回の要求で渡すことがあるため、生成器用のサーバー設定はすべて `limit-mm-per-prompt` を画像2枚にしています。パイロットのルーター用の設定は1枚のままです。

### 固定した重みを取得する

サーバー設定は `models/` から重みを読み込み、リポジトリIDの名前で配信します。`models/` はローカル専用で、コミットしません。サーバーは必ずリポジトリ直下から起動してください。固定した revision を、共有ストレージへ1回だけ取得します。

```sh
uv run --project runtime/vllm --locked hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir models/Qwen3.8-27B
uv run --project runtime/vllm --locked hf download google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --local-dir models/gemma-4-31B-it
```

Gemma は Hugging Face でライセンスへの同意と、環境変数のトークンが必要です。配信する前に、固定した revision について Hub が公開している SHA-256 の一覧と全ファイルを照合し、その記録を重みと一緒に残してください。設定ファイルも、モデル lock 用に同じ revision を固定しています。vLLM はネットワークファイルシステムを検出すると、チェックポイントをページキャッシュへ先読みします。計測した NFS では、各モデルの重みの読み込みは30秒未満で、compile を含めてサーバーが応答できるまで4〜5分でした。

## 3. 長時間の GPU 処理を tmux 内で動かす

standard 構成では、次のサーバー設定を使います。

```text
runtime/vllm/generator-a.yaml   -> Qwen3.8-27B、port 8002、tensor parallel size 2（ルーターを兼ねる）
runtime/vllm/generator-b.yaml   -> Gemma 4 31B、port 8003、tensor parallel size 2
```

起動直前に `doctor --config configs/standard.yaml` を実行し、空いていると判定された GPU だけを割り当てます。各サーバーは、リポジトリ直下から別々の `tmux` ウィンドウで起動します。次の例では、GPU番号を実際に空いていた番号へ置き換えてください。

```sh
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a.yaml

CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-b.yaml
```

ここで示したGPU番号は例です。必要なメモリは、ハードウェアと推論環境によって変わります。`doctor` による静的検査と、実際の起動確認の両方を行ってください。96 GiB の GPU なら、分割配置の設定のように `tensor-parallel-size` を1にして、1モデルあたり1台で足ります。

### ホストをまたぐ分割配置

分割配置の設定は、GPU の種類が違うホストを組み合わせるためのものです。Pixelogue は 48 GiB の GPU 2台があるホストで動かし、そこで Gemma を起動します。Qwen はもう一方のホストの 96 GiB の GPU で起動し、そのポートを手元の同じポートへ転送します。どちらのサーバーも 127.0.0.1 だけで待ち受けます。

```sh
# 96 GiB の GPU があるホスト（UUID は確認済みの空き GPU に置き換える）
CUDA_VISIBLE_DEVICES=GPU-QWEN HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a-split.yaml

# 48 GiB の GPU 2台があり、Pixelogue も動かすホスト
CUDA_VISIBLE_DEVICES=GPU-GEMMA-0,GPU-GEMMA-1 HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-b-split.yaml
ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -L 127.0.0.1:18702:127.0.0.1:18702 QWEN-HOST
uv run --locked pixelogue doctor --config configs/split-pilot.yaml --check-servers
```

転送は専用の `tmux` ウィンドウで動かし、終了したら再起動します。転送では、例のように SSH の接続共有を無効にしてください。共有したマスター接続を使うと、転送したポートはマスター側が保持し、転送を頼んだ ssh が終了しても残ります。残った場合は `ssh -O cancel -L 127.0.0.1:18702:127.0.0.1:18702 QWEN-HOST` で取り消します。

### 一時的な1 GPU用 pilot

生成器、ルーター、pilot、監視用にウィンドウを分けます。

```sh
tmux new-session -d -s pixelogue-qwen35-pilot -n generator
tmux new-window -t pixelogue-qwen35-pilot -n control
tmux new-window -t pixelogue-qwen35-pilot -n router
tmux new-window -t pixelogue-qwen35-pilot -n pilot
```

直前に確認した空き GPU を明示します。次のGPU番号は例です。このコマンドで起動するQwen3.5-9Bは一時的な代替モデルであり、standard構成の生成器ではありません。

先に大きいモデルを起動します。

```sh
CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-qwen35-9b.yaml
```

生成モデルの一覧を取得できるまで待ち、同じGPUでルーターを起動します。

```sh
curl -fsS http://127.0.0.1:8002/v1/models

CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/router-default.yaml
```

Gitに含まれるGPUメモリ使用率の上限は、9Bサーバーが0.68、2Bのルーターが0.20です。48 GiB級のメモリを持つRTX 6000 Adaで確認した値なので、別のGPUへそのまま適用せず、容量を再確認してください。

各ウィンドウの出力は、一意な名前のローカルログへ保存します。`pane_dead=0` だけでは正常と判断できません。モデル一覧、ログ、GPU使用状況、出力行数、model callの状態も確認します。

### 上限付きの空きGPU監視

`gpu_watch.py` は `nvidia-smi` でローカルの各GPUを2回調べ、計算プロセスも確認します。1 GPUのpilot設定について `doctor` が実行可能と判定した場合だけ予約します。メモリ予約には添付されたCUDA割り当てスクリプトの方式を使い、連続した行列積の負荷は省きます。モデル起動中は予約用プロセスがGPUメモリの25%を保持したまま生成モデルを起動します。生成モデルの `/v1/models` 応答後に予約を解放し、引き継ぎを確認してからルーターを起動します。予約のない隙間を作りません。pilot終了後は90%を確保するプロセスで空きGPUを再予約し、停止指示または累積予算の上限まで保持します。Slurmのノード表示が空きでも、他の計算プロセスが載るGPUは使いません。

通常のルーターportは別サービスが使用している場合があるため、この監視経路では18002と18000を使います。固定した非量子化Qwen3.5-9BとQwen3.5-2Bを使います。1回のjobは画像1枚の確認、ユーザーが異なる視覚質問に答えられると確認した2枚での意図的な中断・再開、Open Imagesの20画像、Webから用意した60大分類の評価画像を順に処理します。完了runの再利用・replay、モデル名、`doctor --check-servers`、出力行数、実行エラーも確認します。確認済み2枚には独立した回答ラベルはありません。サーバーとrunのログを分けて保存し、標準モデル組や専門タスクの校正を合格扱いにはしません。

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
継続的な検証を利用者が許可した場合、名前付きの延長1件につき最大24 GPU時間を指定できます。モデル読み込み、推論、予約の時間を同じ台帳へ算入します。延長してもモデル名と空きGPUの判定条件は変わりません。

失敗したpilotを調査する間は `--reserve-only` を加えます。空きGPUを保持してpilotを起動しません。修正後にwatcherを停止し、同じキャンペーンIDでこの指定を外して再起動します。両方の段階を同じ累積GPU台帳に算入します。

pilotが失敗し、予算が残っている場合は `--retry-failed-pilot --campaign-id ID` で、そのキャンペーンの試行済みフラグだけを戻せます。GPU時間は追加しません。元の失敗ログを保存し、修正コードには新しいrun IDを使います。

診断pilotが正常終了した後も、同じキャンペーンに未使用のGPU時間が10分以上あれば `--rerun-completed-pilot --campaign-id ID` で現行コードを再検証できます。予算は増やしません。実行前にジョブ後の予約プロセスを正常停止してください。再実行には新しいrun IDを使い、ジョブ終了後は再度予約します。

## 4. 実際に応答できるまで待つ

vLLMは重みを読み込んだ後も、compile、CUDA graphの準備、画像入力のウォームアップを行います。PIDが存在し、GPUメモリを確保していても、起動完了とは限りません。

```sh
curl -fsS http://127.0.0.1:8002/v1/models
curl -fsS http://127.0.0.1:8003/v1/models
uv run --locked pixelogue doctor \
  --config configs/standard.yaml \
  --check-servers
```

一時的な1 GPU用プロファイルを検証するときは `configs/pilot.yaml` を使い、port 8002と8000を確認します。

50画像の処理を始める前に、本番と同じ構造化出力を使って1画像だけ試します。モデル一覧を返せても、guided decodingが特定のJSON Schemaを受理できない場合があるためです。

トークン上限で終了した応答に必須項目が欠けている場合は、単なるSchema不一致ではなく未完了として記録します。チャートとグラフの原文読み取りは、出力上限を最大2倍にして再試行します。上限は8,192トークンで、設定値がすでにそれより大きい場合は増やしません。表の読み取りを含むその他の段階は上限を変えません。再試行でも全Schemaと領域制約を満たす必要があり、欠けた内容を補いません。

他の項目が揃っていても、欠けたJSONの閉じ括弧を付け足しません。元の応答自体が完全なJSONオブジェクトであることを要求します。

トークン上限で終わったJSONの末尾に512文字以上の空白が続き、内容が無効な場合は `MODEL_WHITESPACE_RUNAWAY` と記録します。再試行には終了方法を具体的に伝え、出力上限は元の値に据え置きます。末尾に空白があってもSchemaを満たす完全なJSONなら受理します。不正な応答は非公開artifactに保持し、欠けた項目や視覚的事実を推測で補いません。

Qwenには、サーバーの既定値と各要求の両方で `enable_thinking=false` を指定します。Gemmaには各要求で `reasoning_effort: none` を指定します。thinkingの文章や内部の制御情報を公開対話へ含めてはいけません。

### 要求のタイムアウト

各要求の読み取りタイムアウトは出力上限に応じて伸び、`min(900, max(request_timeout_seconds, 60 + max_tokens × timeout_seconds_per_1k_output_tokens / 1000))` 秒になります。接続タイムアウトは最大30秒です。既定値の180と45では、1,024トークンの回答は180秒のまま、4,096トークンの原文読み取りは約244秒になります。`runtime.timeout_seconds_per_1k_output_tokens: null` を指定すると、固定の `request_timeout_seconds`（最大600）に戻ります。タイムアウトは通信の失敗として `runtime.transport_max_attempts` の範囲で再試行し、判定として扱うことはありません。

## 5. GPUの全割当時間を計上する

GPU時間は「経過時間 × 使用したGPU枚数」です。1枚を45分使うと0.75 GPU時間、2枚なら1.5 GPU時間です。モデルの読み込み、動作確認、予約プロセスを含めます。初期pilotは4 GPU時間でしたが、その後に許可された検証は名前付きの延長として同じ台帳へ記録します。開始・終了時刻をGit対象外の `_docs/` に記録します。

Pixelogueは要求とトークン使用量を記録しますが、外部vLLMプロセスの起動時間は観測できません。その分は運用者が加算します。サーバーを用意できなかった検査は、合格ではなく未実行です。

## 6. 正解付きの能力検査を行う

必要なサーバーがすべて正常になってから実行します。

```sh
uv run --locked pixelogue evaluate-capabilities \
  --config configs/pilot.yaml \
  --fixtures data/fixtures/fixtures.jsonl \
  --fixture-root data/fixtures \
  --output artifacts/capability-report.json
```

期待する採否はコントローラー内だけで使い、評価呼び出しへ渡しません。developmentとconfirmationの件数も分けます。少数fixtureの結果で確認できるのは接続であり、自然画像に対する正答率ではありません。

各サーバーが生成と評価のどこで使われるかは、[パイプラインガイド](pipeline/README_ja.md)で説明しています。

表・チャート・図の抽出と回答解析では、デコーダーの制約と同じ出力Schemaをモデルが読む本文にも渡します。回答解析はnullを許す結果欄も明示し、`MET` は文面を解析できたことを表します。正誤は別の画像抽出結果との計算で判定します。画像抽出には回答を、回答解析には画像を渡しません。不正出力は上限付き再試行後も拒否します。

## 評価回帰テスト

GPUを確認して `doctor --check-servers` が通った後、起動済みの標準モデルサーバーを使って次の任意検査を実行できます。別のモデルサーバーを起動・確保することはありません。tmux内でログを残して実行します。

```sh
PIXELOGUE_LIVE_RUBRIC=1 uv run --locked pytest -q -s tests/test_rubric_semantics.py
```

分割配置（port 18702と18703）では、`PIXELOGUE_LIVE_CONFIG=configs/split-pilot.yaml` も指定します。変わるのはテストの接続先だけで、モデルの識別は標準構成のままです。

通常のテスト実行では、この外部検査をスキップします。総合評価の検査には、画像に支持される回答と、意図的に誤った画像上の主張の両方を含めます。`rate-existing` は変更不能な保存済みの質問と回答について総合評価を比較できますが、生成されなかったターンを補うことはありません。大規模な本番コーパスに採用する前に、採用率だけでなく人手で確認した欠陥とも照合してください。


## サーバー条件を実行識別に固定する

実験用endpointの `serving_runtime` には `vllm_version`、`structured_output_backend`、`disable_any_whitespace`、`server_manifest_sha256` を指定できます。manifestには実際の起動設定・固定したruntime lock・起動コマンド・版確認を保存します。これらを変更するとrunと応答キャッシュの識別が変わります。宣言だけでは実サーバーの設定を確認したことにならないため、制御された起動とログで照合します。固定vLLMで空白抑制を使うには `xgrammar` または `guidance` を明示し、backendも変えた条件は設定一式の比較として扱います。

## JSONの空白暴走を抑える

生成器のサーバー設定は、JSONトークン間の空白を制限しません（`xgrammar`、`disable_any_whitespace: false`）。空白の抑制は以前に評価し、採用しませんでした。再び試す場合は、サーバーの起動時に `disable_any_whitespace: true` を指定します。固定したvLLM 0.29.0では起動時の設定だけが有効で、リクエストに同名のフィールドを送っただけでは有効化を確認できません。どちらの場合も、文字列内の通常の空白は維持します。既存サーバーを再起動し、起動manifestを記録したうえで比較してください。高速化した条件は、根拠が支持する候補数とタスク範囲も維持できた場合に採用します。

エンジンの反復停止は既定で有効です。

```yaml
runtime:
  repetition_detection:
    min_pattern_size: 1
    max_pattern_size: 4
    min_count: 64
    recovery: retry
    stages: ["*_source", question_draft, image_profile]
```

出力の確率を変えず、連続するトークン列の反復を検知して停止します。
段階の指定はシェル形式のパターンで、対象にできるのは盲検の原文読み取り（`*_source`）、
質問案の起草、画像の分析だけです。回答、修復、回答の解析、すべての評価器は
通常のデコードのままで、これらに一致するパターンは検証エラーになります。
閾値は実験用で、正しい長い反復も停止する可能性があります。
無効にするには `runtime.repetition_detection: null` を指定します。
`configs/repetition-detection-pilot.yaml` は既定の反復停止を明記した設定で、
それ以外は `configs/split-pilot.yaml` と同じです。

`finish_reason: repetition` は `MODEL_OUTPUT_REPETITION` として記録し、
生の応答と使用トークンを保存し、構文が正しいJSONでも途中停止した応答は拒否します。
`recovery: retry` では、対象段階が `structured_output_max_attempts` の範囲で
修正指示付きの完全な応答をもう一度求めます。元の出力上限を維持し、反復を理由に
上限を増やしたり途中出力を採用したりしません。`recovery: abstain` では最初の停止で
確定し、起草呼び出しは案なし、画像の分析は既定の系統、原文読み取りはその画像の保留になります。
設定変更後は異なるrun IDを使います。この機能は固定したvLLMの反復検知APIに
対応するendpointが必要です。

[vLLM 0.29.0の起動設定](https://docs.vllm.ai/en/v0.29.0/cli/serve/)、
[SamplingParams](https://docs.vllm.ai/en/v0.29.0/api/vllm/sampling_params/)、
[構造化出力](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/)で
利用可能な設定を確認できます。導入済み版の全起動設定は
`uv run --project runtime/vllm --locked vllm serve --help=all` で確認してください。
設定項目の存在だけで、全モデル・バックエンドでの動作を保証するものではありません。

## 表と逐語転写の検証範囲

単純な見出し付き表の1セル検索では、回答を見ない2回の読み取りから、全行・全列の見出しと対象セルだけを照合します。見出し集合の不足、選択見出しの曖昧さ、行列の位置ずれ、領域逸脱、読み取りの不一致は棄権します。選択した行・列の見出しの位置と対象セルを照合し、無関係なセルの値や全行列の座標は生成しません。結合・階層見出し、値に影響する脚注、複数セルの要求は表全体の検証に戻します。条件抽出・結合・復元も全体検証を維持します。

逐語転写・読み順・コード転写では、候補回答を渡さずに原文を行ごとの配列として2回読み取り、controllerが回答全体を比較します。外側の引用符・コード囲みだけを許容し、部分一致による欠落や追加の見逃しを防ぎます。原文の読み取りには元の全画像を渡し、指定領域の外に続く文字も確認します。controllerは質問で指定した単位全体が、結び付けた画像領域に収まることを要求します。逐語転写ではない文書QAとラベル・値対応は既存の範囲照合を維持します。Schema合格と画像読み取りの正確さは別であり、実画像での確認が必要です。

局所的な物体識別・属性取得・逐語転写・読み順・コード転写の質問ゲートと総合評価には、対象領域の切り出し画像を渡します。対象領域が省略されている場合はscope領域を使います。`evaluation.judge_views: full_and_crop`（既定）では1枚目に画像全体、2枚目に切り出し画像を渡し、`crop` では切り出し画像だけを渡します。画像全体での再評価は、切り出し画像を見た評価器だけが対象です。切り出しには元view ID、実際の元座標、画像hashを記録し、丸めによる領域拡張はしません。空の切り出しでターンが合格することはありません。質問・回答の生成は元の全画像を維持します。表全体の再試行も元の出力予算を維持し、不完全な表を閉じた集合として合格にしません。

UIの位置を答える操作で、回答が公開された対象名を完全に繰り返すだけの場合は、視覚評価前に棄却し、元の文章と理由を非公開artifactへ保存します。位置の説明を含む回答は引き続き評価します。

[開発点検用100画像のmanifest](../validation/diverse_100_20261004_manifest.json)には、画像の識別情報、種別、ホスト・モデルの割当、seedを固定しています。使用済みの評価画像であり、独立したholdoutではありません。実画像の結果と未確認タスクを、共有CPU契約テストの結果と分けて報告します。
