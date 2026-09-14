# モデルとGPUの確認

Pixelogue は、指示選択と対話生成を別の役割に分けます。standard プロファイルでは、生成器と評価器に異なる2つのモデル系列を使います。一時的な1 GPU用の pilot では、パイプライン全体の動作を確認するため、小さい代替モデルを使います。

| 役割 | モデル | 既定のエンドポイント |
|---|---|---|
| 指示選択器 | `Qwen/Qwen3.5-2B` | `http://127.0.0.1:8000/v1` |
| 明示的に切り替える選択器 | `Qwen/Qwen3.6-35B-A3B` | `http://127.0.0.1:8001/v1` |
| 生成器・評価器A | `Qwen/Qwen3.8-27B` | `http://127.0.0.1:8002/v1` |
| 生成器・評価器B | `google/gemma-4-31B-it` | `http://127.0.0.1:8003/v1` |

`configs/standard.yaml` では、Qwen3.8-27B と Gemma 4 31B が対話を同数ずつ生成します。割り当てたモデルは、1回だけ許される修復を含め、対話が終わるまで変えません。さらに、すべての対話を両方のモデルが別々に評価します。互いの判定は入力へ含めません。

`configs/pilot.yaml` は、一時的な検証用の設定です。2つの論理的な役割を、port 8002で動く1つの `Qwen/Qwen3.5-9B` サーバーへ割り当てます。評価要求は別々に送りますが、同じ重みを使うため、確認できるのはパイプラインの接続です。異なるモデルによる評価の多様性は確認できません。

## 1. GPUを使う直前に調べる

モデルを起動する直前に実行します。

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

`doctor` が空きとみなすのは、使用率が0%で、使用メモリが1 GiB未満のGPUです。`nvidia-smi` のプロセス一覧も確認し、他の処理が使っているGPUは選びません。

静的検査の `ready: true` は、固定したBF16モデルサーバーを現在の空きGPUへ割り当てられるという意味です。同じリポジトリ、revision、エンドポイントを共有する生成器の役割は、1つのサーバーとして数えます。この検査だけでは、実際の推論成功を確認できません。

起動後の `doctor --check-servers` は、設定した配信モデル名（served model name）を確認します。この時点では、選んだGPUが使用中に見えるのが正常です。既に正常に動いているサーバーへ、別のGPUを割り当て直す必要はありません。

## 2. 推論用のuv環境を分けて導入する

GPU用パッケージがCPU開発環境を暗黙に変えないよう、vLLMは別のlock fileで管理します。

```sh
uv sync --project runtime/vllm --locked
```

vLLMは0.29.0に固定しています。サーバー設定には、モデルのrevision、BF16、context長、tensor parallel数、GPUメモリ使用率、生成時の既定値を記録しています。量子化は使いません。

## 3. 長時間の GPU 処理を tmux 内で動かす

standard 構成では、次のサーバー設定を使います。

```text
runtime/vllm/generator-a.yaml        -> Qwen3.8-27B、port 8002、tensor parallel size 2
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

Qwenには、サーバーの既定値と各要求の両方で `enable_thinking=false` を指定します。thinkingの文章や内部の制御情報を公開対話へ含めてはいけません。

## 5. 代替選択器は別に確認する

`Qwen/Qwen3.6-35B-A3B` は、設定のコピーで次の値を明示した場合だけ使います。

```yaml
models:
  active_selector: alternative
```

メモリ計画では、総パラメータ数35Bのモデルとして扱います。十分な空き容量があるときに、port 8001で別runとして確認します。既定の選択器と同時には動かさず、失敗時の自動切り替え先にも使いません。

## 6. 4 GPU時間の上限を守る

GPU時間は「経過時間 × 使用したGPU枚数」です。1枚を45分使うと0.75 GPU時間、2枚なら1.5 GPU時間です。モデルの読み込みと1画像の動作確認も含めます。開始・終了時刻をGit対象外の `_docs/` に記録し、累計4.0へ達する前に停止します。

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
