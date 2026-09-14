# モデルと GPU の確認

Pixelogue は指示の選択と質問・回答の生成を分けます。既定の選択器は
`Qwen/Qwen3.5-2B` です。`Qwen/Qwen3.6-35B-A3B` は設定で明示的に切り替える場合だけ使います。
後者は active parameter が小さくても総パラメータ数は 35B なので、35B の重みとしてメモリを
見積もります。Qwen3.8 と Gemma は対話を分担して生成し、生成担当にかかわらず両方が評価します。

## 1. GPU を使う直前に調べる

モデルを起動する直前に毎回実行します。

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

`doctor` が空きとみなすのは、使用率 0% かつ使用メモリ 1 GiB 未満の GPU だけです。
`nvidia-smi` のプロセス一覧も確認し、他の処理が使っている GPU は選びません。
`doctor --check-servers` は有効な設定に必要な 3 つの endpoint も確認します。

静的検査の `ready: true` は、同時に必要な各 server へ重複しない空き GPU を割り当てたうえで、
固定した BF16 重みの概算量が収まるという意味です。起動前に各 `assigned_gpu_indices` を確認します。
実推論の成功ではありません。サーバー検査では、設定した served name がモデル一覧に存在する
場合だけ READY になります。

## 2. 推論用 uv 環境を分けて導入する

GPU パッケージが CPU 開発環境を暗黙に変更しないよう、vLLM は別の lock file で管理します。

```sh
uv sync --project runtime/vllm --locked
```

vLLM は 0.29.0 に固定しています。各 server 設定では model revision、BF16、context 長、tensor
parallel 数、vLLM の sampling default も固定しています。

## 3. 確認した GPU 番号を明示して起動する

次は例です。実行直前の空き状況に合わせて番号を変更してください。

```sh
CUDA_VISIBLE_DEVICES=3 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/selector-default.yaml

CUDA_VISIBLE_DEVICES=0,3 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a.yaml
```

オンライン生成では、有効な選択器と 2 つの大型モデル endpoint を同時に到達可能にします。
設定例の port は 8000、8002、8003 です。同時起動する server には別々の空き GPU を割り当てます。
不足する場合は pilot を保留し、量子化へ変えたり使用中 GPU を流用したりしません。代替選択器は
pilot 設定のコピーで `models.active_selector: alternative` として port 8001 で個別に確認します。

Qwen には server と request の両方で `enable_thinking=false` を指定します。Gemma には
`reasoning_effort=none` を指定し、公開出力に reasoning 用制御文字列があれば拒否します。
詳細は vLLM の [server 引数](https://docs.vllm.ai/en/latest/configuration/serve_args/)と
[thinking 制御](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)を参照してください。

## 4. 4 GPU 時間を超えない

GPU 時間は「経過時間 × 使用 GPU 枚数」です。2 枚を 30 分使うと 1 GPU 時間です。モデルの
読み込みと能力検査も含めます。開始・終了時刻を Git 対象外の `_docs/` に記録し、累計 4.0 に
達する前に止めます。

Pixelogue は request と token 使用量を保存しますが、外部 vLLM process の起動時間は観測できません。
その時間は運用者が加算します。GPU や server が用意できない検査は合格ではなく未実行です。

## 5. 正解付き能力検査を行う

必要な server がすべて正常になってから実行します。

```sh
uv run --locked pixelogue evaluate-capabilities \
  --config configs/pilot.yaml \
  --fixtures data/fixtures/fixtures.jsonl \
  --fixture-root data/fixtures \
  --output artifacts/capability-report.json
```

期待する採否は controller 内だけで使い、評価器へ渡しません。development と confirmation の集計も
分けます。少数 fixture の結果は接続確認であり、能力認定ではありません。
