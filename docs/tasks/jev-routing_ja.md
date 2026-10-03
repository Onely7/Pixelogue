# Jevを使う実験用ルーティング

[English](jev-routing.md)

この経路は、根拠抽出と候補具体化を対象とする明示的な試験機能です。
まず生成モデルが、領域付きの短い観測、または具体的な公開パラメータを提案します。
次にJevが、各観測・適格性条件を個別に`MET / NOT_MET / UNKNOWN`で判定します。
controllerが固定IDを付与し、組み立てた記録を既存の検証処理に渡します。

分類器だけで座標の発見、文字転写、表復元、回答生成を行う仕組みではありません。
観測の欠落や閉じていない集合を`MET`へ補完しません。
未対応タスクの候補具体化には、上限を設けた従来経路を使います。
既定selector、生成モデルの組、質問ゲート、回答検証、評価専用画像の扱いは維持します。

## 段階を分けて有効化する

検証されるタスク設定は次の形式です。

```yaml
tasks:
  decision_routing:
    evidence_enabled: false
    binding_enabled: true
    model_repository: autotrust/JEV-27B-VL
    model_revision: f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc
    proposal_max_tokens: 1536
```

両フラグの既定値は`false`です。`synthesize` CLIは、有効な設定に対して指定した分類器だけを作成し、終了・失敗・中断時にclientを閉じます。
JEVでは`base_url`へ専用HTTP endpointを指定してください。Omniでは、
`runtime_python`と`model_revision`をdirectory名に持つ`snapshot_path`を絶対pathで指定し、
起動前に`CUDA_VISIBLE_DEVICES`へ検査済みの1枚のGPU UUIDを設定します。
モデルserverの起動とGPU予約はjobの管理側で行います。`rate-existing`は分類器を起動しません。
Pythonからは`SynthesisCoordinator(..., decision_client=client)`へclientを渡せます。

## JEV-27B-VL：専用HTTPサービス

固定snapshotの確認済み`serve_decide.py`を、分離したvLLM環境で起動します。
snapshotに含まれるdecision LoRAとheadを使用します。
専用endpointは`POST /v1/decide`です。通常のchat endpointとは異なります。

```python
from pixelogue.decision_serving import JevHttpDecisionClient

client = JevHttpDecisionClient(
    "http://127.0.0.1:18104/v1",
    "autotrust/JEV-27B-VL",
    "f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc",
)
try:
    # 設定済みのSynthesisCoordinatorへclientを渡す。
    pass
finally:
    client.close()
```

adapterは選択肢を3つに固定し、生成によるthinkingを無効化します。
応答のモデル名、専用protocol、選択肢の順序、選択されたラベル、
正規化済みの全確率を検査します。APIが返すのはモデル名なので、
実際のweight revisionはサーバー起動manifestでも確認してください。
既定ではloopbackだけを許可します。別ホストへは、loopbackで終端する
明示的なSSH tunnelを推奨します。プログラムからの外部host許可は、
アプリの現在のローカル限定runtime方針とは別の設定です。

## Jev-Omni：分離された常駐worker

ML依存はアプリ環境へ追加せず、`runtime/vllm`に保持します。
アプリ側は、明示したruntime PythonからローカルJSONL workerを1つ起動します。
workerは固定snapshotをBF16で一度だけ読み込み、ネイティブのforward-pass headを使用します。
hookのcapture状態を共有するため、リクエストを直列に処理します。
`.generate()`は呼びません。

```python
from pathlib import Path
from pixelogue.decision_serving import SubprocessOmniDecisionClient

revision = "5addda86ddee081a68fb067477ea100c221b8917"
client = SubprocessOmniDecisionClient(
    Path("runtime/vllm/.venv/bin/python").resolve(),
    Path("models/decision-classifiers/Jev-Omni", revision).resolve(),
    revision,
    env={"CUDA_VISIBLE_DEVICES": "GPU-REPLACE-WITH-INSPECTED-IDLE-UUID"},
    stderr_path=Path("artifacts/jev-worker/worker.log"),
)
try:
    # 設定済みのSynthesisCoordinatorへclientを渡す。
    pass
finally:
    client.close()
```

GPU UUIDはclientを作る前に指定します。adapterはGPU探索・予約、量子化変更、
snapshotダウンロードを行いません。workerは`local_files_only=True`とoffline設定を使います。
job管理側で`nvidia-smi`、`doctor`、tmux実行、成否を問わない終了後の再予約、
モデル読み込みを含む全割当GPUの時間記録を実施してください。
必要メモリは、実際の起動と画像入力の確認で測定します。

workerだけをrepository rootから起動する場合は次の形式です。

```sh
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-INSPECTED-IDLE-UUID \
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
runtime/vllm/.venv/bin/python -u -m pixelogue.decision_worker \
  --snapshot-path /absolute/path/to/Jev-Omni/5addda86ddee081a68fb067477ea100c221b8917 \
  --revision 5addda86ddee081a68fb067477ea100c221b8917 \
  2>artifacts/jev-worker/worker.log
```

ログ用directoryは先に作成してください。標準入力は1行ごとに
`DecisionRequest`をJSONで渡します。標準出力は起動時の`ready`と、
各要求の`result`または明示的な`error`だけに使い、モデルの進捗はstderrへ出します。
workerの失敗や不正出力を合格ラベルへ変換しません。
終了・timeout時は自動再起動せず、保存したrunへ新しい明示的なclientで再開します。

## 非公開記録と計測

`DecisionRequest`は画像bytesの識別、view、正規化座標の領域、公開context、
契約版、独立した試行IDを保持します。参照ラベル、候補回答、未来ターン、
他評価者の結果は入力へ含めません。ローカルpathは所在を示し、画像の識別には使いません。
Omni adapterはhash検査済みbytesを不変の一時コピーとしてpredictorへ渡します。

`DecisionResult`は3ラベルすべてのscoreと、取得できたtoken usageを保存します。
scoreは校正合格の証明ではありません。usageの欠測をゼロにしません。
結果を再利用できるのは、試行・公開履歴・画像/view・契約・モデル版が一致する場合です。
異なる実験試行は別呼び出しにします。判定を非公開で保存し、従来の公開出力検証を続けます。

初回は約24画像groupを使い、接続・調整用6画像と比較用18画像へ分けます。
分割はモデル結果を見る前に固定してください。同じ有限質問で4つの固定モデルを比較し、
独立に取得したGPT-6.1 Sol Ultra参照票との3値一致、誤った`MET`、`MET`の見逃し、
`UNKNOWN`、失敗、質問可能な操作の捕捉率を画像系統別に報告します。
median/p95の時間を記録します。開発用画像で根拠の正しさを維持できる条件が見つかった場合に、
完成した2ターン会話数／GPU時間を測定します。条件が見つからなければ拡大せず、
不採用の理由と未実施の比較を記録します。
Qwen/Gemmaの短いラベル出力を比較へ含め、出力短縮と専用分類器への変更を区別します。
少数pilotは問題の診断に使い、一般的な精度や専門タスクの校正合格の根拠にはしません。
