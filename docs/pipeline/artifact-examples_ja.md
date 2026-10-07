# 中間生成物と最終出力の実例

両評価器に合格した往復でも、その対話が途中で止まることがあります。`QUALITY_CANDIDATE` に到達した対話でも、学習用出力には入れられない場合があります。これらは別々の境界であり、保存された記録を見ると違いを確認できます。

[前へ：選抜して出力する](selection-and-export_ja.md) · [目次へ戻る](README_ja.md) · [英語版](artifact-examples.md)

## 実例の読み方

各JSONブロックの項目構成は、現在のコードが書き出すものと同じです。1節の画像識別情報を除き、値は説明用に作成したもので、実測runから写したものではありません。`conv-example-001` や `<sha256>` は長い識別子の代わりです。以下の各記録は、run store内の `artifacts/<kind>/<16進数の先頭2文字>/<sha256>` に正規化JSONとして保存されます。見出しのコード表記は、その `<kind>` を示します。

ここでは、検証専用の街頭写真について3往復を予定した1つの対話を追います。1往復目は画像全体での再評価を経て合格し、2往復目は1回の回答修復を経て合格します。3往復目は操作検証で不合格になるため、対話は2往復分の確定部分を保持します。

## 1. 準備済み画像

次の識別情報は、赤い四角が5つ並ぶ生成済みの評価用fixtureについて記録されたものです。

```json
{
  "image_id": "a96e0109e4d1f21e661ddbd4ef28d8ba6aaf0d149d77006a791f5b349f3b0caa",
  "purpose": "evaluation",
  "dataset": "Pixelogue procedural fixtures",
  "dataset_split": "development",
  "canonical_pixel_sha256": "1e3a955fe64c866075e76bb9a250300b869073bf4357820f7d1e626cb0487302",
  "visual_group_id": "ab7efa8dd4295b3afb992ae39e7e1d5d8a203e3e8105336f8000eb5ad3627ac1",
  "full_view": {
    "relative_path": "images/21/21e9a7efd94d6b56d32fe0751db4a0791e1c62f4ca784381fb4e03b216a77ae4.png",
    "width": 640,
    "height": 480,
    "media_type": "image/png"
  }
}
```

image ID、正規化後の画素hash、visual groupは、それぞれ別のことを表します。順に、この画像台帳の行、正規化した見た目の内容、複製の漏れを防ぐためのグループを識別します。

## 2. 画像の分析結果と割り当て

ルーターの応答は、そのモデル呼び出しの記録として保存されます。

```json
{
  "image_kind": "photo",
  "readable_text": "some",
  "supported_families": ["visual_description", "reference_spatial", "text_reading", "counting_and_sets"],
  "reason": "Street scene with people at a bus stop and a readable shop sign."
}
```

1往復目は先頭ターンなので、controllerは軽い操作だけを提示します。割り当ては `turn_route` テーブルに保存され、`question-drafts` にも同じものが入ります。

```json
{
  "primary_family": "text_reading",
  "primary_task_ids": ["text_transcription", "label_value_lookup", "text_object_binding"],
  "secondary_family": "visual_description",
  "secondary_task_ids": ["attribute_lookup", "object_identification"],
  "basis": "profile"
}
```

分析結果がない場合や、実行可能な系統を1つも挙げていない場合、`basis` は `fallback` になります。分析結果が不正なままの場合は、`image_id`・`reason`・`message` を持つ `image-profile-abstentions` が残ります。

## 3. `question-drafts`

会話を担当する生成器は、この割り当てに対して2つの案を返しました。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 1,
  "call_index": 0,
  "route": {
    "primary_family": "text_reading",
    "primary_task_ids": ["text_transcription", "label_value_lookup", "text_object_binding"],
    "secondary_family": "visual_description",
    "secondary_task_ids": ["attribute_lookup", "object_identification"],
    "basis": "profile"
  },
  "batch": {
    "drafts": [
      {
        "task_id": "text_transcription",
        "question": "What does the sign above the shop entrance say?",
        "target": "the sign above the shop entrance",
        "public_parameters": [],
        "scope_region": {"left": 0.08, "top": 0.05, "right": 0.46, "bottom": 0.21},
        "target_region": null,
        "fact_key": {"subject": "shop sign", "dimension": "text"}
      },
      {
        "task_id": "attribute_lookup",
        "question": "What color is the jacket worn by the person standing next to the bus?",
        "target": "the person standing next to the bus",
        "public_parameters": [{"name": "attribute", "value": "jacket color"}],
        "scope_region": {"left": 0.40, "top": 0.15, "right": 0.75, "bottom": 0.98},
        "target_region": {"left": 0.46, "top": 0.20, "right": 0.64, "bottom": 0.95},
        "fact_key": {"subject": "person next to the bus", "dimension": "jacket colour"}
      }
    ],
    "reason": null
  }
}
```

2つ目の案の非公開 `request_key` は `person next to bus|jacket color` になります。冠詞を除き、`colour` を正規化するため、後の案が表記を変えて同じ事実を尋ねることはできません。

## 4. `public-text-rejections` と `draft-rejections`

1つ目の案は相対位置で文字を指定しており、転写の根拠では検証できないため、評価器には渡りません。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 1,
  "field": "question",
  "reason": "TEXT_RELATION_UNVERIFIED",
  "content": "What does the sign above the shop entrance say?"
}
```

操作契約に反する案は別に記録します。3往復目では、`group_by_attribute` の案が必須の `return` を指定していませんでした。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 3,
  "call_index": 0,
  "draft_index": 0,
  "reason": "DRAFT_PARAMETER_MISSING",
  "message": "Missing parameters ['return']"
}
```

## 5. `question-gate-decisions`

1往復目の2つ目の案を両評価器が審査しました。切り出し画像には対象の人物が写っており、両者のラベルは案の操作と一致しました。

```json
{
  "question_message_id": "conv-example-001:q:1",
  "candidate_id": "cand-example-1",
  "drafted_task_id": "attribute_lookup",
  "votes": [
    {
      "local_anchor": "MET",
      "operation_coherent": "MET",
      "useful_request": "MET",
      "reason": "Asks for the jacket color of the bound person without stating it.",
      "realized_task_id": "attribute_lookup"
    },
    {
      "local_anchor": "MET",
      "operation_coherent": "MET",
      "useful_request": "MET",
      "reason": "The person is visible in the crop and the color is not given.",
      "realized_task_id": "attribute_lookup"
    }
  ],
  "label_rule": "exact",
  "fit": "MET",
  "decided_task_id": "attribute_lookup",
  "verdict": "MET"
}
```

`decided_task_id` が `drafted_task_id` と異なるのは、`same_contract` 方式で `relabel` が起きた場合だけです。

## 6. `rating-decisions`

回答は “The jacket is yellow.” でした。一方の評価器は切り出し画像にバスが写っていないため `UNKNOWN` を返し、画像全体でもう1回評価しました。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 1,
  "template_id": "Q_HOLISTIC",
  "subject": null,
  "votes": [
    {"verdict": "MET", "reason": "The person beside the bus wears a yellow jacket."},
    {"verdict": "UNKNOWN", "reason": "The crop does not show whether this person stands next to the bus."}
  ],
  "tiebreak_vote": {"verdict": "MET", "reason": "The full image shows the person next to the bus in a yellow jacket."},
  "repair_objection": null,
  "controller_reason": null,
  "verdict": "MET"
}
```

`votes` には1回目の票を評価器の順に残します。評価器を呼ばずにcontrollerが判定した場合、`controller_reason` は `INVALID_PUBLIC_TEXT` または `EMPTY_FOCUS_VIEW` になります。`attribute_lookup` には2回の評価以外の検証契約がないため、1往復目は確定しました。

## 7. `answer-attempts`

2往復目の “What is the person on the left doing?” には、最初に “The person on the left is reading a newspaper.” と回答しました。一方の評価器は `MET`、もう一方は `NOT_MET` でした。この評価の `rating-decisions` では `repair_objection` に異議が入り、置き換えられた回答はここに残ります。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 2,
  "attempt": 0,
  "answer": {
    "message_id": "conv-example-001:a:2",
    "turn_index": 2,
    "role": "assistant",
    "content": "The person on the left is reading a newspaper."
  },
  "rating": {
    "items": [
      {
        "item_id": "<sha256>",
        "template_id": "Q_HOLISTIC",
        "axis": "factual_correctness",
        "verdict": "UNKNOWN",
        "reason": "generator_a:The person on the left is reading. | generator_b:The person on the left holds a phone, not a newspaper.",
        "actor": "dual-consensus",
        "history_hash": "<sha256>"
      }
    ],
    "aggregate": "ABSTAIN"
  },
  "objection": "The person on the left holds a phone, not a newspaper."
}
```

生成器が受け取ったのは、画像・履歴・質問のほかには元の回答と異議だけです。置き換えた “The person on the left is looking at a phone.” は両評価器による新しい評価に合格し、2往復目は確定しました。

## 8. `operation-checks`

3往復目は “How many people are waiting at the bus stop?” という質問に “There are three people waiting at the bus stop.” と回答しました。総合評価は両方とも `MET` でしたが、`entity_count` には `closed_set_check` も必要です。2つの読み手は、それぞれ独立に4人と数えました。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 3,
  "contract": "closed_set_check",
  "candidate_id": "cand-example-3",
  "question_hash": "<sha256>",
  "answer_hash": "<sha256>",
  "verdict": "NOT_MET",
  "independent_evidence": [
    {
      "coverage": "MET",
      "mode": "count",
      "counts": [{"scope": "all", "expected": 4, "reported": 3}],
      "expected_members": [],
      "reported_members": [],
      "empty_scope_is_explicit": false,
      "reason": "Four people wait at the stop; the answer reports three."
    },
    {
      "coverage": "MET",
      "mode": "count",
      "counts": [{"scope": "all", "expected": 4, "reported": 3}],
      "expected_members": [],
      "reported_members": [],
      "empty_scope_is_explicit": false,
      "reason": "One person behind the shelter pole is also waiting."
    }
  ]
}
```

2つの読み取りが一致したため、controllerが画像側と回答側の個数を照合しました。このターンの集約結果は `FAIL` になります。

## 9. 停止記録と保持した確定部分

不合格のターンでループは終わります。

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 3,
  "status": "REJECTED",
  "stage": "answer_verification",
  "reason": "RATING_FAIL"
}
```

これは `conversation-stop-reasons` の記録です。すでに2往復が確定しており、`evaluation.retain_accepted_prefix` がtrueなので、3往復分の対話全体を `"retained_turns": 2` とともに非公開の `conversation-stops` へ残し、最終的な `conversations` の行には確定部分だけを入れます。次の行は省略・平坦化したもので、message ID、操作契約、履歴hash、評価項目を省いています。

```json
{
  "conversation_id": "conv-example-001",
  "generation_model": "Qwen/Qwen3.8-27B",
  "target_language": "en",
  "status": "QUALITY_CANDIDATE",
  "image": {
    "image_id": "image-example-001",
    "purpose": "evaluation",
    "dataset": "Example evaluation images"
  },
  "turns": [
    {
      "turn_index": 1,
      "task_id": "attribute_lookup",
      "question": "What color is the jacket worn by the person standing next to the bus?",
      "answer": "The jacket is yellow.",
      "rating": "PASS",
      "status": "COMMITTED"
    },
    {
      "turn_index": 2,
      "task_id": "visible_action",
      "question": "What is the person on the left doing?",
      "answer": "The person on the left is looking at a phone.",
      "rating": "PASS",
      "status": "COMMITTED"
    }
  ]
}
```

`QUALITY_CANDIDATE` は、対話の品質と往復数に関する状態です。この画像は `purpose: evaluation` のままなので、出典の条件によって学習用の選抜とexportから除外されます。

## 10. 学習利用可能なレコードの構造

検証専用の行を、公開済みの学習データとして示してはいけません。次の例は、学習利用可能な画像による2往復の対話について、`TrainingRecord` と同じ項目構造を持つ有効なJSONです。値は説明用です。

```json
{
  "record_id": "conversation-example-001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "text": null, "image": "images/ab/abcdef.png", "width": 640, "height": 480},
        {"type": "text", "text": "What color are the squares in this image?", "image": null, "width": null, "height": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "They are red.", "image": null, "width": null, "height": null}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "How are the squares arranged?", "image": null, "width": null, "height": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "They form a single horizontal row.", "image": null, "width": null, "height": null}
      ]
    }
  ]
}
```

画像は1回だけ現れ、出力した画像ファイルの横と縦のピクセル数も記録されます。後の質問は保存された公開履歴に依存します。操作契約、質問案、評価器の票、評価記録は含まれません。

対応する来歴は、別の行として保存します。

```json
{
  "conversation_id": "conversation-example-001",
  "source_id": "local:example-001",
  "image_id": "image-example-001",
  "visual_group_id": "visual-group-example-001",
  "generation_model": "Qwen/Qwen3.8-27B",
  "operation_ids": ["attribute_lookup", "grounded_description"],
  "catalog_versions": ["8.0", "8.0"],
  "primary_operation_id": "grounded_description"
}
```

`training.jsonl` に入るのは、standardプロファイルで学習利用が許可され、選抜監査に合格し、同じ公開情報だけの構造を持つ対話だけです。

## 11. 非公開記録の種類

| 種類 | 書き込む時点 | 主な項目 |
|---|---|---|
| `question-drafts` | 起草呼び出しが正常な案の一覧を返した | `conversation_id`、`turn_index`、`call_index`、`route`、`batch` |
| `draft-rejections` | 案が操作契約に違反した | `call_index`、`draft_index`、`reason`、`message` |
| `draft-abstentions` | 起草呼び出しが再試行後も不正だった | `call_index`、`reason`、`message` |
| `public-text-rejections` | 決定的検査が質問または回答の文を棄却した | `field`、`reason`、`content` |
| `question-gate-decisions` | 両評価器が質問を審査した | `votes`、`label_rule`、`fit`、`drafted_task_id`、`decided_task_id`、`verdict` |
| `image-profile-abstentions` | ルーターの分析結果が再試行後も不正だった | `image_id`、`reason`、`message` |
| `focus-views`、`focus-images` | 評価器用の切り出し画像を作った | 切り出しの大きさ、`source_view_id`、`source_left`〜`source_bottom`、PNGの本体 |
| `focus-view-abstentions` | 結び付けた領域に完全な画素がない | `candidate_id`、`region` |
| `rating-decisions` | 総合評価の判定が決まった | `votes`、`tiebreak_vote`、`repair_objection`、`controller_reason`、`verdict` |
| `answer-attempts` | 修復で回答を置き換えた | `answer`、`rating`、`objection` |
| `operation-checks` | 検証器が終了した | `contract`、`verdict`、`independent_evidence` |
| `structured-output-failures` | モデル出力が契約に違反した | `stage`、`attempt`、`reason`、`next_retry_feedback` |
| `turns` | 往復を確定した | ターンのartifact全体 |
| `conversation-stop-reasons` | 予定ターン数に達する前に対話が止まった | `turn_index`、`status`、`stage`、`reason` |
| `conversation-stops` | 止まった対話が確定部分を保持した | `conversation`、`retained_turns` |
| `conversations` | 対話が最終状態になった | 対話のartifact全体 |
| `model-output-abstentions`、`errors` | 画像、または再評価中のターンが、モデル出力の不正や実行障害で終わった | `conversation_id`、`image_id`、`reason`、`message` |
| `requests`、`responses`、`request-images` | すべてのモデル呼び出し | 要求の全体、生の応答、重複を除いた画像データ |
| `transport-errors`、`cache-accesses` | HTTPの失敗と保存済み応答の再利用 | 状態コードまたはエラーの種類、試行番号、要求hash |

廃止したscopedプランナーが書いたstoreには、根拠抽出、binding、意図判定、公開要求の記録も含まれる場合があります。現在のコードはこれらを書き込みませんが、そのプランナーが保存したターンは引き続き読み込めます。
