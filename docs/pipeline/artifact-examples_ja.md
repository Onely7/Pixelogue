# 中間生成物と最終出力の実例

1つの回答がすべての評価を通っても、その回答を含む対話全体が棄却されることがあります。別の対話は`QUALITY_CANDIDATE` になっても、学習用には出力できません。この違いは、保存されたレコードを見るとはっきりします。

[前へ：選抜して出力する](selection-and-export_ja.md) · [目次へ戻る](README_ja.md) · [English version](artifact-examples.md)

## 実例を取得した条件

以下の値は、2026年9月15日にBF16のQwen3.5-9BとQwen3.5-2Bで実行した、一時的な1 GPU用pilotから得たものです。この実行では、本来の生成・評価モデル2つの代わりにQwen3.5-9Bを使っているため、Qwen3.8-27BとGemma 4 31Bによる結果ではありません。長いハッシュを残しているのは、生成物同士のつながりを確認できるようにするためです。各JSONには説明に必要な項目だけを載せ、内容ハッシュで保存した完全なレコードには残りの項目も保持しています。

使用した画像はプログラムで作った検証用fixtureです。パイプラインの確認には使えますが、学習用には出力できません。

## 1. 準備済み画像

最初の画像には、赤い四角が5個描かれています。`ingest` は次の識別情報を作りました。

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

image ID、正規化後の画素ハッシュ、visual group IDは、それぞれ異なる対象を識別します。順に、画像台帳と正規化画素の組み合わせ、正規化後の見た目、類似画像の混入を防ぐためのグループです。

## 2. 画像から抽出したcapability

Qwen3.5-9Bは、画像全体を数え上げられる範囲として次のように報告しました。

```json
{
  "image_id": "a96e0109e4d1f21e661ddbd4ef28d8ba6aaf0d149d77006a791f5b349f3b0caa",
  "capabilities": [
    "bounded_complete_scope",
    "countable_entities",
    "multiple_entities",
    "visible_attribute",
    "visible_entity"
  ],
  "visible_scopes": [
    "The entire image is visible and contains exactly five red squares arranged horizontally in a single row."
  ],
  "scope_limited": false
}
```

この説明は内部の判断材料です。指示候補を作るために使いますが、学習用対話へコピーしません。

## 3. 指示候補と選択結果

コントローラーは6つの操作候補を作りました。その中の1つが、次の個数確認タスクです。

```json
{
  "candidate_id": "88225d6b078ceef51d30756f4b22d97f315a6b2fc3ad80ff9c94ea92c111b50d",
  "task_id": "visible_count",
  "family": "visible_count",
  "instruction_summary": "Count all qualifying visible instances within a complete bounded scope.",
  "required_capabilities": ["bounded_complete_scope", "countable_entities"]
}
```

Qwen3.5-2Bは、そのIDを選びました。

```json
{
  "candidate_id": "88225d6b078ceef51d30756f4b22d97f315a6b2fc3ad80ff9c94ea92c111b50d",
  "reason": "The image contains exactly five red squares arranged horizontally. The candidate 'visible_count' with task_id 'visible_count' is the most appropriate as it specifically counts all qualifying visible instances within a complete bounded scope"
}
```

これはモデルが返した選択結果の原文です。内容ハッシュで保存した応答アーティファクトには、vLLMの応答全体とトークン使用量も残っています。

## 4. 生成した質問、公開要求、回答

質問生成と回答前の検査を経て、次の内容が得られました。

```json
{
  "question": "How many red squares are there in the image?",
  "requirement": {
    "kind": "content",
    "description": "How many red squares are there in the image?",
    "start": 0,
    "end": 44,
    "lifetime": "current_turn"
  },
  "answer": "There are 5 red squares in the image."
}
```

2つの評価呼び出しは、回答全体を1つの事実主張として抽出しました。画像側と回答側の要素もそれぞれ5個へ結び付け、往復全体の評価は `PASS` になりました。

## 5. 合格した往復を含む棄却対話

最初の往復は合格しましたが、次の計画では新しい適切な往復を確定できませんでした。品質候補には最低2往復が必要なため、保存された対話全体は `REJECTED` です。

```json
{
  "conversation_id": "5769049f502ce1c43f87a09ef2d82c088663ee552238c7c18201b8413385d0c5",
  "generation_model": "Qwen/Qwen3.5-9B",
  "target_language": "en",
  "status": "REJECTED",
  "turns": [
    {
      "turn_index": 1,
      "question": {"role": "user", "content": "How many red squares are there in the image?"},
      "answer": {"role": "assistant", "content": "There are 5 red squares in the image."},
      "rating": {"aggregate": "PASS"},
      "status": "COMMITTED"
    }
  ]
}
```

完全なJSONL行には、message ID、指示の詳細、公開要求、履歴ハッシュ、個々の評価結果も入っています。最初の回答は保存しつつ、複数往復の学習候補であるかのようには扱いません。

## 6. 実際に得られた品質候補

同じpilotでは、次の3往復の対話も得られました。ここに示す往復は、すべて `PASS` で確定しています。

```json
{
  "conversation_id": "0400eaad5f5e73ff5be1fca146e5bea27ddd0dea7a8233a94e1e573ee4ead54f",
  "generation_model": "Qwen/Qwen3.5-9B",
  "target_language": "en",
  "status": "QUALITY_CANDIDATE",
  "image": {
    "image_id": "80287bd0bc2a5882ee5c3ebdea624ddbc816fcda95710b1055bd8a858ac22b85",
    "purpose": "evaluation",
    "dataset": "Pixelogue procedural fixtures"
  },
  "turns": [
    {
      "turn_index": 1,
      "task_id": "visible_count",
      "question": "How many objects are visible?",
      "answer": "There are 4 objects visible.",
      "rating": "PASS",
      "status": "COMMITTED"
    },
    {
      "turn_index": 2,
      "task_id": "object_identification",
      "question": "What are the values shown for A and B?",
      "answer": "The values shown are A = 3 and B = 1.",
      "rating": "PASS",
      "status": "COMMITTED"
    },
    {
      "turn_index": 3,
      "task_id": "visible_count",
      "question": "How many green squares are visible?",
      "answer": "There are 4 green squares visible.",
      "rating": "PASS",
      "status": "COMMITTED"
    }
  ]
}
```

`QUALITY_CANDIDATE` が表すのは、対話の品質と往復数です。この画像は `purpose: evaluation` のため、画像の利用区分によって学習用の選抜とexportから除外されます。

## 7. 学習利用できる最終レコードの形

検証専用pilotの出力を、公開可能な学習データとして示すべきではありません。次のJSONは、学習利用できる2往復の入力に対する `TrainingRecord` の正確な項目構造です。値は説明用です。

```json
{
  "record_id": "conversation-example-001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "text": null, "image": "images/ab/abcdef.png"},
        {"type": "text", "text": "How many red squares are there in the image?", "image": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "There are five red squares.", "image": null}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What color are they?", "image": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "They are red.", "image": null}
      ]
    }
  ]
}
```

画像参照は最初の1回だけです。2つ目の質問では、それまでの公開履歴を使います。指示候補、公開要求の内部表現、主張、評価結果は含まれません。

対応する来歴情報は、別の `provenance.jsonl` に残ります。

```json
{
  "conversation_id": "conversation-example-001",
  "source_id": "local:example-001",
  "image_id": "image-example-001",
  "visual_group_id": "visual-group-example-001",
  "generation_model": "Qwen/Qwen3.8-27B",
  "student_processor_lock": {
    "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
    "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "min_pixels": 16384,
    "max_pixels": 4194304
  }
}
```

上の2対話は、どちらも動作確認用のデータです。standardプロファイル、学習可能な利用条件、合格した選抜監査、同じ公開用構造が揃った対話だけが `training.jsonl` に入ります。
