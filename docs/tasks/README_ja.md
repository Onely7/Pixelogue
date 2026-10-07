# タスクカタログ v8

[English](README.md) · [78タスクの定義](TASKS.md) · [185 subsetの調査対応表](finevision_mapping_185.md)

専門タスクの環境・校正と研究用CLIは[専門検証と研究実験](specialist-and-research_ja.md)を参照してください。

カタログは **71の標準タスク＋7の条件付き拡張、合計78タスク・18系統**で構成されています。[TASKS.md](TASKS.md) は `src/pixelogue/resources/task_catalog.yaml` から生成します。カタログを編集したら必ず再生成してください。コミット済みの内容がカタログと違う場合はテストが失敗します。

```bash
uv run --locked pixelogue compile --tasks-markdown docs/tasks/TASKS.md
```

## タスクの定義の形

各タスクは次の項目を持ちます。

- `label` は動詞で始まる短い英語名です（例「Read a value from a chart」）。
- `definition` は、ユーザーが何を尋ね、回答が何を返すかを平易な英語で述べます。紛らわしい隣接タスクがある場合は、最後の1文でそれを示します（例「Use count_comparison instead to compare two counts」）。
- `parameters` は、質問文自体が明示すべき選択です。それぞれ種類（`text`・`choice`・`list`・`integer`）、説明、`required` を持ちます。`choice` は値の一覧を持ち、`list` は項目数の上下限を持てます。
- `answer_format` は、検証器が特定の回答形式を照合する場合にだけあります。項目抽出やチャートデータのJSON、決まった判定語などです。
- `do_not_infer` は回答が主張してはいけないことです。加えて、必要な画像の能力、適格性検査、検証契約を列挙します。
- `example_question` と `related_finevision_subsets` は人間向けの情報で、モデルには渡しません。

質問を起草するモデルには、定義、`do_not_infer`、適格性検査、パラメーター契約、回答形式を渡します。評価器にも、同じ定義と回答形式を操作契約として渡します。モデルに見えるカタログの文はすべてASCIIの英語で内部用語を使わないこと、各定義が質問と回答の内容を述べることを、テストで確認しています。

## 定義済みタスクと実行可能なタスク

`pixelogue compile` は全定義、厳密なJSON Schema、`task_admission` を出力します。各タスクについて、実行可能かどうか、必要な検証器とその実行環境、校正済みの領域、使えない具体的な理由を確認できます。

**標準71タスクすべて**に検証経路があり、合成で質問案を作るのも標準タスクだけです。そのうち35タスクは、回答を見せない2回の構造抽出の後、有限集合・算術・表・チャート・文書・数式・図・目盛り・明示された幾何条件・有限パターンを計算で照合します。知識系統の5タスクは、盲検の2つの読み手が同じ短い答えを返した場合だけ確定し、存在の2タスクは、盲検の2つの読み手が対象の有無で一致した場合だけ確定します。物体の矩形は盲検の2つの矩形と重なる必要があります。パネルの比較、パネルの順序、UI要素の位置は、盲検の2回の視覚的な契約審査で確認します。残りの25タスクは軽いタスクで、割り当ての規則で説明します。未対応記法、抽出不足、解の曖昧さ、抽出結果の不一致は棄権します。`grounded_arithmetic` も他の数値タスクと同じ厳密な数値基盤を使います。どの計算を受け付けるかは、質問が明示する精度と単位の規則で決まります。

表、表の1セル検索、チャート、グラフ、専門タスクの原文読み取りには `tasks.source_max_tokens`（既定4096）を使います。チャートとグラフの読み取りが上限でJSONを完了できなかった場合は、上限を最大2倍にして再試行します。上限は8,192トークンで、設定値がすでにそれより大きい場合は増やしません。表の読み取りは上限を変えず、表全体が収まらない場合は `UNKNOWN` を返す必要があります。候補回答からセル内容を補うことはありません。

チャートでは、棒に正確な値が印字されていて数値軸の目盛り表示がない形式も検証します。この軸は `unmarked` として記録し、画素位置から値を推定しません。対象の値にはそれぞれ直接印字されたラベルが必要で、ラベルのないマークは未検証です。線形・対数軸の校正には、順序の正しい目盛り表示を2個以上要求します。

専門拡張7タスクの初版検証器を実装し、`tasks.enabled_extensions` に指定できます。`task_admission` が有効化した拡張を実行可能と報告するのは、対象モデルに一致する校正証明と、動作する検証環境が揃った場合だけです。7タスクをすべて指定した例は `configs/specialist-pilot.yaml` です。直接起草は拡張を提示しないため、拡張は `evaluate-specialist` で検証します。OS隔離が使えないホストでは、コード復元の2タスクを環境不足として報告します。静的SVG・限定TikZ・HTML/CSSは隔離環境内だけで描画し、UI操作は指定を検査するだけで実行しません。

## 知識・専門・創作のタスク

[FineVisionの行単位の監査](finevision_mapping_185.md#row-level-audit-2026-10-07)で、画像を読むだけでは答えられない3つの系統を加えました。

- `knowledge_recognition` は、広く知られた名所・作品・様式・地図上の地域を世界知識で特定します。人物は特定しません。
- `domain_reasoning` は、教科書的な専門知識を使います。ラベル付きの科学・技術の図を説明し、標準的な記法を解釈し、画像に印刷された数学の問題を解きます。
- `grounded_creation` は短い創作文を書きます。雰囲気や筋書きは創作できますが、画像に写っているものについての記述はすべて正しい必要があります。

短い知識の答えを返す5タスク（`named_entity_recognition`、`style_recognition`、`map_region_identification`、`notation_interpretation`、`math_word_problem`）は `answer_consensus_check` を使います。2つの読み手が候補回答を見ずに同じ質問へ答え、大文字・小文字、句読点、冒頭の冠詞、数値の書式を正規化した2つの短い答えが候補回答と一致した場合だけ確定します。読み手どうしの不一致や、訳語などの別名は確定しません。2つのモデルの一致は根拠であって証明ではないため、これらのタスクも2つの盲検の評価器を通る必要があります。`concept_explanation` と `grounded_creative_writing` は、評価器とevidence bindingで確認する軽いタスクです。

監査では既存の系統にも3タスクを加えました。`object_box_grounding` は正規化した矩形で答え、`box_iou_check` が盲検の2つの矩形の読み取りとIoU 0.5以上で一対一に対応付けます。`chart_value_arithmetic` は2つのチャートの読み取りから結果を計算し、印字値なら厳密な値、目盛りからの推定なら範囲で確認します。`visible_text_translation` は、範囲を区切った可視の文字を翻訳します。医療画像、多肢選択の回答形式、1つの質問に複数の画像を使う形式は加えていません。subsetごとの決定は監査の表に記録しています。

## 存在と誤った前提のタスク

`presence_and_premises` 系統は、答える前に対象が写っているかを確かめる振る舞いを教えます。POPE などの物体幻覚の評価や、HaloQuest などの誤った前提のデータが測る失敗です。

- `object_presence` は、名前を挙げた種類の物が画像や名前のある領域にあるかを問います（例: 「Is there a fork on the table?」）。質問の約半分は、写っていない物を挙げます。起草は、写っている物とよく一緒に現れる物（皿の横のフォーク。POPE の adversarial negative）を優先し、次にどこにでもある物（人。popular negative）、それ以外は無関係な物（random negative）を選びます。写っている物を挙げる質問が yes の答えになります。
- `false_premise_question` は、画像にないものを、あるかのように前提にした質問です。数、属性、位置、動作、種類のいずれかを問います。前提は、写っていない物（女性がバッグを持っていないのに「What color is the woman's purse?」）か、写っている物が持たない属性・関係・動作（茶色の犬しかいないのに黒い犬）です。`asked_detail` に5つのどれを問うたかを記録します。

どちらの回答も、まず関連する見えている物を述べ、次に問われた物や前提の物があるかを述べ、結論（yes か no、数なら0、それ以外は判断できないこと）で結びます。`premise_check` は、候補回答を見ない2つの読み手に対象が写っているかを判定させ、画像を見ない2つの解析で回答の結論を読み取ります。読み手どうしが一致し、回答の結論とも一致した場合だけ確定します。読み手が隠れた・切れた・小さすぎる領域を理由に迷った場合は確定せず、前提が実は成り立っていた誤った前提の質問も確定しません。写っていないものを問えるのはこの2つの操作だけです。`entity_count`、`described_object_lookup`、`visual_claim_verification`、`answerability_assessment` の定義は、写っていない物についてはこれらを使うよう案内します。質問ゲートが写っていない対象を認めるのはこの2つの操作だけで、回答の指示は、どの操作でも誤った前提を訂正し、ない物を作らないよう求めます。

## 画像ごとの処理

各手順の詳細は[生成ガイド](../pipeline/generation-and-evaluation_ja.md)で説明しています。

1. ルーター（生成器Aの `Qwen/Qwen3.8-27B` サーバー）が画像を1回だけ分析し、見えている内容から対応できる系統を列挙します。出力が不正な場合は、汎用的な既定の系統を使います。
2. 予定した各ターンについて、controllerが最大4タスクの主系統と最大2タスクの副系統を割り当てます。会話内で未使用の系統と、run全体の目標比率に対する不足が大きい系統を優先します。割り当ては保存されるため、再開したターンにも同じ候補を提示します。
3. 会話を担当する生成器が、提示されたタスクについて最大 `tasks.draft_count` 件の質問案を書きます。各案は対象とパラメーター、scopeと対象の領域、非公開のfact keyを持ちます。controllerはパラメーターをカタログで検証し、公開文の決定的検査を行います。
4. 受け付けた質問ごとに、回答がまだない状態で両評価器がそれぞれ1回ずつ審査します。局所的な根拠、タスクとの整合、有用性と、独立したタスクラベルを返します。
5. 最初に審査を通った質問だけに回答します。両評価器は期待する操作契約とともに回答を総合的に評価します。追加の処理は、画像全体での再評価1回と回答修復1回だけです。
6. 該当するすべての検証器を実行してから、ターンを確定します。構造化されたタスクでは独立した2回の読み取りを行い、根拠は非公開で保存します。

### 割り当ての規則

最初の `tasks.anchor_turns` ターン（既定2）では、検証契約が `dual_visual_review`・`evidence_binding_check`・`transcript_alignment` の範囲に収まる軽いタスクだけを提示します。現在のカタログでは、標準69タスクのうち25タスクが該当します。[TASKS.md](TASKS.md) には各タスクが軽い検証か構造化された検証かを記載しています。これらのターンで会話が最低2ターンに届くかが決まり、構造抽出型の検証器は棄権が多いためです。

`screen_ui` のタスクは、ルーターが `screen` と分析した画像にだけ提示します。注釈付きの図、ダイアグラム、写真は画面として扱いません。分析結果がない場合や、実行可能な系統を1つも挙げていない場合は、`visual_description`、`text_reading`、`reference_spatial`、`counting_and_sets`、`evidence_verification` を使います。

`tasks.family_targets: uniform` では、実行可能な系統すべてに同じ目標比率を与えます。系統IDと正の重みの対応を指定すると、別の目標比率になります。指定しなかった系統の目標比率は0ですが、割り当て自体は可能です。`tasks.task_weights` には0より大きく1以下の重みを指定し、タスクを並べるときに確定数をこの重みで割ります。既定では表・チャート・文書の構造復元を0.25とし、提示される頻度を下げつつ候補には残します。これらの検証器は表や系列の全体を必要とし、特定箇所の検索より棄権が多いためです。

### 質問案の受付

各案は、提示されたタスクを1つだけ指定します。パラメーターはタスクのパラメーター名か `target` だけを使い、必須のパラメーターをすべて含み、種類に合う値を持つ必要があります。`choice` は値の一覧から選び、`list` は項目数の範囲内で重複のない文字列を並べ、`integer` は整数です。欠けた選択をcontrollerが補うことはありません。属性の照会では、値を含めずに尋ねる属性名を `attribute` として必須にします。場面分類では `category_set` に2〜5個の異なる選択肢が必要です。文字転写（複数の文字ブロックやコードを含む）では、案の領域全体を文字の対象領域として結び付けます。

対象と観点からなるfact keyは、正規化した非公開のrequest keyになります。冠詞や `colour`／`color` の表記差では別の事実になりません。会話内の確定済みターンとrequest keyが一致する案は棄却し、起草時には確定済みのfact keyを一覧で渡して避けられるようにします。同じ対象でも毛色と鼻の色のように観点が違えば、別の事実として扱います。

物体同定の案では、対象を位置か、種類名以外の特徴で指します。回答の短い物体名が自分の質問文に既に含まれている場合は、評価の前に棄却します。再評価ではさらに、今回の対象名が確定済みの同定質問に既に現れ、今回の質問がそのターンの回答名を繰り返す相互参照も棄却します。このような双方向の公開がない別の対象は引き続き候補になります。

`visible_action` は静止画像に写る動作や接触を要求します。物体が何をできるかという明示的な質問は、案の受付で棄却します。車両の構造から想像できる動作も含みます。明示的でない能力の推測は両評価器でも確認し、静止した姿勢だけで移動を確定しません。身体の姿勢を尋ねる質問は属性照会です。ブレがないことだけを「静止している」根拠とはみなしません。

文字転写の質問では、画像内の絶対領域または案の領域を指定します。「ロゴの下」などの相対位置は、転写の根拠に座標がなく関係を確認できないため、案の受付で棄却します。保存済み会話の再評価でも同じ検査を適用します。場面分類の質問には、選択肢をすべて書く必要があります。

`evaluation.question_gate_label_policy: same_contract` では、もう一方のラベルが案のタスクと一致していれば、検証契約が同一の隣接タスクを評価器のラベルとして受け入れます。この場合、ラベルの違いは来歴の記録にしか影響しないためです。両評価器が同じ隣接タスクを挙げ、案がそのタスクの必須パラメーターをすでに持つ場合は、ターンのラベルを付け替えます。`strict` では両方のラベルが案のタスクと一致する必要があります。

2往復目という理由だけで履歴依存とは扱いません。依頼された再分類は、新しい画像上の事実がなくても有用です。画像から分からないことを尋ねる質問は `answerability_assessment` タスクです。「見つからない」だけで不在とは判断しません。

## 設定と予算

```yaml
tasks:
  catalog_version: "8.0"
  source_max_tokens: 4096
  answer_max_tokens: 1024
  enabled_extensions: []
  draft_count: 2
  draft_max_tokens: 1024
  extra_draft_calls_per_turn: 1
  profile_max_tokens: 384
  anchor_turns: 2
  family_targets: uniform
  task_weights:
    table_reconstruction: 0.25
    chart_data_reconstruction: 0.25
    document_structure_reconstruction: 0.25
```

`tasks.calibration_manifest` には専門タスクの校正manifestを指定でき、相対パスは設定ファイルの場所を基準に解決します。`family_targets` の未知の系統、`task_weights` の未知のタスク、未知または重複した拡張は検証エラーになります。廃止したscopedプランナーの設定も同様です。トークン上限で終了した公開質問・回答は、JSONの括弧を補えても拒否します。構造が正しいことや2モデルの一致だけでは、視覚的な正しさは保証できません。

物体同定、属性取得、動作、場面分類では、答えになる対象ラベルと自由記述の範囲説明をモデル向けの操作契約から除外します。操作契約には、画像view、scope領域、対象領域を含めます。人物の一般カテゴリは同定の対象にできますが、個人の特定は禁止です。座標の整合性だけでは画像内容の正しさを証明できず、動作・参照・分類粒度・閉じた集合の条件は引き続き盲検の質問・回答検証で確認します。

### 実行時の制約

- 想定される出力は、設定した応答と文脈の予算に収まる必要があります。収まらない場合は、質問で小さい領域を指定するか、確認済みの大きい予算を設定するか、そのタスクを使いません。途中で切れた回答を完全とは認定しません。
- 適格性は、モデルが実際に受け取る画像viewで判断し、座標も同じviewを基準にします。
- 該当する検証器が使えないタスクは提示しません。カタログに定義があるだけでは実行可能な実装にはなりません。
- パターンの規則は `pattern_rule` が列挙する規則の種類に限ります。妥当な答えが2つ当てはまる場合、検証器は棄権します。

## カタログ7.0からの変更

カタログ8.0では、すべての定義を平易な英語で書き直し、必須のパラメーターをすべて明示し、検証器が特定の回答形式を必要とするタスクに `answer_format` を加えました。残したタスクの検証の意味は変えていません。例外は `diagram_element_lookup` で、`graph_check` を外して軽いタスクにしました。

| 7.0のタスクID | 8.0のタスクID |
|---|---|
| `visible_action_relation` | `visible_action` |
| `visual_summary`、`screen_summary` | `grounded_description` に統合 |
| `referring_object_resolution` | `described_object_lookup` |
| `referring_expression_generation` | `distinguishing_description` |
| `predicate_selection`、`set_operation` | `select_by_conditions` |
| `attribute_grouping` | `group_by_attribute` |
| `set_cardinality_comparison` | `count_comparison` |
| `quantified_statement_verification` | `quantified_claim_verification` |
| `grounded_hypothetical_update` | `hypothetical_set_update` |
| `text_reading_order`、`code_transcription` | `text_transcription` に統合 |
| `label_value_linking` | `label_value_lookup` |
| `text_visual_binding` | `text_object_binding` |
| `cross_region_consistency_check` | `stated_value_consistency` |
| `document_layout_role` | `document_element_role` |
| `table_predicate_selection` | `table_row_selection` |
| `table_structure_reconstruction` | `table_reconstruction` |
| `table_cross_reference` | `table_join` |
| `grounded_aggregation` | `value_aggregation` |
| `graph_connectivity` | `diagram_connectivity` |
| `graph_path_tracing` | `diagram_path_tracing` |
| `diagram_branch_evaluation` | `flowchart_evaluation` |
| `geometric_relation_analysis` | `geometric_relations` |
| `pattern_rule_identification` | `pattern_rule` |
| `rule_based_exception` | `pattern_exception` |
| `ui_element_grounding` | `ui_element_location` |
| `visible_sequence_description` | `panel_sequence_description` |
| `evidence_localization` | `visual_claim_verification` に統合 |

6つの系統を改名しました。`set_logic` は `counting_and_sets`、`chart_map_understanding` は `charts_and_maps`、`diagram_graph` は `diagrams`、`pattern_geometry` は `patterns_and_geometry`、`multi_region` は `multi_panel`、`validated_extensions` は `specialist_notation` です。タスクの状態は `core` か `extension` です。

カタログからは、タスク番号、件数、facet、分類規則、割り当ての優先規則、回答プロファイル（`limitation`・`false_premise`）、方針だけのパラメーター、着想元subsetの番号、出典URLを削除しました。割り当ての優先規則は各定義へ、実行時の制約はこのガイドへ移し、出典URLは調査対応表に残しています。使われていなかった能力 `readable_code` も削除しました。

## 移行と集計

**新しいrun IDを使ってください。** 実行識別に設定、コード、プロンプト、Schema、カタログ、タスク契約の版（`scope-operations-v4`）、専門環境のロック、モデル・processorの版、seed、入力と権利manifestを含めます。カタログ8.0より前に書いた記録は8.0では検証を通りません。保存した指示はすべて削除した `profile` 項目を持ち、版付きの記録は7.0のタスクIDを使うためです。そのようなrunの確認、`replay`、再評価には、それを書いたコード（例: commit `6f224e0`）を使ってください。

生成・再評価は `.operations.json` へ全確定タスクと対話の最後の確定タスクを集計します。失敗した後半は含めません。選抜では最後の確定系統を主分類とし、意味的な識別には全タスク列を残します。exportのタスクID・カタログ版はprovenanceへ記録し、学習メッセージへ内部情報を入れません。検証用出典の除外と利用条件の検査も引き続き適用します。

選抜設定例は現在の系統名を使います。小さいquotaは制約の例です。既定の均等な系統目標も、実測で決めた最適配分ではありません。本番quotaは入力画像と実際の採用状況から決め、自然写真だけで文書や図の網羅性を保証しないでください。

## 来歴

カタログは、Onely7/Pixelogue の基盤ブランチ上の設計案と、FineVision の185 subsetの調査から始まりました。[調査対応表](finevision_mapping_185.md)には、その調査のタスク句と出典URLを残しています。各タスクの `related_finevision_subsets` は似たタスクを含むsubsetの一覧です。設計の記録であり、正解データではなく、モデルへの入力にも含めません。

## 確認範囲

CPUテストは共通契約、各構造化タスクのfixture、Schema・参照の拒否、質問案のパラメーター契約、割り当てと系統台帳の規則、質問ゲートのラベル規則、総合評価の追加処理、出力上限、実行識別、情報境界、生成したタスク一覧を確認します。`pixelogue task-status --junit artifacts/cpu-junit.xml` で共有fixtureの実行状況をタスク別に記録できますが、自然画像での精度を証明するものではありません。専門校正、評価者の誤り、言語間の品質、費用、学習効果は外部評価が必要です。既存のlive評価検査も元の少数事例だけを対象とします。
