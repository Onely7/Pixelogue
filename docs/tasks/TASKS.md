# Pixelogue task catalog 8.0

<!-- Generated from src/pixelogue/resources/task_catalog.yaml. Do not edit by hand; run: uv run --locked pixelogue compile --tasks-markdown docs/tasks/TASKS.md -->

[Runtime admission guide](README.md)

The catalog defines 78 tasks in 18 families: 71 core tasks that the question drafter may propose and 7 extensions that stay off until their specialized validator is configured and calibrated. A task is offered only when the image supports it and every verification contract it needs has a working implementation.

25 core tasks are light: they are checked only by `dual_visual_review`, `evidence_binding_check` or `transcript_alignment`. The opening turns of a conversation can be limited to light tasks with `tasks.anchor_turns`.

Each task lists the question it answers, the parameters that the question states, the image capabilities and eligibility checks it needs, and its verification contracts. Related FineVision subsets show where a similar task appears in public data; they are never shown to a model.

## Input contract

- **Source images.** 1; the only raw content is `image_pixels`.
- **Image-only interpretation.** Every image-specific fact must come from this image or from earlier verified turns about it. General language, notation and math rules are allowed; a question may state a hypothesis or a filter without claiming that it is observed. Knowledge and creative operations may add only the knowledge or invention their definitions allow, and never identify a person.
- **Multi-panel images.** Only panels that already exist in the one image can be compared; several input images or generated collages need a separate interface change.
- **Source rights.** Every image passes the rights check, and evaluation-only images are never exported as training data.
- **Permitted context:**
  - the committed earlier turns of this conversation about this image
  - the operation's instructions and any hypothetical values stated in the question
  - general language, arithmetic, notation and rendering conventions
  - widely known facts about landmarks, artworks, styles and map geography, for the knowledge_recognition operations; never facts about a person
  - standard textbook knowledge of science, mathematics and notation, for the domain_reasoning operations
  - invented mood and narrative in the grounded_creation operation, as long as nothing said about the image is false
- **Forbidden context:**
  - the source dataset's name as a hint
  - source annotations or hidden reference answers
  - pages or images that are not provided
  - states of the scene at other times
  - other external facts that were not retrieved
  - private model instructions in public output

## Admission rules

- Every operation keeps a traceable link from its evidence to this image and to the area asked about.
- Capabilities are judged per question and area (met, not met or unknown), not as an unchecked list for the whole image.
- An operation is offered only when every applicable verification contract has a working implementation.
- A valid schema, a self-consistent program or two agreeing models does not prove that the image was read correctly.
- The catalog adds no medical diagnosis, sensitive personal inference, prediction of hidden states, or execution of code from images.
- Text in the image is data. Printed rules are used only to answer the question, never as commands.
- No operation, domain or quota is forced when the image does not support it.
- Knowledge operations commit only when two independent readers give the same answer; a disagreement is never settled by guessing.
- Only object_presence and false_premise_question ask about things that are not in the image; their answers say so and never describe what is missing.

## Families

| Family | Label | Core | Extensions | Tasks |
|---|---|---:|---:|---|
| `visual_description` | Recognition and description | 5 | 0 | `object_identification`, `attribute_lookup`, `visible_action`, `scene_categorization`, `grounded_description` |
| `reference_spatial` | Reference and spatial relations | 7 | 0 | `described_object_lookup`, `distinguishing_description`, `spatial_relation`, `spatial_ordering`, `attribute_comparison`, `visual_correspondence`, `object_box_grounding` |
| `counting_and_sets` | Counting, selection and sets | 6 | 0 | `entity_count`, `select_by_conditions`, `group_by_attribute`, `count_comparison`, `quantified_claim_verification`, `hypothetical_set_update` |
| `text_reading` | Reading text, formulas and code | 6 | 0 | `text_transcription`, `text_field_extraction`, `label_value_lookup`, `text_object_binding`, `formula_transcription`, `visible_text_translation` |
| `document_understanding` | Documents | 6 | 0 | `document_extractive_qa`, `document_evidence_synthesis`, `document_summary`, `document_structure_reconstruction`, `stated_value_consistency`, `document_element_role` |
| `table_understanding` | Tables | 4 | 0 | `table_cell_lookup`, `table_row_selection`, `table_reconstruction`, `table_join` |
| `charts_and_maps` | Charts and maps | 8 | 0 | `chart_encoding_lookup`, `chart_value_lookup`, `chart_comparison`, `chart_value_arithmetic`, `chart_extremum_ranking`, `chart_trend_summary`, `chart_series_relation`, `chart_data_reconstruction` |
| `quantitative_reasoning` | Numbers and measurements | 5 | 0 | `quantity_comparison`, `grounded_arithmetic`, `value_aggregation`, `unit_conversion`, `measurement_reading` |
| `diagrams` | Diagrams and flowcharts | 5 | 1 | `diagram_element_lookup`, `diagram_connectivity`, `diagram_path_tracing`, `diagram_process_description`, `flowchart_evaluation`, `diagram_to_code` |
| `patterns_and_geometry` | Patterns and geometry | 4 | 1 | `geometric_relations`, `pattern_rule`, `pattern_completion`, `pattern_exception`, `geometric_constraint_solving` |
| `screen_ui` | Screens and user interfaces | 2 | 2 | `ui_element_location`, `ui_state_reading`, `screen_to_code`, `ui_action_specification` |
| `multi_panel` | Multi-panel images | 2 | 0 | `panel_comparison`, `panel_sequence_description` |
| `evidence_verification` | Claims and answerability | 2 | 0 | `visual_claim_verification`, `answerability_assessment` |
| `presence_and_premises` | Object presence and false premises | 2 | 0 | `object_presence`, `false_premise_question` |
| `knowledge_recognition` | Recognition with world knowledge | 3 | 0 | `named_entity_recognition`, `style_recognition`, `map_region_identification` |
| `domain_reasoning` | Reasoning with specialist knowledge | 3 | 0 | `concept_explanation`, `notation_interpretation`, `math_word_problem` |
| `grounded_creation` | Creative writing from the image | 1 | 0 | `grounded_creative_writing` |
| `specialist_notation` | Music, chemistry and circuit notation | 0 | 3 | `music_notation_reading`, `chemical_structure_reading`, `circuit_structure_reading` |

## Tasks

### Recognition and description (`visual_description`)

#### `object_identification`: Identify an object

The user asks what a visible thing is; the answer names its ordinary category as precisely as its appearance shows, such as 'a golden retriever' or 'a coffee mug'. People are named only by a generic category such as 'a child' or 'a cyclist'. Use text_transcription instead for reading a printed name.

- **Status.** core; light verification.
- **Example question.** What is the object on the left side of the desk?
- **Do not infer.** A person's identity, a brand or exact product model that is not printed, or a category that the visible features do not support.
- **Required capabilities.** `visible_entity`.
- **Eligibility checks.** `scope_resolved`, `visible_category_supported`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `cocoqa`, `lvis_instruct4v`, `vqav2`, `objects365_qa`.
- **Parameters.** None.

#### `attribute_lookup`: Read a visible attribute

The user names one property of a visible object, such as its color, shape, material, relative size, posture or state (open, lit, broken); the answer states that property. Use visible_action instead for what a person or animal is doing.

- **Status.** core; light verification.
- **Example question.** What color is the umbrella that the woman is holding?
- **Do not infer.** Feelings, intentions, health, age, real-world measurements, or any hidden or sensitive personal attribute.
- **Required capabilities.** `visible_entity`, `visible_attribute`.
- **Eligibility checks.** `scope_resolved`, `attribute_visible`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `coco_colors`, `cocoqa`, `lnqa`, `vqav2`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `attribute` | yes | text | The property being asked about, such as 'color' or 'what is on the person's head'; never the property's value. |

#### `visible_action`: Describe a visible action

The user asks what a visible person or animal is doing, or how it interacts with an object; the answer describes only the action or contact shown in this still image. Use attribute_lookup instead for posture alone and spatial_relation for where things are placed.

- **Status.** core; light verification.
- **Example question.** What is the man in the red jacket doing with the rope?
- **Do not infer.** What happened before or will happen next, intentions, speed, or abilities such as what an object could do.
- **Required capabilities.** `visible_entity`, `visible_interaction`.
- **Eligibility checks.** `scope_resolved`, `action_visually_supported`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `sharegpt4v(coco)`, `LLaVA_Instruct_150K`, `vqav2`, `drivelm`.
- **Parameters.** None.

#### `scene_categorization`: Classify the scene

The user lists two to five distinct kinds of scene, one of which matches the image, such as 'kitchen, office or street', and asks which one it shows; the answer picks the option that the visible evidence supports.

- **Status.** core; light verification.
- **Example question.** Is this scene a kitchen, an office or a street?
- **Do not infer.** A city, a landmark name or an exact location from outside knowledge.
- **Required capabilities.** `scene_context`.
- **Eligibility checks.** `scope_resolved`, `visible_category_supported`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `indoor_qa`, `vision_flan(filtered)`, `vqav2`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `category_set` | yes | list of 2 to 5 | The scene options that the question offers, written exactly as in the question, for example ['kitchen', 'office', 'street']. |

#### `grounded_description`: Describe the image or a region

The user asks for a brief, standard or detailed description or summary of the whole image, a screen or a named region; the answer covers only visible content, such as objects, people, text, layout and interface controls, and how they are arranged.

- **Status.** core; light verification.
- **Example question.** Briefly describe what this screenshot shows.
- **Do not infer.** Backstories, purposes, identities, emotions, or objects and events that are not visible.
- **Required capabilities.** `resolvable_region`, `multiple_facts`.
- **Eligibility checks.** `scope_resolved`, `description_claims_visible`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `densefusion_1m`, `image_textualization(filtered)`, `laion_gpt4v`, `localized_narratives`, `sharegpt4o`, `sharegpt4v(llava)`, `sharegpt4v(sam)`, `textcaps`, `LLaVA_Instruct_150K`, `websight`, `screen2words`, `screenqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `detail_level` | yes | one of `brief`, `standard`, `detailed` | How much detail the question asks for: brief is one or two sentences, detailed covers all notable visible content. |

### Reference and spatial relations (`reference_spatial`)

#### `described_object_lookup`: Find the object that matches a description

The user describes one object by its appearance, position or relation to other objects, such as 'the cup right of the plate with a blue handle'; the answer says which object that is, by its name, label or position, or says that the description fits several objects. Use false_premise_question when no object fits.

- **Status.** core; light verification.
- **Example question.** Which cup is to the right of the plate and has a blue handle?
- **Do not infer.** Picking one of several matching objects without saying that the description is ambiguous.
- **Required capabilities.** `multiple_entities`, `discriminating_attributes`.
- **Eligibility checks.** `scope_resolved`, `unique_referent`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `visual7w`, `lvis_instruct4v`, `clevr`, `super_clevr(mathv360k)`.
- **Parameters.** None.

#### `distinguishing_description`: Describe an object so it can be told apart

The user points to one object by a visible label, color or position and asks for a short description that distinguishes it from the other similar objects; the answer gives features or relations that only that object has.

- **Status.** core; light verification.
- **Example question.** Describe the dog lying on the bench so that it cannot be confused with the other dogs.
- **Do not infer.** Hidden IDs, invisible markers, or features that the other similar objects share.
- **Required capabilities.** `multiple_entities`, `discriminating_attributes`.
- **Eligibility checks.** `scope_resolved`, `unique_referent`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `localized_narratives`, `lvis_instruct4v`, `visual7w`.
- **Parameters.** None.

#### `spatial_relation`: State where one object is relative to another

The user asks where one visible object is relative to another, such as left of, above, in front of or inside; the answer states the relation from the viewpoint that the question names.

- **Status.** core; light verification.
- **Example question.** As seen in the image, is the bicycle to the left or to the right of the tree?
- **Do not infer.** Distances in real units, or left and right that depend on a viewpoint the question does not state.
- **Required capabilities.** `multiple_entities`, `spatial_layout`.
- **Eligibility checks.** `scope_resolved`, `relation_frame_defined`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `spatialsense`, `vsr`, `clevr`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `frame` | yes | one of `image`, `object`, `depth` | The viewpoint: image means left, right, above and below as seen in the picture; object means relative to the named object's own front and sides; depth means nearer to or farther from the camera. |

#### `spatial_ordering`: Order objects by position

The user asks to list every object of a stated kind in a given direction, such as left to right or nearest to farthest; the answer gives the complete order and mentions any ties.

- **Status.** core; structured verification.
- **Example question.** List the labeled boxes from left to right.
- **Do not infer.** An order for objects that overlap or whose positions cannot be told apart.
- **Required capabilities.** `multiple_entities`, `spatial_layout`, `closed_scope`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`.
- **Related FineVision subsets.** `spatialsense`, `clevr`, `super_clevr(mathv360k)`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `direction` | yes | one of `left_to_right`, `right_to_left`, `top_to_bottom`, `bottom_to_top`, `nearest_to_farthest`, `farthest_to_nearest` | The direction of the order, as seen in the image. |

#### `attribute_comparison`: Compare an attribute across objects

The user asks how two or more visible objects compare in a visible property, such as color, shape or relative size; the answer states their similarities or differences.

- **Status.** core; light verification.
- **Example question.** How do the two bags differ in color and shape?
- **Do not infer.** Real-world size from apparent size, quality, value, or other properties that are not visible.
- **Required capabilities.** `multiple_entities`, `comparable_attributes`.
- **Eligibility checks.** `scope_resolved`, `comparable_basis`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `mimic_cgd`, `mmra`, `vqav2`, `clevr`.
- **Parameters.** None.

#### `visual_correspondence`: Follow a drawn link or key

The user asks what a visible line, arrow, number, letter or legend entry connects or refers to; the answer follows that explicit link, for example from label B to the picture it points at.

- **Status.** core; light verification.
- **Example question.** Which picture is connected to label B?
- **Do not infer.** A link based only on closeness, alignment or similar appearance.
- **Required capabilities.** `explicit_mapping`.
- **Eligibility checks.** `scope_resolved`, `association_explicit`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `CoSyn_400k_graphic`, `spatialsense`, `clevr`, `ai2d_merged`.
- **Parameters.** None.

#### `object_box_grounding`: Give the box of an object

The user describes one object, or every object of a stated kind, and asks for its bounding box; the answer gives one tight box per object as fractions of the image width and height. Use described_object_lookup instead to say which object a description means.

- **Status.** core; structured verification.
- **Example question.** Give the bounding box of the red car on the left.
- **Answer format.** One JSON object of the form {"boxes": [[left, top, right, bottom]]} with one box per requested object, listed from left to right; each value is a fraction from 0 to 1 of the image width or height, with left smaller than right and top smaller than bottom.
- **Do not infer.** Boxes for hidden or cut-off parts, or for objects that only resemble the description.
- **Required capabilities.** `box_targets`.
- **Eligibility checks.** `scope_resolved`, `boxes_definable`.
- **Verification contracts.** `dual_visual_review`, `box_iou_check`.
- **Related FineVision subsets.** `objects365_qa`, `groundui`, `drivelm`, `lvis_instruct4v`.
- **Parameters.** None.

### Counting, selection and sets (`counting_and_sets`)

#### `entity_count`: Count objects

The user asks how many objects of a stated kind are in the image or a named area; the answer gives the exact number, zero only when that kind is visible but none meets a stated condition. Objects can also be chart marks, table rows or symbols in a labeled region of a diagram. Use false_premise_question if that kind is absent; count_comparison compares two counts.

- **Status.** core; structured verification.
- **Example question.** How many red cups are on the table?
- **Do not infer.** Objects hidden or cut off by the frame, or a different unit than the one asked about, such as petals instead of flowers.
- **Required capabilities.** `closed_scope`, `countable_entities`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `count_unit_defined`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`.
- **Related FineVision subsets.** `cocoqa`, `oodvqa`, `tallyqa`, `clevr`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `count_unit` | yes | text | What counts as one item, such as 'each apple' or 'each person, including partly hidden ones'. |

#### `select_by_conditions`: List the objects that meet stated conditions

The user states one or more visible conditions, such as color, shape, printed text or position, combined with and, or and not; the answer lists every object in the named area that meets them. Objects can also be chart marks, table rows or symbols in the regions of a Venn diagram.

- **Status.** core; structured verification.
- **Example question.** Which shapes are blue but not circles?
- **Do not infer.** Conditions that cannot be checked by looking, or a partial list when the area is not fully visible.
- **Required capabilities.** `multiple_entities`, `interpretable_predicates`, `closed_scope`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `predicates_observable`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`.
- **Related FineVision subsets.** `tallyqa`, `clevr`, `clevr_math`, `super_clevr(mathv360k)`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `conditions` | yes | text | The visible conditions an object must meet, combined with and, or, not, for example 'red and not round'. |

#### `group_by_attribute`: Group objects by an attribute

The user asks to group all objects of a stated kind by one visible attribute, such as color; the answer names every group together with its members or its count.

- **Status.** core; structured verification.
- **Example question.** Group the buttons by color and give the number of buttons in each group.
- **Do not infer.** Group names without saying which objects belong to them, or groups based on properties that are not visible.
- **Required capabilities.** `multiple_entities`, `visible_attribute`, `closed_scope`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `grouping_key_defined`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`.
- **Related FineVision subsets.** `coco_colors`, `tallyqa`, `clevr`, `super_clevr(mathv360k)`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `group_key` | yes | text | The visible attribute that defines the groups, such as 'color' or 'shape'. |
| `return` | yes | one of `members`, `counts` | Whether each group lists its members or only how many members it has. |

#### `count_comparison`: Compare two counts

The user asks whether there are more, fewer or equally many objects of one kind than of another; the answer compares the two exact counts, not the space the objects take up.

- **Status.** core; structured verification.
- **Example question.** Are there more cups than plates on the table?
- **Do not infer.** A comparison based on area or density instead of counting.
- **Required capabilities.** `closed_scope`, `countable_entities`, `interpretable_predicates`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `count_unit_defined`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `iconqa`, `tallyqa`, `clevr`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `count_unit` | yes | text | What counts as one item in each group, such as 'each cup' and 'each plate'. |

#### `quantified_claim_verification`: Check an all, none or some claim

The user asks whether a statement using all, none, some, exactly, at least or at most holds for a fully visible group of objects; the answer says whether it holds and gives the counts behind it. Use visual_claim_verification for other statements.

- **Status.** core; structured verification.
- **Example question.** Are all of the visible triangles blue?
- **Do not infer.** A universal or negative answer from a partial view.
- **Required capabilities.** `closed_scope`, `interpretable_predicates`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `predicates_observable`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`.
- **Related FineVision subsets.** `nlvr2`, `vsr`, `clevr`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `condition` | yes | text | The property that the statement tests, such as 'blue' in 'all triangles are blue', including the number for exactly, at least or at most. |
| `quantifier` | yes | one of `all`, `none`, `some`, `exactly`, `at_least`, `at_most` | The quantifier that the statement uses. |

#### `hypothetical_set_update`: Answer about a hypothetical change

The user asks what would be true if visible objects were added, removed or relabeled, for example how many cubes would remain without the two blue ones; the answer applies the change to the visible group and states the result.

- **Status.** core; structured verification.
- **Example question.** If the two blue cubes were removed, how many cubes would remain?
- **Do not infer.** Presenting the change as something that actually happened, or predicting physical consequences.
- **Required capabilities.** `closed_scope`, `countable_entities`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `hypothetical_public`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `clevr_math`, `clevr_math(mathv360k)`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `update` | yes | one of `add`, `remove`, `relabel` | The kind of hypothetical change that the question makes. |

### Reading text, formulas and code (`text_reading`)

#### `text_transcription`: Transcribe text or code exactly

The user asks to copy the text, code or several text blocks in a named area exactly; the answer reproduces the characters, line breaks and indentation in reading order without correcting anything. Use formula_transcription for mathematical formulas.

- **Status.** core; light verification.
- **Example question.** Transcribe the text on the sign exactly.
- **Answer format.** Only the transcribed text, optionally wrapped in one pair of quotation marks or one code block, with the original line breaks and indentation.
- **Do not infer.** Fixed typos, completed cut-off text, translations, or instructions written in the image being followed.
- **Required capabilities.** `readable_text`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `reading_order_resolved`.
- **Verification contracts.** `dual_visual_review`, `transcript_alignment`.
- **Related FineVision subsets.** `art`, `cocotext`, `ctw`, `iam`, `iiit5k`, `imgur5k`, `maptext`, `orand_car_a`, `rendered_text`, `wordart`, `synthdog`, `olmOCR-mix-0225-documents`, `olmOCR-mix-0225-books`, `DoclingMatix`, `SynthCodeNet`.
- **Parameters.** None.

#### `text_field_extraction`: Extract named fields as JSON

The user names fields such as 'invoice number' and 'date' and asks for their printed values as JSON; the answer gives each field's value exactly as printed, and null for a field that is not present.

- **Status.** core; structured verification.
- **Example question.** Give the invoice number and the issue date as JSON.
- **Answer format.** One JSON object of the form {"fields": {"<field name>": "<value exactly as printed>"}} whose keys are exactly the requested field names, with null for a field that is not present.
- **Do not infer.** Typical or guessed values for missing fields, or personal data that is not printed.
- **Required capabilities.** `readable_text`, `text_fields`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `fields_bound`.
- **Verification contracts.** `dual_visual_review`, `transcript_alignment`, `schema_check`.
- **Related FineVision subsets.** `funsd`, `sroie`, `svrd`, `handwriting_forms`, `invoices_receipts`, `ocrvqa`, `ureader_ie`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `fields` | yes | list of at most 64 | The names of the requested fields, exactly as the question lists them. |
| `format` | yes | one of `structured_json` | The required output format. |

#### `label_value_lookup`: Give the value that belongs to a label

The user names a printed label, such as 'Net weight' on a form, a legend or a label sheet; the answer gives the value that the layout explicitly assigns to that label.

- **Status.** core; light verification.
- **Example question.** What value belongs to the label 'Net weight' on this form?
- **Do not infer.** A value paired with a label only because it is nearby.
- **Required capabilities.** `readable_text`, `explicit_mapping`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `association_explicit`.
- **Verification contracts.** `dual_visual_review`, `transcript_alignment`.
- **Related FineVision subsets.** `funsd`, `sroie`, `svrd`, `handwriting_forms`, `ureader_ie`.
- **Parameters.** None.

#### `text_object_binding`: Relate text to what it labels

The user asks which pictured object a piece of text names or describes, or whether the text agrees with the picture; the answer links the text to the object or region using visible evidence.

- **Status.** core; light verification.
- **Example question.** Which product on the shelf carries the label 'Green tea'?
- **Do not infer.** Jokes, the author's intentions, or links between text and objects that the layout does not show.
- **Required capabilities.** `readable_text`, `text_object_alignment`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `association_explicit`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `textcaps`, `chinesememe`, `llavar_gpt4_20k`, `est_vqa`, `st_vqa`, `textocr(gpt4v)`, `textvqa`.
- **Parameters.** None.

#### `formula_transcription`: Transcribe a formula

The user asks to write a visible mathematical formula in LaTeX or Unicode math; the answer copies its symbols, grouping, fractions, roots and sub- and superscripts exactly, without solving or simplifying it.

- **Status.** core; structured verification.
- **Example question.** Write the displayed equation in LaTeX without solving it.
- **Answer format.** Use only symbols, numbers, grouping, arithmetic, equality, sub- and superscripts, fractions and square roots; LaTeX display delimiters around the whole formula are allowed. Matrices, chemical diagrams and other commands are not supported, so choose another operation for them.
- **Do not infer.** A corrected formula, or an equivalent formula written differently.
- **Required capabilities.** `readable_formula`.
- **Eligibility checks.** `scope_resolved`, `notation_resolved`.
- **Verification contracts.** `dual_visual_review`, `formula_structure_check`.
- **Related FineVision subsets.** `chrome_writting`, `hme100k`, `k12_printing`, `latex_handwritten`, `latexformulas`, `mathwriting-google`, `SynthFormulaNet`, `tal_ocr_eng`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `notation` | yes | one of `latex`, `unicode_math` | The notation that the answer must use. |

#### `visible_text_translation`: Translate visible text

The user asks to translate a clearly delimited piece of visible text into a stated language; the answer gives a faithful translation that keeps names, numbers and meaning without adding content. Use text_transcription instead to copy the text in its original language.

- **Status.** core; light verification.
- **Example question.** Translate the shop sign above the door into English.
- **Answer format.** Only the translation, optionally followed by the original text in parentheses.
- **Do not infer.** Unreadable characters, text that is not shown, or explanations presented as part of the translation.
- **Required capabilities.** `readable_text`, `translatable_text`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `translation_language_stated`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `art`, `ctw`, `est_vqa`, `chinesememe`, `k12_printing`, `tal_ocr_eng`, `svrd`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `into_language` | yes | text | The language the text is translated into, such as 'English' or 'Japanese'. |

### Documents (`document_understanding`)

#### `document_extractive_qa`: Answer with the exact text from a document

The user asks a question that a visible document answers directly; the answer quotes the shortest complete passage that answers it, in the document's language, keeping its qualifiers and negations.

- **Status.** core; light verification.
- **Example question.** According to this notice, when does registration close?
- **Answer format.** The exact passage from the document, in its language; only the case of the first letter and a final period may differ. No translation, paraphrase or added claims.
- **Do not infer.** Answers from outside knowledge or from pages that are not shown.
- **Required capabilities.** `readable_prose`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `answer_evidence_present`.
- **Verification contracts.** `dual_visual_review`, `transcript_alignment`.
- **Related FineVision subsets.** `hw_squad`, `bentham`, `docvqa`, `pdfvqa`, `screenqa`, `ureader_qa_processed`.
- **Parameters.** None.

#### `document_evidence_synthesis`: Answer by combining parts of a document

The user asks a question whose answer needs two or more separate parts of the same document, such as a table and a note below it; the answer combines them instead of copying one passage.

- **Status.** core; light verification.
- **Example question.** Using the table and the note below it, which entries are included in the total?
- **Do not infer.** Facts from other pages, or causes and conclusions that the document does not state.
- **Required capabilities.** `multiple_evidence_regions`, `readable_prose`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `all_evidence_on_canvas`.
- **Verification contracts.** `dual_visual_review`, `evidence_binding_check`.
- **Related FineVision subsets.** `finqa`, `multihiertt`, `tat_dqa`, `tat_qa`, `infographic_vqa`, `slidevqa`, `visualmrc`.
- **Parameters.** None.

#### `document_summary`: Summarize a document

The user asks for the main points of a visible document or of one of its sections; the answer summarizes what the document says, attributing the claims to the document.

- **Status.** core; light verification.
- **Example question.** Summarize the three main points of this notice.
- **Do not infer.** Background that the document does not give, dropped negations, or the document's claims treated as verified facts.
- **Required capabilities.** `readable_prose`, `multiple_facts`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `summary_entails_evidence`.
- **Verification contracts.** `dual_visual_review`, `evidence_binding_check`.
- **Related FineVision subsets.** `DoclingMatix`, `sujet_finance`, `ureader_cap`, `ureader_kg_processed`, `visualmrc`.
- **Parameters.** None.

#### `document_structure_reconstruction`: Convert a document page to structured JSON

The user asks to convert a visible page, or a named part of it, into a JSON outline of its headings, paragraphs, lists, tables, formulas, captions and footnotes in reading order; the answer keeps the text and the hierarchy.

- **Status.** core; structured verification.
- **Example question.** Convert this page into a JSON outline of its headings, paragraphs and lists in reading order.
- **Answer format.** One JSON object of the form {"nodes": [{"node_id": "n1", "kind": "heading", "text": "...", "parent_id": null, "order": 0}]}. kind is one of heading, paragraph, list, list_item, table, table_row, table_cell, formula, caption or footnote; order is the reading position as a unique integer; parent_id is null or names a node that comes earlier.
- **Do not infer.** Sections that are not on the page, unseen table cells, or document metadata.
- **Required capabilities.** `document_layout`, `readable_text`, `reading_order`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `reading_order_resolved`, `schema_public`.
- **Verification contracts.** `dual_visual_review`, `transcript_alignment`, `schema_check`.
- **Related FineVision subsets.** `synthdog`, `olmOCR-mix-0225-documents`, `olmOCR-mix-0225-books`, `DoclingMatix`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `format` | yes | one of `structured_json` | The required output format. |

#### `stated_value_consistency`: Check whether two stated values agree

The user points to two places in the same image that state the same quantity or fact, such as a total in a paragraph and in a table; the answer says whether they agree, keeping units, dates and scope.

- **Status.** core; structured verification.
- **Example question.** Does the total in the paragraph match the total in the table?
- **Do not infer.** Different periods or units treated as the same, or a judgment of whether a claim is true in the real world.
- **Required capabilities.** `multiple_evidence_regions`, `explicit_cross_references`.
- **Eligibility checks.** `scope_resolved`, `all_evidence_on_canvas`, `comparable_basis`.
- **Verification contracts.** `dual_visual_review`, `evidence_binding_check`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `finqa`, `multihiertt`, `tat_dqa`, `tat_qa`, `infographic_vqa`, `sujet_finance`.
- **Parameters.** None.

#### `document_element_role`: Identify the role of a document element

The user asks what role a part of a document plays, or which part has a given role, such as title, heading, caption, footnote or table header; the answer decides from the layout and the text together.

- **Status.** core; light verification.
- **Example question.** Which text is the caption of the figure?
- **Do not infer.** A role based on font size alone, or document structure that is not visible.
- **Required capabilities.** `document_layout`, `readable_text`.
- **Eligibility checks.** `scope_resolved`, `text_legible`, `role_visually_supported`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `olmOCR-mix-0225-documents`, `olmOCR-mix-0225-books`, `DoclingMatix`, `pdfvqa`.
- **Parameters.** None.

### Tables (`table_understanding`)

#### `table_cell_lookup`: Look up a table cell

The user names a row and a column of a visible table, including group headers when present; the answer gives the cell's value exactly as printed, with its unit, without calculating anything.

- **Status.** core; structured verification.
- **Example question.** What value is listed for Europe in 2024 under Revenue?
- **Do not infer.** An ignored parent header, unit, footnote or merged cell.
- **Required capabilities.** `readable_table`, `table_headers`.
- **Eligibility checks.** `scope_resolved`, `table_headers_bound`, `text_legible`.
- **Verification contracts.** `dual_visual_review`, `table_structure_check`.
- **Related FineVision subsets.** `CoSyn_400k_table`, `hitab`, `robut_wikisql`, `robut_wtq`, `vqaonbd`.
- **Parameters.** None.

#### `table_row_selection`: Select table rows by conditions

The user states conditions on the columns of a visible table, optionally with a sort order; the answer lists every row that meets them, in that order.

- **Status.** core; structured verification.
- **Example question.** Which rows have a quantity above 10? List their names from the largest to the smallest quantity.
- **Answer format.** The labels of the selected rows, in the requested order.
- **Do not infer.** Conditions on columns that the table does not have, or only some of the matching rows.
- **Required capabilities.** `readable_table`, `table_headers`, `closed_scope`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `table_headers_bound`, `predicates_observable`.
- **Verification contracts.** `dual_visual_review`, `table_structure_check`, `closed_set_check`.
- **Related FineVision subsets.** `CoSyn_400k_table`, `hitab`, `robut_wikisql`, `robut_wtq`, `tabmwp`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `conditions` | yes | text | The column conditions and any requested order, for example 'quantity above 10, largest quantity first'. |

#### `table_reconstruction`: Convert a table to HTML, JSON or Markdown

The user asks to convert a visible table into HTML, JSON or simple Markdown; the answer reproduces every cell, header level, merged cell and blank cell.

- **Status.** core; structured verification.
- **Example question.** Convert this table to HTML, keeping its grouped column headings.
- **Answer format.** For html_table, one static table using only table, thead, tbody, tfoot, tr, th and td tags, with rowspan and colspan, br for line breaks inside a cell, scope on th (row, col, rowgroup or colgroup), numeric border, cellpadding and cellspacing on table, and only these styles: border-collapse (collapse or separate), text-align (left, right, center, start, end or justify) and vertical-align (top, middle or bottom); no scripts, event handlers, external assets or other tags. For structured_json, one object {"table_id", "rows", "cols", "cells", "data_rows", "closed"} whose cells each have "row", "col", "rowspan", "colspan", "text" and "kind" ("header" or "data"). Keep blank cells, header and data roles, and merged-cell spans.
- **Do not infer.** Invented cells, collapsed group headers, or blank cells turned into zeros.
- **Required capabilities.** `readable_table`, `table_headers`.
- **Eligibility checks.** `scope_resolved`, `table_headers_bound`, `schema_public`.
- **Verification contracts.** `dual_visual_review`, `table_structure_check`, `schema_check`.
- **Related FineVision subsets.** `CoSyn_400k_table`, `hitab`, `vqaonbd`, `DoclingMatix`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `format` | yes | one of `html_table`, `structured_json`, `markdown_simple_only` | The output format; markdown_simple_only is only for tables without merged cells and with one header row. |

#### `table_join`: Join two tables by a shared key

The user asks to match the rows of two or more tables in the same image through a shared identifier, such as a product ID; the answer pairs the matching records.

- **Status.** core; structured verification.
- **Example question.** Using the product IDs, which category in the right table belongs to each item in the left table?
- **Answer format.** The matched pairs, each naming a row of one table and the matching row or value of the other.
- **Do not infer.** Rows matched by position, or a table that is not shown.
- **Required capabilities.** `multiple_tables`, `table_join_keys`.
- **Eligibility checks.** `scope_resolved`, `all_evidence_on_canvas`, `join_keys_unique`.
- **Verification contracts.** `dual_visual_review`, `table_structure_check`, `evidence_binding_check`.
- **Related FineVision subsets.** `multihiertt`, `tat_dqa`, `vqaonbd`.
- **Parameters.** None.

### Charts and maps (`charts_and_maps`)

#### `chart_encoding_lookup`: Explain an axis, legend or map key

The user asks what an axis, legend color, symbol, map shade or series label stands for; the answer reads it from the chart's own labels and legend.

- **Status.** core; light verification.
- **Example question.** What range does the darkest color on the map represent?
- **Do not infer.** Geographic or subject knowledge that the chart's labels and legend do not give.
- **Required capabilities.** `readable_chart`, `chart_encoding`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `dvqa`, `mmc_instruct`, `plotqa`, `mapqa`, `mapqa(mathv360k)`.
- **Parameters.** None.

#### `chart_value_lookup`: Read a value from a chart

The user asks for the value of one bar, point, slice or map region; the answer gives the printed value, or an estimate at the stated precision.

- **Status.** core; structured verification.
- **Example question.** What value does the chart show for series A in March?
- **Do not infer.** Exact decimals made up from pixel positions, ignored log axes, or exact values from a color range.
- **Required capabilities.** `readable_chart`, `chart_encoding`, `legible_values`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`, `precision_declared`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`.
- **Related FineVision subsets.** `chartqa`, `dvqa`, `plotqa`, `Unichart`, `mapqa`, `mapqa(mathv360k)`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `precision` | yes | one of `explicit_label`, `calibrated_estimate`, `interval` | explicit_label means a value printed on the chart; calibrated_estimate means a value read against labeled axis ticks; interval means a range such as 'between 20 and 30'. |

#### `chart_comparison`: Compare chart values

The user asks which of two or more named bars, points or categories is larger or smaller on the same chart; the answer compares them without inventing an exact difference.

- **Status.** core; structured verification.
- **Example question.** Which series is higher in 2024?
- **Do not infer.** Heights compared across different axes or scales.
- **Required capabilities.** `readable_chart`, `comparable_series`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`, `comparable_basis`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`.
- **Related FineVision subsets.** `chartqa`, `figureqa`, `figureqa(mathv360k)`, `plotqa`, `Unichart`, `mapqa`.
- **Parameters.** None.

#### `chart_value_arithmetic`: Calculate with chart values

The user asks for the sum, difference, product, ratio or mean of two or more values read from the same chart, naming each value by its series and category; the answer gives the result, exact for printed values and inside the range the readings allow otherwise.

- **Status.** core; structured verification.
- **Example question.** What is the difference between the 2015 and the 2010 values of the blue series?
- **Answer format.** One number with the chart's unit (no unit for a ratio). For printed values give the exact result or round it as the question states; for estimates give one value inside the range the readings allow.
- **Do not infer.** Exact decimals made up from pixel positions, values that are not plotted, or operands from another chart.
- **Required capabilities.** `readable_chart`, `chart_encoding`, `legible_values`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`, `expression_defined`, `precision_declared`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`.
- **Related FineVision subsets.** `dvqa`, `plotqa`, `chartqa`, `figureqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `operator` | yes | one of `add`, `subtract`, `multiply`, `divide`, `sum`, `mean` | The calculation: add, subtract, multiply or divide two values, or the sum or mean of several. |
| `precision` | yes | one of `explicit_label`, `calibrated_estimate`, `interval` | explicit_label means every value is printed on the chart; calibrated_estimate means values read against labeled axis ticks; interval means coarse ranges. |

#### `chart_extremum_ranking`: Find the largest, smallest or ranked entries

The user asks for the highest or the lowest entry, both, or a full ranking of all entries of a chart; the answer names them and includes any ties.

- **Status.** core; structured verification.
- **Example question.** Which category has the highest value? Include any ties.
- **Answer format.** Name the entries in the requested order and include every tie; for max_min give only the largest and then the smallest.
- **Do not infer.** A largest value taken from a partial view of the series.
- **Required capabilities.** `readable_chart`, `complete_series`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `encoding_resolved`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`, `closed_set_check`.
- **Related FineVision subsets.** `chart2text`, `CoSyn_400k_chart`, `figureqa`, `figureqa(mathv360k)`, `vistext`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `rank_mode` | yes | one of `all`, `max`, `min`, `max_min` | max asks for the largest entry or entries, min for the smallest, max_min for both (largest first), and all for the complete order. |
| `rank_order` | yes | one of `descending`, `ascending` | The order of the answer when rank_mode is all. |

#### `chart_trend_summary`: Describe a trend

The user asks how a series changes over the chart's range, such as rises, falls, plateaus and turning points; the answer describes the visible trend.

- **Status.** core; structured verification.
- **Example question.** Describe how the values change across the displayed years.
- **Do not infer.** Forecasts beyond the shown range, or causes of the trend.
- **Required capabilities.** `readable_chart`, `ordered_series`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`, `summary_entails_evidence`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`.
- **Related FineVision subsets.** `chart2text`, `CoSyn_400k_chart`, `Unichart`, `vistext`.
- **Parameters.** None.

#### `chart_series_relation`: Describe how two series relate

The user asks whether two or more series on the same axes cross, which one stays higher, or how their variation compares; the answer states the relation at the precision the chart allows.

- **Status.** core; structured verification.
- **Example question.** Do the two lines cross within the displayed years? If so, approximately where?
- **Do not infer.** Exact crossing coordinates or areas that the chart's precision does not support.
- **Required capabilities.** `readable_chart`, `comparable_series`, `complete_series`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`, `comparable_basis`, `precision_declared`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`.
- **Related FineVision subsets.** `figureqa`, `figureqa(mathv360k)`, `mmc_instruct`, `plotqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `precision` | yes | one of `explicit_label`, `calibrated_estimate`, `interval` | explicit_label means printed values; calibrated_estimate means values read against labeled axis ticks; interval means ranges. |
| `relation` | yes | one of `intersection`, `dominance`, `variation` | intersection asks whether and where the series cross; dominance asks which stays higher; variation asks which changes more. |

#### `chart_data_reconstruction`: Convert a chart to JSON data

The user asks to turn a chart's categories, series and values into JSON; the answer lists every visible value, marking estimates or ranges where exact values are not printed.

- **Status.** core; structured verification.
- **Example question.** Convert the labeled bars and their values into JSON.
- **Answer format.** One JSON object of the form {"unit": "<axis unit or null>", "marks": [{"series": "...", "category": "...", "lower": "12.5", "upper": "12.5", "precision": "explicit_label"}]} with one entry per visible mark. lower and upper are plain decimal strings; they are equal for a printed value (explicit_label) and give the lowest and highest possible reading for calibrated_estimate or interval.
- **Do not infer.** An exact recovery of source data from an approximate graphic.
- **Required capabilities.** `readable_chart`, `chart_encoding`, `complete_series`.
- **Eligibility checks.** `scope_resolved`, `encoding_resolved`, `precision_declared`, `schema_public`.
- **Verification contracts.** `dual_visual_review`, `chart_encoding_check`, `schema_check`.
- **Related FineVision subsets.** `mmc_instruct`, `SynthChartNet`, `Unichart`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `format` | yes | one of `structured_json` | The required output format. |
| `precision` | yes | one of `explicit_label`, `calibrated_estimate`, `interval` | explicit_label means printed values; calibrated_estimate means values read against labeled axis ticks; interval means ranges. |

### Numbers and measurements (`quantitative_reasoning`)

#### `quantity_comparison`: Compare printed quantities

The user asks which of two or more quantities printed in the image is larger, smaller or equal, such as the weights on two labels; the answer compares them after matching units, signs and dates.

- **Status.** core; structured verification.
- **Example question.** Which package has the larger net weight?
- **Do not infer.** Comparisons across different units, currencies or periods without a conversion given in the question.
- **Required capabilities.** `typed_operands`, `explicit_units`.
- **Eligibility checks.** `scope_resolved`, `operands_grounded`, `comparable_basis`.
- **Verification contracts.** `dual_visual_review`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `chartqa`, `finqa`, `robut_wtq`, `vqaonbd`, `CoSyn_400k_nutrition`.
- **Parameters.** None.

#### `grounded_arithmetic`: Compute with printed numbers

The user asks for the result of one addition, subtraction, multiplication or division of numbers printed in the image, such as the total price of two items; the answer gives the exact result with the requested number of decimal places.

- **Status.** core; structured verification.
- **Example question.** What is the total price of the two drinks printed on the menu?
- **Answer format.** One exact number from a single addition, subtraction, multiplication or division, rounded only as the question states and with the unit that the calculation gives; no percentages or other derived forms.
- **Do not infer.** Prices or rates that are not printed, an implied denominator, or unstated intermediate values.
- **Required capabilities.** `typed_operands`.
- **Eligibility checks.** `scope_resolved`, `operands_grounded`, `expression_defined`, `precision_declared`.
- **Verification contracts.** `dual_visual_review`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `chartqa`, `finqa`, `multihiertt`, `plotqa`, `tabmwp`, `tat_dqa`, `tat_qa`, `vqaonbd`, `CoSyn_400k_nutrition`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `operators` | yes | one of `add`, `subtract`, `multiply`, `divide` | The single operation that the question asks for. |
| `precision` | yes | integer | The number of decimal places that the answer must use, such as 0 or 2. |

#### `value_aggregation`: Sum or average a complete set of values

The user asks for the sum, mean, median, minimum, maximum or weighted mean of a complete set of printed values, such as all rows of a column; the answer computes it over every value in the set.

- **Status.** core; structured verification.
- **Example question.** What is the mean of the values in the three labeled rows?
- **Do not infer.** Skipped rows, blank cells counted as zero, or weights that are not given.
- **Required capabilities.** `typed_operands`, `closed_scope`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `operands_grounded`, `expression_defined`.
- **Verification contracts.** `dual_visual_review`, `closed_set_check`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `CoSyn_400k_table`, `finqa`, `hitab`, `multihiertt`, `robut_wikisql`, `robut_wtq`, `tabmwp`, `vqaonbd`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `operator` | yes | one of `sum`, `mean`, `median`, `min`, `max`, `weighted_mean` | The reduction that the question asks for. |
| `weights` | no | text | Only for weighted_mean: where the weights are printed, or the weights that the question states. |

#### `unit_conversion`: Convert a printed unit

The user asks to express a printed quantity in another unit, using a standard exact conversion such as meters to centimeters or a conversion stated in the question; the answer gives the converted value.

- **Status.** core; structured verification.
- **Example question.** Convert the printed length of 2.5 m to centimeters.
- **Do not infer.** Exchange rates, densities, serving sizes or other conversion factors that are not standard or stated.
- **Required capabilities.** `typed_operands`, `explicit_units`.
- **Eligibility checks.** `scope_resolved`, `operands_grounded`, `unit_rule_available`.
- **Verification contracts.** `dual_visual_review`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `tabmwp`, `vqaonbd`, `CoSyn_400k_nutrition`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `target_unit` | yes | text | The unit to convert into, such as 'cm'. |

#### `measurement_reading`: Read a scale, gauge or clock

The user asks for the reading of a visible ruler, gauge, scale or analog clock; the answer reads it from the marks and the pointer at the precision that the marks allow.

- **Status.** core; structured verification.
- **Example question.** What time does the analog clock show, to the nearest minute?
- **Do not infer.** A precision finer than the marks allow, or a measurement without a printed scale.
- **Required capabilities.** `calibrated_scale`.
- **Eligibility checks.** `scope_resolved`, `scale_and_pointer_resolved`, `precision_declared`.
- **Verification contracts.** `dual_visual_review`, `scale_check`.
- **Related FineVision subsets.** `iconqa`, `iconqa(mathv360k)`, `spark`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `precision` | yes | text | The precision that the question asks for, following the marks, such as 'to the nearest 0.5 cm' or 'to the nearest minute'. |

### Diagrams and flowcharts (`diagrams`)

#### `diagram_element_lookup`: Identify a labeled diagram element

The user asks which part of a diagram carries a given label or symbol, or what a labeled part is; the answer reads it from the diagram's labels and its key or notation.

- **Status.** core; light verification.
- **Example question.** Which component is labeled B in the diagram?
- **Do not infer.** Functions or names that the diagram does not label or define.
- **Required capabilities.** `readable_diagram`, `notation_context`.
- **Eligibility checks.** `scope_resolved`, `notation_resolved`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `geo170k(align)`, `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `ai2d_merged`.
- **Parameters.** None.

#### `diagram_connectivity`: List what connects to a diagram element

The user asks which elements of a diagram or network connect directly to a named element, or which connections go into or out of it; the answer follows the drawn lines and arrows.

- **Status.** core; structured verification.
- **Example question.** Which nodes are directly connected to node A?
- **Do not infer.** A connection at every line crossing, or a link inferred from closeness alone.
- **Required capabilities.** `graph_nodes_edges`.
- **Eligibility checks.** `scope_resolved`, `edges_resolved`.
- **Verification contracts.** `dual_visual_review`, `graph_check`.
- **Related FineVision subsets.** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `diagram_image_to_text`, `ai2d_merged`, `CoSyn_400k_circuit`.
- **Parameters.** None.

#### `diagram_path_tracing`: Trace a path through a diagram

The user asks for the route from one named element to another along the drawn connections; the answer lists the elements in order, respecting arrow directions, and gives every valid path when there are several.

- **Status.** core; structured verification.
- **Example question.** Trace the directed path from Start to the output labeled C.
- **Do not infer.** Connections that are not drawn, or a shortest path when no distance is defined.
- **Required capabilities.** `graph_nodes_edges`, `edge_directions`.
- **Eligibility checks.** `scope_resolved`, `edges_resolved`, `path_objective_defined`.
- **Verification contracts.** `dual_visual_review`, `graph_check`.
- **Related FineVision subsets.** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `diagram_image_to_text`, `ai2d_merged`.
- **Parameters.** None.

#### `diagram_process_description`: Explain the process a diagram shows

The user asks what sequence, cycle or branching process a diagram's arrows show; the answer explains the steps as they are drawn.

- **Status.** core; structured verification.
- **Example question.** Explain the sequence of steps shown by the arrows in this diagram.
- **Do not infer.** Causes or mechanisms that the diagram does not show, or every arrow read as causation.
- **Required capabilities.** `readable_diagram`, `graph_nodes_edges`, `edge_directions`.
- **Eligibility checks.** `scope_resolved`, `edges_resolved`.
- **Verification contracts.** `dual_visual_review`, `graph_check`.
- **Related FineVision subsets.** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `diagram_image_to_text`, `ai2d_merged`.
- **Parameters.** None.

#### `flowchart_evaluation`: Follow a flowchart for a given input

The user gives an input value and asks which outcome a visible flowchart or decision diagram reaches; the answer applies the printed conditions step by step and names the end point.

- **Status.** core; structured verification.
- **Example question.** For an input of 8, which output does this flowchart reach?
- **Do not infer.** Instructions written in the image being followed, invented missing conditions, or steps that are not shown.
- **Required capabilities.** `graph_nodes_edges`, `explicit_branch_conditions`.
- **Eligibility checks.** `scope_resolved`, `edges_resolved`, `public_rule_input_defined`.
- **Verification contracts.** `dual_visual_review`, `graph_check`, `exact_arithmetic_check`.
- **Related FineVision subsets.** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `diagram_image_to_text`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `input_values` | yes | list of exactly 1 | The single input value that the question gives, for example ['8']. |

#### `diagram_to_code`: Recreate a diagram as SVG or TikZ

The user asks for SVG or TikZ code that redraws a visible diagram; the answer gives code that renders the same shapes, labels and arrows in a sandbox.

- **Status.** extension; structured verification.
- **Example question.** Recreate this diagram in SVG, including its labels and arrows.
- **Do not infer.** The original source code, external images or fonts, or any file or network access.
- **Required capabilities.** `readable_diagram`, `rendered_layout`.
- **Eligibility checks.** `scope_resolved`, `rendering_contract_available`.
- **Verification contracts.** `dual_visual_review`, `sandbox_render_validator`.
- **Related FineVision subsets.** `datik`, `datikz`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `format` | yes | one of `svg`, `tikz_subset` | The code language of the answer. |

### Patterns and geometry (`patterns_and_geometry`)

#### `geometric_relations`: Read marked geometric relations

The user asks about a shape's type or symmetry, or which sides, angles or lines are marked as equal, parallel or perpendicular; the answer relies only on drawn marks, labels and clear shape structure.

- **Status.** core; structured verification.
- **Example question.** Which sides of the triangle are marked as equal?
- **Do not infer.** Exact equality, angles or lengths judged from how the drawing looks.
- **Required capabilities.** `geometric_marks`.
- **Eligibility checks.** `scope_resolved`, `geometry_evidence_sufficient`.
- **Verification contracts.** `dual_visual_review`, `geometry_check`.
- **Related FineVision subsets.** `CoSyn_400k_graphic`, `iconqa`, `iconqa(mathv360k)`, `geo170k(align)`, `geo170k(qa)`, `geo3k`.
- **Parameters.** None.

#### `pattern_rule`: State the rule of a visual pattern

The user asks what change repeats across a sequence or grid of panels; the answer names the one rule that fits every example: constant, translation, rotation, reflection, count progression, attribute cycle or set composition.

- **Status.** core; structured verification.
- **Example question.** What change repeats from one panel to the next?
- **Answer format.** Name one rule type (constant, translation, rotation, reflection, count progression, attribute cycle or set composition) and say what it changes.
- **Do not infer.** A rule chosen while another rule fits the examples equally well.
- **Required capabilities.** `repeated_structure`.
- **Eligibility checks.** `scope_resolved`, `pattern_rule_unique`.
- **Verification contracts.** `dual_visual_review`, `pattern_check`.
- **Related FineVision subsets.** `iconqa`, `iconqa(mathv360k)`, `raven`.
- **Parameters.** None.

#### `pattern_completion`: Complete a visual pattern

The user asks which of the visible answer options completes a pattern or matrix; the answer picks the one option that follows the rule shown by all of the examples.

- **Status.** core; structured verification.
- **Example question.** Which of the visible options completes the matrix?
- **Do not infer.** A hidden answer key, or one option chosen while another fits equally well.
- **Required capabilities.** `repeated_structure`, `answer_options_visible`.
- **Eligibility checks.** `scope_resolved`, `pattern_rule_unique`, `candidate_options_public`.
- **Verification contracts.** `dual_visual_review`, `pattern_check`.
- **Related FineVision subsets.** `iconqa`, `iconqa(mathv360k)`, `raven`.
- **Parameters.** None.

#### `pattern_exception`: Find the element that breaks a rule

The user states or points to a rule that a complete set of elements follows and asks which element breaks it; the answer names that element.

- **Status.** core; structured verification.
- **Example question.** Under the color-alternation rule, which panel breaks the pattern?
- **Do not infer.** An odd-one-out choice without a clear rule, or invented defects.
- **Required capabilities.** `repeated_structure`, `closed_scope`.
- **Eligibility checks.** `scope_resolved`, `complete_scope`, `exception_rule_defined`.
- **Verification contracts.** `dual_visual_review`, `pattern_check`.
- **Related FineVision subsets.** `iconqa`, `iconqa(mathv360k)`, `raven`.
- **Parameters.** None.

#### `geometric_constraint_solving`: Solve a geometry problem from marked facts

The user asks for a length, an angle or another value in a geometry figure; the answer derives it only from printed values, drawn marks and an allowed set of theorems.

- **Status.** extension; structured verification.
- **Example question.** Using the marked right angle and the printed side lengths, how long is side AC?
- **Do not infer.** Measurements read from an unscaled drawing, or assumptions that are not marked.
- **Required capabilities.** `geometric_marks`, `readable_formula`.
- **Eligibility checks.** `scope_resolved`, `geometry_evidence_sufficient`, `formal_rules_available`, `unique_solution`.
- **Verification contracts.** `dual_visual_review`, `formal_geometry_validator`.
- **Related FineVision subsets.** `CoSyn_400k_math`, `geo170k(qa)`, `geo3k`, `geometry3k(mathv360k)`, `geomverse`, `geoqa+(mathv360k)`, `geos(mathv360k)`, `intergps`, `mavis_math_metagen`, `mavis_math_rule_geo`, `unigeo(mathv360k)`.
- **Parameters.** None.

### Screens and user interfaces (`screen_ui`)

#### `ui_element_location`: Locate a control on a screen

The user asks where a visible button, field, link or other control is on a screenshot; the answer identifies it and describes its location in words.

- **Status.** core; structured verification.
- **Example question.** Where is the search box on this screen?
- **Do not infer.** Controls that are off screen or hidden, or claims that an action was performed.
- **Required capabilities.** `ui_controls`.
- **Eligibility checks.** `scope_resolved`, `unique_referent`, `ui_target_visible`.
- **Verification contracts.** `dual_visual_review`, `ui_grounding_check`.
- **Related FineVision subsets.** `aguvis-stage-1`, `groundui`, `screenqa`.
- **Parameters.** None.

#### `ui_state_reading`: Read the state of a control

The user asks about the visible state of an interface, such as which tab is selected, whether a box is checked, or which error is shown; the answer reads it from the screenshot.

- **Status.** core; light verification.
- **Example question.** Which tab is currently selected?
- **Do not infer.** Server state, permissions, or an enabled state judged from color alone.
- **Required capabilities.** `ui_controls`, `ui_state_indicators`.
- **Eligibility checks.** `scope_resolved`, `ui_state_explicit`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `aguvis-stage-1`, `groundui`, `screenqa`.
- **Parameters.** None.

#### `screen_to_code`: Recreate a screenshot as HTML and CSS

The user asks for static HTML and CSS that reproduce a visible screen layout; the answer gives markup that renders the same layout in a sandbox under the stated font and asset rules.

- **Status.** extension; structured verification.
- **Example question.** Recreate the visible layout as static HTML and CSS.
- **Do not infer.** Hidden scripts, app behavior, the original page source, credentials or network assets.
- **Required capabilities.** `ui_controls`, `rendered_layout`.
- **Eligibility checks.** `scope_resolved`, `rendering_contract_available`.
- **Verification contracts.** `dual_visual_review`, `sandbox_render_validator`.
- **Related FineVision subsets.** `websight`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `format` | yes | one of `html_css` | The code language of the answer. |

#### `ui_action_specification`: Specify one action on a screen

The user states a goal on the visible screen and asks for one click, focus or text input that starts it; the answer gives the action and the target's coordinates in the image, without performing it.

- **Status.** extension; structured verification.
- **Example question.** To focus the search field, where should one click? Give the point as fractions of the image width and height.
- **Do not infer.** Several steps through unseen screens, assumed results, or performing the action.
- **Required capabilities.** `ui_controls`.
- **Eligibility checks.** `scope_resolved`, `unique_referent`, `local_goal_public`, `action_schema_available`, `coordinates_verifiable`.
- **Verification contracts.** `dual_visual_review`, `ui_action_validator`.
- **Related FineVision subsets.** `aguvis-stage-1`, `groundui`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `action` | yes | one of `click`, `focus`, `input` | The single action that the answer specifies. |
| `input_text` | no | text | For the input action only, the exact text to type. |

### Multi-panel images (`multi_panel`)

#### `panel_comparison`: Compare panels

The user asks how two or more panels of the same image differ or agree, such as the left and right halves of a comparison figure; the answer states the differences and similarities seen in those panels.

- **Status.** core; structured verification.
- **Example question.** What differs between the left and the right panel?
- **Do not infer.** A second image, or before-and-after claims for panels that are not presented that way.
- **Required capabilities.** `panels_resolvable`.
- **Eligibility checks.** `scope_resolved`, `all_evidence_on_canvas`, `panels_publicly_identified`.
- **Verification contracts.** `dual_visual_review`, `panel_comparison_check`.
- **Related FineVision subsets.** `mimic_cgd`, `mmra`, `nlvr2`, `spot_the_diff`, `yesbut`.
- **Parameters.** None.

#### `panel_sequence_description`: Describe an ordered panel sequence

The user asks what changes across panels whose order is numbered or shown by arrows, such as a comic strip or a life cycle; the answer describes the changes in that order.

- **Status.** core; structured verification.
- **Example question.** What changes from panel 1 to panel 3?
- **Do not infer.** An order taken from unlabeled placement, events between panels, or causes.
- **Required capabilities.** `panels_resolvable`, `visible_sequence_order`.
- **Eligibility checks.** `scope_resolved`, `all_evidence_on_canvas`, `sequence_order_supported`.
- **Verification contracts.** `dual_visual_review`, `panel_comparison_check`.
- **Related FineVision subsets.** `spot_the_diff`, `yesbut`, `ai2d_merged`.
- **Parameters.** None.

### Claims and answerability (`evidence_verification`)

#### `visual_claim_verification`: Verify a claim against the image

The user states a claim about the image and asks whether it holds, or where the image supports it; the answer says supported, contradicted or cannot be determined and points to the visible evidence. Use quantified_claim_verification for claims with all, none or some, and object_presence for whether an object is there.

- **Status.** core; light verification.
- **Example question.** Is the statement 'the box is to the left of the chair' true for this image? Point to the evidence.
- **Answer format.** Supported, contradicted or cannot be determined, followed by the visible evidence: a region, a label or a quoted text.
- **Do not infer.** A yes-or-no verdict when the evidence is missing, or facts that are not visible.
- **Required capabilities.** `resolvable_region`.
- **Eligibility checks.** `scope_resolved`, `claim_public_and_local`.
- **Verification contracts.** `dual_visual_review`, `evidence_binding_check`.
- **Related FineVision subsets.** `idk`, `lnqa`, `lrv_normal(filtered)`, `nlvr2`, `spatialsense`, `vsr`, `est_vqa`, `infographic_vqa`, `pdfvqa`, `slidevqa`, `visualmrc`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `claim` | yes | text | The statement being checked, as the question quotes it. |

#### `answerability_assessment`: Say whether a question can be answered from the image

The user asks whether a specific question about a visible object or field can be answered from the image; the answer says whether it can and, if not, whether the needed part is unreadable, cut off or ambiguous. Use false_premise_question instead when the question assumes something that is not in the image.

- **Status.** core; light verification.
- **Example question.** Can the exact price be read on this cropped label? Explain what limits it.
- **Answer format.** One of answerable, unreadable, cropped or ambiguous, followed by the visible reason.
- **Do not infer.** A refusal of an answerable question, or unreadable text treated as evidence that something is absent.
- **Required capabilities.** `resolvable_region`.
- **Eligibility checks.** `scope_resolved`, `local_question_supported`.
- **Verification contracts.** `dual_visual_review`.
- **Related FineVision subsets.** `idk`, `vizwiz(mathv360k)`, `screenqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `local_question` | yes | text | The question whose answerability is judged, as the user asks it. |

### Object presence and false premises (`presence_and_premises`)

#### `object_presence`: Say whether an object is in the image

The user asks whether an object of a named kind is in the image or a named area, such as 'Is there a fork on the table?'; the answer names the relevant visible objects, says whether the object asked about is there, and ends with yes or no. Absent objects are often ones that usually go with what is visible.

- **Status.** core; structured verification.
- **Example question.** Is there a fork on the table?
- **Answer format.** First the relevant objects that are visible, then whether the object asked about is there and, if it is, where; end with yes or no.
- **Do not infer.** A no for an object that could be hidden, cut off or too small to see, or a yes because the object usually goes with the scene.
- **Required capabilities.** `decidable_presence`.
- **Eligibility checks.** `scope_resolved`, `presence_decidable`, `presence_not_hinted`.
- **Verification contracts.** `dual_visual_review`, `premise_check`.
- **Related FineVision subsets.** `lrv_normal(filtered)`, `sketchyvqa`, `oodvqa`, `objects365_qa`.
- **Parameters.** None.

#### `false_premise_question`: Correct a question about something that is not there

The user asks about something that the image does not show, such as 'What color is the woman's purse?' when she has none, or 'How many horses are there?' in a street scene; the answer names what is visible, says that the assumed object or detail is absent, and ends with zero for a count or that the detail cannot be determined.

- **Status.** core; structured verification.
- **Example question.** What type of bird is sitting on the elephant's back?
- **Answer format.** First the relevant objects that are visible, then that the assumed object or detail is not there, ending with 0 for a count or with the statement that the detail cannot be determined.
- **Do not infer.** Details of the missing object, a guess that it is hidden somewhere, or an absence claimed for an area that is cut off or too small to see.
- **Required capabilities.** `decidable_presence`.
- **Eligibility checks.** `scope_resolved`, `premise_clearly_false`, `presence_decidable`.
- **Verification contracts.** `dual_visual_review`, `premise_check`.
- **Related FineVision subsets.** `idk`, `lrv_normal(filtered)`, `oodvqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `asked_detail` | yes | one of `count`, `attribute`, `location`, `action`, `kind` | What the question asks about the assumed object: how many there are, a property such as its color, where it is, what it is doing, or what kind it is. |

### Recognition with world knowledge (`knowledge_recognition`)

#### `named_entity_recognition`: Name a well-known landmark or artwork

The user asks which well-known landmark, building, artwork, flag or emblem is shown; the answer names it from world knowledge and points to the visible features that identify it. People are never identified. Use object_identification instead for an ordinary category such as 'a cathedral'.

- **Status.** core; structured verification.
- **Example question.** Which famous bridge is shown in this photo?
- **Answer format.** The name of the entity, followed by the visible features that identify it.
- **Do not infer.** Any person's identity, a name suggested only by the question, or a guess when the visible features fit several places.
- **Required capabilities.** `recognizable_entity`.
- **Eligibility checks.** `scope_resolved`, `entity_widely_known`.
- **Verification contracts.** `dual_visual_review`, `answer_consensus_check`.
- **Related FineVision subsets.** `google_landmarks`, `densefusion_1m`, `sharegpt4v(knowledge)`, `lnqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `entity_kind` | yes | one of `landmark`, `building`, `artwork`, `flag_or_emblem`, `natural_feature` | The kind of entity the question asks to name. |

#### `style_recognition`: Recognize a style or genre

The user asks which well-known style, movement, period or genre a visible artwork, building, design or publication belongs to, optionally choosing from listed options; the answer names it and the visible traits that support it. Use scene_categorization instead for kinds of places.

- **Status.** core; structured verification.
- **Example question.** Which art movement does the brushwork of this painting suggest?
- **Answer format.** The style or genre name, followed by the visible traits that support it.
- **Do not infer.** The specific artist, the date of creation or a judgment of quality, unless printed.
- **Required capabilities.** `style_traits`.
- **Eligibility checks.** `scope_resolved`, `style_traits_support_answer`.
- **Verification contracts.** `dual_visual_review`, `answer_consensus_check`.
- **Related FineVision subsets.** `sharegpt4v(knowledge)`, `ocrvqa`, `google_landmarks`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `style_kind` | yes | one of `art_movement`, `architectural_style`, `design_period`, `genre` | The kind of style the question asks about. |

#### `map_region_identification`: Identify a region on a map

The user points to a highlighted, outlined or marked area on a map and asks which country, state, city or other region it is; the answer names it using geographic knowledge of shapes and positions. Use chart_value_lookup instead to read a value from a map's legend.

- **Status.** core; structured verification.
- **Example question.** Which country is outlined in green on this map?
- **Answer format.** The region's name, followed by the map features that place it.
- **Do not infer.** Borders, names or data that the map does not show, or a name chosen when the outline fits several regions.
- **Required capabilities.** `map_geography`.
- **Eligibility checks.** `scope_resolved`, `region_locatable`.
- **Verification contracts.** `dual_visual_review`, `answer_consensus_check`.
- **Related FineVision subsets.** `mapqa`, `mapqa(mathv360k)`, `scienceqa(nona_context)`, `scienceqa`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `region_level` | yes | one of `country`, `state_or_province`, `city`, `continent_or_ocean`, `other_region` | The kind of region the question asks to name. |

### Reasoning with specialist knowledge (`domain_reasoning`)

#### `concept_explanation`: Explain the concept a diagram shows

The user asks what scientific or technical process, principle or structure a labeled diagram, schematic or figure illustrates, or what role a labeled part plays; the answer explains it with standard textbook knowledge and ties each point to the visible labels.

- **Status.** core; light verification.
- **Example question.** Using the labels in this diagram, explain how carbon moves between the air, plants and animals.
- **Do not infer.** Facts beyond standard textbook knowledge, claims about the specific source or experiment, or medical advice.
- **Required capabilities.** `explanatory_diagram`.
- **Eligibility checks.** `scope_resolved`, `concept_shown_by_labels`.
- **Verification contracts.** `dual_visual_review`, `evidence_binding_check`.
- **Related FineVision subsets.** `ai2d_merged`, `tqa`, `CoSyn_400k_circuit`, `CoSyn_400k_chemical`, `arxivqa`.
- **Parameters.** None.

#### `notation_interpretation`: Interpret a standard notation

The user asks what marks in a standard notation mean, such as a key or time signature, a functional group or molecular formula, a circuit symbol or a stem-and-leaf plot; the answer applies the notation's conventions to the visible marks and gives a short result.

- **Status.** core; structured verification.
- **Example question.** Which key does the key signature on this staff indicate?
- **Answer format.** A short result, such as 'D major' or 'C7H14O6', followed by the marks it is based on.
- **Do not infer.** Marks that are not visible, how the music sounds, or chemical or electrical behavior that the notation does not state.
- **Required capabilities.** `standard_notation`.
- **Eligibility checks.** `scope_resolved`, `notation_standard_and_complete`.
- **Verification contracts.** `dual_visual_review`, `answer_consensus_check`.
- **Related FineVision subsets.** `CoSyn_400k_music`, `CoSyn_400k_chemical`, `CoSyn_400k_circuit`, `tabmwp`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `notation_field` | yes | one of `music`, `chemistry`, `electronics`, `statistics`, `mathematics` | The field whose notation the question asks about. |

#### `math_word_problem`: Solve a printed math problem

The user asks to solve a math problem whose numbers and conditions are printed in the image or stated in the question, possibly in several steps; the answer shows the steps and gives one final number or expression. Use grounded_arithmetic instead for a single calculation.

- **Status.** core; structured verification.
- **Example question.** Solve the problem printed on the card and give the average yearly rise in millimeters.
- **Answer format.** The steps, then a final line of the form 'Answer: <number or expression>'.
- **Do not infer.** Numbers or conditions that are neither printed nor stated, or a rounding rule the question does not give.
- **Required capabilities.** `printed_problem`.
- **Eligibility checks.** `scope_resolved`, `problem_fully_given`.
- **Verification contracts.** `dual_visual_review`, `answer_consensus_check`.
- **Related FineVision subsets.** `CoSyn_400k_math`, `tabmwp`, `tabmwp(mathv360k)`, `mavis_math_metagen`, `infographic_vqa_llava_format`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `precision` | no | integer | The number of decimal places for a decimal result, when the question asks for rounding. |

### Creative writing from the image (`grounded_creation`)

#### `grounded_creative_writing`: Write a creative text about the image

The user asks for a short creative text about the image, such as a caption, poem, story opening or product blurb, in a stated form and length; the answer may invent mood and narrative, but everything it says about what is shown must match the image.

- **Status.** core; light verification.
- **Example question.** Write a four-line poem about this harbor at sunset.
- **Answer format.** Only the creative text, in the requested form and length.
- **Do not infer.** Names, identities or real events for the people shown, or invented facts presented as observations.
- **Required capabilities.** `multiple_facts`.
- **Eligibility checks.** `scope_resolved`, `creative_brief_stated`.
- **Verification contracts.** `dual_visual_review`, `evidence_binding_check`.
- **Related FineVision subsets.** `laion_gpt4v`, `sharegpt4o`, `image_textualization(filtered)`, `wildvision`, `textcaps`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `form` | yes | one of `caption`, `poem`, `story_opening`, `product_blurb`, `dialogue` | The form of the creative text. |
| `max_words` | no | integer | The length limit in words, when the question gives one. |

### Music, chemistry and circuit notation (`specialist_notation`)

#### `music_notation_reading`: Read music notation

The user asks for the pitches, durations, rests or note order in named complete measures of a single-voice score with a visible clef, key and meter; the answer reads them in a structured form.

- **Status.** extension; structured verification.
- **Example question.** What are the pitches of the notes in the first complete measure?
- **Do not infer.** A missing clef or accidental, how the music sounds, or the composer's intent.
- **Required capabilities.** `notation_context`, `readable_diagram`.
- **Eligibility checks.** `scope_resolved`, `music_context_complete`.
- **Verification contracts.** `dual_visual_review`, `music_notation_validator`.
- **Related FineVision subsets.** `CoSyn_400k_music`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `bar_range` | yes | text | The complete measures asked about, such as '1-2'. |

#### `chemical_structure_reading`: Read a chemical structure

The user asks about the atoms, bonds, rings or a marked part of a drawn molecule; the answer reads them using non-stereo SMILES conventions.

- **Status.** extension; structured verification.
- **Example question.** Which atoms does the marked double bond connect?
- **Do not infer.** Reactions, synthesis instructions, biological effects, or stereochemistry that is not drawn.
- **Required capabilities.** `notation_context`, `readable_diagram`.
- **Eligibility checks.** `scope_resolved`, `chemical_notation_resolved`.
- **Verification contracts.** `dual_visual_review`, `chemical_graph_validator`.
- **Related FineVision subsets.** `CoSyn_400k_chemical`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `notation` | yes | one of `nonstereo_smiles` | The chemical notation that the answer uses. |

#### `circuit_structure_reading`: Read a circuit diagram

The user asks which components of a drawn circuit are connected, in series or in parallel, or for its netlist; the answer follows the drawn wires, junction dots and symbols. In a netlist, pin a of each two-terminal part is its left end (its upper end when vertical) and pin b the other end; a netlist question states this convention.

- **Status.** extension; structured verification.
- **Example question.** Which components are connected in parallel in this circuit?
- **Do not infer.** Electrical behavior from missing values or wires, or safety conditions that are not shown.
- **Required capabilities.** `notation_context`, `graph_nodes_edges`.
- **Eligibility checks.** `scope_resolved`, `circuit_notation_resolved`.
- **Verification contracts.** `dual_visual_review`, `circuit_graph_validator`.
- **Related FineVision subsets.** `CoSyn_400k_circuit`.

| Parameter | Required | Form | Meaning |
|---|---|---|---|
| `notation` | yes | one of `two_terminal_netlist` | The circuit notation that the answer uses. |
| `operation` | yes | one of `netlist`, `parallel_pairs`, `series_pairs` | What the answer reports. |

## Capabilities

A capability is something the image must show for a task to apply.

| Capability | Meaning |
|---|---|
| `answer_options_visible` | All answer options for the pattern are visible. |
| `box_targets` | Each target is fully visible with clear edges, so a tight box can be drawn around it. |
| `calibrated_scale` | The scale's labels, tick spacing, units and pointer are visible. |
| `chart_encoding` | The needed axes, legend, scale type and units are visible. |
| `closed_scope` | Every member of the group asked about is fully visible, so a complete list or count is possible. |
| `comparable_attributes` | The same visible property can be seen on every compared object. |
| `comparable_series` | The series can be compared on a shared scale. |
| `complete_series` | The whole series or set of categories asked about is visible. |
| `countable_entities` | Every relevant object can be counted under one clear counting unit. |
| `decidable_presence` | The scene is clear enough to tell whether an object of a named kind is there or not. |
| `discriminating_attributes` | Visible features or relations single out the intended object among similar ones. |
| `document_layout` | Document elements and their boundaries are visible. |
| `edge_directions` | Arrow directions, or the absence of direction, can be seen. |
| `explanatory_diagram` | A labeled diagram, schematic or figure shows a process, structure or principle. |
| `explicit_branch_conditions` | The condition on each branch is printed. |
| `explicit_cross_references` | Labels, keys or wording link the parts being compared. |
| `explicit_mapping` | A visible line, key, label or legend links the items. |
| `explicit_units` | The units are printed, or the values are clearly unitless. |
| `geometric_marks` | The shapes, equality and angle marks, and labels are visible. |
| `graph_nodes_edges` | The diagram's boxes, connections, junctions and end points can be told apart. |
| `interpretable_predicates` | Each condition can be checked by looking. |
| `legible_values` | The requested values are printed or can be estimated at the stated precision. |
| `map_geography` | Coastlines, borders or a locator view show enough geography to place a region. |
| `multiple_entities` | At least two relevant objects can be told apart. |
| `multiple_evidence_regions` | At least two relevant parts of the same image hold evidence. |
| `multiple_facts` | At least two separate visible facts are available in the requested area. |
| `multiple_tables` | At least two separate tables are visible. |
| `notation_context` | The symbols, keys and notation the drawing uses are visible. |
| `ordered_series` | The order along the chart's axis is visible. |
| `panels_resolvable` | The image contains at least two panels that can be told apart. |
| `printed_problem` | A math problem and the numbers it needs are printed in the image or stated in the question. |
| `readable_chart` | The chart or map supports the requested reading. |
| `readable_diagram` | The diagram's elements, labels and marks can be read. |
| `readable_formula` | The formula's symbols and two-dimensional structure can be read. |
| `readable_prose` | Enough document text can be read for the requested operation. |
| `readable_table` | The needed rows, columns and cells can be read. |
| `readable_text` | The relevant text can be read at the image's actual resolution. |
| `reading_order` | The reading order of the text blocks is visible or marked. |
| `recognizable_entity` | A well-known landmark, building, artwork, flag or emblem is shown clearly enough to recognize. |
| `rendered_layout` | The visible layout can be redrawn with the allowed code under fixed rendering rules. |
| `repeated_structure` | Several examples show a repeating structure. |
| `resolvable_region` | The area can be described in words without inventing a marker or coordinate. |
| `scene_context` | Enough of the setting is visible to say what kind of scene it is. |
| `spatial_layout` | The positions needed for the relation can be read in the image. |
| `standard_notation` | Marks follow a standard notation, such as a music staff, a structural formula, circuit symbols or a statistical plot. |
| `style_traits` | Visible technique, form, ornament or layout shows a style, period or genre. |
| `table_headers` | Row and column headers, including group headers, can be read. |
| `table_join_keys` | The tables share a visible identifier for matching rows. |
| `text_fields` | The labeled fields and their values can be located. |
| `text_object_alignment` | The text can be linked to the object or region it refers to. |
| `translatable_text` | The text to translate is readable and clearly separated from other text. |
| `typed_operands` | The numbers needed are printed, with enough context to know what they measure. |
| `ui_controls` | The relevant buttons, fields and other controls can be located. |
| `ui_state_indicators` | The interface shows the state asked about explicitly. |
| `visible_attribute` | The property asked about can be seen; it is not hidden or inferred. |
| `visible_entity` | The object asked about is visible and can be told apart from its surroundings. |
| `visible_interaction` | Body pose or contact shows the action or interaction. |
| `visible_sequence_order` | The order of the panels is numbered, marked by arrows, or otherwise clear. |

## Eligibility checks

An eligibility check is a condition on the question and the image that the drafter must satisfy and the question judges confirm.

| Check | Condition |
|---|---|
| `action_schema_available` | An allowed action and coordinate format is configured; the action is described, never performed. |
| `action_visually_supported` | Body pose or contact visibly shows the action; nothing about time, intent or what happens next. |
| `all_evidence_on_canvas` | Everything needed (panels, pages, labels, options and text) is in this one image. |
| `answer_evidence_present` | The visible document contains the answer; pages that are not shown do not count. |
| `association_explicit` | A visible line, key, label or layout links the items; being close together is not enough. |
| `attribute_visible` | The property asked about can be seen, and it is not a hidden or sensitive personal trait. |
| `boxes_definable` | Every requested target is fully visible with clear edges, so each one gets exactly one tight box; cut-off or hidden targets make the question unsuitable. |
| `candidate_options_public` | All answer options are visible in the image or listed in the question. |
| `chemical_notation_resolved` | The drawing uses atom, bond and hydrogen conventions that the validator supports. |
| `circuit_notation_resolved` | The drawing's symbols and junction conventions are supported by the validator. |
| `claim_public_and_local` | The question quotes the claim, and the claim concerns visible content. |
| `comparable_basis` | The compared items use the same property, unit, scale and period. |
| `complete_scope` | The whole group or area asked about is visible, so a complete answer is possible; unreadable or cut-off parts make the question unanswerable, not empty. |
| `concept_shown_by_labels` | The diagram's labels or structure identify the concept or process, so the explanation rests on standard textbook knowledge rather than speculation. |
| `coordinates_verifiable` | Coordinates refer to the image as shown and to a verified target area. |
| `count_unit_defined` | The question makes clear what counts as one item and which items are included. |
| `creative_brief_stated` | The question states the form of the creative text and, if it matters, its length; nothing it asks for contradicts the image or concerns a person's identity. |
| `description_claims_visible` | Every statement in the description is supported by something visible in the requested area. |
| `edges_resolved` | The connected elements, arrow directions and junctions or crossings can be read. |
| `encoding_resolved` | The axes, legend, scale type (linear or log), baseline and units are clear. |
| `entity_widely_known` | The entity is widely known, it is not a person, and its visible features identify it without help from the question; a guess between similar places is not enough. |
| `exception_rule_defined` | The rule is stated or clearly shown before asking which element breaks it. |
| `expression_defined` | The question states the calculation, the order of the operands and any rounding. |
| `fields_bound` | Each requested field has a visible label or value area; a missing field is reported as missing, never invented. |
| `formal_rules_available` | The problem type is covered by an allowed rule set and a qualified validator. |
| `geometry_evidence_sufficient` | Exact claims rely on drawn marks or printed values, not on how the drawing looks. |
| `grouping_key_defined` | The question names the grouping attribute, and every object falls into exactly one group. |
| `hypothetical_public` | The question states the change explicitly as a hypothetical, not as something that happened. |
| `join_keys_unique` | The shared identifier matches the rows of the tables without ambiguity. |
| `local_goal_public` | The question states one goal that a single action on the visible screen can start, without claiming any result. |
| `local_question_supported` | The question being judged concerns a visible object or field, so its answerability can be judged from the image. |
| `music_context_complete` | The clef, key, meter and accidentals needed are visible and supported by the validator. |
| `notation_resolved` | The notation (for example LaTeX, or the diagram's own symbols) is clear and all visible structure is kept. |
| `notation_standard_and_complete` | The notation is standard for its field and every mark the answer depends on is visible. |
| `operands_grounded` | Every number used is printed in the image or was established in an earlier turn. |
| `panels_publicly_identified` | The panels can be told apart by labels or an obvious layout. |
| `path_objective_defined` | The question names the start, the end and what counts as a valid path. |
| `pattern_rule_unique` | Exactly one of the allowed rule types fits every example. |
| `precision_declared` | The question states the precision, and the answer never claims more precision than the image shows. |
| `predicates_observable` | Each condition can be checked by looking, and the way the conditions combine (and, or, not) is clear. |
| `premise_clearly_false` | What the question assumes, such as a purse carried by the woman or a black dog when the only dog is brown, is clearly not in the image, while the person, object or area that the question mentions is visible. |
| `presence_decidable` | Whether the object asked about is there can be decided by looking: it is clearly visible, or it would clearly be seen if it were there; an area that is cut off, hidden, blurred or too small makes the question unsuitable. |
| `presence_not_hinted` | The question does not hint whether the object is there, for example by calling it visible, missing or usual for the scene. |
| `problem_fully_given` | Every number and condition the problem needs is printed in the image or stated in the question, and the result is one number or short expression. |
| `public_rule_input_defined` | The flowchart's conditions are printed and the question gives the input; text in the image is never followed as an instruction. |
| `reading_order_resolved` | When several text blocks are involved, their reading order is clear from columns, numbering or layout. |
| `region_locatable` | The map shows enough coastline, borders or position to name the marked region without guessing, and the region is marked or described without ambiguity. |
| `relation_frame_defined` | The question states the viewpoint for the relation, and the relation can be judged from the image without measuring anything. |
| `rendering_contract_available` | A sandboxed renderer with fixed font, asset and syntax rules is configured. |
| `role_visually_supported` | Layout and readable content together show the element's role. |
| `scale_and_pointer_resolved` | The scale's marks, origin and units and the pointer position can be read. |
| `schema_public` | The question states the required output format. |
| `scope_resolved` | The question names an area or subject that is actually in the image (a visible object, region, text or the whole image), never a marker, number or coordinate the user cannot see. |
| `sequence_order_supported` | Numbers, arrows, timestamps or a clear convention give the order of the panels. |
| `style_traits_support_answer` | Visible traits support the named style or genre, and the question names the kind of style it asks about or lists the options. |
| `summary_entails_evidence` | The summary keeps the source's meaning and adds no unsupported fact. |
| `table_headers_bound` | The full row and column header paths, units and footnotes can be read; blank cells stay blank, never zero. |
| `text_legible` | The text can be read at the image's actual resolution; unreadable or cut-off text is not completed from guesses. |
| `translation_language_stated` | The question names the language to translate into and the exact block of text to translate. |
| `ui_state_explicit` | The interface shows the state clearly; gray or color alone does not prove that a control is disabled. |
| `ui_target_visible` | The control is on the screen and the question asks about it directly. |
| `unique_referent` | The description matches exactly one object; if several match, the question must be rewritten or the answer must say so. |
| `unique_solution` | The visible facts determine exactly one answer under the allowed rules. |
| `unit_rule_available` | The conversion is a standard exact one, such as meters to centimeters, or is stated in the question. |
| `visible_category_supported` | The visible features justify the category named in the answer; no guessed identities and no finer category than the appearance supports. |

## Verification contracts

| Contract | What it checks | Applies when |
|---|---|---|
| `answer_consensus_check` | Two independent readers answer the question without seeing the candidate answer, and the controller requires both short answers to match the candidate's after normalizing case, punctuation and number format. Any disagreement or abstention leaves the turn uncommitted. | The answer is a short name, style, region, notation reading or number that world or specialist knowledge determines. |
| `box_iou_check` | Two independent readers draw a box around each target without seeing the answer, and the controller matches every answer box one-to-one to each reader's boxes with an overlap of at least 0.5 (intersection over union). | The answer gives bounding boxes for visible targets. |
| `chart_encoding_check` | Two independent readers recover the axes, legend, scale and the relevant marks, and the controller checks the answer and its precision against them. | The answer depends on a chart's or map's encoding or on plotted values. |
| `chemical_graph_validator` | A specialized validator compares the read molecule graph with two independent extractions; it never proposes chemistry procedures. | The chemical_structure_reading extension is enabled. |
| `circuit_graph_validator` | A specialized validator compares the read circuit topology with two independent extractions under the stated symbol and junction conventions. | The circuit_structure_reading extension is enabled. |
| `closed_set_check` | Two independent readers list the members or per-group counts of the visible group, and the controller compares them with the answer. Duplicates, unknown members and incomplete groups stay explicit. | The answer depends on a complete list, a count, an order, or an all, none or some claim. |
| `dual_visual_review` | Two independent judges each decide whether the question and the answer are correct, complete, supported by the image and in the requested language and format. Their agreement does not prove truth. | Always. |
| `evidence_binding_check` | Two independent reviewers link every essential part of the answer to a visible region or an exact quote, and check that the linked evidence agrees, without using outside facts. | The answer identifies or combines evidence from parts of the image or from earlier verified turns. |
| `exact_arithmetic_check` | Two independent readers extract the operands, and the controller recomputes the result exactly with the stated units and rounding. A correct calculation does not prove that the operands were read correctly. | The answer depends on a calculation or a numeric comparison. |
| `formal_geometry_validator` | A specialized validator checks every visual premise and a formal derivation under an allowed set of theorems. | The geometric_constraint_solving extension is enabled. |
| `formula_structure_check` | Two independent readers transcribe the formula's structure, and the controller compares symbols, grouping and scripts. An algebraically equivalent formula does not count as correct. | The answer transcribes a visible mathematical formula. |
| `geometry_check` | Two independent readers extract the drawn marks and printed values; the controller never treats how the drawing looks as an exact fact. | The answer states a geometric relation or a shape property. |
| `graph_check` | Two independent readers extract the diagram's elements and connections, and the controller computes neighbors, paths and branch outcomes from them. | The answer depends on connections, directions, paths or branches. |
| `music_notation_validator` | A specialized validator checks the read notes, rests and durations against the visible notation; sound is never inferred. | The music_notation_reading extension is enabled. |
| `panel_comparison_check` | Two independent reviewers compare only the panels the question names and link every stated difference or change to those panels. | The answer compares panels or describes their sequence. |
| `pattern_check` | Two independent readers describe every panel and option, and the controller searches the allowed rule types, requiring exactly one fitting rule and answer. | The answer names a pattern rule, a completion or an exception. |
| `premise_check` | Two independent readers decide, without seeing the candidate answer, whether the image shows what the question asks about or assumes, and the controller requires them to agree and the answer's conclusion to match. A reader's doubt about hidden, cut-off or very small areas leaves the turn uncommitted. | The question asks whether something is in the image, or assumes something that may not be there. |
| `sandbox_render_validator` | A locked-down renderer without network, file access, scripts or unsafe TeX renders the answer's code and compares it with the image under fixed fidelity rules. | The diagram_to_code or screen_to_code extension is enabled. |
| `scale_check` | Two independent readers recover the scale's labels, ticks, units and pointer, and the controller recomputes the reading and its tolerance. | The answer reads a ruler, gauge, scale or clock. |
| `schema_check` | The controller checks that the answer follows the requested machine-readable format. A valid format does not make the content correct. | The question requires JSON, HTML, Markdown or another fixed format. |
| `table_structure_check` | Two independent readers recover the table's header paths and cells, and the controller checks the answer's alignment and completeness against them. | The answer depends on row, column or header alignment, or rebuilds a table. |
| `transcript_alignment` | Two independent readers transcribe the requested source text, and the controller compares the answer with it exactly, including punctuation and meaningful whitespace. Source errors are not corrected. | The answer copies or extracts visible text, code or labels. |
| `ui_action_validator` | A specialized validator checks one allowed, unexecuted action, its target and its coordinates against the image as shown; it never assumes the action succeeded. | The ui_action_specification extension is enabled. |
| `ui_grounding_check` | Two independent reviewers find the visible control that the question refers to and check the answer's description of it and of its location. | The answer locates or identifies an interface control. |
