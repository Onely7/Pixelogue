# Pixelogue task catalog v7

**Supplied design reference.** The operation definitions below are preserved from the proposal. The [runtime admission guide](README.md) distinguishes implemented checks from blocked requirements; the design document itself is not an implementation claim.

72 operation contracts: 65 core candidates and 7 opt-in extensions. These counts are design choices, not an empirical optimum. Every task still requires per-image eligibility and applicable validation.

Base: Onely7/Pixelogue, feat/pixelogue-foundation-tsubame-edit, commit 50ed89a60c9a7b6a724cd6a7ea03e8d57696b026.

## Scope

All image-specific factual premises must come from this canvas or validated image-derived public history. General language/notation/math rules are permitted; a question may publicly choose a filter or hypothesis without asserting it is an observed fact.

Only one source image is assumed. Existing panels can be compared; missing pages or a second image cannot be invented. The source table is a prior research artifact, not complete QA-record verification.

## Catalog overview

| No. | Task ID | Operation | Family | Status |
|---:|---|---|---|---|
| 1 | `object_identification` | Object identification | visual_description | core_candidate |
| 2 | `attribute_lookup` | Visible attribute lookup | visual_description | core_candidate |
| 3 | `visible_action_relation` | Visible action or interaction | visual_description | core_candidate |
| 4 | `scene_categorization` | Scene categorization | visual_description | core_candidate |
| 5 | `grounded_description` | Grounded scene or region description | visual_description | core_candidate |
| 6 | `visual_summary` | Visual synthesis and summary | visual_description | core_candidate |
| 7 | `referring_object_resolution` | Resolve a referring expression | reference_spatial | core_candidate |
| 8 | `referring_expression_generation` | Generate a discriminating reference | reference_spatial | core_candidate |
| 9 | `spatial_relation` | Spatial relation lookup | reference_spatial | core_candidate |
| 10 | `spatial_ordering` | Spatial ordering | reference_spatial | core_candidate |
| 11 | `attribute_comparison` | Visible attribute comparison | reference_spatial | core_candidate |
| 12 | `visual_correspondence` | Explicit visual correspondence | reference_spatial | core_candidate |
| 13 | `entity_count` | Count a bounded set | set_logic | core_candidate |
| 14 | `predicate_selection` | Predicate-based selection | set_logic | core_candidate |
| 15 | `set_operation` | Union, intersection, or difference | set_logic | core_candidate |
| 16 | `attribute_grouping` | Grouping and partitioning | set_logic | core_candidate |
| 17 | `set_cardinality_comparison` | Compare set cardinalities | set_logic | core_candidate |
| 18 | `quantified_statement_verification` | Verify a quantified proposition | set_logic | core_candidate |
| 19 | `grounded_hypothetical_update` | Explicit hypothetical set update | set_logic | core_candidate |
| 20 | `text_transcription` | Verbatim text transcription | text_reading | core_candidate |
| 21 | `text_field_extraction` | Extract specified text fields | text_reading | core_candidate |
| 22 | `text_reading_order` | Reconstruct reading order | text_reading | core_candidate |
| 23 | `label_value_linking` | Link labels and values | text_reading | core_candidate |
| 24 | `text_visual_binding` | Relate text and visual content | text_reading | core_candidate |
| 25 | `formula_transcription` | Transcribe mathematical notation | text_reading | core_candidate |
| 26 | `code_transcription` | Transcribe visible code | text_reading | core_candidate |
| 27 | `document_extractive_qa` | Extractive document question answering | document_understanding | core_candidate |
| 28 | `document_evidence_synthesis` | Synthesize distributed document evidence | document_understanding | core_candidate |
| 29 | `document_summary` | Document summary or key points | document_understanding | core_candidate |
| 30 | `document_structure_reconstruction` | Reconstruct document structure | document_understanding | core_candidate |
| 31 | `cross_region_consistency_check` | Cross-region consistency checking | document_understanding | core_candidate |
| 32 | `document_layout_role` | Identify document element roles | document_understanding | core_candidate |
| 33 | `table_cell_lookup` | Table lookup with hierarchical headers | table_understanding | core_candidate |
| 34 | `table_predicate_selection` | Filter and order table records | table_understanding | core_candidate |
| 35 | `table_structure_reconstruction` | Reconstruct a table | table_understanding | core_candidate |
| 36 | `table_cross_reference` | Join explicitly linked tables | table_understanding | core_candidate |
| 37 | `chart_encoding_lookup` | Read chart or map encodings | chart_map_understanding | core_candidate |
| 38 | `chart_value_lookup` | Read an encoded value | chart_map_understanding | core_candidate |
| 39 | `chart_comparison` | Compare chart values | chart_map_understanding | core_candidate |
| 40 | `chart_extremum_ranking` | Find extrema or rank chart entries | chart_map_understanding | core_candidate |
| 41 | `chart_trend_summary` | Summarize a chart trend | chart_map_understanding | core_candidate |
| 42 | `chart_series_relation` | Analyze relations between series | chart_map_understanding | core_candidate |
| 43 | `chart_data_reconstruction` | Reconstruct chart data | chart_map_understanding | core_candidate |
| 44 | `quantity_comparison` | Compare or rank quantities | quantitative_reasoning | core_candidate |
| 45 | `grounded_arithmetic` | Grounded arithmetic expression | quantitative_reasoning | core_candidate |
| 46 | `grounded_aggregation` | Aggregate a complete set of quantities | quantitative_reasoning | core_candidate |
| 47 | `unit_conversion` | Convert displayed units | quantitative_reasoning | core_candidate |
| 48 | `measurement_reading` | Read a visual measuring scale | quantitative_reasoning | core_candidate |
| 49 | `diagram_element_lookup` | Identify diagram elements | diagram_graph | core_candidate |
| 50 | `graph_connectivity` | Recover graph connectivity | diagram_graph | core_candidate |
| 51 | `graph_path_tracing` | Trace a visible path | diagram_graph | core_candidate |
| 52 | `diagram_process_description` | Describe an explicit diagram process | diagram_graph | core_candidate |
| 53 | `diagram_branch_evaluation` | Apply a depicted branch rule | diagram_graph | core_candidate |
| 54 | `geometric_relation_analysis` | Analyze depicted geometry | pattern_geometry | core_candidate |
| 55 | `pattern_rule_identification` | Identify a bounded visual rule | pattern_geometry | core_candidate |
| 56 | `pattern_completion` | Complete a visual pattern | pattern_geometry | core_candidate |
| 57 | `rule_based_exception` | Find a rule violation | pattern_geometry | core_candidate |
| 58 | `ui_element_grounding` | Locate a UI target | screen_ui | core_candidate |
| 59 | `ui_state_reading` | Read visible UI state | screen_ui | core_candidate |
| 60 | `screen_summary` | Summarize a screen | screen_ui | core_candidate |
| 61 | `panel_comparison` | Compare panels already in one image | multi_region | core_candidate |
| 62 | `visible_sequence_description` | Describe an explicitly ordered panel sequence | multi_region | core_candidate |
| 63 | `visual_claim_verification` | Verify a visible claim | evidence_verification | core_candidate |
| 64 | `evidence_localization` | Locate supporting evidence | evidence_verification | core_candidate |
| 65 | `answerability_assessment` | Assess answerability | evidence_verification | core_candidate |
| 66 | `geometric_constraint_solving` | Solve explicit geometric constraints | validated_extensions | validator_gated_extension |
| 67 | `diagram_to_code` | Reconstruct a diagram as code | validated_extensions | validator_gated_extension |
| 68 | `screen_to_code` | Reconstruct a screenshot as markup | validated_extensions | validator_gated_extension |
| 69 | `music_notation_reading` | Read music notation | validated_extensions | validator_gated_extension |
| 70 | `chemical_structure_reading` | Read a chemical structure | validated_extensions | validator_gated_extension |
| 71 | `circuit_structure_reading` | Read a circuit topology | validated_extensions | validator_gated_extension |
| 72 | `ui_action_specification` | Specify one local UI action | validated_extensions | validator_gated_extension |

## Detailed operation contracts

### 01. object_identification

Name a resolved visible entity at the finest category supported by its appearance, including generic person categories; do not infer an individual's identity or an exact product model.

**Required image capabilities:** `visible_entity`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. The chosen ordinary category is justified by the image; do not guess named identities or obscure fine-grained classes.

**Question example (only when its referenced objects and conditions are actually present):** What object is on the left of the desk?

**Do not infer:** Unseen brand, exact model, named identity, or private personal attributes.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `cocoqa`, `lvis_instruct4v`, `vqav2`, `objects365_qa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1505.02074 ; https://arxiv.org/abs/2311.07574 ; https://arxiv.org/abs/1612.00837 ; https://www.objects365.org/overview.html ; https://arxiv.org/html/2510.17269v2

### 02. attribute_lookup

Report a visible property of a resolved object, including color, shape, material appearance, posture, or externally visible state.

**Required image capabilities:** `visible_entity`, `visible_attribute`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. The requested property is visually supported and not a hidden or sensitive personal attribute.

**Question example (only when its referenced objects and conditions are actually present):** What color is the lid?

**Do not infer:** Internal emotion, material composition, intent, health, real-world dimensions, or hidden state.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `coco_colors`, `cocoqa`, `lnqa`, `vqav2`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://huggingface.co/datasets/hazal-karakus/mscoco-controlnet-canny-less-colors ; https://arxiv.org/abs/1505.02074 ; https://vikhyat.net/posts/2024-08-17-lnqa.html ; https://arxiv.org/abs/1612.00837

### 03. visible_action_relation

Describe the action or interaction directly supported by body configuration and object contact in this still image.

**Required image capabilities:** `visible_entity`, `visible_interaction`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Visible pose/contact supports the description; temporal or intentional claims are excluded.

**Question example (only when its referenced objects and conditions are actually present):** What is the person holding?

**Do not infer:** A preceding event, future action, intent, or speed inferred from a still frame.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `sharegpt4v(coco)`, `LLaVA_Instruct_150K`, `vqav2`, `drivelm`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2311.12793 ; https://arxiv.org/abs/2304.08485 ; https://arxiv.org/abs/1612.00837 ; https://arxiv.org/abs/2312.14150

### 04. scene_categorization

Classify the broad visible scene using declared ordinary scene categories and observable evidence.

**Required image capabilities:** `scene_context`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. The chosen ordinary category is justified by the image; do not guess named identities or obscure fine-grained classes.

**Question example (only when its referenced objects and conditions are actually present):** Does this image show a kitchen, a bedroom, or an outdoor scene?

**Do not infer:** A precise geographical location or landmark name from outside knowledge.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {"category_set": "must be public in the question or a versioned ordinary-category vocabulary"}

**FineVision inspirations:** `indoor_qa`, `vision_flan(filtered)`, `vqav2`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://huggingface.co/datasets/keremberke/indoor-scene-classification ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2402.11690 ; https://arxiv.org/abs/1612.00837

### 05. grounded_description

Describe the objects, attributes, visible interactions, spatial relations, and readable text relevant to the specified scene or region.

**Required image capabilities:** `resolvable_region`, `multiple_facts`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Every description clause has a visible basis within scope.

**Question example (only when its referenced objects and conditions are actually present):** Describe the objects and their arrangement in the lower half of the image.

**Do not infer:** Invented backstory, identity, hidden objects, or reasons for events.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {"scope": "whole image or resolved region", "detail_level": ["brief", "standard", "detailed"]}

**FineVision inspirations:** `densefusion_1m`, `image_textualization(filtered)`, `laion_gpt4v`, `localized_narratives`, `sharegpt4o`, `sharegpt4v(llava)`, `sharegpt4v(sam)`, `textcaps`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/20ffc2b42c7de4a1960cfdadf305bbe2-Abstract-Datasets_and_Benchmarks_Track.html ; https://arxiv.org/abs/2406.07502 ; https://huggingface.co/datasets/laion/gpt4v-dataset ; https://link.springer.com/10.1007/978-3-030-58558-7_38 ; https://sharegpt4o.github.io/ ; https://arxiv.org/abs/2311.12793 ; https://arxiv.org/abs/2003.12462

### 06. visual_summary

Select and compress the main visible facts into a requested high-level account instead of enumerating every detail.

**Required image capabilities:** `scene_context`, `multiple_facts`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Compression preserves source meaning and does not introduce an ungrounded fact.

**Question example (only when its referenced objects and conditions are actually present):** Summarize what is visible in this image in one sentence.

**Do not infer:** An unstated event purpose or unsupported emotional narrative.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {"detail_level": ["brief", "standard"]}

**FineVision inspirations:** `densefusion_1m`, `image_textualization(filtered)`, `sharegpt4o`, `LLaVA_Instruct_150K`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/20ffc2b42c7de4a1960cfdadf305bbe2-Abstract-Datasets_and_Benchmarks_Track.html ; https://arxiv.org/abs/2406.07502 ; https://sharegpt4o.github.io/ ; https://arxiv.org/abs/2304.08485

### 07. referring_object_resolution

Identify the unique visible target satisfying a combination of attributes and relations in the public referring expression.

**Required image capabilities:** `multiple_entities`, `discriminating_attributes`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify that the public reference identifies one target; if multiple match, revise the question or use an ambiguity profile.

**Question example (only when its referenced objects and conditions are actually present):** Which cup is to the right of the plate and has a blue handle?

**Do not infer:** Choosing among indistinguishable targets without acknowledging ambiguity.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `visual7w`, `lvis_instruct4v`, `clevr`, `super_clevr(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1511.03416 ; https://arxiv.org/abs/2311.07574 ; https://arxiv.org/abs/1612.06890 ; https://github.com/Lizw14/Super-CLEVR ; https://arxiv.org/abs/2406.17294

### 08. referring_expression_generation

Describe a visibly identified target so another reader can distinguish it from the other candidates in the same scope.

**Required image capabilities:** `multiple_entities`, `discriminating_attributes`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify that the public reference identifies one target; if multiple match, revise the question or use an ambiguity profile.

**Question example (only when its referenced objects and conditions are actually present):** Describe the highlighted object so it cannot be confused with the other objects.

**Do not infer:** Using private object IDs, invisible highlights, or non-discriminating descriptions.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {"target_binding": "must already be publicly resolvable; no invented marker"}

**FineVision inspirations:** `localized_narratives`, `lvis_instruct4v`, `visual7w`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://link.springer.com/10.1007/978-3-030-58558-7_38 ; https://arxiv.org/abs/2311.07574 ; https://arxiv.org/abs/1511.03416

### 09. spatial_relation

State a relation between resolved visible targets in an explicitly declared image-plane or supported depth frame.

**Required image capabilities:** `multiple_entities`, `spatial_layout`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Specify image-plane/depth frame and orientation; reject unsupported metric or viewpoint-invariant claims.

**Question example (only when its referenced objects and conditions are actually present):** Where is the bicycle relative to the tree?

**Do not infer:** Metric distance, camera-independent left/right, or uncertain depth ordering.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `spatialsense`, `vsr`, `clevr`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SpatialSense_An_Adversarially_Crowdsourced_Benchmark_for_Spatial_Relation_Recognition_ICCV_2019_paper.html ; https://arxiv.org/abs/2205.00363 ; https://arxiv.org/abs/1612.06890

### 10. spatial_ordering

Order a finite set of visible targets by a specified spatial key and preserve ties or ambiguity.

**Required image capabilities:** `multiple_entities`, `spatial_layout`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Specify image-plane/depth frame and orientation; reject unsupported metric or viewpoint-invariant claims.

**Question example (only when its referenced objects and conditions are actually present):** List the labeled boxes from left to right.

**Do not infer:** Invented order between overlapping or unresolved positions.

**Verification contracts:** `dual_visual_review`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `spatialsense`, `clevr`, `super_clevr(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SpatialSense_An_Adversarially_Crowdsourced_Benchmark_for_Spatial_Relation_Recognition_ICCV_2019_paper.html ; https://arxiv.org/abs/1612.06890 ; https://github.com/Lizw14/Super-CLEVR ; https://arxiv.org/abs/2406.17294

### 11. attribute_comparison

Compare the same directly observable attribute of two or more resolved objects, reporting relevant similarities or differences.

**Required image capabilities:** `multiple_entities`, `comparable_attributes`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Resolve units, scale, attribute, period, and comparison scope before comparing.

**Question example (only when its referenced objects and conditions are actually present):** How do the two bags differ in color and shape?

**Do not infer:** Comparing real-world size from projected size, or unseen quality.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `mimic_cgd`, `mmra`, `vqav2`, `clevr`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://huggingface.co/datasets/pufanyi/MIMICIT ; https://arxiv.org/abs/2306.05425 ; https://arxiv.org/abs/2407.17379 ; https://arxiv.org/abs/1612.00837 ; https://arxiv.org/abs/1612.06890

### 12. visual_correspondence

Match targets using an explicit line, key, identifier, legend, or otherwise unambiguous visual association.

**Required image capabilities:** `explicit_mapping`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. A visible association establishes the link; proximity alone is insufficient.

**Question example (only when its referenced objects and conditions are actually present):** Which picture is connected to label B?

**Do not infer:** Assuming repetition, proximity, or alignment alone establishes a semantic mapping.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `CoSyn_400k_graphic`, `spatialsense`, `clevr`, `ai2d_merged`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SpatialSense_An_Adversarially_Crowdsourced_Benchmark_for_Spatial_Relation_Recognition_ICCV_2019_paper.html ; https://arxiv.org/abs/1612.06890 ; https://huggingface.co/datasets/andito/ai2d-merged ; https://arxiv.org/html/2510.17269v2

### 13. entity_count

Count every member satisfying the public predicates within a visible, closed scope; support verified zero as distinct from unreadable scope.

**Required image capabilities:** `closed_scope`, `countable_entities`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Fix the unit and inclusion rules, then independently count all members.

**Question example (only when its referenced objects and conditions are actually present):** How many red cups are visible on the table?

**Do not infer:** Counting unseen instances or changing the unit from flowers to buds.

**Verification contracts:** `dual_visual_review`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `cocoqa`, `oodvqa`, `tallyqa`, `clevr`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1505.02074 ; https://arxiv.org/abs/2311.16101 ; https://ojs.aaai.org/index.php/AAAI/article/view/4815 ; https://arxiv.org/abs/1612.06890

### 14. predicate_selection

Return the complete set satisfying declared attribute, text, or spatial predicates within a closed visible scope.

**Required image capabilities:** `multiple_entities`, `interpretable_predicates`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Each selection predicate has a visible truth condition; retain AND/OR/NOT scope.

**Question example (only when its referenced objects and conditions are actually present):** Which objects are red and to the left of the bowl?

**Do not infer:** Unspecified criteria, hidden membership, or omitted qualifying members.

**Verification contracts:** `dual_visual_review`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"boolean_operator": ["and", "or", "not"], "negation_scope": "explicit"}

**FineVision inspirations:** `tallyqa`, `clevr`, `super_clevr(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://ojs.aaai.org/index.php/AAAI/article/view/4815 ; https://arxiv.org/abs/1612.06890 ; https://github.com/Lizw14/Super-CLEVR ; https://arxiv.org/abs/2406.17294

### 15. set_operation

Apply a stated set operation to two publicly defined, image-grounded sets; resolve overlapping instances once.

**Required image capabilities:** `multiple_entities`, `interpretable_predicates`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Each selection predicate has a visible truth condition; retain AND/OR/NOT scope.

**Question example (only when its referenced objects and conditions are actually present):** Which shapes are blue but not circles?

**Do not infer:** A complement over an unbounded universe or double-counting the same instance.

**Verification contracts:** `dual_visual_review`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"operator": ["union", "intersection", "difference"]}

**FineVision inspirations:** `clevr`, `clevr_math`, `super_clevr(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1612.06890 ; https://arxiv.org/abs/2208.05358 ; https://github.com/Lizw14/Super-CLEVR ; https://arxiv.org/abs/2406.17294

### 16. attribute_grouping

Partition a specified complete set by a visible key, associating every group with its members or count.

**Required image capabilities:** `multiple_entities`, `visible_attribute`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Define the key, membership rule, and whether groups are mutually exclusive and exhaustive.

**Question example (only when its referenced objects and conditions are actually present):** Group the buttons by color and give the count in each group.

**Do not infer:** A list of category names without member associations or unsupported categories.

**Verification contracts:** `dual_visual_review`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"return": ["members", "counts"], "group_key": "visible and explicit"}

**FineVision inspirations:** `coco_colors`, `tallyqa`, `clevr`, `super_clevr(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://huggingface.co/datasets/hazal-karakus/mscoco-controlnet-canny-less-colors ; https://ojs.aaai.org/index.php/AAAI/article/view/4815 ; https://arxiv.org/abs/1612.06890 ; https://github.com/Lizw14/Super-CLEVR ; https://arxiv.org/abs/2406.17294

### 17. set_cardinality_comparison

Compare the sizes of two completely observable sets, without substituting area or visual density for counting.

**Required image capabilities:** `closed_scope`, `countable_entities`, `interpretable_predicates`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Fix the unit and inclusion rules, then independently count all members.

**Question example (only when its referenced objects and conditions are actually present):** Are there more cups than plates?

**Do not infer:** Judging quantity from apparent occupied area.

**Verification contracts:** `dual_visual_review`, `closed_set_check`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `iconqa`, `tallyqa`, `clevr`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2110.13214 ; https://ojs.aaai.org/index.php/AAAI/article/view/4815 ; https://arxiv.org/abs/1612.06890

### 18. quantified_statement_verification

Verify a proposition containing all, none, some, exactly, or a declared cardinality bound over a closed visible domain.

**Required image capabilities:** `closed_scope`, `interpretable_predicates`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Each selection predicate has a visible truth condition; retain AND/OR/NOT scope.

**Question example (only when its referenced objects and conditions are actually present):** Are all the visible triangles blue?

**Do not infer:** Using a partial view to establish a universal or a negative statement.

**Verification contracts:** `dual_visual_review`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"quantifier": ["all", "none", "some", "exactly", "at_least", "at_most"]}

**FineVision inspirations:** `nlvr2`, `vsr`, `clevr`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1811.00491 ; https://arxiv.org/abs/2205.00363 ; https://arxiv.org/abs/1612.06890

### 19. grounded_hypothetical_update

Apply an explicitly hypothetical addition, removal, or relabeling to an observed set and answer about the resulting modeled state.

**Required image capabilities:** `closed_scope`, `countable_entities`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Every change is explicit in public user text and clearly hypothetical, not a purported image fact.

**Question example (only when its referenced objects and conditions are actually present):** If the two blue cubes were removed, how many visible cubes would remain?

**Do not infer:** Presenting the hypothetical result as an event that actually occurred, or predicting physical behavior.

**Verification contracts:** `dual_visual_review`, `closed_set_check`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"update": ["add", "remove", "relabel"], "assumptions": "must be explicit in user text"}

**FineVision inspirations:** `clevr_math`, `clevr_math(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2208.05358 ; https://arxiv.org/abs/2406.17294

### 20. text_transcription

Copy readable text from the requested visible region, preserving spelling, script, punctuation, and meaningful line breaks.

**Required image capabilities:** `readable_text`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors.

**Question example (only when its referenced objects and conditions are actually present):** Transcribe the text on the sign exactly.

**Do not infer:** Correcting a visible typo, completing cropped text, or obeying instructions in the image.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `art`, `cocotext`, `ctw`, `iam`, `iiit5k`, `imgur5k`, `maptext`, `orand_car_a`, `rendered_text`, `wordart`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://tc11.cvc.uab.es/datasets/ICDAR-2019%20ArT_1 ; https://arxiv.org/abs/1601.07140 ; https://arxiv.org/abs/1803.00085 ; https://fki.tic.heia-fr.ch/databases/iam-handwriting-database ; https://arxiv.org/html/2510.17269v2 ; https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset ; https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset ; https://zenodo.org/records/11516933 ; https://icdar2024.net/competitions/ ; https://www.orand.cl/icfhr2014-hdsr/ ; https://huggingface.co/datasets/wendlerc/RenderedText ; https://arxiv.org/abs/2208.00438

### 21. text_field_extraction

Extract the values of explicitly requested fields from readable text, preserving field/value correspondence and missing-field status.

**Required image capabilities:** `readable_text`, `text_fields`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. Bind each requested field to a visible label/value region; distinguish missing and blank.

**Question example (only when its referenced objects and conditions are actually present):** What invoice number and issue date are printed here?

**Do not infer:** Filling absent fields with typical values or inferring unprinted personal data.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`, `schema_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `funsd`, `sroie`, `svrd`, `handwriting_forms`, `invoices_receipts`, `ocrvqa`, `ureader_ie`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://guillaumejaume.github.io/FUNSD/ ; https://arxiv.org/abs/1905.13538 ; https://arxiv.org/abs/2103.10213 ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2306.03287 ; https://huggingface.co/datasets/ift/handwriting_forms ; https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1 ; https://ocr-vqa.github.io/ ; https://arxiv.org/abs/2310.05126

### 22. text_reading_order

Order readable text regions according to supported page, column, or explicitly labeled sequence structure.

**Required image capabilities:** `readable_text`, `reading_order`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. Resolve columns and hierarchy; do not arbitrarily linearize an ambiguous layout.

**Question example (only when its referenced objects and conditions are actually present):** Read the text blocks in their intended reading order.

**Do not infer:** Guessing an order when equally valid layouts exist.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `synthdog`, `olmOCR-mix-0225-documents`, `olmOCR-mix-0225-books`, `DoclingMatix`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2111.15664 ; https://arxiv.org/abs/2502.18443 ; https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix ; https://arxiv.org/abs/2503.11576

### 23. label_value_linking

Recover explicit associations between readable labels and their values, including separated form fields and legends.

**Required image capabilities:** `readable_text`, `explicit_mapping`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. A visible association establishes the link; proximity alone is insufficient.

**Question example (only when its referenced objects and conditions are actually present):** Which value belongs to the label Net weight?

**Do not infer:** Pairing nearby values with labels when the layout does not establish the link.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `funsd`, `sroie`, `svrd`, `handwriting_forms`, `ureader_ie`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://guillaumejaume.github.io/FUNSD/ ; https://arxiv.org/abs/1905.13538 ; https://arxiv.org/abs/2103.10213 ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2306.03287 ; https://huggingface.co/datasets/ift/handwriting_forms ; https://arxiv.org/abs/2310.05126

### 24. text_visual_binding

Associate readable text with depicted objects or regions, or explain a directly visible agreement or contrast between text and depiction.

**Required image capabilities:** `readable_text`, `text_object_alignment`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. A visible association establishes the link; proximity alone is insufficient.

**Question example (only when its referenced objects and conditions are actually present):** Which displayed product is identified by the label Green tea?

**Do not infer:** An inferred cultural joke, author intent, or unshown product-price association.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `textcaps`, `chinesememe`, `llavar_gpt4_20k`, `est_vqa`, `st_vqa`, `textocr(gpt4v)`, `textvqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2003.12462 ; https://huggingface.co/datasets/REILX/chinese-meme-description-dataset ; https://arxiv.org/abs/2306.17107 ; https://arxiv.org/abs/2002.10215 ; https://arxiv.org/abs/1905.13648 ; https://huggingface.co/datasets/jimmycarter/textocr-gpt4v ; https://arxiv.org/abs/1904.08920

### 25. formula_transcription

Convert a readable two-dimensional formula to a declared notation while preserving symbols, grouping, scripts, and structure; do not solve it.

**Required image capabilities:** `readable_formula`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Declare the representation convention and preserve all visible structure.

**Question example (only when its referenced objects and conditions are actually present):** Transcribe the displayed equation into LaTeX without solving it.

**Do not infer:** Repairing a source equation or substituting an equivalent but differently written formula.

**Verification contracts:** `dual_visual_review`, `formula_structure_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"notation": ["latex", "unicode_math"]}

**FineVision inspirations:** `chrome_writting`, `hme100k`, `k12_printing`, `latex_handwritten`, `latexformulas`, `mathwriting-google`, `SynthFormulaNet`, `tal_ocr_eng`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/html/2510.17269v2 ; https://ieeexplore.ieee.org/document/6628849 ; https://ai.100tal.com/dataset ; https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data ; https://sujayr91.github.io/Im2Latex/ ; https://huggingface.co/datasets/OleehyO/latex-formulas ; https://arxiv.org/abs/2404.10690 ; https://arxiv.org/abs/2503.11576

### 26. code_transcription

Copy visible program text with its indentation, strings, punctuation, and line structure; retain visible errors.

**Required image capabilities:** `readable_code`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors.

**Question example (only when its referenced objects and conditions are actually present):** Transcribe the code in the image, preserving indentation.

**Do not infer:** Executing code, silently fixing it, or adding hidden imports.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`. These are requirements, not provided runtime implementations.

**Parameters:** {"execution": "forbidden", "source_errors": "preserve"}

**FineVision inspirations:** `SynthCodeNet`, `DoclingMatix`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2503.11576 ; https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix

### 27. document_extractive_qa

Find and return text in the visible document that answers the question, preserving qualifiers and the requested scope.

**Required image capabilities:** `readable_prose`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. The visible document contains answer evidence; an unprovided page is not evidence.

**Question example (only when its referenced objects and conditions are actually present):** According to this notice, when does registration close?

**Do not infer:** Answering from world knowledge or from a page not in the image.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `hw_squad`, `bentham`, `docvqa`, `pdfvqa`, `screenqa`, `ureader_qa_processed`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://www.docvqa.org/datasets/benthamqa-and-hw-squad ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2007.00398 ; https://arxiv.org/abs/2304.06447 ; https://arxiv.org/abs/2209.08199 ; https://arxiv.org/abs/2310.05126

### 28. document_evidence_synthesis

Combine at least two distinct visible text, table, or figure regions to answer one question whose required result is not a single copied span.

**Required image capabilities:** `multiple_evidence_regions`, `readable_prose`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. All indispensable panels, pages, labels, choices, and text are present in the one input image.

**Question example (only when its referenced objects and conditions are actually present):** Using the table and the note below it, explain which entries are included in the total.

**Do not infer:** Importing another page or causal claims not established by the visible document.

**Verification contracts:** `dual_visual_review`, `evidence_binding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `finqa`, `multihiertt`, `tat_dqa`, `tat_qa`, `infographic_vqa`, `slidevqa`, `visualmrc`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2109.00122 ; https://arxiv.org/abs/2206.01347 ; https://nextplusplus.github.io/TAT-DQA/ ; https://arxiv.org/abs/2207.11871 ; https://arxiv.org/abs/2105.07624 ; https://arxiv.org/abs/2104.12756 ; https://arxiv.org/abs/2301.04883 ; https://arxiv.org/abs/2101.11272

### 29. document_summary

Summarize the important claims of a visible document in the requested scope, attributing claims to the document rather than asserting external truth.

**Required image capabilities:** `readable_prose`, `multiple_facts`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. Compression preserves source meaning and does not introduce an ungrounded fact.

**Question example (only when its referenced objects and conditions are actually present):** Summarize the three main points of this notice.

**Do not infer:** Fabricated background, omitted negation, or interpreting document claims as independently verified facts.

**Verification contracts:** `dual_visual_review`, `evidence_binding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `DoclingMatix`, `sujet_finance`, `ureader_cap`, `ureader_kg_processed`, `visualmrc`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix ; https://arxiv.org/abs/2503.11576 ; https://huggingface.co/datasets/sujet-ai/Sujet-Finance-QA-Vision-100k ; https://arxiv.org/abs/2310.05126 ; https://arxiv.org/abs/2101.11272

### 30. document_structure_reconstruction

Serialize visible headings, paragraphs, lists, tables, formulas, and their hierarchy in a declared schema while retaining content and reading order.

**Required image capabilities:** `document_layout`, `readable_text`, `reading_order`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. Resolve columns and hierarchy; do not arbitrarily linearize an ambiguous layout. The required output structure is stated in the user instruction; private schema details must not leak.

**Question example (only when its referenced objects and conditions are actually present):** Convert this visible page into structured Markdown, preserving headings, lists, and tables.

**Do not infer:** Inventing off-page sections, unseen table cells, or unsupported document metadata.

**Verification contracts:** `dual_visual_review`, `transcript_alignment`, `schema_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"format": ["markdown", "html_subset", "structured_json"], "scope": "visible page or explicit bounded region"}

**FineVision inspirations:** `synthdog`, `olmOCR-mix-0225-documents`, `olmOCR-mix-0225-books`, `DoclingMatix`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2111.15664 ; https://arxiv.org/abs/2502.18443 ; https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix ; https://arxiv.org/abs/2503.11576

### 31. cross_region_consistency_check

Compare two explicitly linked statements or values within the same image and report whether they agree, preserving units, dates, and scope.

**Required image capabilities:** `multiple_evidence_regions`, `explicit_cross_references`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. All indispensable panels, pages, labels, choices, and text are present in the one input image. Resolve units, scale, attribute, period, and comparison scope before comparing.

**Question example (only when its referenced objects and conditions are actually present):** Does the total stated in the paragraph match the total labeled in the table?

**Do not infer:** Equating unlike reporting periods or judging whether a financial claim is true outside the document.

**Verification contracts:** `dual_visual_review`, `evidence_binding_check`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `finqa`, `multihiertt`, `tat_dqa`, `tat_qa`, `infographic_vqa`, `sujet_finance`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2109.00122 ; https://arxiv.org/abs/2206.01347 ; https://nextplusplus.github.io/TAT-DQA/ ; https://arxiv.org/abs/2207.11871 ; https://arxiv.org/abs/2105.07624 ; https://arxiv.org/abs/2104.12756 ; https://huggingface.co/datasets/sujet-ai/Sujet-Finance-QA-Vision-100k

### 32. document_layout_role

Identify visible titles, section headings, captions, body paragraphs, table headers, lists, or footnotes from layout and content.

**Required image capabilities:** `document_layout`, `readable_text`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors. Use layout and readable content jointly to assign a document role.

**Question example (only when its referenced objects and conditions are actually present):** Which text is the caption for the figure?

**Do not infer:** Inferring hidden document structure or assigning a role based only on font size.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `olmOCR-mix-0225-documents`, `olmOCR-mix-0225-books`, `DoclingMatix`, `pdfvqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.18443 ; https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix ; https://arxiv.org/abs/2503.11576 ; https://arxiv.org/abs/2304.06447

### 33. table_cell_lookup

Resolve row and column paths, including multilevel headers, and retrieve the requested visible cell or cells without performing an aggregate.

**Required image capabilities:** `readable_table`, `table_headers`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Resolve full row/column header paths, spans, units, and footnotes; preserve blanks separately from zero. Read at actual delivered resolution; preserve unknown/cropped text instead of completing it from priors.

**Question example (only when its referenced objects and conditions are actually present):** What value is listed for Europe, 2024, under Revenue?

**Do not infer:** Ignoring a parent header, unit, footnote, or merged-cell scope.

**Verification contracts:** `dual_visual_review`, `table_structure_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `CoSyn_400k_table`, `hitab`, `robut_wikisql`, `robut_wtq`, `vqaonbd`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://arxiv.org/abs/2108.06712 ; https://arxiv.org/abs/2306.14321 ; https://ilocr.iiit.ac.in/vqabd/dataset.html

### 34. table_predicate_selection

Select and optionally order complete table rows or columns according to explicit predicates and visible cells.

**Required image capabilities:** `readable_table`, `table_headers`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Resolve full row/column header paths, spans, units, and footnotes; preserve blanks separately from zero. Each selection predicate has a visible truth condition; retain AND/OR/NOT scope.

**Question example (only when its referenced objects and conditions are actually present):** Which rows have quantity above 10? Return their names in descending quantity order.

**Do not infer:** Filtering on a missing column or returning only some qualifying records.

**Verification contracts:** `dual_visual_review`, `table_structure_check`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `CoSyn_400k_table`, `hitab`, `robut_wikisql`, `robut_wtq`, `tabmwp`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://arxiv.org/abs/2108.06712 ; https://arxiv.org/abs/2306.14321 ; https://arxiv.org/abs/2209.14610

### 35. table_structure_reconstruction

Recover the visible table cells, row/column alignment, header hierarchy, and spans in a declared structured representation.

**Required image capabilities:** `readable_table`, `table_headers`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Resolve full row/column header paths, spans, units, and footnotes; preserve blanks separately from zero. The required output structure is stated in the user instruction; private schema details must not leak.

**Question example (only when its referenced objects and conditions are actually present):** Convert the table to JSON while preserving its grouped column headings.

**Do not infer:** Inventing cells, collapsing meaningful headers, or treating blank as zero.

**Verification contracts:** `dual_visual_review`, `table_structure_check`, `schema_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"format": ["html_table", "structured_json", "markdown_simple_only"]}

**FineVision inspirations:** `CoSyn_400k_table`, `hitab`, `vqaonbd`, `DoclingMatix`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://arxiv.org/abs/2108.06712 ; https://ilocr.iiit.ac.in/vqabd/dataset.html ; https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix ; https://arxiv.org/abs/2503.11576

### 36. table_cross_reference

Follow explicit shared keys between two or more tables visible in the same image to retrieve or assemble related records.

**Required image capabilities:** `multiple_tables`, `table_join_keys`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. All indispensable panels, pages, labels, choices, and text are present in the one input image. Validate shared keys, multiplicity, and the join relation before joining.

**Question example (only when its referenced objects and conditions are actually present):** Using the product IDs, match the items in the left table with their categories in the right table.

**Do not infer:** Joining unrelated rows by proximity or relying on another missing page.

**Verification contracts:** `dual_visual_review`, `table_structure_check`, `evidence_binding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `multihiertt`, `tat_dqa`, `vqaonbd`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2206.01347 ; https://nextplusplus.github.io/TAT-DQA/ ; https://arxiv.org/abs/2207.11871 ; https://ilocr.iiit.ac.in/vqabd/dataset.html

### 37. chart_encoding_lookup

Identify the meaning of an axis, legend color, symbol, map bin, or series label from an explicit readable encoding.

**Required image capabilities:** `readable_chart`, `chart_encoding`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings.

**Question example (only when its referenced objects and conditions are actually present):** What range does the darkest map color represent?

**Do not infer:** Geographical or domain knowledge that is not provided by the labels or legend.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `dvqa`, `mmc_instruct`, `plotqa`, `mapqa`, `mapqa(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1801.08163 ; https://arxiv.org/abs/2311.10774 ; https://arxiv.org/abs/1909.00997 ; https://arxiv.org/abs/2211.08545 ; https://arxiv.org/abs/2406.17294

### 38. chart_value_lookup

Retrieve a displayed value or a declared-precision estimate from a resolved chart mark or thematic-map region.

**Required image capabilities:** `readable_chart`, `chart_encoding`, `legible_values`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings. Use printed exact values or public approximation/rounding/interval rules; never invent precision.

**Question example (only when its referenced objects and conditions are actually present):** What value is shown for series A in March?

**Do not infer:** Fabricating exact decimals from pixels, ignoring logarithmic axes, or converting a color interval into an exact value.

**Verification contracts:** `dual_visual_review`, `chart_encoding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"precision": ["explicit_label", "calibrated_estimate", "interval"]}

**FineVision inspirations:** `chartqa`, `dvqa`, `plotqa`, `Unichart`, `mapqa`, `mapqa(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2203.10244 ; https://arxiv.org/abs/1801.08163 ; https://arxiv.org/abs/1909.00997 ; https://arxiv.org/abs/2305.14761 ; https://arxiv.org/abs/2211.08545 ; https://arxiv.org/abs/2406.17294

### 39. chart_comparison

Compare explicitly specified values or categories in a shared compatible chart encoding, without inventing an exact difference.

**Required image capabilities:** `readable_chart`, `comparable_series`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings. Resolve units, scale, attribute, period, and comparison scope before comparing.

**Question example (only when its referenced objects and conditions are actually present):** Which series is higher at the labeled year 2024?

**Do not infer:** Comparing pixel height across incompatible axes.

**Verification contracts:** `dual_visual_review`, `chart_encoding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `chartqa`, `figureqa`, `figureqa(mathv360k)`, `plotqa`, `Unichart`, `mapqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2203.10244 ; https://arxiv.org/abs/1710.07300 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/1909.00997 ; https://arxiv.org/abs/2305.14761 ; https://arxiv.org/abs/2211.08545

### 40. chart_extremum_ranking

Identify all minima, maxima, ties, or a requested order over a complete visible series or set of chart categories.

**Required image capabilities:** `readable_chart`, `complete_series`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings.

**Question example (only when its referenced objects and conditions are actually present):** Which category has the maximum value? Include any ties.

**Do not infer:** Selecting a local extremum as the global maximum when parts of the series are missing.

**Verification contracts:** `dual_visual_review`, `chart_encoding_check`, `closed_set_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `chart2text`, `CoSyn_400k_chart`, `figureqa`, `figureqa(mathv360k)`, `vistext`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2203.06486 ; https://arxiv.org/abs/2502.14846 ; https://huggingface.co/datasets/HuggingFaceM4/FineVision ; https://arxiv.org/abs/1710.07300 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/2307.05356

### 41. chart_trend_summary

Describe visible rises, falls, plateaus, turning regions, or major comparisons over the declared axis range.

**Required image capabilities:** `readable_chart`, `ordered_series`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings. Compression preserves source meaning and does not introduce an ungrounded fact.

**Question example (only when its referenced objects and conditions are actually present):** Describe the trend across the displayed years without speculating about its cause.

**Do not infer:** Forecasting outside the range or attributing a cause to correlation.

**Verification contracts:** `dual_visual_review`, `chart_encoding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `chart2text`, `CoSyn_400k_chart`, `Unichart`, `vistext`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2203.06486 ; https://arxiv.org/abs/2502.14846 ; https://huggingface.co/datasets/HuggingFaceM4/FineVision ; https://arxiv.org/abs/2305.14761 ; https://arxiv.org/abs/2307.05356

### 42. chart_series_relation

Determine intersections, relative dominance, or visibly supported variation of complete series on comparable axes.

**Required image capabilities:** `readable_chart`, `comparable_series`, `complete_series`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings. Resolve units, scale, attribute, period, and comparison scope before comparing. Use printed exact values or public approximation/rounding/interval rules; never invent precision.

**Question example (only when its referenced objects and conditions are actually present):** Do the two curves cross within the displayed interval? Where approximately?

**Do not infer:** Exact intersection coordinates or curve integrals not recoverable at the available precision.

**Verification contracts:** `dual_visual_review`, `chart_encoding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"relation": ["intersection", "dominance", "variation"], "numeric_area": "route to grounded_arithmetic with verified samples and explicit integration rule"}

**FineVision inspirations:** `figureqa`, `figureqa(mathv360k)`, `mmc_instruct`, `plotqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1710.07300 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/2311.10774 ; https://arxiv.org/abs/1909.00997

### 43. chart_data_reconstruction

Recover visible category/series/value associations as a structured table, explicitly marking intervals or approximations where exact values are not legible.

**Required image capabilities:** `readable_chart`, `chart_encoding`, `complete_series`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Check axes, legends, linear/log scales, zero baselines, units, and interval encodings. Use printed exact values or public approximation/rounding/interval rules; never invent precision. The required output structure is stated in the user instruction; private schema details must not leak.

**Question example (only when its referenced objects and conditions are actually present):** Convert the labeled bars and their values into a table.

**Do not infer:** Pretending an exact recovery of hidden source data from approximate graphics.

**Verification contracts:** `dual_visual_review`, `chart_encoding_check`, `schema_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `mmc_instruct`, `SynthChartNet`, `Unichart`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2311.10774 ; https://arxiv.org/abs/2503.11576 ; https://arxiv.org/abs/2305.14761

### 44. quantity_comparison

Compare quantities extracted from visible sources after resolving units, signs, scales, dates, and the requested ordering.

**Required image capabilities:** `typed_operands`, `explicit_units`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Each operand refers to a visible value or a verified image-derived public-history value. Resolve units, scale, attribute, period, and comparison scope before comparing.

**Question example (only when its referenced objects and conditions are actually present):** Which labeled package has the larger net weight?

**Do not infer:** Comparing incomparable quantities, unlike periods, or currencies without a supplied exchange rule.

**Verification contracts:** `dual_visual_review`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"dimensionless_values": "allowed when explicitly identified as dimensionless"}

**FineVision inspirations:** `chartqa`, `finqa`, `robut_wtq`, `vqaonbd`, `CoSyn_400k_nutrition`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2203.10244 ; https://arxiv.org/abs/2109.00122 ; https://arxiv.org/abs/2306.14321 ; https://ilocr.iiit.ac.in/vqabd/dataset.html ; https://arxiv.org/abs/2502.14846 ; https://arxiv.org/html/2510.17269v2

### 45. grounded_arithmetic

Compute a finite expression over image-grounded operands, with an explicit public question defining the operation, denominator, units, and rounding.

**Required image capabilities:** `typed_operands`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Each operand refers to a visible value or a verified image-derived public-history value. The requested computation or reduction, operand order, denominator, and domain are explicit. Use printed exact values or public approximation/rounding/interval rules; never invent precision.

**Question example (only when its referenced objects and conditions are actually present):** Using the two printed values, what is the percentage increase from the first to the second?

**Do not infer:** Unseen prices or conversion rates, an implicit denominator, or unsupported intermediate values.

**Verification contracts:** `dual_visual_review`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"operators": ["add", "subtract", "multiply", "divide"], "derived_forms": ["ratio", "percentage", "percentage_change"], "division_by_zero": "not answerable"}

**FineVision inspirations:** `chartqa`, `finqa`, `multihiertt`, `plotqa`, `tabmwp`, `tat_dqa`, `tat_qa`, `vqaonbd`, `CoSyn_400k_nutrition`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2203.10244 ; https://arxiv.org/abs/2109.00122 ; https://arxiv.org/abs/2206.01347 ; https://arxiv.org/abs/1909.00997 ; https://arxiv.org/abs/2209.14610 ; https://nextplusplus.github.io/TAT-DQA/ ; https://arxiv.org/abs/2207.11871 ; https://arxiv.org/abs/2105.07624 ; https://ilocr.iiit.ac.in/vqabd/dataset.html ; https://arxiv.org/abs/2502.14846 ; https://arxiv.org/html/2510.17269v2

### 46. grounded_aggregation

Compute a declared reduction over a complete image-grounded set of values, retaining selection criteria, units, and any visible weights.

**Required image capabilities:** `typed_operands`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. Each operand refers to a visible value or a verified image-derived public-history value. The requested computation or reduction, operand order, denominator, and domain are explicit.

**Question example (only when its referenced objects and conditions are actually present):** What is the mean of the values in the three labeled rows?

**Do not infer:** Omitted rows, treating blanks as zero, or supplying unprinted weights.

**Verification contracts:** `dual_visual_review`, `closed_set_check`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"operator": ["sum", "mean", "median", "min", "max", "weighted_mean"], "weights": "all visible or explicitly supplied as hypothetical parameters"}

**FineVision inspirations:** `CoSyn_400k_table`, `finqa`, `hitab`, `multihiertt`, `robut_wikisql`, `robut_wtq`, `tabmwp`, `vqaonbd`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://arxiv.org/abs/2109.00122 ; https://arxiv.org/abs/2108.06712 ; https://arxiv.org/abs/2206.01347 ; https://arxiv.org/abs/2306.14321 ; https://arxiv.org/abs/2209.14610 ; https://ilocr.iiit.ac.in/vqabd/dataset.html

### 47. unit_conversion

Express a visible quantity in another unit using a versioned exact conversion rule or a conversion rule stated publicly in the question.

**Required image capabilities:** `typed_operands`, `explicit_units`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Each operand refers to a visible value or a verified image-derived public-history value. Only use a versioned permitted conversion or a public hypothetical conversion; reject missing/live-world factors.

**Question example (only when its referenced objects and conditions are actually present):** Convert the printed length of 2.5 m to centimeters.

**Do not infer:** Exchange rates, serving sizes, density, calibration, or physical constants absent from the permitted rule set.

**Verification contracts:** `dual_visual_review`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"conversion_rule": "versioned allowlist; no live-world values"}

**FineVision inspirations:** `tabmwp`, `vqaonbd`, `CoSyn_400k_nutrition`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2209.14610 ; https://ilocr.iiit.ac.in/vqabd/dataset.html ; https://arxiv.org/abs/2502.14846 ; https://arxiv.org/html/2510.17269v2

### 48. measurement_reading

Read a clock, ruler, gauge, or scale from unambiguous marks and pointers at the available precision.

**Required image capabilities:** `calibrated_scale`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Resolve scale, pointer, origin, and units at the delivered resolution. Use printed exact values or public approximation/rounding/interval rules; never invent precision.

**Question example (only when its referenced objects and conditions are actually present):** What time is shown on the analog clock?

**Do not infer:** An exact physical measurement without calibration or pretending a value is more precise than the marks.

**Verification contracts:** `dual_visual_review`, `scale_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"precision": "must follow ticks, labels, and visible resolution"}

**FineVision inspirations:** `iconqa`, `iconqa(mathv360k)`, `spark`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2110.13214 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/2408.12114

### 49. diagram_element_lookup

Identify labeled components, geometric marks, nodes, or symbols using visible labels and a declared notation convention.

**Required image capabilities:** `readable_diagram`, `notation_context`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Declare the representation convention and preserve all visible structure.

**Question example (only when its referenced objects and conditions are actually present):** Which component is labeled B in the diagram?

**Do not infer:** Unlabeled specialist functions or an unprovided legend.

**Verification contracts:** `dual_visual_review`, `graph_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `geo170k(align)`, `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `ai2d_merged`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2312.11370 ; https://aclanthology.org/2022.findings-aacl.15/ ; https://arxiv.org/abs/2502.14846 ; https://huggingface.co/datasets/andito/ai2d-merged ; https://arxiv.org/html/2510.17269v2

### 50. graph_connectivity

Determine adjacency, incoming/outgoing connections, or a visible edge set from resolved nodes, arrows, and junction conventions.

**Required image capabilities:** `graph_nodes_edges`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify endpoint identities, edge direction, junction/crossing conventions, and all traversed links.

**Question example (only when its referenced objects and conditions are actually present):** Which nodes are directly connected to A?

**Do not infer:** Treating every line crossing as a junction or using proximity as an edge.

**Verification contracts:** `dual_visual_review`, `graph_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `diagram_image_to_text`, `ai2d_merged`, `CoSyn_400k_circuit`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://aclanthology.org/2022.findings-aacl.15/ ; https://arxiv.org/abs/2502.14846 ; https://huggingface.co/datasets/Kamizuru00/diagram_image_to_text ; https://huggingface.co/datasets/andito/ai2d-merged ; https://arxiv.org/html/2510.17269v2

### 51. graph_path_tracing

Follow one or more declared graph edges between visible endpoints, respecting direction and explicitly reporting multiple valid paths.

**Required image capabilities:** `graph_nodes_edges`, `edge_directions`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify endpoint identities, edge direction, junction/crossing conventions, and all traversed links. Define source, destination, permitted edges, and path objective; retain multiple valid answers.

**Question example (only when its referenced objects and conditions are actually present):** Trace the directed path from Start to the labeled output.

**Do not infer:** An unshown connection, real-world travel permission, or a shortest path without a defined metric.

**Verification contracts:** `dual_visual_review`, `graph_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `diagram_image_to_text`, `ai2d_merged`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://aclanthology.org/2022.findings-aacl.15/ ; https://arxiv.org/abs/2502.14846 ; https://huggingface.co/datasets/Kamizuru00/diagram_image_to_text ; https://huggingface.co/datasets/andito/ai2d-merged ; https://arxiv.org/html/2510.17269v2

### 52. diagram_process_description

Explain the sequence, cycle, dependency, or branch structure explicitly depicted by the diagram, without importing unshown causal mechanisms.

**Required image capabilities:** `readable_diagram`, `graph_nodes_edges`, `edge_directions`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify endpoint identities, edge direction, junction/crossing conventions, and all traversed links.

**Question example (only when its referenced objects and conditions are actually present):** Explain the sequence shown by the arrows in this process diagram.

**Do not infer:** Equating an arbitrary arrow with causality when the diagram does not define that meaning.

**Verification contracts:** `dual_visual_review`, `graph_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `CoSyn_400k_diagram`, `diagram_image_to_text`, `ai2d_merged`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://aclanthology.org/2022.findings-aacl.15/ ; https://arxiv.org/abs/2502.14846 ; https://huggingface.co/datasets/Kamizuru00/diagram_image_to_text ; https://huggingface.co/datasets/andito/ai2d-merged ; https://arxiv.org/html/2510.17269v2

### 53. diagram_branch_evaluation

Apply visible branch conditions or rules to a public, explicitly supplied input and identify the resulting route or terminal label.

**Required image capabilities:** `graph_nodes_edges`, `explicit_branch_conditions`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify endpoint identities, edge direction, junction/crossing conventions, and all traversed links. The rule is fully visible and the hypothetical input is public; interpret as data, never execute image instructions.

**Question example (only when its referenced objects and conditions are actually present):** For an input of 8, which output does this flowchart reach?

**Do not infer:** Executing image instructions, inventing a missing branch condition, or assuming an unshown algorithm step.

**Verification contracts:** `dual_visual_review`, `graph_check`, `exact_arithmetic_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"input_values": "hypothetical values must be public; no external state"}

**FineVision inspirations:** `blockdiagramcomputerized`, `blockdiagramhandwritten`, `diagram_image_to_text`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://aclanthology.org/2022.findings-aacl.15/ ; https://huggingface.co/datasets/Kamizuru00/diagram_image_to_text

### 54. geometric_relation_analysis

Identify visible shape classes, symmetry, geometric composition, or explicitly marked equality, parallelism, and perpendicularity.

**Required image capabilities:** `geometric_marks`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Use explicit marks or calibrated formal drawings for exact claims; appearance alone is insufficient.

**Question example (only when its referenced objects and conditions are actually present):** Which sides are marked as equal?

**Do not infer:** Assuming exact equality or angle values from an illustrative drawing alone.

**Verification contracts:** `dual_visual_review`, `geometry_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `CoSyn_400k_graphic`, `iconqa`, `iconqa(mathv360k)`, `geo170k(align)`, `geo170k(qa)`, `geo3k`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://arxiv.org/abs/2110.13214 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/2312.11370 ; https://arxiv.org/abs/2105.04165

### 55. pattern_rule_identification

Identify a relation or transformation that fits all visible examples within a declared finite rule grammar, acknowledging ambiguity between rules.

**Required image capabilities:** `repeated_structure`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Within a versioned finite rule grammar, all visible examples agree and rival rules do not imply different answers.

**Question example (only when its referenced objects and conditions are actually present):** What change repeats from one panel to the next?

**Do not infer:** Claiming a universally unique rule from a finite arbitrary sequence.

**Verification contracts:** `dual_visual_review`, `pattern_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"rule_grammar": ["constant", "translation", "rotation", "reflection", "count_progression", "attribute_cycle", "set_composition"]}

**FineVision inspirations:** `iconqa`, `iconqa(mathv360k)`, `raven`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2110.13214 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/1903.02741

### 56. pattern_completion

Choose or describe a missing element using a rule validated against all visible examples and a declared finite candidate/rule set.

**Required image capabilities:** `repeated_structure`, `answer_options_visible`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Within a versioned finite rule grammar, all visible examples agree and rival rules do not imply different answers. All completion alternatives used in the answer are visible or explicitly defined in the public instruction.

**Question example (only when its referenced objects and conditions are actually present):** Which of the visible options completes the matrix?

**Do not infer:** Using a hidden answer key or accepting one option while another fits equally well.

**Verification contracts:** `dual_visual_review`, `pattern_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `iconqa`, `iconqa(mathv360k)`, `raven`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2110.13214 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/1903.02741

### 57. rule_based_exception

Identify the element that violates a visible or publicly stated rule within a complete comparison set.

**Required image capabilities:** `repeated_structure`, `closed_scope`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Determine the closed universe and ensure no relevant area is cropped, hidden, or unreadable. Unknown is not empty. State or uniquely establish the rule before choosing an exception; no subjective odd-one-out.

**Question example (only when its referenced objects and conditions are actually present):** Under the displayed color-alternation rule, which panel breaks the pattern?

**Do not infer:** Subjective odd-one-out choices without an agreed rule or invented abnormalities.

**Verification contracts:** `dual_visual_review`, `pattern_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `iconqa`, `iconqa(mathv360k)`, `raven`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2110.13214 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/1903.02741

### 58. ui_element_grounding

Identify a visible control that matches a public referring expression or locally supported goal; return a textual location by default.

**Required image capabilities:** `ui_controls`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify that the public reference identifies one target; if multiple match, revise the question or use an ambiguity profile. The target is on-screen and tied to a public local request.

**Question example (only when its referenced objects and conditions are actually present):** Where is the search box in this screenshot?

**Do not infer:** An off-screen control, hidden DOM state, or a claim that an action was executed.

**Verification contracts:** `dual_visual_review`, `ui_grounding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"representation": "textual_reference", "coordinate_output": "use ui_action_specification with a coordinate validator"}

**FineVision inspirations:** `aguvis-stage-1`, `groundui`, `screenqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2412.04454 ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2403.17918 ; https://arxiv.org/abs/2209.08199

### 59. ui_state_reading

Report the visibly indicated selected tab, checked option, displayed error, or control state using the screenshot only.

**Required image capabilities:** `ui_controls`, `ui_state_indicators`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. A state indication is explicit enough; gray/color alone does not prove backend enablement.

**Question example (only when its referenced objects and conditions are actually present):** Which tab is currently selected?

**Do not infer:** Backend state, user permissions, or disabled/enabled status inferred from color alone.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `aguvis-stage-1`, `groundui`, `screenqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2412.04454 ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2403.17918 ; https://arxiv.org/abs/2209.08199

### 60. screen_summary

Summarize the main visible content and apparent interface function from explicit controls, headings, and text.

**Required image capabilities:** `ui_controls`, `scene_context`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Compression preserves source meaning and does not introduce an ungrounded fact.

**Question example (only when its referenced objects and conditions are actually present):** What information and controls are shown on this screen?

**Do not infer:** Hidden app features, the user goal, or a workflow not shown.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `websight`, `screen2words`, `screenqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2403.09029 ; https://arxiv.org/abs/2108.03353 ; https://arxiv.org/abs/2209.08199

### 61. panel_comparison

Describe visible similarities and differences between two or more explicitly resolved panels of the same input canvas.

**Required image capabilities:** `panels_resolvable`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. All indispensable panels, pages, labels, choices, and text are present in the one input image. Identify existing panels by labels or unambiguous layout; do not create unseen comparison images.

**Question example (only when its referenced objects and conditions are actually present):** What differs between the left and right panels?

**Do not infer:** Inventing a second image or presenting unrelated crops as before/after evidence.

**Verification contracts:** `dual_visual_review`, `panel_comparison_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"canvas_policy": "existing panels only; no implicit multi-image support"}

**FineVision inspirations:** `mimic_cgd`, `mmra`, `nlvr2`, `spot_the_diff`, `yesbut`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://huggingface.co/datasets/pufanyi/MIMICIT ; https://arxiv.org/abs/2306.05425 ; https://arxiv.org/abs/2407.17379 ; https://arxiv.org/abs/1811.00491 ; https://arxiv.org/abs/1808.10584 ; https://arxiv.org/abs/2409.13592

### 62. visible_sequence_description

Describe the depicted changes across panels whose order is explicitly labeled or unambiguously encoded, without filling in unseen events.

**Required image capabilities:** `panels_resolvable`, `visible_sequence_order`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. All indispensable panels, pages, labels, choices, and text are present in the one input image. Require visible numbering, arrows, timestamps, or an explicit sequence convention.

**Question example (only when its referenced objects and conditions are actually present):** Describe what changes from panel 1 to panel 3.

**Do not infer:** Inferring chronology from unlabeled juxtaposition, hidden intermediate actions, or a causal story.

**Verification contracts:** `dual_visual_review`, `panel_comparison_check`. These are requirements, not provided runtime implementations.

**Parameters:** {}

**FineVision inspirations:** `spot_the_diff`, `yesbut`, `ai2d_merged`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/1808.10584 ; https://arxiv.org/abs/2409.13592 ; https://huggingface.co/datasets/andito/ai2d-merged ; https://arxiv.org/html/2510.17269v2

### 63. visual_claim_verification

Determine whether a public claim is supported, contradicted, or not decidable from the image, and name the relevant visible evidence when requested.

**Required image capabilities:** `resolvable_region`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. The proposition is public and relates to visible targets; it is not a hidden desired answer.

**Question example (only when its referenced objects and conditions are actually present):** Is the statement "the box is left of the chair" consistent with the image?

**Do not infer:** A forced binary label when evidence is missing, or guessing unseen facts.

**Verification contracts:** `dual_visual_review`, `evidence_binding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"verdicts": ["supported", "contradicted", "not_determined"]}

**FineVision inspirations:** `idk`, `lnqa`, `lrv_normal(filtered)`, `nlvr2`, `spatialsense`, `vsr`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2402.09717 ; https://vikhyat.net/posts/2024-08-17-lnqa.html ; https://arxiv.org/abs/2306.14565 ; https://arxiv.org/abs/1811.00491 ; https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SpatialSense_An_Adversarially_Crowdsourced_Benchmark_for_Spatial_Relation_Recognition_ICCV_2019_paper.html ; https://arxiv.org/abs/2205.00363

### 64. evidence_localization

Identify the visible region, label, sentence, or connected elements that support an already public question or proposition.

**Required image capabilities:** `resolvable_region`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. The proposition is public and relates to visible targets; it is not a hidden desired answer.

**Question example (only when its referenced objects and conditions are actually present):** Which part of the image shows the closing time?

**Do not infer:** Private evidence IDs, an invented source, or a location without the claimed evidence.

**Verification contracts:** `dual_visual_review`, `evidence_binding_check`. These are requirements, not provided runtime implementations.

**Parameters:** {"default_output": "natural-language region description or exact visible quote"}

**FineVision inspirations:** `est_vqa`, `infographic_vqa`, `pdfvqa`, `slidevqa`, `visualmrc`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2002.10215 ; https://arxiv.org/abs/2104.12756 ; https://arxiv.org/abs/2304.06447 ; https://arxiv.org/abs/2301.04883 ; https://arxiv.org/abs/2101.11272

### 65. answerability_assessment

Determine whether a specific locally anchored question can be answered from the image and identify missing, unreadable, cropped, or ambiguous evidence.

**Required image capabilities:** `resolvable_region`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. The question concerns a visible object/field; the limitation can be locally grounded.

**Question example (only when its referenced objects and conditions are actually present):** Can the exact price be read from this cropped label? Explain the visible limitation.

**Do not infer:** Generic refusal to any unrelated question or using unreadable evidence as a negative fact.

**Verification contracts:** `dual_visual_review`. These are requirements, not provided runtime implementations.

**Parameters:** {"verdicts": ["answerable", "unreadable", "cropped", "ambiguous", "not_present"]}

**FineVision inspirations:** `idk`, `vizwiz(mathv360k)`, `screenqa`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2402.09717 ; https://arxiv.org/abs/1802.08218 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/2209.08199

### 66. geometric_constraint_solving

Solve a numerical or logical geometry question using only visibly stated or marked constraints and a versioned theorem/rule set.

**Required image capabilities:** `geometric_marks`, `readable_formula`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Use explicit marks or calibrated formal drawings for exact claims; appearance alone is insufficient. A versioned formal rule set and qualified validator cover the exact problem type. All visible premises jointly determine the requested solution under the permitted rule set.

**Question example (only when its referenced objects and conditions are actually present):** Using the marked right angle and printed side lengths, find the requested length.

**Do not infer:** Reading exact measurements from an uncalibrated drawing or adding unstated geometric assumptions.

**Verification contracts:** `dual_visual_review`, `formal_geometry_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false}

**FineVision inspirations:** `CoSyn_400k_math`, `geo170k(qa)`, `geo3k`, `geometry3k(mathv360k)`, `geomverse`, `geoqa+(mathv360k)`, `geos(mathv360k)`, `intergps`, `mavis_math_metagen`, `mavis_math_rule_geo`, `unigeo(mathv360k)`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846 ; https://arxiv.org/abs/2312.11370 ; https://arxiv.org/abs/2105.04165 ; https://arxiv.org/abs/2406.17294 ; https://arxiv.org/abs/2312.12241 ; https://aclanthology.org/2022.coling-1.130/ ; https://aclanthology.org/D15-1171/ ; https://arxiv.org/abs/2407.08739 ; https://arxiv.org/abs/2212.02746

### 67. diagram_to_code

Generate sandbox-renderable TikZ or SVG that reproduces the specified visible diagram, accepting equivalent code instead of requiring the hidden original.

**Required image capabilities:** `readable_diagram`, `rendered_layout`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. A sandboxed renderer, asset/font policy, syntax allowlist, and output comparison procedure are configured.

**Question example (only when its referenced objects and conditions are actually present):** Recreate this diagram in SVG, including the visible labels and arrows.

**Do not infer:** Claiming recovery of the original source code, external assets, or arbitrary file/network access.

**Verification contracts:** `dual_visual_review`, `sandbox_render_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false, "code_languages": ["svg", "tikz_subset"]}

**FineVision inspirations:** `datik`, `datikz`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2310.00367 ; https://huggingface.co/datasets/nllg/datikz ; https://huggingface.co/datasets/HuggingFaceM4/datikz

### 68. screen_to_code

Generate sandbox-renderable static HTML/CSS that reproduces the visible layout under declared font, asset, viewport, and approximation rules.

**Required image capabilities:** `ui_controls`, `rendered_layout`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. A sandboxed renderer, asset/font policy, syntax allowlist, and output comparison procedure are configured.

**Question example (only when its referenced objects and conditions are actually present):** Recreate the visible layout as static HTML and CSS.

**Do not infer:** Inferring hidden JavaScript, app behavior, original DOM, credentials, or network assets.

**Verification contracts:** `dual_visual_review`, `sandbox_render_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false, "javascript": "disabled", "asset_policy": "local allowlist or explicit placeholders"}

**FineVision inspirations:** `websight`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2403.09029

### 69. music_notation_reading

Read pitch, duration, rests, or note order from a score with visible clef, key, meter, and accidental context, using a declared notation schema.

**Required image capabilities:** `notation_context`, `readable_diagram`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Required clef, key, meter, accidentals, and bar context are available and supported by the verifier.

**Question example (only when its referenced objects and conditions are actually present):** What are the pitches of the notes in the first complete bar?

**Do not infer:** Guessing an omitted clef or accidental context, audio properties, or musical intent.

**Verification contracts:** `dual_visual_review`, `music_notation_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false}

**FineVision inspirations:** `CoSyn_400k_music`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846

### 70. chemical_structure_reading

Recover visible atom, bond, ring, or explicitly defined substructure information using a declared chemical notation convention.

**Required image capabilities:** `notation_context`, `readable_diagram`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Bond, atom, implicit-hydrogen, and stereochemical conventions are covered by the chosen validator.

**Question example (only when its referenced objects and conditions are actually present):** Which atoms are connected by the marked double bond?

**Do not infer:** Reaction conditions, synthesis instructions, biological effects, or unresolvable stereochemistry.

**Verification contracts:** `dual_visual_review`, `chemical_graph_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false}

**FineVision inspirations:** `CoSyn_400k_chemical`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846

### 71. circuit_structure_reading

Identify circuit symbols and explicit electrical connections using known junction, crossover, and component conventions.

**Required image capabilities:** `notation_context`, `graph_nodes_edges`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. All symbol and junction conventions needed for the topology are supported by the verifier.

**Question example (only when its referenced objects and conditions are actually present):** Which components are connected in parallel in the shown circuit?

**Do not infer:** Electrical performance from missing values or connections, or unshown safety conditions.

**Verification contracts:** `dual_visual_review`, `circuit_graph_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false}

**FineVision inspirations:** `CoSyn_400k_circuit`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2502.14846

### 72. ui_action_specification

Produce a single action-schema record targeting a visible control for an explicit public local goal, without executing it or claiming a result.

**Required image capabilities:** `ui_controls`.

**Eligibility:** Bind a public region/target expression to this image; reject nonexistent markers or unbound targets. Verify that the public reference identifies one target; if multiple match, revise the question or use an ambiguity profile. The goal is explicit and achievable as a single target specification on the visible screen; no outcome claim. An explicit allowlisted action and coordinate schema is configured; this is a data-generation task only. Coordinates use the delivered view dimensions and a verified target region; arbitrary points cannot pass.

**Question example (only when its referenced objects and conditions are actually present):** To focus the visible search field, specify one click target in normalized coordinates.

**Do not infer:** Multi-step planning through unseen screens, assumed action outcomes, or actual tool execution.

**Verification contracts:** `dual_visual_review`, `ui_action_validator`. These are requirements, not provided runtime implementations.

**Parameters:** {"enabled_by_default": false, "action_scope": "single locally grounded action; no execution"}

**FineVision inspirations:** `aguvis-stage-1`, `groundui`. These are abstractions/projections, not claims of identical supervision.

**Sources:** https://arxiv.org/abs/2412.04454 ; https://arxiv.org/html/2510.17269v2 ; https://arxiv.org/abs/2403.17918

## Facets, not extra task IDs

### input_domain
photo, sketch, document, table, chart, thematic_map, diagram, ui, math_notation, code_listing, score, chemical_diagram, circuit

Infer from image only. Domain is not a new semantic operation and must not enable an unsupported task.

### source_appearance
printed, handwritten, stylized, rotated, low_contrast, dense_layout

Difficulty tags, not separate task IDs. Never degrade legibility to manufacture diversity.

### output_form
short_answer, sentence, list, table, json, markdown, latex, static_code

Require only public output constraints. Code, coordinates, and specialized schemas have additional validators.

### answer_language
en, ja, zh-Hans

Retain source script for verbatim OCR; target language applies to generated explanation. A requested translation is an optional transformation, not an OCR correction.

### scope
single_target, bounded_region, whole_canvas, existing_panels

Binding and closure are region-specific; multi-page/multi-image context is not silently supplied.

### composition
direct, filter_then_lookup, relational_chain, multi_region_synthesis, verified_derived_value

Label the requested final semantic operation; multiple independent operations require explicit subtasks instead of ambiguous single labels.

### dialogue_relation
new_observation, refinement, reference_to_prior, regroup, reformat, verify, explicit_hypothetical_update

Use exact committed public history. A later turn alone does not prove dependency. Reformats are modifiers, not new visual tasks.

### answerability_profile
normal, limitation, false_premise

A response behavior orthogonal to task ID. Alternative profile guards below replace, not blindly reuse, normal answerability prerequisites.

## Legacy migration

| Existing ID | New IDs | Parameter changes |
|---|---|---|
| `object_identification` | `object_identification` | {} |
| `attribute_lookup` | `attribute_lookup` | {} |
| `region_description` | `grounded_description` | {"scope": "bounded_region"} |
| `visual_summary` | `visual_summary` | {} |
| `relation_lookup` | `spatial_relation` | {"frame": "explicit image-plane or supported depth frame"} |
| `attribute_comparison` | `attribute_comparison` | {} |
| `spatial_ordering` | `spatial_ordering` | {} |
| `correspondence_matching` | `visual_correspondence` | {} |
| `visible_count` | `entity_count` | {"count_unit": "required"} |
| `conditional_selection` | `predicate_selection` | {"predicate": "explicit Boolean predicate"} |
| `exclusion_selection` | `set_operation` | {"operator": "difference"} |
| `attribute_grouping` | `attribute_grouping` | {} |
| `text_transcription` | `text_transcription` | {} |
| `text_field_extraction` | `text_field_extraction` | {} |
| `text_reading_order` | `text_reading_order` | {} |
| `label_value_linking` | `label_value_linking` | {} |
| `table_lookup` | `table_cell_lookup` | {} |
| `table_selection` | `table_predicate_selection` | {} |
| `chart_lookup` | `chart_encoding_lookup`, `chart_value_lookup` | {"migration": "reclassify by actual requested output; not both by default"} |
| `chart_comparison` | `chart_comparison` | {} |
| `grounded_sum` | `grounded_arithmetic` | {"operator": "add"} |
| `grounded_difference` | `grounded_arithmetic` | {"operator": "subtract"} |
| `grounded_product_quotient` | `grounded_arithmetic` | {"operator": "multiply or divide, read from question"} |
| `table_aggregation` | `grounded_aggregation` | {"operand_source": "table", "aggregation": "read from question"} |

## Verification limits

Only catalog structure, cross-references, source-table preservation, and migration coverage are checked by the bundled validation tests. No Pixelogue runtime integration, model call, image test, expert-domain validation, or SFT improvement measurement is included. See INTEGRATION.md for the implementation handoff.

## Conditional verification and routing

Verification contracts are conditional on the actual requested operation and public parameters. For example, a textual consistency check does not automatically trigger arithmetic, and a schema check applies only when a public output format requires it. Applicability is controller-owned; missing applicable validators block admission.

### Routing precedence

- Use formula_transcription or code_transcription for structural transcription of formulas/code, not generic text_transcription.
- Use table_cell_lookup for row/column/header-path cell addressing, and chart_encoding_lookup/chart_value_lookup for chart addressing.
- Use ui_element_grounding for UI targets and diagram_element_lookup for labeled diagram elements.
- Use grounded_aggregation for a closed-set reduction, grounded_arithmetic for an explicit expression, and entity_count for member enumeration.
- Use quantified_statement_verification for quantified propositions; visual_claim_verification is for other visible propositions.
- Use panel_comparison for corresponding regions in explicitly distinguishable panels; attribute_comparison compares resolved objects.
- Use graph_path_tracing for paths, graph_connectivity for adjacency/reachability, and diagram_process_description for a process summary.
- Bind the requested final operation, not the answer format; genuinely independent subtasks need explicit composition.
- If the operation is still ambiguous, mark UNRESOLVED rather than silently substituting an easier neighboring task.

### Runtime feasibility

- The expected output must fit the configured response and context budget. Otherwise choose a publicly explicit bounded region, allocate a reviewed larger budget, or skip. Never truncate and certify completeness.
- Check eligibility using the actual model/student image resolution and record any view transformation. Coordinate checks must use the same view.
- An unavailable applicable validator blocks admission; a registry entry in this proposal is not a running implementation.
- Pattern uniqueness is relative to an explicitly versioned finite rule grammar, made public when it changes the task interpretation. Competing reasonable answers require abstention.
