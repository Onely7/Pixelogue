"""Report synthesis yield and render actual examples for every catalog task.

The report separates committed turns from completed automatic quality candidates.
It reads public dialogue only; human quality and semantic duplication remain unmeasured.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from pixelogue.catalog import load_task_catalog
from pixelogue.contracts import ConversationArtifact, ImageArtifact
from pixelogue.io import read_jsonl, write_json, write_jsonl


def normalized_question(text: str) -> str:
    """Normalize typography and whitespace for exact-repeat diagnostics."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def diversity(counts: Counter[str]) -> dict[str, Any]:
    """Describe an observed distribution without interpreting it as quality."""
    total = counts.total()
    probabilities = [value / total for value in counts.values()] if total else []
    entropy = -sum(value * math.log(value) for value in probabilities)
    return {
        "counts": dict(counts.most_common()),
        "represented": len(counts),
        "top_task_share": max(probabilities) if probabilities else None,
        "top_three_share": sum(sorted(probabilities, reverse=True)[:3]) if total else None,
        "effective_categories": math.exp(entropy) if total else None,
    }


def build_report(
    images: list[dict[str, Any]],
    conversations: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count outcomes and retain example provenance from a bounded campaign.

    Raises:
        ValueError: If identities, catalog references or candidate states are inconsistent.
    """
    sources = {image["source_id"] for image in images}
    if len(sources) != len(images):
        raise ValueError("Prepared source IDs must be unique")
    task_by_id = {task["id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Catalog task IDs must be unique")
    seen: set[str] = set()
    statuses: Counter[str] = Counter()
    attempted: Counter[str] = Counter()
    verified: Counter[str] = Counter()
    candidate: Counter[str] = Counter()
    families: Counter[str] = Counter()
    generators: dict[str, Counter[str]] = defaultdict(Counter)
    languages: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    within_repeats = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        source_id = conversation["image"]["source_id"]
        if source_id not in sources:
            raise ValueError("Conversation refers to an unprepared source")
        if source_id in seen:
            raise ValueError("Conversation source IDs must be unique")
        seen.add(source_id)
        status = conversation["status"]
        if status not in {"QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"}:
            raise ValueError("Unknown conversation status")
        turns = conversation["turns"]
        adopted = status == "QUALITY_CANDIDATE"
        if adopted and (
            not 2 <= len(turns) <= 6 or any(turn["status"] != "COMMITTED" for turn in turns)
        ):
            raise ValueError("Quality candidates require two to six committed turns")
        statuses[status] += 1
        generators[conversation["generation_model"]][status] += 1
        committed = [turn for turn in turns if turn["status"] == "COMMITTED"]
        if adopted:
            lengths[str(len(committed))] += 1
            languages[conversation["target_language"]] += 1
        local_questions: Counter[str] = Counter()
        history: list[dict[str, str]] = []
        for turn in turns:
            task_id = turn["instruction"]["task_id"]
            if task_id not in task_by_id:
                raise ValueError("Turn refers to an unknown catalog task")
            attempted[task_id] += 1
            if turn["status"] != "COMMITTED":
                continue
            verified[task_id] += 1
            question = turn["question"]["content"]
            answer = turn["answer"]["content"]
            if adopted:
                candidate[task_id] += 1
                families[task_by_id[task_id]["family"]] += 1
                key = normalized_question(question)
                questions[key] += 1
                local_questions[key] += 1
            examples[task_id].append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "source_id": source_id,
                    "conversation_status": status,
                    "turn_index": turn["turn_index"],
                    "task_id": task_id,
                    "question": question,
                    "answer": answer,
                    "public_history": list(history),
                    "image": conversation["image"],
                }
            )
            history.extend(
                ({"role": "user", "content": question}, {"role": "assistant", "content": answer})
            )
        within_repeats += sum(value - 1 for value in local_questions.values())
    count = len(conversations)
    candidates = statuses["QUALITY_CANDIDATE"]
    task_rows = []
    for number, task in enumerate(tasks, start=1):
        task_id = task["id"]
        actual = sorted(
            examples[task_id],
            key=lambda example: example["conversation_status"] != "QUALITY_CANDIDATE",
        )[:3]
        task_rows.append(
            {
                "id": task_id,
                "number": number,
                "label": task["label"],
                "definition": task["definition"],
                "family": task["family"],
                "catalog_status": task["status"],
                "attempted_turns": attempted[task_id],
                "committed_turns": verified[task_id],
                "candidate_turns": candidate[task_id],
                "examples": actual,
                "example_status": "automatic_candidate"
                if candidate[task_id]
                else "diagnostic_prefix"
                if actual
                else "no_observed_example",
            }
        )
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "requested_images": len(images),
        "processed_images": count,
        "complete": count == len(images),
        "unprocessed_images": len(images) - count,
        "status_counts": dict(statuses),
        "automatic_quality_candidates": candidates,
        "automatic_candidate_rate": candidates / count if count else None,
        "not_selected_images": count - candidates,
        "not_selected_rate": (count - candidates) / count if count else None,
        "persisted_attempted_turns": attempted.total(),
        "committed_turns_all_outcomes": verified.total(),
        "candidate_turns": candidate.total(),
        "attempted_task_counts": dict(attempted),
        "committed_task_counts": dict(verified),
        "candidate_task_diversity": diversity(candidate),
        "candidate_family_diversity": diversity(families),
        "catalog_tasks": len(tasks),
        "core_catalog_tasks": sum(task["status"] == "core" for task in tasks),
        "candidate_lengths": dict(lengths),
        "candidate_languages": dict(languages),
        "generator_outcomes": {name: dict(values) for name, values in generators.items()},
        "exact_question_repeats_within_candidates": within_repeats,
        "normalized_unique_candidate_questions": len(questions),
        "cross_image_exact_question_repeats": sum(value - 1 for value in questions.values()),
        "most_frequent_candidate_questions": questions.most_common(15),
        "human_approved_conversations": None,
        "semantic_question_diversity": None,
        "tasks": task_rows,
        "limits": [
            "Automatic quality candidates are not independently human approved or training exports.",
            "Committed prefixes of rejected/abstained/error conversations are diagnostic examples only.",
            "Pre-turn stops are absent from persisted_attempted_turns; use run-diagnostics for full stage reach.",
            "Exact cross-image question repetition can be valid; semantic duplication requires human review.",
            "Missing examples remain missing; catalog example questions are never substituted for actual output.",
            "Photographs do not establish table/chart/music/circuit/UI or all-72-task coverage.",
        ],
    }


def write_reports(report: dict[str, Any], artifact_root: Path, destination: Path) -> None:
    """Write JSON, CSV, Markdown and an escaped local image/Q/A gallery."""
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "report.json", report)
    fields = (
        "number",
        "id",
        "label",
        "family",
        "catalog_status",
        "attempted_turns",
        "committed_turns",
        "candidate_turns",
        "example_status",
    )
    with (destination / "task-coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["tasks"])
    e = html.escape
    cards, rows, markdown = [], [], []
    dataset_root = artifact_root.resolve(strict=True)
    thumbnail_root = destination / "thumbnails"
    thumbnail_root.mkdir(exist_ok=True)
    for task in report["tasks"]:
        task_id = task["id"]
        examples_html = []
        rows.append(
            f"| {task['number']} | [{task_id}](#{task_id}) | {task['attempted_turns']} | {task['committed_turns']} | {task['candidate_turns']} | {task['example_status']} |"
        )
        markdown.extend(
            (f"## {task['number']}. {task_id}", "", task["label"], "", task["definition"], "")
        )
        for example in task["examples"]:
            view = example["image"]["full_view"]
            original = (artifact_root / view["relative_path"]).resolve(strict=True)
            if not original.is_relative_to(dataset_root):
                raise ValueError("Image path leaves the prepared artifact root")
            thumbnail = thumbnail_root / f"{view['encoded_sha256']}.jpg"
            if not thumbnail.exists():
                with Image.open(original) as image:
                    image.thumbnail((640, 480))
                    image.convert("RGB").save(thumbnail, quality=85)
            image_data_uri = "data:image/jpeg;base64," + base64.b64encode(
                thumbnail.read_bytes()
            ).decode("ascii")
            kind = (
                "自動品質候補（人手未確認）"
                if example["conversation_status"] == "QUALITY_CANDIDATE"
                else "未採用会話の検証済みprefix（診断用）"
            )
            history = "".join(
                f"<p><b>{e(message['role'])}</b>: {e(message['content'])}</p>"
                for message in example["public_history"]
            )
            examples_html.append(
                f'<article><p class="kind">{kind}</p><img loading="lazy" src="{e(image_data_uri, quote=True)}" alt="評価画像">'
                f"<p><b>指示</b>: {e(example['question'])}</p><p><b>回答</b>: {e(example['answer'])}</p>"
                f"<details><summary>先行する公開履歴・出典ID</summary>{history}<p>{e(example['source_id'])} / {e(example['conversation_id'])} / turn {example['turn_index']}</p></details></article>"
            )
            markdown.extend(
                (
                    f"### {kind} / turn {example['turn_index']}",
                    "",
                    f"![評価画像]({thumbnail.resolve()})",
                    "",
                    f"**指示:** {example['question']}",
                    "",
                    f"**回答:** {example['answer']}",
                    "",
                    f"source: {example['source_id']} / conversation: {example['conversation_id']}",
                    "",
                )
            )
            if example["public_history"]:
                markdown.append("先行する公開履歴:")
                markdown.extend(
                    f"- {message['role']}: {message['content']}"
                    for message in example["public_history"]
                )
                markdown.append("")
        if not examples_html:
            reason = (
                "校正待ちの専門拡張。通常選択対象外。"
                if task["catalog_status"] == "extension"
                else "今回の画像群で検証済みの実例は得られていません。"
            )
            examples_html.append(f"<p>{reason}</p>")
            markdown.extend((reason, ""))
        cards.append(
            f'<section id="{e(task_id, quote=True)}" data-family="{e(task["family"], quote=True)}"><h2>{task["number"]}. {e(task["label"])} <code>{e(task_id)}</code></h2><p>{e(task["definition"])}</p><p>試行 {task["attempted_turns"]} / 確定 {task["committed_turns"]} / 候補会話内 {task["candidate_turns"]}</p>{"".join(examples_html)}</section>'
        )
    count = report["processed_images"]
    candidates = report["automatic_quality_candidates"]
    rate = f"{report['automatic_candidate_rate']:.1%}" if count else "未測定"
    introduction = f"# Open Images V7 合成結果とタスク別実例\n\n処理 {count}/{report['requested_images']}画像。自動品質候補 {candidates}件（{rate}）。人手品質は未確認。\n\n採用候補会話のタスク種類: {report['candidate_task_diversity']['represented']}/{report['core_catalog_tasks']}標準タスク。\n\n"
    overview = (
        "| # | タスク | 試行turn | 確定turn | 候補内turn | 実例 |\n|---:|---|---:|---:|---:|---|\n"
        + "\n".join(rows)
    )
    (destination / "examples.md").write_text(
        introduction + overview + "\n\n" + "\n".join(markdown), encoding="utf-8"
    )
    families = sorted({task["family"] for task in report["tasks"]})
    options = '<option value="">全分野</option>' + "".join(
        f"<option>{e(family)}</option>" for family in families
    )
    page = f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pixelogue タスク別合成実例</title><style>body{{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px}}section{{border-top:1px solid #aaa;padding:20px 0}}article{{background:#f5f7fa;padding:20px;margin:16px 0;overflow-wrap:anywhere}}img{{max-width:100%;height:auto}}input,select{{font:inherit;padding:8px}}.kind{{font-weight:bold}}code{{font-size:13px}}nav{{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #aaa}}</style><h1>Pixelogue タスク別合成実例</h1><p>処理 {count}/{report["requested_images"]}画像・自動品質候補 {candidates}件（{rate}）。人手未確認。実例のないタスクは明示しています。画像は評価用で、学習exportは行っていません。</p><p>候補会話内のタスク種類 {report["candidate_task_diversity"]["represented"]}/{report["core_catalog_tasks"]}標準タスク。未採用会話のprefixは診断用です。</p><nav><input id="query" placeholder="タスク・指示・回答を検索"><select id="family">{options}</select><span id="count"></span></nav>{"".join(cards)}<script>function filter(){{const q=document.getElementById("query").value.toLowerCase(),f=document.getElementById("family").value;let n=0;document.querySelectorAll("section").forEach(s=>{{s.hidden=!(s.textContent.toLowerCase().includes(q)&&(!f||s.dataset.family===f));if(!s.hidden)n++;}});document.getElementById("count").textContent="表示 "+n+"タスク";}}document.getElementById("query").addEventListener("input",filter);document.getElementById("family").addEventListener("change",filter);filter();</script></html>'
    (destination / "examples.html").write_text(page, encoding="utf-8")
    write_jsonl(
        destination / "examples.jsonl",
        [example for task in report["tasks"] for example in task["examples"]],
    )
    summary = {key: value for key, value in report.items() if key != "tasks"}
    (destination / "summary.md").write_text(
        introduction
        + "```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n\n[画像・指示・回答の実例](examples.html)\n",
        encoding="utf-8",
    )


def main() -> None:
    """Read typed saved outputs and create an evaluation-only campaign report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    images = [image.model_dump(mode="json") for image in read_jsonl(args.images, ImageArtifact)]
    conversations = [
        conversation.model_dump(mode="json")
        for conversation in read_jsonl(args.conversations, ConversationArtifact)
    ]
    report = build_report(images, conversations, load_task_catalog()["tasks"])
    write_reports(report, args.artifact_root, args.output_dir)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "processed_images",
                    "complete",
                    "status_counts",
                    "automatic_candidate_rate",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
