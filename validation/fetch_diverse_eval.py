#!/usr/bin/env python3
"""Fetch a small, provenance-recorded evaluation set from Wikimedia Commons.

The selected raster is a thumbnail of a Commons file. Its file page and API
metadata are retained privately; neither search terms nor descriptions enter
Pixelogue's model-visible source records. This is a test input collector, not
an answer-label generator.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/diverse-web-eval"
PINNED_MANIFEST = ROOT / "validation/diverse_web_eval_manifest.jsonl"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "PixelogueEvaluation/0.1 (https://github.com/Onely7/Pixelogue)"
PUBLIC_DOMAIN = "https://creativecommons.org/publicdomain/mark/1.0/"
SEARCHES = (
    (1, "写真", "File:street photography people"),
    (2, "人物・人体", "File:person portrait full body"),
    (3, "文書", "File:scanned document page"),
    (4, "身分証・カード", "File:sample identity card specimen"),
    (5, "手書き", "File:handwritten note"),
    (6, "テキスト主体", "File:street sign text"),
    (7, "表", "File:statistical table"),
    (8, "グラフ・チャート", "File:bar chart data"),
    (9, "数学・科学プロット", "File:function graph plot"),
    (10, "ダイアグラム", "File:flowchart process diagram"),
    (11, "ITダイアグラム", "File:network architecture diagram"),
    (12, "工学図面", "File:electronic circuit schematic"),
    (13, "組織・業務図", "File:organizational chart"),
    (14, "地図", "File:subway map"),
    (15, "地理空間", "File:satellite image earth"),
    (16, "医用", "File:chest x ray"),
    (17, "顕微鏡", "File:microscope cells"),
    (18, "天文", "File:Hubble galaxy"),
    (19, "センサー", "File:thermal infrared image"),
    (20, "産業・検査", "File:printed circuit board defect"),
    (21, "防犯・監視", "File:CCTV surveillance camera view"),
    (22, "生体認証", "File:fingerprint image"),
    (23, "スクリーンショット", "File:computer screenshot"),
    (24, "UI・UX", "File:mobile app wireframe"),
    (25, "プレゼンテーション", "File:presentation slide"),
    (26, "インフォグラフィック", "File:infographic statistics"),
    (27, "イラスト", "File:scientific illustration"),
    (28, "漫画", "File:comic strip page"),
    (29, "美術", "File:painting art museum"),
    (30, "広告", "File:advertising poster"),
    (31, "ロゴ", "File:logo symbol"),
    (32, "タイポグラフィ", "File:calligraphy lettering"),
    (33, "パターン", "File:seamless geometric pattern"),
    (34, "CGI・3DCG", "File:3d render computer graphics"),
    (35, "ゲーム", "File:video game screenshot"),
    (36, "AI生成", "File:AI generated image"),
    (37, "3D関連2D表現", "File:normal map texture"),
    (38, "パノラマ", "File:panorama 360 photograph"),
    (39, "動画由来静止画", "File:film still frame"),
    (40, "複数フレーム表現", "File:animation contact sheet frames"),
    (41, "コード画像", "File:QR code"),
    (42, "数式・記号", "File:mathematical equation handwritten"),
    (43, "化学・生物構造", "File:chemical structure formula"),
    (44, "教育・説明", "File:geometry diagram educational"),
    (45, "出版物", "File:magazine page layout"),
    (46, "商品・EC", "File:product photo white background"),
    (47, "ファッション", "File:fashion clothing model"),
    (48, "食品", "File:food dish photograph"),
    (49, "農業", "File:crop field aerial"),
    (50, "交通", "File:road intersection traffic"),
    (51, "ロボティクス", "File:robot camera view"),
    (52, "法医学", "File:forensic footprint evidence"),
    (53, "災害", "File:flood damage photograph"),
    (54, "気象", "File:weather radar image"),
    (55, "CV注釈", "File:bounding boxes annotated image"),
    (56, "CV中間表現", "File:edge detection image"),
    (57, "マスク", "File:segmentation mask image"),
    (58, "画像処理", "File:before after image processing"),
    (59, "比較・検証", "File:before after comparison image"),
    (60, "特殊・破損", "File:blurred photograph"),
)

# These file pages were checked against a contact sheet after broad search.
OVERRIDE_TITLES = {
    3: "File:Mingyao Dept. Store uniform invoice NY33509209.jpg",
    7: "File:Stats Table Career.png",
    10: "File:Gasification Process Flowchart.PNG",
    13: "File:Fundamentals of Business - Fig. 8.3 - Organizational Chart.jpg",
    19: "File:Thermal Image Test of Space Shuttle Main Engine - GPN-2000-000554.jpg",
    20: "File:DAMAGED INJECTOR SERIAL NUMBER 283 - CRACKED WELD AND INJECTOR SERIAL NUMBER 273 ON TEST STAND - LOX LIQUID OXYGEN COOLING PROGRAM - NARA - 17500261.jpg",
    21: "File:Ringgold CCTV Screengrab.jpg",
    22: "File:Fingerprint - Central Pocket Loop Whorl.jpg",
    39: "File:Ben Hur (1907) - film frame.jpg",
    40: "File:B&W 35mm Film Contact Sheet. Surrey UK.jpg",
    41: "File:Lipu tenpo - QR Code.png",
    42: "File:Quadratic Formula Scroll.jpg",
    43: "File:Ethanol-structure.png",
    45: "File:The botanical magazine = (Page 126) BHL5431892.jpg",
    46: "File:Roland SH32 - Product Photography.jpg",
    49: "File:Pixillated pattern in a crop field South of Kettlethorpe, aerial 2016 - geograph.org.uk - 5022255.jpg",
    52: "File:Crime scene shoeprint oblique.jpg",
    54: "File:Radar image of the 2023 Amory EF3 tornado.png",
    55: "File:Intersection over Union - object detection bounding boxes.jpg",
    56: "File:Subpixel edge detection.png",
    57: "File:Ceratotherium-simum-simum-mask.png",
    59: "File:Portrait restoration, before and after.jpg",
    60: "File:Blurry photo.jpg",
}


def _request(url: str, *, attempts: int = 3) -> bytes:
    """Read a bounded response and retry transient remote errors."""
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=25) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError("response exceeds 20 MiB")
            return data
        except (OSError, ValueError, urllib.error.URLError) as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {error}")


def _plain(value: object) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()


def _candidates(query: str) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 12,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1200,
        "format": "json",
    }
    data = json.loads(_request(f"{API}?{urllib.parse.urlencode(params)}"))
    return sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p["index"])


def _page(title: str) -> dict:
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1200,
        "format": "json",
    }
    data = json.loads(_request(f"{API}?{urllib.parse.urlencode(params)}"))
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page:
        raise RuntimeError(f"Commons file missing: {title}")
    return page


def _fetch_one(number: int, category: str, query: str, pinned: dict | None = None) -> dict:
    """Select the first decodable raster result and preserve its provenance."""
    if pinned is not None:
        candidates = [_page(pinned["title"])]
    elif number in OVERRIDE_TITLES:
        candidates = [_page(OVERRIDE_TITLES[number])]
    else:
        candidates = _candidates(query)
    for page in candidates:
        info = page.get("imageinfo", [{}])[0]
        if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        url = info.get("thumburl")
        if not url:
            continue
        try:
            data = _request(url)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    continue
                if getattr(image, "n_frames", 1) != 1 or min(image.size) < 128:
                    continue
                if image.width * image.height > 40_000_000:
                    continue
                extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image.format]
                dimensions = image.size
        except (OSError, ValueError, RuntimeError):
            continue
        metadata = {
            key: _plain(value.get("value", ""))
            for key, value in info.get("extmetadata", {}).items()
        }
        license_uri = metadata.get("LicenseUrl")
        if not license_uri and metadata.get("LicenseShortName") == "Public domain":
            license_uri = PUBLIC_DOMAIN
        if license_uri and license_uri.startswith("http://creativecommons.org/"):
            license_uri = "https://" + license_uri.removeprefix("http://")
        if not license_uri or not license_uri.startswith("https://"):
            continue
        raw_sha256 = hashlib.sha256(data).hexdigest()
        if pinned is not None and (
            page["pageid"] != pinned["commons_page_id"]
            or raw_sha256 != pinned["raw_sha256"]
            or license_uri != pinned["license_uri"]
        ):
            raise RuntimeError(f"pinned Commons identity changed: {number} {page['title']}")
        image_path = OUTPUT / "images" / f"{number:02d}.{extension}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(data)
        return {
            "category_number": number,
            "category": category,
            "search_query": query,
            "commons_page_id": page["pageid"],
            "title": page["title"],
            "file_page_url": info["descriptionurl"],
            "thumbnail_url": url,
            "source_file_url": info["url"],
            "image_path": image_path.relative_to(OUTPUT).as_posix(),
            "raw_sha256": raw_sha256,
            "width": dimensions[0],
            "height": dimensions[1],
            "license_uri": license_uri,
            "license_name": metadata.get("LicenseShortName"),
            "artist": metadata.get("Artist") or metadata.get("Credit") or page["title"],
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    raise RuntimeError(f"no valid raster candidate: {number} {category}: {query}")


def _matches_local_pin(row: dict, pinned: dict) -> bool:
    path = OUTPUT / row["image_path"]
    return (
        path.is_file()
        and row["title"] == pinned["title"]
        and row["raw_sha256"] == pinned["raw_sha256"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == pinned["raw_sha256"]
    )


def main() -> None:
    """Fetch missing categories and write model-safe manifests in stable order."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pinned = (
        {
            row["category_number"]: row
            for row in map(json.loads, PINNED_MANIFEST.read_text().splitlines())
        }
        if PINNED_MANIFEST.exists()
        else {}
    )
    metadata_path = OUTPUT / "private-metadata.jsonl"
    existing = {}
    if metadata_path.exists():
        existing = {
            row["category_number"]: row
            for row in map(json.loads, metadata_path.read_text().splitlines())
        }
    pending = [
        (n, label, query)
        for n, label, query in SEARCHES
        if n not in existing
        or (n in pinned and not _matches_local_pin(existing[n], pinned[n]))
        or (n not in pinned and not (OUTPUT / existing[n]["image_path"]).exists())
        or (not pinned and n in OVERRIDE_TITLES and existing[n]["title"] != OVERRIDE_TITLES[n])
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one, *item, pinned.get(item[0])): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                row = future.result()
                existing[item[0]] = row
                print(f"{item[0]:02d} {item[1]}: {row['title']}", flush=True)
            except (RuntimeError, KeyError, ValueError, TypeError) as exc:
                print(f"FAILED {exc}", flush=True)
            metadata_path.write_text(
                "".join(
                    json.dumps(existing[n], ensure_ascii=False) + "\n" for n in sorted(existing)
                )
            )
    if len(existing) != len(SEARCHES) or any(
        not _matches_local_pin(existing[n], pin) for n, pin in pinned.items()
    ):
        raise RuntimeError("The pinned 60-image evaluation set is incomplete")
    sources = []
    rights = []
    for n in sorted(existing):
        row = existing[n]
        source_id = f"commons-eval:{row['commons_page_id']}"
        sources.append(
            {
                "source_id": source_id,
                "image_path": row["image_path"],
                "source_group_ids": [f"commons:{row['commons_page_id']}"],
                "rights_record_id": source_id,
                "purpose": "evaluation",
                "dataset": "Wikimedia Commons diverse evaluation",
                "dataset_split": "web-eval-2026-09",
                "dataset_image_id": str(row["commons_page_id"]),
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": source_id,
                "license_uri": row["license_uri"],
                "attribution": row["artist"],
                "processing_allowed": True,
                "qa_redistribution_allowed": False,
                "image_redistribution_allowed": False,
                "training_allowed": False,
                "valid_from": "2026-09-01T00:00:00Z",
                "valid_until": None,
            }
        )
    for name, rows in (("sources.jsonl", sources), ("rights.jsonl", rights)):
        (OUTPUT / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    print(f"selected={len(existing)} of {len(SEARCHES)}", flush=True)


if __name__ == "__main__":
    main()
