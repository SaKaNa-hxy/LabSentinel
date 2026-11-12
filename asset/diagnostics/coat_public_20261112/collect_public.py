"""Collect small, attributed previews for manual garment and label review."""
import concurrent.futures
import html
import io
import json
import random
import re
import threading
import time
from pathlib import Path

import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
FOLDER = Path(__file__).resolve().parent
STAGING = FOLDER / "previews"
API_CACHE = FOLDER / ".api_cache"
API_LOCK = threading.Lock()
UA = {"User-Agent": "LabSentinelResearch/1.0 (graduate research, Wikimedia Commons sample review)"}
GROUPS = {
    "lab": ["People with lab coats", "Laboratory personnel", "Medical laboratory scientists"],
    "shirt": ["People wearing white button-down shirts", "Men wearing white button-down shirts",
              "Men wearing white button-down shirts in the United States", "Women wearing white shirts"],
    "outer": ["Women wearing white coats", "Male people wearing white coats", "Women wearing beige coats",
              "Females wearing trenchcoats", "Male people wearing trenchcoats"],
}
SEEDS = {
    "lab": ["Pipetting a sample University of Pennsylvania.jpg", "Technician performing laboratory test.jpg",
            "Drug synthesis.jpg", "Biochimiste 01.jpg", "Facce da laboratorio.jpg"],
    "shirt": ["(Man in white shirt in Bangkok, Thailand) - DPLA - 0466e3a34404f93fc25258676c4843f0.jpg",
              "Harajuku Fashion Street Snap (2017-09-16 14.34.41 by Dick Thomas Johnson).jpg",
              "MICHAEL PRESS web.jpg"],
    "outer": ["Earthly Color.jpg", "Casual Fashion 2022 - 51907662707.jpg", "Woman wearing trench coat.jpg"],
}


def fetch(params):
    import hashlib
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    entry = API_CACHE / (key + ".json")
    with API_LOCK:
        if entry.exists():
            return json.loads(entry.read_text(encoding="utf-8"))["query"]
        time.sleep(1.5)
        for delay in (8, 20, 40, None):
            response = requests.get(API, params={"format": "json", "formatversion": 2, **params}, headers=UA, timeout=25)
            if response.status_code != 429 or delay is None:
                break
            retry_after = response.headers.get("Retry-After", "")
            time.sleep(max(delay, min(120, int(retry_after))) if retry_after.isdigit() else delay)
        response.raise_for_status()
        data = response.json()
        if "query" in data:
            entry.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["query"]


def plain(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value or "")).split())


def category_titles(category):
    data = fetch({"action": "query", "list": "categorymembers", "cmtitle": "Category:" + category,
                  "cmnamespace": 6, "cmlimit": 500})
    return [x["title"] for x in data["categorymembers"] if x["title"].lower().endswith((".jpg", ".jpeg"))]


def get_files(group, titles):
    records = []
    for start in range(0, len(titles), 30):
        pages = fetch({"action": "query", "titles": "|".join(titles[start:start+30]), "prop": "imageinfo",
                       "iiprop": "url|size|sha1|extmetadata", "iiurlwidth": 330})["pages"]
        for page in pages:
            if not page.get("imageinfo"):
                continue
            info = page["imageinfo"][0]
            fields = info.get("extmetadata", {})
            def field(name):
                return plain(fields.get(name, {}).get("value"))
            license_name = field("LicenseShortName")
            license_url = field("LicenseUrl")
            if license_name not in {"CC BY 4.0", "CC BY 3.0", "CC BY 2.0", "CC BY-SA 4.0",
                                     "CC BY-SA 3.0", "CC BY-SA 2.0", "CC0", "Public domain", "PDM"}:
                continue
            if min(info["width"], info["height"]) < 420:
                continue
            # Reject credit or rights fields where a restriction beyond standard attribution is flagged.
            restrictions = field("Restrictions")
            if restrictions:
                continue
            records.append({"id": f"{group}-{page['pageid']}", "group": group, "title": page["title"],
                            "file_page": info["descriptionurl"], "url": info["url"],
                            "thumb": info.get("thumburl"), "width": info["width"], "height": info["height"],
                            "commons_sha1": info["sha1"], "author": field("Artist"), "credit": field("Credit"),
                            "license": license_name, "license_url": license_url,
                            "description": field("ImageDescription")[:500]})
    return records


def download_preview(item):
    target = STAGING / (item["id"] + ".jpg")
    if target.exists():
        return True
    for url in [item["thumb"], item["url"] if item["width"] * item["height"] < 12_000_000 else None]:
        if not url:
            continue
        try:
            response = requests.get(url.split("?")[0], headers=UA, timeout=22)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as picture:
                from PIL import ImageOps
                picture = ImageOps.exif_transpose(picture)
                picture.thumbnail((380, 380))
                picture.convert("RGB").save(target, quality=92)
            return True
        except (requests.RequestException, OSError):
            continue
    return False


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    API_CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    records = []
    by_group = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {category: pool.submit(category_titles, category) for categories in GROUPS.values() for category in categories}
        for group, categories in GROUPS.items():
            titles = ["File:" + name for name in SEEDS[group]]
            for category in categories:
                try:
                    choices = futures[category].result()
                except (requests.RequestException, RuntimeError) as error:
                    print("Category skipped", category, error, flush=True)
                    continue
                rng.shuffle(choices)
                titles.extend(choices[:{"lab": 108, "shirt": 37, "outer": 30}[group]])
            titles = list(dict.fromkeys(titles))
            by_group[group] = pool.submit(get_files, group, titles)
        for group, future in by_group.items():
            group_records = future.result()
            records.extend(group_records)
            print(group, len(group_records), "licensed candidate files", flush=True)
        (FOLDER / "candidates.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        available = [row for row, ok in zip(records, pool.map(download_preview, records)) if ok]
    (FOLDER / "available.json").write_text(json.dumps(available, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Preview files available", len(available), flush=True)


if __name__ == "__main__":
    main()
