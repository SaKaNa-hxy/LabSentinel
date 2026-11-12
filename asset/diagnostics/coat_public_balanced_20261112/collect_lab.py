"""Collect additional color photographs of real lab coats for the second round."""
import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
FIRST = FOLDER.parent / "coat_public_20261112"
sys.path.insert(0, str(FIRST))
import collect_public as commons


def main():
    commons.API_CACHE = FIRST / ".api_cache"
    commons.STAGING = FOLDER / "previews"
    commons.STAGING.mkdir(parents=True, exist_ok=True)
    prior = json.loads((FIRST / "candidates.json").read_text(encoding="utf-8"))
    excluded = {item["title"] for item in prior if (FIRST / "previews" / (item["id"] + ".jpg")).is_file()}
    rng = random.Random(137)
    titles = ["File:" + title for title in (
        "Pipetting a sample University of Pennsylvania.jpg", "Drug synthesis.jpg",
        "Monoclonal antibodies (2).jpg", "CSIRO ScienceImage 11169 CSIRO scientist.jpg",
        "CSIRO ScienceImage 11172 CSIRO scientist.jpg", "Medical Laboratory Scientist US NIH.jpg",
        "Female in laboratory.jpg", "Barbara Askins, Chemist - GPN-2004-00022.jpg",
    )]
    categories = (
        ("People with lab coats", 95),
        ("People using laboratory equipment", 60),
        ("Biologists at work", 55),
        ("Chemists at work", 45),
        ("Laboratory technicians", 45),
    )
    for name, maximum in categories:
        try:
            candidates = commons.category_titles(name)
        except (RuntimeError, commons.requests.RequestException) as error:
            print("Unavailable category", name, str(error)[:160], flush=True)
            continue
        candidates = [
            title for title in candidates if title not in excluded
            and not re.search(r"Bundesarchiv|Wellcome [LV]\d|Fortepan|stamp|Caricature|gravure|Coat of arms|portrait of Sir", title, re.I)
        ]
        rng.shuffle(candidates)
        titles.extend(candidates[:maximum])
        print(name, len(candidates), flush=True)
    titles = list(dict.fromkeys(title for title in titles if title not in excluded))
    records = commons.get_files("lab", titles)
    (FOLDER / "candidates.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        available = [row for row, ok in zip(records, pool.map(commons.download_preview, records)) if ok]
    (FOLDER / "available.json").write_text(json.dumps(available, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Additional lab coat candidates", len(available), "of", len(records), flush=True)


if __name__ == "__main__":
    main()
