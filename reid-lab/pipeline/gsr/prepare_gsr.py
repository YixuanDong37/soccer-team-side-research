"""Pick the 50 SN-GSR-2025 train clips for the re-entry test and extract only those.

Usage (gbf's .venv): python -I prepare_gsr.py TRAIN_ZIP ZIP_SHA256 OUT_DIR SELECTION_JSON

The order is the one prereg/gates.v1.json fixes for every GSR split (gsr_evaluation.split):
clips sorted by sha256("gbf-gsr-v1/" + clip name). In train the first 4 are the pre-registered
smoothing_fit clips, so they are left out and the next 50 are taken. The test split (whose first
12 clips are the pre-registered holdout) is not touched.

The archive is untrusted: only members named <...>/<SNGS-nnn>/img1/<6 digits>.jpg and
<...>/<SNGS-nnn>/Labels-GameState.json of the selected clips are extracted, to
OUT_DIR/<SNGS-nnn>/img1/ and OUT_DIR/<SNGS-nnn>/Labels-GameState.json; any other name is
ignored. Nothing is opened as an image here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

SEED = "gbf-gsr-v1"
SMOOTHING_FIT = 4
TAKE = 50
MEMBER = re.compile(r"^(?:[^/]+/)*(SNGS-\d+)/(img1/\d{6}\.jpg|Labels-GameState\.json)$")


def main() -> int:
    archive, zip_sha, out, selection = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != zip_sha:
        raise SystemExit(f"{archive}: sha256 {digest.hexdigest()} is not the expected {zip_sha}")
    with zipfile.ZipFile(archive) as z:
        members: dict[str, list[zipfile.ZipInfo]] = {}
        for info in z.infolist():
            m = MEMBER.match(info.filename)
            if m:
                members.setdefault(m.group(1), []).append(info)
        names = sorted(members)
        order = sorted(names, key=lambda n: hashlib.sha256(f"{SEED}/{n}".encode()).hexdigest())
        chosen = order[SMOOTHING_FIT:SMOOTHING_FIT + TAKE]
        if len(chosen) != TAKE:
            raise SystemExit(f"only {len(chosen)} clips available after the smoothing_fit ones")
        report = {"archive": str(archive), "archive_sha256": zip_sha, "seed": SEED, "clips_in_archive": len(names),
                  "order": order, "smoothing_fit_left_out": order[:SMOOTHING_FIT], "selected": chosen,
                  "unused": order[SMOOTHING_FIT + TAKE:], "clips": {}}  # fmt: skip
        for name in chosen:
            images = [i for i in members[name] if i.filename.endswith(".jpg")]
            labels = [i for i in members[name] if i.filename.endswith("Labels-GameState.json")]
            if len(labels) != 1:
                raise SystemExit(f"{name}: {len(labels)} label files")
            target = out / name
            dests = [info.filename.split(f"{name}/", 1)[1] for info in images + labels]
            if len(set(dests)) != len(dests):
                raise SystemExit(f"{name}: two archive members map to the same file")
            if len(images) != 750:
                raise SystemExit(f"{name}: {len(images)} images, expected 750")
            (target / "img1").mkdir(parents=True, exist_ok=True)
            for info in images + labels:
                rel = info.filename.split(f"{name}/", 1)[1]
                dest = (target / rel).resolve()
                if target.resolve() not in dest.parents:
                    raise SystemExit(f"refusing member path {info.filename}")
                if dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                with z.open(info) as src, dest.open("wb") as dst:
                    while block := src.read(1 << 20):
                        dst.write(block)
            numbers = sorted(int(Path(i.filename).stem) for i in images)
            report["clips"][name] = {"images": len(images), "first": numbers[0], "last": numbers[-1],
                                     "contiguous": numbers == list(range(numbers[0], numbers[-1] + 1)),
                                     "labels_bytes": labels[0].file_size}  # fmt: skip
    report["restrictions"] = ["publish numbers only, never frames (registries/data.toml soccernet-gsr-2025)"]
    selection.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(names)} clips in the archive; selected {len(chosen)}: {chosen[0]} ... {chosen[-1]}")
    bad = {n: c for n, c in report["clips"].items() if c["images"] != 750 or not c["contiguous"]}
    print("clips without exactly 750 contiguous images:", bad or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
