#!/usr/bin/env python3
r"""relocate.py: point the hard-coded paths of the reid-lab package at this machine.

The lab and the input pipeline were written with absolute paths of the machine they were
developed on. The shipped files still hold those paths: they are the "from" values this script
replaces. Standard library only; run it with any Python 3.9+.

    python relocate.py --runs D:/data/gsr-runs --runs-leadcfg D:/data/gsr-runs-leadcfg \
        --gsr-train D:/data/sn-gsr-2025/train --gbf-repo D:/code/gbf --sngs D:/code/sn-gamestate \
        [--python D:/code/gbf/.venv/Scripts/python.exe] [--tools-root DIR] [--config paths.json]

    python relocate.py --check          list absolute paths that are not under a configured place
    python relocate.py --show           print the keys with their original and current values
    python relocate.py --dry-run ...    say what would change, write nothing
    python relocate.py --dest DIR ...   copy the package to DIR first and relocate the copy
    python relocate.py --verify-manifest   compare the files with manifest-sha256.txt
    python relocate.py make-lab         build the lead-settings copy of the lab (see below)

What is rewritten: every text file of the package (.py .sh .md .json .txt .log .csv .out
.stdout), in place. Not rewritten: *.patch (the provenance files record their sha256), this
script, relocate-log.json and manifest-sha256.txt. Each run writes relocate-log.json: the values
now in the files, what was replaced where, and the sha256 of every changed file before and
after. A later run reads that log, so the script can be run again with other values.

Keys (flag, meaning, default when the flag is not given):
  TOOLS_ROOT        --tools-root     folder that contains reid_lab/            <package>/method
  PIPELINE_ROOT     --pipeline-root  folder with gsr/, analysis/, the scripts  <package>/pipeline
  EVIDENCE_ROOT     --evidence-root  folder with fuse/, m_keep/, v_*/ ...      <package>/evidence
  PATCH_DIR         --patch-dir      folder with the *.patch files             <package>/pipeline/patches
  LEADCFG_ROOT      --leadcfg-root   lead-settings lab copy (make-lab)         <package>/_leadcfg
  PYTHON            --python         interpreter of the gbf venv               <gbf-repo>/.venv/Scripts/python.exe
  GBF_REPO          --gbf-repo       checkout of the gbf repository            (kept)
  RUNS_DIR          --runs           run outputs, earlier tracker settings     (kept)
  RUNS_LEADCFG_DIR  --runs-leadcfg   run outputs, lead's tracker settings      (kept)
  GSR_TRAIN_DIR     --gsr-train      extracted SN-GSR train clips (labels)     (kept)
  GSR_ROOT          --gsr-root       dataset folder (train.zip under download) parent of --gsr-train
  SNGS_DIR          --sngs           sn-gamestate checkout                     (kept)
"(kept)" means: the paths of that key stay as they are and --check keeps listing them.

Forms. A path is replaced in the form the file used: C:/a/b, C:\a\b, C:\\a\\b (escaped),
/c/a/b (Git Bash, SNGS_DIR only) and /a/b without the drive (RUNS_DIR and RUNS_LEADCFG_DIR only:
tracklab splits "vid://path:frame" at the colon, so that form must stay without a drive letter;
it then only works when the shell's current drive is the drive of the runs folder).
Two things that are not paths are adjusted too: the drive in "C:$CLIP" of track_clip*.sh (drive
of the runs folder) and the folder name in the process filter of run_gsr_batch_leadcfg.sh.

make-lab. The lead-settings evaluation uses a second copy of the lab that reads the lead-settings
runs. It is not shipped twice; after relocating, build it with

    python relocate.py make-lab [--dest DIR] [--runs-dir DIR] [--force]

which copies <TOOLS_ROOT>/reid_lab (code and fitted parameters of m_cuts, m_appear, m_space and
fuse_r1, no caches) to DIR/reid_lab, rewrites TOOLS_ROOT to DIR and RUNS_DIR to the lead-settings
runs in that copy, and points pipeline/leadcfg/run_leadcfg_eval.sh at it. DIR defaults to
LEADCFG_ROOT, --runs-dir to RUNS_LEADCFG_DIR.

Limits: paths with spaces are not supported (several shell lines use them unquoted). The
pipeline scripts are Git Bash on Windows scripts (.venv/Scripts/python.exe, powershell.exe);
the lab itself (method/, evidence/) is plain Python and also runs with POSIX paths. Going back
from POSIX paths to drive-letter paths cannot restore the special forms: start from a fresh
copy in that case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

LOG_NAME = "relocate-log.json"
MANIFEST_NAME = "manifest-sha256.txt"
LAB_DIRNAME = "_leadcfg"

# The paths of the machine the package was built on: the "from" values of a first run.
ORIGINAL = {
    "TOOLS_ROOT": "C:/Research/plugins/tools",
    "PIPELINE_ROOT": "C:/Research/plugins/tools",
    "EVIDENCE_ROOT": "C:/Research/plugins/tools/reid_lab/work",
    "PATCH_DIR": "C:/Research/plugins/patches",
    "LEADCFG_ROOT": "C:/Research/plugins/tools_leadcfg",
    "PYTHON": "C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe",
    "GBF_REPO": "C:/Research/soccer-team-side-research/gbf",
    "RUNS_DIR": "C:/Research/gbf-data/gsr-runs",
    "RUNS_LEADCFG_DIR": "C:/Research/gbf-data/gsr-runs-leadcfg",
    "GSR_TRAIN_DIR": "C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7/train",
    "GSR_ROOT": "C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7",
    "SNGS_DIR": "C:/Research/plugins/sn-gamestate",
}
KEYS = tuple(ORIGINAL)
FLAGS = {
    "TOOLS_ROOT": "--tools-root", "PIPELINE_ROOT": "--pipeline-root", "EVIDENCE_ROOT": "--evidence-root",
    "PATCH_DIR": "--patch-dir", "LEADCFG_ROOT": "--leadcfg-root", "PYTHON": "--python",
    "GBF_REPO": "--gbf-repo", "RUNS_DIR": "--runs", "RUNS_LEADCFG_DIR": "--runs-leadcfg",
    "GSR_TRAIN_DIR": "--gsr-train", "GSR_ROOT": "--gsr-root", "SNGS_DIR": "--sngs",
}
MEANING = {
    "TOOLS_ROOT": "folder that contains reid_lab/",
    "PIPELINE_ROOT": "folder with gsr/, analysis/ and the pipeline scripts",
    "EVIDENCE_ROOT": "folder with the history and verification folders",
    "PATCH_DIR": "folder with the *.patch files",
    "LEADCFG_ROOT": "lead-settings copy of the lab (built by make-lab)",
    "PYTHON": "python of the gbf venv",
    "GBF_REPO": "checkout of the gbf repository",
    "RUNS_DIR": "run outputs, earlier tracker settings",
    "RUNS_LEADCFG_DIR": "run outputs, lead's tracker settings",
    "GSR_TRAIN_DIR": "extracted SN-GSR train clips (with the labels)",
    "GSR_ROOT": "SN-GSR dataset folder",
    "SNGS_DIR": "sn-gamestate checkout",
}
# folders that sat next to each other under reid_lab/work on the original machine
ENGINE = ("m_cuts", "m_appear", "m_space", "fuse_r1")
EVIDENCE = ("fuse", "m_keep", "diag_cues", "diag_failures", "critic", "v_leak_0", "v_leak_1",
            "v_repro_0", "v_repro_1", "v_robust_0", "v_robust_1")
# what lives directly under PIPELINE_ROOT (on the original machine it was the same folder as TOOLS_ROOT)
PIPELINE_CHILDREN = ("gsr", "analysis", "run_tracklab.py", "run_tracklab_leadcfg.py", "track_clip.sh",
                     "track_clip_leadcfg.sh", "exact_reader.py", "gbf_frame_hashes.py", "export_sngs.py",
                     "export_embeddings.py", "ball_detect.py", "fetch_weights.sh")
MSYS_KEYS = {"SNGS_DIR"}                            # also written as /c/...
DRIVELESS_KEYS = {"RUNS_DIR", "RUNS_LEADCFG_DIR"}   # also written without the drive letter

TEXT_EXT = {".py", ".sh", ".md", ".json", ".txt", ".log", ".csv", ".out", ".stdout"}
VERBATIM_EXT = {".patch"}
SKIP_FILES = {"relocate.py", LOG_NAME, MANIFEST_NAME}
SKIP_DIRS = {".git", "__pycache__", "cache", "cache_stress", "cache_audit"}
LAB_SKIP_DIRS = {"__pycache__", "cache", "cache_stress", "cache_audit"}
LAB_SKIP_EXT = {".pkl", ".pyc"}

AFTER = r"(?![A-Za-z0-9_\-])(?!\.[A-Za-z0-9_])"      # the match must end at the end of a path component
BEFORE_DRIVE = r"(?<![A-Za-z0-9])"
BEFORE_SLASH = r"(?<![A-Za-z0-9_.:/\\$}\-])"
TAIL = r"[^\s\"'<>|*?,;)\]}`]*"
CANDIDATE = re.compile(
    BEFORE_DRIVE + r"[A-Za-z]:(?:\\\\|\\|/)(?:Research|Users)(?![A-Za-z0-9_\-])" + TAIL
    + "|" + BEFORE_SLASH + r"/[A-Za-z]/(?:Research|Users)/" + TAIL
    + "|" + BEFORE_SLASH + r"/Research/" + TAIL
    + "|" + BEFORE_SLASH + r"/(?:home|Users)/" + TAIL
)


# ----------------------------------------------------------------------------- small helpers
def norm(p: str) -> str:
    """Forward slashes, no trailing slash, no doubled slashes."""
    p = str(p).strip().replace("\\", "/")
    p = re.sub(r"/{2,}", "/", p)
    if len(p) > 3:
        p = p.rstrip("/")
    return p


def user_path(value: str) -> str:
    """A path typed by the user: absolute as given, or made absolute from the current folder."""
    v = str(value).strip().replace("\\", "/")
    if os.name == "nt" and re.match(r"^/[A-Za-z]/", v):     # /c/x typed in Git Bash
        v = v[1].upper() + ":" + v[2:]
    if re.match(r"^[A-Za-z]:/", v) or v.startswith("/"):
        return norm(v)
    return norm(os.path.abspath(v))


def split_drive(p: str):
    m = re.match(r"^([A-Za-z]):(/.*)?$", p)
    return (m.group(1), m.group(2) or "/") if m else (None, p)


def under(path: str, root: str) -> bool:
    a, b = path, root
    da, db = split_drive(a)[0], split_drive(b)[0]
    if da and db:                                            # drive letters compare without case
        a, b = a[0].upper() + a[1:], b[0].upper() + b[1:]
    return a == b or a.startswith(b + "/")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fmt(to: str, style: str) -> str:
    """Write the new path in the form the file used."""
    drive, rest = split_drive(to)
    if drive is None or style in ("fwd", "posix"):
        return to
    if style == "back":
        return to.replace("/", "\\")
    if style == "dback":
        return to.replace("/", "\\\\")
    if style == "msys":
        return "/" + drive.lower() + rest
    if style == "driveless":
        return rest
    raise ValueError(style)


def forms(key: str, path: str):
    """[(regex source, style)] for one current path: every form it can have in a file."""
    drive, rest = split_drive(path)
    if drive is None:
        return [(BEFORE_SLASH + re.escape(path) + AFTER, "posix")]
    segs = [s for s in rest.split("/") if s]
    if not segs:
        raise SystemExit(f"{key}: a drive root ({path}) cannot be used as a location")
    dl = "[" + drive.upper() + drive.lower() + "]:"
    out = [
        (BEFORE_DRIVE + dl + "".join("/" + re.escape(s) for s in segs) + AFTER, "fwd"),
        (BEFORE_DRIVE + dl + "".join(r"\\\\" + re.escape(s) for s in segs) + AFTER, "dback"),
        (BEFORE_DRIVE + dl + "".join(r"\\" + re.escape(s) for s in segs) + AFTER, "back"),
    ]
    if key in MSYS_KEYS:
        out.append((BEFORE_SLASH + "/[" + drive.upper() + drive.lower() + "]" + "".join("/" + re.escape(s) for s in segs) + AFTER, "msys"))
    if key in DRIVELESS_KEYS:
        out.append((BEFORE_SLASH + "".join("/" + re.escape(s) for s in segs) + AFTER, "driveless"))
    return out


# ----------------------------------------------------------------------------- rule table
def build_rules(cur: dict, new: dict, in_pipeline: bool):
    """[(from, to, key)], longest "from" first. Rules that change nothing stay in the table:
    they keep a shorter rule from rewriting the front part of a longer path."""
    rules = []
    merged = cur["EVIDENCE_ROOT"] == cur["TOOLS_ROOT"] + "/reid_lab/work"   # the original layout
    for name in EVIDENCE:
        rules.append((cur["EVIDENCE_ROOT"] + "/" + name, new["EVIDENCE_ROOT"] + "/" + name, "EVIDENCE_ROOT"))
    if not merged:
        rules.append((cur["EVIDENCE_ROOT"], new["EVIDENCE_ROOT"], "EVIDENCE_ROOT"))
    rules.append((cur["TOOLS_ROOT"] + "/reid_lab", new["TOOLS_ROOT"] + "/reid_lab", "TOOLS_ROOT"))
    for child in PIPELINE_CHILDREN:
        rules.append((cur["PIPELINE_ROOT"] + "/" + child, new["PIPELINE_ROOT"] + "/" + child, "PIPELINE_ROOT"))
    if cur["TOOLS_ROOT"] == cur["PIPELINE_ROOT"]:
        # one string for two folders: the bare string means the pipeline folder in pipeline files
        key = "PIPELINE_ROOT" if in_pipeline else "TOOLS_ROOT"
        rules.append((cur["TOOLS_ROOT"], new[key], key))
    else:
        rules.append((cur["TOOLS_ROOT"], new["TOOLS_ROOT"], "TOOLS_ROOT"))
        rules.append((cur["PIPELINE_ROOT"], new["PIPELINE_ROOT"], "PIPELINE_ROOT"))
    for key in ("PATCH_DIR", "LEADCFG_ROOT", "PYTHON", "GBF_REPO", "RUNS_DIR", "RUNS_LEADCFG_DIR",
                "GSR_TRAIN_DIR", "GSR_ROOT", "SNGS_DIR"):
        rules.append((cur[key], new[key], key))
    seen, out = set(), []
    for f, t, k in sorted(rules, key=lambda r: (-len(r[0]), r[0])):
        if f not in seen:
            seen.add(f)
            out.append((f, t, k))
    return out


def compile_rules(rules):
    """[(regex, to, style, key, anchor)]; the anchor is a plain substring every form contains."""
    comp = []
    for f, t, key in rules:
        anchor = max((s for s in f.split("/") if s and not s.endswith(":")), key=len)
        for src, style in forms(key, f):
            comp.append((re.compile(src), t, style, key, anchor))
    return comp


def apply_rules(text: str, comp, counts: dict) -> str:
    """Replace every known path. Matches are parked as numbered slots so that a shorter rule
    never touches the result of a longer one, or a path that was already replaced."""
    slots = []

    def run(rx, to, style, key):
        def cb(m):
            new = fmt(to, style)
            if new != m.group(0):
                counts[key] = counts.get(key, 0) + 1
            slots.append(new)
            return "\x00%d\x00" % (len(slots) - 1)
        return rx.sub(cb, text)

    for rx, to, style, key, anchor in comp:
        if anchor in text:
            text = run(rx, to, style, key)
    if not slots:
        return text
    return re.sub("\x00(\\d+)\x00", lambda m: slots[int(m.group(1))], text)


def apply_special(rel: str, text: str, cur: dict, new: dict, counts: dict, notes: list) -> str:
    """The two strings that are not paths."""
    if rel in ("pipeline/track_clip.sh", "pipeline/track_clip_leadcfg.sh"):
        key = "RUNS_LEADCFG_DIR" if rel.endswith("_leadcfg.sh") else "RUNS_DIR"
        drive = split_drive(new[key])[0]
        if drive is None:
            if re.search(r'"[A-Za-z]:\$CLIP"', text):
                notes.append(f'{rel}: "C:$CLIP" left as it is ({key} has no drive letter); edit the two lines by hand')
        else:
            def cb(m):
                s = '"%s:$CLIP"' % drive.upper()
                if s != m.group(0):
                    counts["CLIP_DRIVE"] = counts.get("CLIP_DRIVE", 0) + 1
                return s
            text = re.sub(r'"[A-Za-z]:\$CLIP"', cb, text)
    if rel == "pipeline/gsr/run_gsr_batch_leadcfg.sh":
        a = "'*%s*'" % cur["RUNS_LEADCFG_DIR"].rsplit("/", 1)[-1]
        b = "'*%s*'" % new["RUNS_LEADCFG_DIR"].rsplit("/", 1)[-1]
        if a != b and a in text:
            counts["RUNS_LEADCFG_NAME"] = counts.get("RUNS_LEADCFG_NAME", 0) + text.count(a)
            text = text.replace(a, b)
    return text


# ----------------------------------------------------------------------------- files and state
def package_files(pkg: str, lab_dirs=()):
    """(relative path, absolute path) of every file of the package, sorted."""
    skip_abs = {norm(d).lower() for d in lab_dirs}
    out = []
    for dp, dn, fn in os.walk(pkg):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and norm(os.path.join(dp, d)).lower() not in skip_abs
                       and not (norm(dp) == norm(pkg) and d == LAB_DIRNAME))
        for f in sorted(fn):
            full = os.path.join(dp, f)
            out.append((norm(os.path.relpath(full, pkg)), full))
    return out


def is_text(rel: str) -> bool:
    return os.path.splitext(rel)[1].lower() in TEXT_EXT and rel not in SKIP_FILES


def load_log(pkg: str):
    p = os.path.join(pkg, LOG_NAME)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def current_state(pkg: str):
    log = load_log(pkg)
    if log is None:
        return dict(ORIGINAL), set(), None
    cur = dict(ORIGINAL)
    cur.update(log.get("current", {}))
    return cur, set(log.get("confirmed", [])), log


def read_text(path: str):
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw, raw.decode("utf-8", errors="surrogateescape")


def write_text(path: str, text: str):
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8", errors="surrogateescape"))


# ----------------------------------------------------------------------------- commands
def wanted_state(pkg: str, args, cur: dict, log):
    """The values after this run; the keys that are settled; the keys the user set by hand."""
    explicit = set((log or {}).get("explicit", []))
    confirmed = set((log or {}).get("confirmed", []))
    given = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
        for k, v in cfg.items():
            if k.startswith("_") or v in (None, ""):
                continue
            if k not in ORIGINAL:
                raise SystemExit(f"--config: unknown key {k!r}; known keys: {', '.join(KEYS)}")
            given[k] = user_path(v)
    for k in KEYS:
        v = getattr(args, k.lower())
        if v:
            given[k] = user_path(v)
    explicit |= set(given)
    new = dict(cur)
    new.update(given)
    done = confirmed | set(given)
    # folders inside the package follow the package unless the user set them by hand
    for k, sub in (("TOOLS_ROOT", "method"), ("PIPELINE_ROOT", "pipeline"), ("EVIDENCE_ROOT", "evidence"),
                   ("PATCH_DIR", "pipeline/patches"), ("LEADCFG_ROOT", LAB_DIRNAME)):
        if k not in explicit:
            new[k] = norm(pkg) + "/" + sub
            done.add(k)
    if "PYTHON" not in explicit and "GBF_REPO" in given:
        windows = split_drive(given["GBF_REPO"])[0] is not None
        new["PYTHON"] = given["GBF_REPO"] + ("/.venv/Scripts/python.exe" if windows else "/.venv/bin/python")
        done.add("PYTHON")
    if "GSR_ROOT" not in explicit and "GSR_TRAIN_DIR" in given:
        new["GSR_ROOT"] = norm(os.path.dirname(given["GSR_TRAIN_DIR"]))
        done.add("GSR_ROOT")
    return new, done, explicit


def validate(new: dict):
    notes = []
    for k, v in new.items():
        if re.search(r"\s", v):
            notes.append(f"{k}: {v!r} contains a space; the shell scripts use these paths unquoted in places")
        if not (re.match(r"^[A-Za-z]:/", v) or v.startswith("/")):
            raise SystemExit(f"{k}: {v!r} is not an absolute path")
    for a, b in (("RUNS_DIR", "RUNS_LEADCFG_DIR"), ("TOOLS_ROOT", "LEADCFG_ROOT"), ("TOOLS_ROOT", "EVIDENCE_ROOT"),
                 ("PIPELINE_ROOT", "EVIDENCE_ROOT"), ("PIPELINE_ROOT", "LEADCFG_ROOT")):
        if new[a] == new[b]:
            raise SystemExit(f"{a} and {b} must be different folders ({new[a]})")
    return notes


def rewrite_tree(root: str, files, cur: dict, new: dict, dry: bool, specials: bool):
    """Apply the rule table to the given (rel, full) files. Returns (per-file records, counts, notes)."""
    comp = {False: compile_rules(build_rules(cur, new, False)), True: compile_rules(build_rules(cur, new, True))}
    records, total, notes = [], {}, []
    for rel, full in files:
        if not is_text(rel):
            continue
        raw, text = read_text(full)
        if "\x00" in text:
            notes.append(f"{rel}: contains NUL bytes, left as it is")
            continue
        counts = {}
        out = apply_rules(text, comp[rel.startswith("pipeline/")], counts)
        if specials:
            out = apply_special(rel, out, cur, new, counts, notes)
        if out == text:
            continue
        data = out.encode("utf-8", errors="surrogateescape")
        records.append({"path": rel, "replacements": sum(counts.values()), "by_key": counts,
                        "sha256_before": sha256_bytes(raw), "sha256_after": sha256_bytes(data)})
        for k, n in counts.items():
            total[k] = total.get(k, 0) + n
        if not dry:
            with open(full, "wb") as fh:
                fh.write(data)
    return records, total, notes


def cmd_relocate(pkg: str, args, home: str = "") -> int:
    """pkg: the files to rewrite. home: where the package will live (differs only for --dry-run --dest)."""
    cur, confirmed, log = current_state(pkg)
    new, done, explicit = wanted_state(home or pkg, args, cur, log)
    notes = validate(new)
    files = package_files(pkg, (cur["LEADCFG_ROOT"], new["LEADCFG_ROOT"]))
    records, total, more = rewrite_tree(pkg, files, cur, new, args.dry_run, specials=True)
    notes += more
    kept = [k for k in KEYS if k not in done]
    print(f"package: {norm(pkg)}")
    print(("would change" if args.dry_run else "changed") + f" {len(records)} file(s), {sum(total.values())} replacement(s)")
    for k in sorted(total):
        print(f"  {k:18s} {total[k]:5d}")
    for k in KEYS:
        mark = "" if k in done else "   (kept: not given; pass %s)" % FLAGS[k]
        print(f"  {k:18s} {new[k]}{mark}")
    for n in notes:
        print("note:", n)
    if args.dry_run:
        if args.verbose:
            for r in records:
                print(f"  {r['replacements']:4d}  {r['path']}")
        return 0
    pristine = dict((log or {}).get("pristine_sha256", {}))
    for r in records:
        pristine.setdefault(r["path"], r["sha256_before"])
    history = list((log or {}).get("history", []))
    if log is not None:
        history.append({k: log.get(k) for k in ("time_utc", "previous", "current", "files_changed", "replacements")})
    out = {
        "tool": "reid-lab relocate.py", "format": 1,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": norm(pkg),
        "original": ORIGINAL, "previous": cur, "current": new,
        "confirmed": sorted(done), "explicit": sorted(explicit), "kept_at_original_value": kept,
        "files_changed": len(records), "replacements": sum(total.values()), "by_key": total,
        "notes": notes, "files": records, "pristine_sha256": pristine, "history": history,
    }
    with open(os.path.join(pkg, LOG_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {LOG_NAME}")
    if kept:
        print("kept at the original machine's value:", ", ".join(kept))
    return 0


def classify(cand: str, rules, ok_keys: set, pkg: str):
    """(ok, key): is this absolute path under a configured location, and which key does it belong to?
    The key is found the way the rewrite finds it: the longest rule that starts the path."""
    c = norm(cand)
    m = re.match(r"^/([A-Za-z])/(Research|Users)(/.*)?$", c)
    if m:                                                    # Git Bash form
        c = m.group(1).upper() + ":/" + m.group(2) + (m.group(3) or "")
    for f, _, key in rules:
        hit = under(c, f)
        if not hit and key in DRIVELESS_KEYS and split_drive(f)[0] and not split_drive(c)[0]:
            hit = under(c, split_drive(f)[1])                # written without the drive letter
        if hit:
            return key in ok_keys, key
    if under(c, norm(pkg)):
        return True, "PACKAGE"
    return False, "UNKNOWN"


def cmd_check(pkg: str, args) -> int:
    cur, confirmed, log = current_state(pkg)
    files = package_files(pkg, (cur["LEADCFG_ROOT"],))
    rules = {False: build_rules(cur, cur, False), True: build_rules(cur, cur, True)}
    flagged, verbatim = {}, {}
    n_files = set()
    for rel, full in files:
        ext = os.path.splitext(rel)[1].lower()
        if rel in SKIP_FILES or ext not in TEXT_EXT | VERBATIM_EXT:
            continue
        _, text = read_text(full)
        for i, line in enumerate(text.splitlines(), 1):
            for m in CANDIDATE.finditer(line):
                if ext in VERBATIM_EXT:
                    verbatim.setdefault(rel, []).append((i, m.group(0)))
                    continue
                ok, key = classify(m.group(0), rules[rel.startswith("pipeline/")], confirmed, pkg)
                if not ok:
                    flagged.setdefault(key, []).append((rel, i, m.group(0)))
                    n_files.add(rel)
    print(f"package: {norm(pkg)}")
    print("state:", "not relocated yet (the files hold the original machine's paths)" if log is None
          else f"relocated {log.get('time_utc')}; configured: {', '.join(sorted(confirmed))}")
    total = sum(len(v) for v in flagged.values())
    if total:
        print(f"{total} absolute path(s) in {len(n_files)} file(s) are not under a configured location:")
        for key in sorted(flagged):
            hits = flagged[key]
            hint = f"  (pass {FLAGS[key]})" if key in FLAGS else ""
            print(f"  {key}: {len(hits)} in {len({h[0] for h in hits})} file(s){hint}")
            for rel, i, s in (hits if args.verbose else hits[:5]):
                print(f"      {rel}:{i}: {s}")
            if not args.verbose and len(hits) > 5:
                print(f"      ... {len(hits) - 5} more (use -v)")
    else:
        print("no absolute path outside the configured locations")
    if verbatim:
        print("kept verbatim, never rewritten (edit after applying the patch):")
        for rel, hits in sorted(verbatim.items()):
            print(f"  {rel}: {len(hits)} path(s), e.g. line {hits[0][0]}: {hits[0][1]}")
    return 1 if total else 0


def cmd_show(pkg: str) -> int:
    cur, confirmed, log = current_state(pkg)
    print(f"package: {norm(pkg)}   ({'relocated ' + str(log.get('time_utc')) if log else 'not relocated yet'})")
    for k in KEYS:
        print(f"{k}  [{FLAGS[k]}]  {MEANING[k]}")
        print(f"    original: {ORIGINAL[k]}")
        print(f"    current:  {cur[k]}{'' if k in confirmed or not log else '   (kept)'}")
    return 0


def cmd_verify_manifest(pkg: str) -> int:
    path = os.path.join(pkg, MANIFEST_NAME)
    if not os.path.isfile(path):
        print(f"no {MANIFEST_NAME}")
        return 2
    log = load_log(pkg) or {}
    pristine = log.get("pristine_sha256", {})
    after = {r["path"]: r["sha256_after"] for r in log.get("files", [])}
    same = moved = 0
    bad = []
    listed = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            want, rel = line.split("  ", 1)
            listed.add(rel)
            full = os.path.join(pkg, rel)
            if not os.path.isfile(full):
                bad.append((rel, "missing"))
                continue
            with open(full, "rb") as f2:
                raw = f2.read()
            got = sha256_bytes(raw)
            if got == want or sha256_bytes(raw.replace(b"\r\n", b"\n")) == want:
                same += 1
            elif pristine.get(rel) == want:
                moved += 1                                    # rewritten by relocate.py
            else:
                bad.append((rel, "different"))
    extra = [rel for rel, _ in package_files(pkg, (log.get("current", {}).get("LEADCFG_ROOT", ""),))
             if rel not in listed and rel not in (LOG_NAME, MANIFEST_NAME)]
    print(f"{same} file(s) as shipped, {moved} rewritten by relocate.py, {len(bad)} different or missing, {len(extra)} not in the manifest")
    for rel, why in bad[:40]:
        print(f"  {why}: {rel}")
    for rel in extra[:40]:
        print(f"  not in the manifest: {rel}")
    return 1 if bad else 0


def cmd_make_lab(pkg: str, args) -> int:
    cur, confirmed, log = current_state(pkg)
    if log is None:
        raise SystemExit("make-lab: relocate the package first (python relocate.py --runs ... --runs-leadcfg ...)")
    dest = user_path(args.dest) if args.dest else cur["LEADCFG_ROOT"]
    runs = user_path(args.runs_dir) if args.runs_dir else cur["RUNS_LEADCFG_DIR"]
    if "RUNS_LEADCFG_DIR" not in confirmed and not args.runs_dir:
        raise SystemExit("make-lab: the lead-settings runs folder is not set (pass --runs-leadcfg to relocate, or --runs-dir here)")
    src = cur["TOOLS_ROOT"] + "/reid_lab"
    if not os.path.isdir(src):
        raise SystemExit(f"make-lab: {src} does not exist")
    if dest == cur["TOOLS_ROOT"] or runs == cur["RUNS_DIR"]:
        raise SystemExit("make-lab: the copy needs its own folder and its own runs folder")
    target = dest + "/reid_lab"
    if os.path.isdir(target) and os.listdir(target) and not args.force:
        raise SystemExit(f"make-lab: {target} is not empty (use --force to copy over it; nothing is deleted)")
    copied = []
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            copied.append((name, os.path.join(src, name)))
    for agent in ENGINE:
        base = os.path.join(src, "work", agent)
        if not os.path.isdir(base):
            raise SystemExit(f"make-lab: {norm(base)} is missing")
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d not in LAB_SKIP_DIRS)
            for f in sorted(fn):
                if os.path.splitext(f)[1].lower() in LAB_SKIP_EXT:
                    continue
                full = os.path.join(dp, f)
                copied.append((norm(os.path.relpath(full, src)), full))
    for rel, full in copied:
        out = os.path.join(target, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(full, out)
    lab_cur = dict(cur)
    lab_new = dict(cur)
    lab_new["TOOLS_ROOT"] = dest
    lab_new["RUNS_DIR"] = runs
    lab_files = [("lab/" + rel, os.path.join(target, rel)) for rel, _ in copied]   # "lab/": never a pipeline file
    records, total, notes = rewrite_tree(target, lab_files, lab_cur, lab_new, False, specials=False)
    with open(os.path.join(dest, "MADE-BY-RELOCATE.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"what": "lead-settings copy of the lab, built by relocate.py make-lab",
                   "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": src,
                   "TOOLS_ROOT": dest, "RUNS_DIR": runs, "files_copied": len(copied),
                   "files_rewritten": len(records), "replacements": total}, fh, indent=1)
        fh.write("\n")
    print(f"copied {len(copied)} file(s) to {target}; rewrote {len(records)} ({sum(total.values())} replacement(s))")
    print(f"  the copy reads its runs from {runs}")
    # point the package (the driver script, the shipped logs) at the copy
    new = dict(cur)
    new["LEADCFG_ROOT"] = dest
    new["RUNS_LEADCFG_DIR"] = runs
    if new != cur:
        files = package_files(pkg, (cur["LEADCFG_ROOT"], dest))
        recs, tot, more = rewrite_tree(pkg, files, cur, new, False, specials=True)
        pristine = dict(log.get("pristine_sha256", {}))
        for r in recs:
            pristine.setdefault(r["path"], r["sha256_before"])
        history = list(log.get("history", []))
        history.append({k: log.get(k) for k in ("time_utc", "previous", "current", "files_changed", "replacements")})
        log.update({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "previous": cur, "current": new,
                    "confirmed": sorted(set(confirmed) | {"LEADCFG_ROOT", "RUNS_LEADCFG_DIR"}),
                    "explicit": sorted(set(log.get("explicit", [])) | ({"LEADCFG_ROOT"} if args.dest else set())
                                       | ({"RUNS_LEADCFG_DIR"} if args.runs_dir else set())),
                    "files_changed": len(recs), "replacements": sum(tot.values()), "by_key": tot, "files": recs,
                    "pristine_sha256": pristine, "history": history, "notes": more})
        log["kept_at_original_value"] = [k for k in KEYS if k not in log["confirmed"]]
        with open(os.path.join(pkg, LOG_NAME), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(log, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"package updated: {len(recs)} file(s) now point at {dest}")
    for n in notes:
        print("note:", n)
    print("next: bash pipeline/leadcfg/run_leadcfg_eval.sh T")
    return 0


def copy_package(pkg: str, dest: str):
    if os.path.exists(dest) and os.listdir(dest):
        raise SystemExit(f"--dest: {dest} exists and is not empty")
    n = 0
    for rel, full in package_files(pkg):
        if rel == LOG_NAME:
            continue
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(full, out)
        n += 1
    print(f"copied {n} file(s) to {dest}")


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Point the hard-coded paths of the reid-lab package at this machine.",
                                 epilog="See the top of this file and docs/ for the details.")
    ap.add_argument("command", nargs="?", choices=["make-lab"], help="make-lab: build the lead-settings copy of the lab")
    for k in KEYS:
        ap.add_argument(FLAGS[k], dest=k.lower(), metavar="PATH", help=MEANING[k])
    ap.add_argument("--config", metavar="JSON", help="a JSON file with the keys (see relocate.example.json)")
    ap.add_argument("--dest", metavar="DIR", help="copy the package to DIR and relocate the copy (make-lab: where to build the lab copy)")
    ap.add_argument("--check", action="store_true", help="list absolute paths that are not under a configured location; change nothing")
    ap.add_argument("--show", action="store_true", help="print the keys with their original and current values")
    ap.add_argument("--dry-run", action="store_true", help="say what would change; write nothing")
    ap.add_argument("--verify-manifest", action="store_true", help="compare the files with manifest-sha256.txt")
    ap.add_argument("--runs-dir", metavar="DIR", help="make-lab: the runs folder the copy reads (default RUNS_LEADCFG_DIR)")
    ap.add_argument("--force", action="store_true", help="make-lab: copy over an existing lab copy")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    pkg = here
    if args.command == "make-lab":
        return cmd_make_lab(pkg, args)
    if args.show:
        return cmd_show(pkg)
    if args.check:
        return cmd_check(pkg, args)
    if args.verify_manifest:
        return cmd_verify_manifest(pkg)
    if args.dest:
        dest = user_path(args.dest)
        if args.dry_run:
            print(f"--dry-run with --dest: would copy the package to {dest} and relocate the copy")
            return cmd_relocate(pkg, args, home=dest)
        copy_package(pkg, dest)
        pkg = dest
    return cmd_relocate(pkg, args)


if __name__ == "__main__":
    sys.exit(main())
