"""fuse_r1 / diag_points.py -- DIAGNOSIS with labels of finished METHOD files (not a method result;
nothing here feeds back into the method).

    python -I diag_points.py METHOD_strict.json METHOD_balanced.json ...   -> out/diag_points.md, out/diag_points.json

For every METHOD file:
  - the wrong returns (clip, annotated person, frames) and the links score.py judges wrong;
  - "right but mixed": returns scored right whose identity also holds >= 5 matched boxes of another
    annotated person (v_leak's reading of precision: wrong + mixed);
  - impure rows per clip against CUTS ONLY (the same cuts, no links): what the linking adds;
  - identities that hold >= 5 boxes of two annotated people;
  - links that score.py cannot judge, by weak labels (majority person of >= 1 matched box);
  - returns with a gap over 10 s;
  - completeness: share of an annotated player's matched boxes that lie in his largest identity.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score  # noqa: E402  LABELS (diagnosis)

HERE = Path(__file__).resolve().parent


def groups_of(ids: np.ndarray, links) -> np.ndarray:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        parent[find(int(a))] = find(int(b))
    return np.array([find(int(x)) if x >= 0 else -1 - i for i, x in enumerate(ids)], dtype=np.int64)


def clip_diag(seq: str, links, cuts) -> dict:
    c = score.clip(seq)
    ids = score._ids_after_cuts(c, cuts)
    g = groups_of(ids, links)
    g0 = groups_of(ids, [])
    res = score.score_clip(seq, links, cuts)
    res0 = score.score_clip(seq, [], cuts)
    tracked = ids >= 0
    out = {"returns": Counter(), "wrong_returns": [], "mixed_right": [], "long": Counter(), "impure": res["impure_rows"],
           "impure_cuts_only": res0["impure_rows"], "rows": res["tracked_rows"], "wrong_links": [], "weak": Counter()}
    # people per identity
    def people(gg):
        d = defaultdict(Counter)
        for x, r in zip(gg[tracked], c.refs[tracked]):
            d[int(x)][int(r)] += 1
        return d
    P = people(g)
    P0 = people(g0)
    two = lambda d: int(sum(1 for cnt in d.values() if sum(1 for v in cnt.values() if v >= 5) >= 2))  # noqa: E731
    out["identities_with_two_people"] = two(P)
    out["identities_with_two_people_cuts_only"] = two(P0)
    for r, o in zip(c.returns, res["per_return"]):
        out["returns"][o] += 1
        if o in ("right", "wrong", "lost"):
            if r["gap"] > 250:
                out["long"][o] += 1
            rec = {"clip": seq, "person": r["ref_id"], "last": r["last"], "first": r["first"], "gap": r["gap"],
                   "piece_a": int(ids[r["a"]]), "piece_b": int(ids[r["b"]])}
            if o == "wrong":
                out["wrong_returns"].append(rec)
            if o == "right":
                cnt = P[int(g[r["b"]])]
                others = {k: v for k, v in cnt.items() if k != r["ref_id"] and v >= 5}
                if others:
                    out["mixed_right"].append({**rec, "others": others})
    ident_firm = {}
    ident_weak = {}
    for tid in np.unique(ids[tracked]):
        top, n = Counter(c.refs[ids == tid].tolist()).most_common(1)[0]
        ident_weak[int(tid)] = int(top)
        if n >= 5:
            ident_firm[int(tid)] = int(top)
    for a, b in links:
        a, b = int(a), int(b)
        if a in ident_firm and b in ident_firm:
            if ident_firm[a] != ident_firm[b]:
                out["wrong_links"].append({"clip": seq, "a": a, "b": b, "person_a": ident_firm[a], "person_b": ident_firm[b]})
        elif a in ident_weak and b in ident_weak:
            out["weak"]["agree" if ident_weak[a] == ident_weak[b] else "disagree"] += 1
        else:
            out["weak"]["no matched box"] += 1
    # completeness: per annotated player/goalkeeper, share of matched tracked boxes in his largest identity
    comp_num = comp_den = 0
    by_person = defaultdict(Counter)
    for x, r in zip(g[tracked], c.refs[tracked]):
        by_person[int(r)][int(x)] += 1
    for r, cnt in by_person.items():
        comp_num += max(cnt.values())
        comp_den += sum(cnt.values())
    out["completeness"] = (comp_num, comp_den)
    return out


def main(files):
    rep = {}
    split = score.evaluate(score.baseline_links("split"))["per_clip"]
    c0 = {s: score.clip(s) for s in data.DEV_CLIPS}
    split_share = {s: split[s]["impure_rows"] / int((c0[s].piece >= 0).sum()) for s in data.DEV_CLIPS}
    L = ["# fuse_r1: diagnosis of the METHOD files with labels (not method results)", ""]
    for f in files:
        m = json.loads((HERE / f).read_text(encoding="utf-8"))
        per = {s: clip_diag(s, m["clips"][s]["links"], m["clips"][s].get("cuts") or []) for s in data.DEV_CLIPS}
        tot = Counter()
        for d in per.values():
            tot.update(d["returns"])
        wrong = [w for d in per.values() for w in d["wrong_returns"]]
        mixed = [w for d in per.values() for w in d["mixed_right"]]
        wl = [w for d in per.values() for w in d["wrong_links"]]
        weak = Counter()
        long = Counter()
        for d in per.values():
            weak.update(d["weak"])
            long.update(d["long"])
        imp = sum(d["impure"] for d in per.values())
        imp0 = sum(d["impure_cuts_only"] for d in per.values())
        rows = sum(d["rows"] for d in per.values())
        cn = sum(d["completeness"][0] for d in per.values())
        cd = sum(d["completeness"][1] for d in per.values())
        rep[f] = {"returns": dict(tot), "wrong_returns": wrong, "right_but_mixed": mixed, "wrong_links": wl, "weak_links": dict(weak),
                  "gap_over_10s": dict(long), "impure_rows": imp, "impure_rows_cuts_only": imp0, "rows": rows,
                  "impure_added_per_clip": {s: per[s]["impure"] - per[s]["impure_cuts_only"] for s in per},
                  "impure_share_per_clip": {s: round(per[s]["impure"] / per[s]["rows"], 5) for s in per},
                  "clips_more_impure_than_gbf_now": [s for s in per if per[s]["impure"] / per[s]["rows"] > split_share[s] + 1e-12],
                  "identities_with_two_people": sum(d["identities_with_two_people"] for d in per.values()),
                  "identities_with_two_people_cuts_only": sum(d["identities_with_two_people_cuts_only"] for d in per.values()),
                  "completeness": round(cn / cd, 4)}
        L += [f"## {f}", "", f"{m.get('name')}", "",
              f"- returns: right {tot['right']}, wrong {tot['wrong']}, lost {tot['lost']} (undetected {tot['undetected']})",
              f"- wrong returns: " + ("; ".join(f"{w['clip']} person {w['person']} (frames {w['last']}>{w['first']}, pieces {w['piece_a']}>{w['piece_b']})" for w in wrong) or "none"),
              f"- right but mixed (identity also holds >= 5 boxes of another person): {len(mixed)}"
              + ("" if not mixed else ": " + "; ".join(f"{w['clip']} person {w['person']} with {w['others']}" for w in mixed)),
              f"- wrong + mixed: {tot['wrong'] + len(mixed)}",
              f"- links judged wrong: " + ("; ".join(f"{w['clip']} {w['a']}>{w['b']} (persons {w['person_a']}, {w['person_b']})" for w in wl) or "none"),
              f"- links score.py cannot judge, by weak labels: agree {weak['agree']}, disagree {weak['disagree']}, a piece without a matched box {weak['no matched box']}",
              f"- returns with a gap over 10 s: right {long['right']}, wrong {long['wrong']}, lost {long['lost']}",
              f"- impure rows: {imp} of {rows} = {100 * imp / rows:.2f}% (cuts only: {imp0} = {100 * imp0 / rows:.2f}%; added by links: {imp - imp0})",
              "- impure rows added by links per clip: " + ", ".join(f"{s.split('-')[1]} {per[s]['impure'] - per[s]['impure_cuts_only']:+d}" for s in per),
              "- clips whose impure-row share is above gbf now (split, no cuts): " + (", ".join(
                  f"{s} {100 * per[s]['impure'] / per[s]['rows']:.2f}% against {100 * split_share[s]:.2f}%" for s in rep[f]["clips_more_impure_than_gbf_now"]) or "none"),
              f"- identities with >= 5 boxes of two annotated people: {rep[f]['identities_with_two_people']} (cuts only: {rep[f]['identities_with_two_people_cuts_only']})",
              f"- share of a person's matched boxes in his largest identity: {rep[f]['completeness']}", ""]
    (HERE / "out" / "diag_points.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    text = "\n".join(L) + "\n"
    (HERE / "out" / "diag_points.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["METHOD_strict.json", "METHOD_balanced.json", "METHOD_recall.json",
                          "METHOD_fixed_strict.json", "METHOD_fixed_balanced.json", "METHOD_fixed_recall.json"])
