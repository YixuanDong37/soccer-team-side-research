"""Write gbf's plugin-sngs export from the tracklab tracker state of one sn-gamestate run.

Runs in the sn-gamestate environment (it unpickles tracklab's own state file, which gbf's core
never does). Inputs: --state (the .pklz of a run on ONE video read through
tools/exact_reader, so tracklab's image id is gbf's canonical frame index), --video (that
video: frame count, size, sha256), --out. Outputs in --out, as gbf.sngs_import reads them
(docs/design/m1-highlights.md, "plugin-sngs export rows"):

- detections.jsonl: one row per detection: frame, state_frame, track_id (-1 without a track),
  box [x1, y1, x2, y2], kf_box (StrongSORT's Kalman box or null), conf, role (the tracklet's
  voted role; null becomes "unknown"), team ("left" | "right" | null), team_cluster, jersey and
  jersey_conf (the per-detection OCR read, not the tracklet vote, so gbf can judge reads);
- calib.jsonl: one row per frame: calib_status "fresh" when NBJW gave a camera on this frame
  (the run uses use_prev_homography=False, so there is no "held"), else "none"; calib_age 0;
  n_keypoints, n_line_classes; K, R (world to camera), C, radial_distortion, width, height;
  rep_err_px: the median pixel distance between this frame's keypoints with p >= 0.5 and their
  model points projected with K, R, C (null with fewer than 4), and rep_err_n_keypoints;
- plugin-receipt.json: sha256 and size of both files, the restrictions, video_frames, and the
  provenance of this local rebuild;
- keypoints.jsonl and keypoints-receipt.json (unless --no-keypoints): NBJW's pitch keypoints of
  every frame as gbf.calib_refine.load_keypoints reads them, [u, v, X, Y, Z, score, id], moved
  to the centre of their heatmap cell (see keypoint_lines), so gbf can refit the calibration
  with a fixed camera centre; the receipt carries their sha256, size, licence and tags;
- embeddings.npz (only with --embeddings): the tracker's re-identification embedding of every
  detection, one row per line of detections.jsonl in the same order: frame (int32), track
  (int64, -1 without a track), box (float64, n x 4: the line's rounded box) and emb (float32,
  n x D: PRTReid's embedding as stored, not normalised; a NaN row where a detection has no
  usable one). It is listed in plugin-receipt.json with its sha256 and size, so gbf's import
  verifies it like the other files. The embeddings are PRTReid products: the flag is refused
  unless hl3 is among --restrictions.

It also checks itself: for every detection sn-gamestate placed on the pitch (bbox_pitch, from
its ground-plane homography), the bottom-centre ground point through the exported K, R, C must
lie close to sn-gamestate's own; the median and P90 distance go into the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from nbjw_calib.utils.utils_calib import keypoint_aux_world_coords_2D, keypoint_world_coords_2D

GOAL_TOP = {12, 15, 16, 19}  # crossbar keypoints, 2.44 m above the grass (z down)
# sn-gamestate's NBJW wrapper resizes every frame to 960x540 (T.Resize((540, 960))) and decodes
# a keypoint at the top-left corner of its heatmap cell: get_keypoints_from_heatmap_batch_maxpool
# returns index * scale with scale = 2 network-input pixels per cell. keypoints.jsonl moves each
# point to the centre of its cell (+1 input pixel); line intersections, built from line ends
# decoded the same way, move by the same amount.
NBJW_INPUT = (960, 540)
NBJW_CELL = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def video_facts(video: Path) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    width, height, frames = (int(v) for v in out.split(",")[:3])
    return width, height, frames


def load_state(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        images = [n for n in names if n.endswith("_image.pkl")]
        dets = [n for n in names if n.endswith(".pkl") and n not in images]
        if len(dets) != 1 or len(images) != 1:
            raise SystemExit(f"expected one video in {path}, found {names}")
        detections = pd.read_pickle(io.BytesIO(zf.read(dets[0])))
        image_pred = pd.read_pickle(io.BytesIO(zf.read(images[0])))
    return detections, image_pred


def world_point(kp: int) -> list[float]:
    x, y = keypoint_world_coords_2D[kp - 1] if kp <= 57 else keypoint_aux_world_coords_2D[kp - 58]
    return [float(x), float(y), -2.44 if kp in GOAL_TOP else 0.0]


def project(K: np.ndarray, R: np.ndarray, C: np.ndarray, points: np.ndarray) -> np.ndarray:
    cam = (points - C) @ R.T
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def ground(K: np.ndarray, R: np.ndarray, C: np.ndarray, u: float, v: float) -> np.ndarray | None:
    ray = R.T @ np.linalg.solve(K, np.array([u, v, 1.0]))
    if abs(ray[2]) < 1e-9:
        return None
    t = -C[2] / ray[2]
    return None if t <= 0 else (C + t * ray)[:2]


def number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def text(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else str(value)
    value = str(value).strip()
    return value or None


def ltrb(box) -> list[float] | None:
    if box is None:
        return None
    arr = np.asarray(box, dtype=float).ravel()
    if arr.size != 4 or not np.isfinite(arr).all() or arr[2] <= 0 or arr[3] <= 0:
        return None
    return [round(float(arr[0]), 3), round(float(arr[1]), 3),
            round(float(arr[0] + arr[2]), 3), round(float(arr[1] + arr[3]), 3)]  # fmt: skip


def embedding(det) -> np.ndarray | None:
    """The detection's stored re-id embedding as a flat float32 vector, or None."""
    value = getattr(det, "embeddings", None)
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return arr if arr.size > 1 else None


def write_embeddings(path: Path, rows: list) -> int:
    """Write embeddings.npz: one row per detection line, in the same order; a NaN row where a
    detection has no usable embedding. Returns the embedding dimension."""
    sizes = {int(e.size) for _f, _t, _b, e in rows if e is not None}
    if len(sizes) != 1:
        raise SystemExit(f"--embeddings: expected embeddings of one size, found sizes {sorted(sizes)}")
    dim = sizes.pop()
    emb = np.full((len(rows), dim), np.nan, dtype=np.float32)
    for i, (_f, _t, _b, e) in enumerate(rows):
        if e is not None:
            emb[i] = e
    np.savez_compressed(
        path,
        frame=np.array([f for f, _t, _b, _e in rows], np.int32),
        track=np.array([t for _f, t, _b, _e in rows], np.int64),
        box=np.array([b for _f, _t, b, _e in rows], np.float64).reshape(len(rows), 4),
        emb=emb,
    )
    return dim


def keypoint_lines(
    images: pd.DataFrame, n_frames: int, width: int, height: int, cams: dict, centred: bool = False
) -> tuple[list[str], dict]:
    """keypoints.jsonl rows (one per canonical frame, possibly with no points) and their facts.

    A point is [u, v, X, Y, Z, score, keypoint_id]: u, v in the export's pixel grid as NBJW
    counts it (the grid of calib.jsonl's K), moved to the centre of the heatmap cell; X, Y, Z
    in the SoccerNet world frame (z down, crossbar keypoints at z = -2.44); score is NBJW's
    heatmap peak (1.0 for a keypoint built from two lines). The stored keypoints must already
    be in pixels (NBJW's calibration denormalises them in place on every frame).

    centred: the plugin run already moved every keypoint to its cell centre before its own
    camera fit (tools/run_tracklab_leadcfg.py). The heatmap points must then lie on the
    cell-centre grid, and they are not moved a second time."""
    sx, sy = width / NBJW_INPUT[0], height / NBJW_INPUT[1]
    shift_u, shift_v = (0.0, 0.0) if centred else (NBJW_CELL / 2 * sx, NBJW_CELL / 2 * sy)
    grid_offset = 0.5 if centred else 0.0  # in cells: corners, or centres when already moved
    lines, frames_with, points_total, strong, on_grid, gridded = [], 0, 0, 0, 0, 0
    errors_raw, errors_moved = [], []
    for frame in range(n_frames):
        row = images.loc[frame] if frame in images.index else None
        kps = row["keypoints"] if row is not None and "keypoints" in row else None
        kps = kps if isinstance(kps, dict) else {}
        coords = [abs(float(v[a])) for v in kps.values() for a in ("x", "y")]
        if coords and max(coords) <= 1.5:
            raise SystemExit(f"frame {frame}: keypoints look normalised, not pixels")
        points, raw, model = [], [], []
        for kp in sorted(int(k) for k in kps):
            v = kps[kp]
            u, w, p = float(v["x"]), float(v["y"]), float(v["p"])
            if not (np.isfinite(u) and np.isfinite(w) and 0.0 <= p <= 1.0):
                raise SystemExit(f"frame {frame}: keypoint {kp} has u={u}, v={w}, p={p}")
            X, Y, Z = world_point(kp)
            points.append([round(u + shift_u, 4), round(w + shift_v, 4), X, Y, Z, round(p, 6), kp])
            strong += p >= 0.4
            if p < 1.0:  # a heatmap peak, not a line intersection
                gridded += 1
                cell_u = u / (NBJW_CELL * sx) - grid_offset
                cell_v = w / (NBJW_CELL * sy) - grid_offset
                on_grid += abs(cell_u - round(cell_u)) < 1e-6 and abs(cell_v - round(cell_v)) < 1e-6
            if p >= 0.5:
                raw.append([u, w])
                model.append([X, Y, Z])
        if frame in cams and len(raw) >= 4:
            pred = project(*cams[frame], np.array(model))
            obs = np.array(raw)
            errors_raw.append(float(np.median(np.hypot(*(pred - obs).T))))
            errors_moved.append(float(np.median(np.hypot(*(pred - obs - [shift_u, shift_v]).T))))
        frames_with += bool(points)
        points_total += len(points)
        lines.append(json.dumps({"frame": frame, "points": points}, separators=(",", ":")))
    grid_name = "cell-centre" if centred else "cell-corner"
    if gridded and on_grid < 0.99 * gridded:
        raise SystemExit(
            f"only {on_grid} of {gridded} heatmap keypoints lie on the {NBJW_CELL} x {sx:.4f} px "
            f"{grid_name} grid: the keypoints were not denormalised once from NBJW's 960x540 "
            "input to this frame size"
            + (", or the plugin did not move them to the cell centres" if centred
               else ", so the cell-centre shift would be wrong")
        )
    facts = {
        "frames": n_frames,
        "frames_with_points": frames_with,
        "points": points_total,
        "points_score_at_least_0.4": int(strong),
        f"heatmap_points_on_the_{grid_name.replace('-', '_')}_grid": f"{on_grid}/{gridded}",
        "centred_by_plugin": centred,
        "shift_px": [round(shift_u, 6), round(shift_v, 6)],
        "plugin_camera_median_rep_err_px": {
            "frames": len(errors_raw),
            "as_decoded": round(float(np.median(errors_raw)), 3) if errors_raw else None,
            "cell_centres": round(float(np.median(errors_moved)), 3) if errors_moved else None,
        },
    }
    return lines, facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="JSON merged into the receipt")
    parser.add_argument("--restrictions", default="no_share,agpl_network_source,hl3")
    parser.add_argument("--no-keypoints", action="store_true",
                        help="do not write keypoints.jsonl and keypoints-receipt.json")
    parser.add_argument("--keypoints-license", default="GPL-2.0-only",
                        help="licence of the NBJW keypoints (registries/models.toml, nbjw-calib)")
    parser.add_argument("--keypoints-restrictions", default="no_share",
                        help="tags of the keypoints: the clip's (NBJW itself adds none)")
    parser.add_argument("--keypoints-centred", action="store_true",
                        help="the plugin run already moved the keypoints to their heatmap cell "
                        "centres (run_tracklab_leadcfg.py): check that, do not move them again")
    parser.add_argument("--note", default=None,
                        help="replaces the receipt's note on how the plugin was run "
                        "(default: the local rebuild with EasyOCR, use_prev_homography=False)")
    parser.add_argument("--embeddings", action="store_true",
                        help="also write embeddings.npz (one PRTReid embedding per detection "
                        "line) and list it in the receipt; needs hl3 among --restrictions")
    args = parser.parse_args()
    if args.embeddings and "hl3" not in [t for t in args.restrictions.split(",") if t]:
        parser.error("--embeddings needs hl3 among --restrictions: the embeddings are PRTReid "
                     "products")

    width, height, n_frames = video_facts(args.video)
    detections, images = load_state(args.state)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- calibration
    cams: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    calib_lines = []
    stats = {"fresh": 0, "none": 0}
    for frame in range(n_frames):
        row = images.loc[frame] if frame in images.index else None
        params = row["parameters"] if row is not None and "parameters" in row else None
        kps = row["keypoints"] if row is not None and "keypoints" in row else None
        lines = row["lines"] if row is not None and "lines" in row else None
        kps = kps if isinstance(kps, dict) else {}
        lines = lines if isinstance(lines, dict) else {}
        record = {"frame": frame, "state_frame": frame, "n_keypoints": len(kps),
                  "n_line_classes": len(lines)}  # fmt: skip
        if isinstance(params, dict) and params.get("rotation_matrix") is not None:
            fx, fy = float(params["x_focal_length"]), float(params["y_focal_length"])
            cx, cy = (float(v) for v in params["principal_point"])
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            R = np.asarray(params["rotation_matrix"], dtype=float).reshape(3, 3)
            C = np.asarray(params["position_meters"], dtype=float).ravel()
            cams[frame] = (K, R, C)
            # keypoints are normalised by NBJW and denormalised in place by its calibration
            xs = [float(v["x"]) for v in kps.values()]
            ys = [float(v["y"]) for v in kps.values()]
            scale = (width, height) if xs and max(xs + ys) <= 1.5 else (1.0, 1.0)
            used = [k for k, v in kps.items() if float(v.get("p", 0.0)) >= 0.5]
            rep_err = None
            if len(used) >= 4:
                obs = np.array([[float(kps[k]["x"]) * scale[0], float(kps[k]["y"]) * scale[1]]
                                for k in used])  # fmt: skip
                pred = project(K, R, C, np.array([world_point(int(k)) for k in used]))
                rep_err = round(float(np.median(np.hypot(*(pred - obs).T))), 3)
            record.update(
                calib_status="fresh", calib_age=0, K=K.ravel().round(6).tolist(),
                R=R.ravel().round(9).tolist(), C=C.round(6).tolist(),
                radial_distortion=[0.0] * 6, width=width, height=height,
                rep_err_px=rep_err, rep_err_n_keypoints=len(used),
            )  # fmt: skip
            stats["fresh"] += 1
        else:
            record.update(calib_status="none", calib_age=0, rep_err_px=None,
                          rep_err_n_keypoints=0)  # fmt: skip
            stats["none"] += 1
        calib_lines.append(json.dumps(record, separators=(",", ":")))

    # ---------------------------------------------------------------- detections
    det_lines = []
    emb_rows = []  # with --embeddings: (frame, track, box, embedding or None) per line
    gaps = []
    roles = {}
    for det in detections.itertuples(index=False):
        frame = int(det.image_id)
        if not 0 <= frame < n_frames:
            raise SystemExit(f"detection on image {frame}, outside the video's {n_frames} frames")
        box = ltrb(det.bbox_ltwh)
        if box is None:
            continue
        track = number(getattr(det, "track_id", None))
        role = text(getattr(det, "role", None)) or "unknown"
        roles[role] = roles.get(role, 0) + 1
        team = text(getattr(det, "team", None))
        record = {
            "frame": frame,
            "state_frame": frame,
            "track_id": int(track) if track is not None else -1,
            "box": box,
            "kf_box": ltrb(getattr(det, "track_bbox_kf_ltwh", None)),
            "conf": round(min(1.0, max(0.0, float(det.bbox_conf))), 5),
            "role": role,
            "team": team if team in ("left", "right") else None,
            "team_cluster": number(getattr(det, "team_cluster", None)),
            "jersey": text(getattr(det, "jersey_number_detection", None)),
            "jersey_conf": number(getattr(det, "jersey_number_confidence", None)),
        }
        det_lines.append(json.dumps(record, separators=(",", ":")))
        if args.embeddings:
            emb_rows.append((frame, record["track_id"], box, embedding(det)))
        pitch = getattr(det, "bbox_pitch", None)
        if isinstance(pitch, dict) and frame in cams:
            mine = ground(*cams[frame], (box[0] + box[2]) / 2, box[3])
            if mine is not None:
                theirs = np.array([pitch["x_bottom_middle"], pitch["y_bottom_middle"]], float)
                gaps.append(float(np.hypot(*(mine - theirs))))

    det_path, calib_path = args.out / "detections.jsonl", args.out / "calib.jsonl"
    det_path.write_bytes(("\n".join(det_lines) + "\n").encode("utf-8"))
    calib_path.write_bytes(("\n".join(calib_lines) + "\n").encode("utf-8"))
    gaps_arr = np.array(gaps)
    check = {
        "detections_compared": len(gaps),
        "median_m": round(float(np.median(gaps_arr)), 3) if gaps else None,
        "p90_m": round(float(np.percentile(gaps_arr, 90)), 3) if gaps else None,
        "max_m": round(float(gaps_arr.max()), 3) if gaps else None,
        "what": "ground point of the box bottom centre through the exported camera vs "
        "sn-gamestate's own bbox_pitch (its ground-plane homography)",
    }
    receipt = {
        "plugin": "plugin-sngs-export",
        "version": "local-rebuild-2026-11-14",
        "license": "GPL-3.0-only",
        "outputs": [
            {"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (det_path, calib_path)
        ],
        "restrictions": [t for t in args.restrictions.split(",") if t],
        "video_frames": n_frames,
        "video_sha256": sha256(args.video),
        "frame_size": [width, height],
        "counts": {"detections": len(det_lines), "frames": n_frames, **stats, "roles": roles},
        "camera_check": check,
        "note": args.note or "Local rebuild of the project's plugin-sngs export (not the original): "
        "sn-gamestate with local patches, NBJW with use_prev_homography=False, EasyOCR jersey "
        "reads, tracklab frames read through tools/exact_reader.",
    }
    if args.embeddings:
        emb_path = args.out / "embeddings.npz"
        dim = write_embeddings(emb_path, emb_rows)
        receipt["outputs"].append(
            {"name": emb_path.name, "sha256": sha256(emb_path), "bytes": emb_path.stat().st_size}
        )
        receipt["embeddings"] = {
            "rows": len(emb_rows), "dim": dim, "dtype": "float32", "normalised": False,
            "model": "PRTReid (prtreid-soccernet-baseline) via sn-gamestate",
        }  # fmt: skip
    if args.provenance is not None:
        receipt["provenance"] = json.loads(args.provenance.read_text(encoding="utf-8"))
    (args.out / "plugin-receipt.json").write_bytes(
        (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    )
    print(json.dumps({"counts": receipt["counts"], "camera_check": check}, indent=2))

    if args.no_keypoints:
        return 0
    kp_lines, kp_facts = keypoint_lines(images, n_frames, width, height, cams, args.keypoints_centred)
    kp_path = args.out / "keypoints.jsonl"
    kp_path.write_bytes(("\n".join(kp_lines) + "\n").encode("utf-8"))
    kp_receipt = {
        "plugin": "plugin-sngs-export",
        "version": "local-rebuild-2026-11-14",
        "license": args.keypoints_license,
        "restrictions": [t for t in args.keypoints_restrictions.split(",") if t],
        "outputs": [{"name": kp_path.name, "sha256": sha256(kp_path),
                     "bytes": kp_path.stat().st_size}],  # fmt: skip
        "video_frames": n_frames,
        "video_sha256": sha256(args.video),
        "frame_size": [width, height],
        "format": "one line per canonical frame: {frame, points: [[u, v, X, Y, Z, score, "
        "keypoint_id], ...]} (gbf.calib_refine.load_keypoints reads the first six). u, v: "
        "pixels of the export's grid as NBJW counts it (the grid of calib.jsonl's K, principal "
        "point at W/2, H/2), moved from the corner to the centre of the heatmap cell (+1 pixel "
        "of NBJW's 960x540 input). The shift assumes NBJW's training annotations count pixel "
        "edges from 0, as its principal point at W/2, H/2 suggests; under a pixel-centre "
        "convention the decode bias would be about half as large. "
        + ("The plugin run moved them before its own camera fit (run_tracklab_leadcfg.py), so "
           "the plugin cameras were fitted on the same points; this export did not move them "
           "again. " if args.keypoints_centred else
           "It cannot be checked against the plugin cameras, which were fitted on the unshifted "
           "points. ")
        + "X, Y, Z: SoccerNet world "
        "frame in metres (origin at the "
        "centre spot, far touchline at y = -34, z down), NBJW's model points, crossbar keypoints "
        "12, 15, 16, 19 at z = -2.44. score: NBJW's heatmap peak; 1.0 for a keypoint built from "
        "two pitch lines.",
        "counts": kp_facts,
        "note": "Keypoints of sn-gamestate's NBJW pitch module (GPL-2.0-only code, weights "
        "NOASSERTION), stored by the run in its tracker state and exported by this local "
        "rebuild; not the project's original plugin-sngs.",
    }
    (args.out / "keypoints-receipt.json").write_bytes(
        (json.dumps(kp_receipt, indent=2) + "\n").encode("utf-8")
    )
    print(json.dumps({"keypoints": kp_facts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
