#!/usr/bin/env python3
"""Extract reviewable video frames with timestamps, quality metrics, and a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    requirements = Path(__file__).with_name("keyframe-requirements.txt").resolve()
    raise SystemExit(
        f"Missing dependency. Install dependencies from {requirements} in the active Python environment. "
        f"Original error: {exc}"
    )


@dataclass
class VideoInfo:
    fps: float
    frames: int
    duration: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Source video")
    p.add_argument("--output-dir", required=True, type=Path, help="Destination directory")
    p.add_argument("--mode", choices=("tail", "timestamps", "uniform", "scene"), default="tail")
    p.add_argument("--timestamps", help="Comma-separated seconds for timestamps mode")
    p.add_argument("--tail-seconds", type=float, default=4.0)
    p.add_argument("--sample-step", type=float, default=0.10)
    p.add_argument("--count", type=int, default=12, help="Frame count for uniform mode")
    p.add_argument("--top-k", type=int, default=5, help="Candidate count for tail mode")
    p.add_argument("--min-gap", type=float, default=0.35, help="Minimum seconds between ranked candidates")
    p.add_argument("--min-luma", type=float, default=24.0)
    p.add_argument("--max-luma", type=float, default=236.0)
    p.add_argument("--max-black-ratio", type=float, default=0.55)
    p.add_argument("--max-white-ratio", type=float, default=0.55)
    p.add_argument("--blur-threshold", type=float, default=35.0)
    p.add_argument("--transition-threshold", type=float, default=42.0)
    p.add_argument("--scene-threshold", type=float, default=30.0)
    p.add_argument("--max-scenes", type=int, default=30)
    p.add_argument("--allow-low-quality", action="store_true")
    p.add_argument("--contact-sheet", action="store_true")
    p.add_argument("--manifest-name", default="manifest.json")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path.resolve()))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frames <= 0:
        cap.release()
        raise SystemExit("Video metadata is invalid or unsupported by this OpenCV build.")
    return cap, VideoInfo(fps, frames, frames / fps, width, height)


def read_frame(cap, info: VideoInfo, t: float):
    safe_t = min(max(float(t), 0.0), max(0.0, info.duration - 1.0 / info.fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, safe_t * 1000.0)
    ok, frame = cap.read()
    return (safe_t, frame) if ok else (safe_t, None)


def small_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 160), interpolation=cv2.INTER_AREA)


def quality_metrics(frame, before=None, after=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 160), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luma = float(gray.mean())
    black_ratio = float((gray < 16).mean())
    white_ratio = float((gray > 245).mean())
    diffs = []
    for neighbor in (before, after):
        if neighbor is not None:
            diffs.append(float(cv2.absdiff(small, small_gray(neighbor)).mean()))
    transition = max(diffs) if diffs else 0.0
    return {
        "sharpness": round(sharpness, 3),
        "meanLuma": round(luma, 3),
        "blackRatio": round(black_ratio, 6),
        "whiteRatio": round(white_ratio, 6),
        "neighborDifference": round(transition, 3),
    }


def warnings_for(metrics, args):
    warnings = []
    if metrics["meanLuma"] < args.min_luma:
        warnings.append("too_dark")
    if metrics["meanLuma"] > args.max_luma:
        warnings.append("too_bright")
    if metrics["blackRatio"] > args.max_black_ratio:
        warnings.append("mostly_black")
    if metrics["whiteRatio"] > args.max_white_ratio:
        warnings.append("mostly_white")
    if metrics["sharpness"] < args.blur_threshold:
        warnings.append("low_sharpness")
    if metrics["neighborDifference"] > args.transition_threshold:
        warnings.append("possible_transition_or_fast_motion")
    return warnings


def score(metrics, t, info: VideoInfo, tail_start=0.0):
    sharp = min(55.0, math.log1p(max(0.0, metrics["sharpness"])) * 8.0)
    exposure = max(0.0, 18.0 - abs(metrics["meanLuma"] - 128.0) / 7.0)
    stability = max(0.0, 22.0 - metrics["neighborDifference"] * 0.55)
    span = max(0.001, info.duration - tail_start)
    recency = max(0.0, min(25.0, 25.0 * (t - tail_start) / span))
    extreme = 25.0 * (metrics["blackRatio"] + metrics["whiteRatio"])
    return round(sharp + exposure + stability + recency - extreme, 3)


def analyze_at(cap, info, t, args, tail_start=0.0):
    actual, frame = read_frame(cap, info, t)
    if frame is None:
        return None
    delta = max(2.0 / info.fps, min(args.sample_step, 0.10))
    _, before = read_frame(cap, info, actual - delta)
    _, after = read_frame(cap, info, actual + delta)
    metrics = quality_metrics(frame, before, after)
    warns = warnings_for(metrics, args)
    return {
        "timeSeconds": round(actual, 6),
        "frameIndex": int(round(actual * info.fps)),
        "quality": metrics,
        "warnings": warns,
        "passesQualityGates": not warns,
        "score": score(metrics, actual, info, tail_start),
        "frame": frame,
    }


def distinct_ranked(candidates, count, min_gap, allow_low):
    pool = [c for c in candidates if c and (c["passesQualityGates"] or allow_low)]
    pool.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    for candidate in pool:
        if all(abs(candidate["timeSeconds"] - old["timeSeconds"]) >= min_gap for old in chosen):
            chosen.append(candidate)
        if len(chosen) >= count:
            break
    return chosen


def select_tail(cap, info, args):
    start = max(0.0, info.duration - max(args.tail_seconds, args.sample_step))
    end = max(0.0, info.duration - 1.0 / info.fps)
    times = np.arange(start, end + args.sample_step * 0.5, args.sample_step)
    candidates = [analyze_at(cap, info, float(t), args, start) for t in times]
    chosen = distinct_ranked(candidates, args.top_k, args.min_gap, args.allow_low_quality)
    return chosen, {"tailStartSeconds": round(start, 6), "candidateCount": len(candidates)}


def parse_times(value):
    if not value:
        raise SystemExit("--timestamps is required for timestamps mode")
    try:
        return [float(x.strip()) for x in value.replace(";", ",").split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --timestamps value: {exc}")


def select_timestamps(cap, info, args):
    requested = parse_times(args.timestamps)
    chosen = [analyze_at(cap, info, t, args) for t in requested]
    return [item for item in chosen if item], {"requestedTimestamps": requested}


def select_uniform(cap, info, args):
    count = max(1, args.count)
    end = max(0.0, info.duration - 1.0 / info.fps)
    times = np.linspace(0.0, end, count)
    chosen = [analyze_at(cap, info, float(t), args) for t in times]
    return [item for item in chosen if item], {"requestedCount": count}


def select_scene(cap, info, args):
    scan_step = max(0.08, 3.0 / info.fps)
    times = np.arange(0.0, max(0.0, info.duration - 1.0 / info.fps), scan_step)
    samples = []
    for t in times:
        actual, frame = read_frame(cap, info, float(t))
        if frame is not None:
            samples.append((actual, small_gray(frame)))
    diffs = [
        (samples[i][0], float(cv2.absdiff(samples[i][1], samples[i - 1][1]).mean()))
        for i in range(1, len(samples))
    ]
    dynamic = float(np.percentile([d for _, d in diffs], 97)) if diffs else args.scene_threshold
    threshold = max(args.scene_threshold, dynamic)
    cuts = []
    for t, difference in sorted(diffs, key=lambda x: x[1], reverse=True):
        if difference >= threshold and all(abs(t - old) >= 0.45 for old in cuts):
            cuts.append(t)
    cuts = sorted(cuts)[: max(0, args.max_scenes - 1)]
    bounds = [0.0] + cuts + [info.duration]
    chosen = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start < 0.20:
            continue
        margin = min(0.25, (end - start) * 0.20)
        latest = max(start + margin, end - margin - 1.0 / info.fps)
        trials = np.linspace(start + margin, latest, 5)
        candidates = [analyze_at(cap, info, float(t), args) for t in trials]
        ranked = distinct_ranked(candidates, 1, 0.0, args.allow_low_quality)
        if ranked:
            ranked[0]["sceneStartSeconds"] = round(start, 6)
            ranked[0]["sceneEndSeconds"] = round(end, 6)
            chosen.extend(ranked)
    return chosen, {"detectedCutsSeconds": [round(x, 6) for x in cuts], "cutThreshold": round(threshold, 3)}


def write_png(path: Path, frame):
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Could not encode frame: {path}")
    path.write_bytes(encoded.tobytes())


def make_contact_sheet(items, path: Path):
    if not items:
        return
    font = ImageFont.load_default()
    thumbs = []
    for item in items:
        rgb = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        thumb_width = 300
        thumb_height = max(1, round(image.height * thumb_width / image.width))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((thumb_width, thumb_height), resampling)
        panel = Image.new("RGB", (thumb_width, thumb_height + 34), "black")
        panel.paste(image, (0, 34))
        label = f'{item["timeSeconds"]:.3f}s  score={item["score"]:.1f}'
        if item["warnings"]:
            label += "  " + ",".join(item["warnings"])
        ImageDraw.Draw(panel).text((7, 10), label, fill="white", font=font)
        thumbs.append(panel)
    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumbs[0].width, rows * thumbs[0].height), (24, 24, 24))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % cols) * image.width, (index // cols) * image.height))
    sheet.save(path, quality=94)


def main():
    args = parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input video not found: {source}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(source)
    try:
        if args.mode == "tail":
            selected, mode_details = select_tail(cap, info, args)
        elif args.mode == "timestamps":
            selected, mode_details = select_timestamps(cap, info, args)
        elif args.mode == "uniform":
            selected, mode_details = select_uniform(cap, info, args)
        else:
            selected, mode_details = select_scene(cap, info, args)
    finally:
        cap.release()
    if not selected:
        raise SystemExit("No frame passed the requested selection. Review the source or rerun with --allow-low-quality.")
    manifest_frames = []
    for rank, item in enumerate(selected, 1):
        name = f'kf_{rank:02d}_{item["timeSeconds"]:08.3f}s.png'
        write_png(output / name, item["frame"])
        clean = {key: value for key, value in item.items() if key != "frame"}
        clean["file"] = name
        clean["machineRank"] = rank
        manifest_frames.append(clean)
    if args.contact_sheet:
        make_contact_sheet(selected, output / "contact_sheet.jpg")
    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "sizeBytes": source.stat().st_size,
        },
        "video": {
            "fps": info.fps,
            "frameCount": info.frames,
            "durationSeconds": info.duration,
            "width": info.width,
            "height": info.height,
        },
        "selection": {
            "mode": args.mode,
            "details": mode_details,
            "qualityGates": {
                "minLuma": args.min_luma,
                "maxLuma": args.max_luma,
                "maxBlackRatio": args.max_black_ratio,
                "maxWhiteRatio": args.max_white_ratio,
                "blurThreshold": args.blur_threshold,
                "transitionThreshold": args.transition_threshold,
                "allowLowQuality": args.allow_low_quality,
            },
            "manualReviewRequired": True,
        },
        "frames": manifest_frames,
    }
    manifest_path = output / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "outputDir": str(output),
        "manifest": str(manifest_path),
        "contactSheet": str(output / "contact_sheet.jpg") if args.contact_sheet else None,
        "selected": [
            {"file": item["file"], "timeSeconds": item["timeSeconds"], "warnings": item["warnings"]}
            for item in manifest_frames
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
