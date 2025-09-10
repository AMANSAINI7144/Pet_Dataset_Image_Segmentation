#!/usr/bin/env python3
"""
export_tb.py

Scan runs/* for TensorBoard event files and export:
 - runs/<run>/logs/metrics.csv    (scalars)
 - runs/<run>/logs/images/...     (optional, if --extract-images)

Usage:
  conda activate unet_pet
  python export_tb.py --logdir runs
  python export_tb.py --logdir runs --extract-images
"""
import os
import glob
import csv
import argparse
import io
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:
    raise SystemExit("Please install tensorboard in your environment (pip install tensorboard). Error: " + str(e))

try:
    from PIL import Image
except Exception:
    Image = None

def collect_event_files(run_path):
    # recursively find event files under run_path
    files = glob.glob(os.path.join(run_path, "**", "events.out.tfevents.*"), recursive=True)
    return sorted(files)

def load_scalars_from_file(evfile):
    # load scalars from one event file
    # keep scalars only to reduce memory
    ea = event_accumulator.EventAccumulator(evfile,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.GRAPH: 0
        })
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(int(ev.step), float(ev.value)) for ev in events]
    return data

def load_images_from_file(evfile):
    # load images from one event file (if PIL available)
    ea = event_accumulator.EventAccumulator(evfile,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.GRAPH: 0
        })
    ea.Reload()
    tags = ea.Tags().get("images", [])
    imgs = {}
    for tag in tags:
        imgs[tag] = ea.Images(tag)  # list of Event with encoded_image_string
    return imgs

def merge_tag_events(all_tag_lists):
    # all_tag_lists: list of [(step, value), ...] from multiple files
    merged = {}
    for lst in all_tag_lists:
        for step, val in lst:
            merged[step] = val  # if same step appears multiple times last wins
    return merged  # dict: step -> value

def export_run_scalars(run_path, ev_files, out_csv_path):
    # ev_files: list of event files for this run
    print(f"  Found {len(ev_files)} event file(s) in {run_path}")
    # collect per-tag lists across files
    per_tag_all = defaultdict(list)  # tag -> list of (step, value)
    for ef in ev_files:
        try:
            data = load_scalars_from_file(ef)
        except Exception as e:
            print(f"    WARNING: failed to read scalars from {ef}: {e}")
            continue
        for tag, lst in data.items():
            per_tag_all[tag].append(lst)

    if not per_tag_all:
        print("  No scalar tags found.")
        return False

    # merge and build step set
    per_tag_merged = {}
    all_steps = set()
    for tag, lists in per_tag_all.items():
        merged = merge_tag_events(lists)
        per_tag_merged[tag] = merged
        all_steps.update(merged.keys())

    if not all_steps:
        print("  No steps found for tags.")
        return False

    steps_sorted = sorted(all_steps)
    tags_sorted = sorted(per_tag_merged.keys())

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + tags_sorted)
        for step in steps_sorted:
            row = [step]
            for tag in tags_sorted:
                v = per_tag_merged[tag].get(step, "")
                row.append(v)
            writer.writerow(row)
    print(f"  Wrote scalars CSV: {out_csv_path} (steps: {len(steps_sorted)}, tags: {len(tags_sorted)})")
    return True

def export_run_images(run_path, ev_files, out_images_dir):
    if Image is None:
        print("  PIL not installed; skipping image export.")
        return False
    os.makedirs(out_images_dir, exist_ok=True)
    exported = 0
    for ef in ev_files:
        try:
            ea = event_accumulator.EventAccumulator(ef)
            ea.Reload()
        except Exception as e:
            print(f"    WARNING: failed to read images from {ef}: {e}")
            continue
        tags = ea.Tags().get("images", [])
        for tag in tags:
            imgs = ea.Images(tag)
            for i, ev in enumerate(imgs):
                b = ev.encoded_image_string
                try:
                    im = Image.open(io.BytesIO(b))
                except Exception as e:
                    print(f"      Failed to decode image for tag {tag} step {ev.step}: {e}")
                    continue
                safe_tag = tag.replace("/", "_").replace(" ", "_")
                out_path = os.path.join(out_images_dir, f"{safe_tag}_step{ev.step}_{i}.png")
                try:
                    im.save(out_path)
                    exported += 1
                except Exception as e:
                    print(f"      Failed to save image to {out_path}: {e}")
    print(f"  Exported {exported} images to {out_images_dir}")
    return exported > 0

def main(root_runs, extract_images):
    # find candidate run directories: immediate children of root_runs that contain run_ or events
    run_dirs = []
    # first, if there are subfolders like runs/run_YYYY..., use those
    for entry in sorted(glob.glob(os.path.join(root_runs, "*"))):
        if os.path.isdir(entry):
            run_dirs.append(entry)
    # if no runs found, try scanning for any folder containing event files
    if not run_dirs:
        for d in glob.glob(os.path.join(root_runs, "**/"), recursive=True):
            if glob.glob(os.path.join(d, "events.out.tfevents.*")):
                run_dirs.append(d)

    if not run_dirs:
        print("No run directories found under", root_runs)
        return

    print(f"Found {len(run_dirs)} run directories under {root_runs}")

    for rd in run_dirs:
        # find event files under this run dir
        ev_files = collect_event_files(rd)
        if not ev_files:
            # maybe nested other/
            # try to find events recursively inside
            ev_files = glob.glob(os.path.join(rd, "**", "events.out.tfevents.*"), recursive=True)
            ev_files = sorted(ev_files)
        if not ev_files:
            # nothing for this run
            continue

        # prepare output paths
        logs_dir = os.path.join(rd, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        metrics_csv = os.path.join(logs_dir, "metrics.csv")

        print(f"Processing run: {rd}")
        try:
            ok = export_run_scalars(rd, ev_files, metrics_csv)
            if not ok:
                print("  No scalars exported for run", rd)
        except Exception as e:
            print("  ERROR exporting scalars for run", rd, ":", e)

        if extract_images:
            images_out = os.path.join(logs_dir, "images")
            try:
                export_run_images(rd, ev_files, images_out)
            except Exception as e:
                print("  ERROR exporting images for run", rd, ":", e)

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export TensorBoard event scalars/images to CSV/PNG per run.")
    p.add_argument("--logdir", default="runs", help="Top-level runs directory (default: runs)")
    p.add_argument("--extract-images", action="store_true", help="Also extract image summaries to PNG")
    args = p.parse_args()
    main(args.logdir, args.extract_images)
