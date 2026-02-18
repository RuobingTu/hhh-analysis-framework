#!/usr/bin/env python3
"""Split large ROOT files into parts of ≤MAX_EVENTS events.

Usage:
    python3 split_large_files.py [--max-events 300000] [--dry-run]

Splits files in parts_SPANET_v9_merged/ that exceed MAX_EVENTS,
creating _part0, _part1, ... files alongside the originals.
Then updates symlinks in inclusive-weights/.
"""
import ROOT
import os
import sys
import time
import argparse

ROOT.gROOT.SetBatch(True)

MAX_EVENTS = 300000
TREE_NAME = "Events"

BASE = "/eos/user/r/rtu/TurbOutputMC2017_v9_with_corr_ak8_option92_2017"
SYMLINK_DIR = os.path.join(BASE, "test_skimm", "inclusive-weights")

# Files known to exceed 300k events
FILES_TO_SPLIT = [
    os.path.join(BASE, "signal", "parts_SPANET_v9_merged",
                 "HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8_tree_tree.root"),
    os.path.join(BASE, "mc", "parts_SPANET_v9_merged",
                 "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tree_tree.root"),
]


def split_file(input_path, max_events, dry_run=False):
    """Split a single ROOT file into parts of ≤max_events."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")

    f = ROOT.TFile.Open(input_path)
    tree = f.Get(TREE_NAME)
    total = tree.GetEntries()
    print(f"  Total events: {total}")

    if total <= max_events:
        print(f"  No split needed (≤{max_events})")
        f.Close()
        return []

    n_parts = (total + max_events - 1) // max_events
    print(f"  Splitting into {n_parts} parts of ≤{max_events} events")
    f.Close()

    base_name = input_path.replace(".root", "")
    output_files = []

    for i in range(n_parts):
        first = i * max_events
        n_this = min(max_events, total - first)
        out_path = f"{base_name}_part{i}.root"
        output_files.append(out_path)

        if dry_run:
            print(f"  [DRY RUN] Part {i}: events [{first}, {first+n_this}) -> {os.path.basename(out_path)}")
            continue

        t0 = time.time()
        print(f"  Part {i}: events [{first}, {first+n_this}) ...", end="", flush=True)

        df = ROOT.RDataFrame(TREE_NAME, input_path)
        df_part = df.Range(first, first + n_this)
        df_part.Snapshot(TREE_NAME, out_path)

        dt = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1048576
        print(f" done ({dt:.1f}s, {size_mb:.0f} MB)")

    return output_files


def update_symlinks(original_path, part_files, dry_run=False):
    """Remove original symlink, add symlinks for each part file."""
    orig_name = os.path.basename(original_path)
    orig_link = os.path.join(SYMLINK_DIR, orig_name)

    if os.path.islink(orig_link):
        if dry_run:
            print(f"  [DRY RUN] Would remove symlink: {orig_name}")
        else:
            os.remove(orig_link)
            print(f"  Removed symlink: {orig_name}")

    for pf in part_files:
        link_name = os.path.basename(pf)
        link_path = os.path.join(SYMLINK_DIR, link_name)
        if dry_run:
            print(f"  [DRY RUN] Would create symlink: {link_name} -> {pf}")
        else:
            os.symlink(pf, link_path)
            print(f"  Created symlink: {link_name} -> {pf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Max events per part: {args.max_events}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")

    for fpath in FILES_TO_SPLIT:
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found, skipping")
            continue

        part_files = split_file(fpath, args.max_events, args.dry_run)
        if part_files:
            update_symlinks(fpath, part_files, args.dry_run)

    print(f"\n{'='*60}")
    print("Done! Symlinks in inclusive-weights/:")
    if not args.dry_run:
        for f in sorted(os.listdir(SYMLINK_DIR)):
            print(f"  {f}")


if __name__ == "__main__":
    main()
