#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Print obs/action shapes for MMBench .pt tasks."
    )
    parser.add_argument(
        "--pt_root",
        default="/data2/minghao/hf_cache/hub/datasets--nicklashansen--mmbench/snapshots/a59d457df617400d3e45a5158c8deac8a52055b4",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task name (without .pt). If omitted, prints all tasks.",
    )
    args = parser.parse_args()

    pt_root = Path(args.pt_root)
    if args.task:
        pt_paths = [pt_root / f"{args.task}.pt"]
    else:
        pt_paths = sorted(pt_root.glob("*.pt"))

    Image.MAX_IMAGE_PIXELS = None

    for pt_path in pt_paths:
        if not pt_path.exists():
            print(f"missing: {pt_path}")
            continue
        td = torch.load(pt_path, map_location="cpu")
        obs = td.get("obs", None)
        action = td.get("action", None)
        if obs is None:
            print(f"{pt_path.stem}: obs missing")
            continue
        obs_shape = tuple(obs.shape)
        act_shape = tuple(action.shape) if action is not None else None

        # Visual shape from sprite sheets (task-0.png, task-1.png, ...)
        visual_shape = None
        total_frames = 0
        sheet_sizes = []
        idx = 0
        while True:
            sheet_path = pt_root / f"{pt_path.stem}-{idx}.png"
            if not sheet_path.exists():
                break
            img = Image.open(sheet_path)
            w, h = img.size
            sheet_sizes.append((w, h))
            if h > 0 and w % h == 0:
                total_frames += w // h
                visual_shape = (h, h, 3)
            idx += 1

        if visual_shape:
            print(
                f"{pt_path.stem}: obs {obs_shape} action {act_shape} "
                f"visual {visual_shape} frames {total_frames}"
            )
        else:
            print(f"{pt_path.stem}: obs {obs_shape} action {act_shape} visual none")


if __name__ == "__main__":
    main()
