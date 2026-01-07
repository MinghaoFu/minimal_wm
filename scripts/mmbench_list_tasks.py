#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="List all MMBench task names from .pt files."
    )
    parser.add_argument(
        "--pt_root",
        default="/data2/minghao/hf_cache/hub/datasets--nicklashansen--mmbench/snapshots/a59d457df617400d3e45a5158c8deac8a52055b4",
    )
    args = parser.parse_args()

    pt_root = Path(args.pt_root)
    tasks = sorted([p.stem for p in pt_root.glob("*.pt")])
    for name in tasks:
        pt_path = pt_root / f"{name}.pt"
        td = torch.load(pt_path, map_location="cpu")
        obs = td.get("obs")
        action = td.get("action")
        obs_dim = obs.shape[1] if obs is not None else None
        act_dim = action.shape[1] if action is not None else None

        if obs is not None:
            obs_np = obs.cpu().numpy()
            obs_mask = np.any(obs_np != 0, axis=0)
            obs_dim = int(obs_mask.sum())
        if action is not None:
            act_np = action.cpu().numpy()
            act_mask = np.any(act_np != 0, axis=0)
            act_dim = int(act_mask.sum())

        print(f"{name}\tobs_dim={obs_dim}\taction_dim={act_dim}")


if __name__ == "__main__":
    main()
