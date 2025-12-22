#!/usr/bin/env python3

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch


def split_episodes(terminals):
    terminals = np.asarray(terminals, dtype=bool)
    end_idxs = np.where(terminals)[0].tolist()
    if not end_idxs or end_idxs[-1] != len(terminals) - 1:
        end_idxs.append(len(terminals) - 1)
    lengths = []
    start = 0
    for end in end_idxs:
        if end < start:
            continue
        lengths.append(end - start + 1)
        start = end + 1
    return lengths


def compute_stats(observations, actions, proprios):
    obs_t = torch.from_numpy(observations)
    act_t = torch.from_numpy(actions)
    prop_t = torch.from_numpy(proprios)
    return {
        "state_mean": obs_t.mean(dim=0),
        "state_std": obs_t.std(dim=0),
        "proprio_mean": prop_t.mean(dim=0),
        "proprio_std": prop_t.std(dim=0),
        "action_mean": act_t.mean(dim=0),
        "action_std": act_t.std(dim=0),
    }


def convert_npz(npz_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "obses").mkdir(exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        observations = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        qpos = data["qpos"].astype(np.float32)
        qvel = data["qvel"].astype(np.float32)
        terminals = data["terminals"].astype(bool)

    lengths = split_episodes(terminals)
    n_eps = len(lengths)
    max_len = max(lengths) if lengths else 0

    state_dim = observations.shape[1]
    action_dim = actions.shape[1]
    proprio = np.concatenate([qpos, qvel], axis=1)

    if proprio.shape[1] != state_dim:
        raise ValueError(
            f"Proprio dim {proprio.shape[1]} != observation dim {state_dim}"
        )

    states = torch.zeros((n_eps, max_len, state_dim), dtype=torch.float32)
    proprios = torch.zeros((n_eps, max_len, state_dim), dtype=torch.float32)
    velocities = torch.zeros((n_eps, max_len, qvel.shape[1]), dtype=torch.float32)
    abs_actions = torch.zeros((n_eps, max_len, action_dim), dtype=torch.float32)
    rel_actions = torch.zeros((n_eps, max_len, action_dim), dtype=torch.float32)

    start = 0
    for i, length in enumerate(lengths):
        end = start + length
        states[i, :length] = torch.from_numpy(observations[start:end])
        proprios[i, :length] = torch.from_numpy(proprio[start:end])
        velocities[i, :length] = torch.from_numpy(qvel[start:end])
        abs_actions[i, :length] = torch.from_numpy(actions[start:end])
        rel_actions[i, :length] = torch.from_numpy(actions[start:end])
        start = end

    torch.save(states, out_dir / "states.pth")
    torch.save(proprios, out_dir / "proprios.pth")
    torch.save(velocities, out_dir / "velocities.pth")
    torch.save(abs_actions, out_dir / "abs_actions.pth")
    torch.save(rel_actions, out_dir / "rel_actions.pth")

    with open(out_dir / "seq_lengths.pkl", "wb") as f:
        pickle.dump(lengths, f)

    stats = compute_stats(observations, actions, proprio)
    torch.save(stats["state_mean"], out_dir / "state_mean.pth")
    torch.save(stats["state_std"], out_dir / "state_std.pth")
    torch.save(stats["proprio_mean"], out_dir / "proprio_mean.pth")
    torch.save(stats["proprio_std"], out_dir / "proprio_std.pth")
    torch.save(stats["action_mean"], out_dir / "action_mean.pth")
    torch.save(stats["action_std"], out_dir / "action_std.pth")

    print(f"Converted {npz_path} -> {out_dir}")
    print(f"Episodes: {n_eps}, max_len: {max_len}")
    print(
        f"state_dim: {state_dim}, action_dim: {action_dim}, "
        f"proprio_dim: {proprio.shape[1]}, vel_dim: {qvel.shape[1]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ogbench antmaze/pointmaze npz to robomimic-style folder."
    )
    parser.add_argument("--train_npz", required=True, help="Path to train npz file.")
    parser.add_argument("--val_npz", required=True, help="Path to val npz file.")
    parser.add_argument("--out_dir", required=True, help="Output directory root.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    convert_npz(args.train_npz, out_dir / "train")
    convert_npz(args.val_npz, out_dir / "val")


if __name__ == "__main__":
    main()
