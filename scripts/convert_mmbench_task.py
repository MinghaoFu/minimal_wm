#!/usr/bin/env python3

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2


def load_instruction(tasks_json, task_name):
    tasks = json.loads(Path(tasks_json).read_text())
    if task_name not in tasks:
        raise KeyError(f"Task {task_name} not found in {tasks_json}")
    return tasks[task_name]["instruction"]


def compute_nonzero_mask(array_2d):
    if array_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array_2d.ndim}D")
    return np.any(array_2d != 0, axis=0)


def split_episodes(episode_ids):
    lengths = []
    start = 0
    for i in range(1, len(episode_ids)):
        if episode_ids[i] != episode_ids[i - 1]:
            lengths.append(i - start)
            start = i
    lengths.append(len(episode_ids) - start)
    return lengths


def drop_first_step_per_episode(obs, actions, episode_ids):
    keep = np.ones(len(episode_ids), dtype=bool)
    keep[0] = False
    keep[1:] = episode_ids[1:] == episode_ids[:-1]
    return obs[keep], actions[keep], episode_ids[keep]


def build_subset_arrays(actions, states_np, lengths, episode_indices):
    max_len = max(lengths[i] for i in episode_indices)
    num_eps = len(episode_indices)
    state_dim = states_np.shape[1]
    action_dim = actions.shape[1]

    states = torch.zeros((num_eps, max_len, state_dim), dtype=torch.float32)
    proprios = torch.zeros((num_eps, max_len, state_dim), dtype=torch.float32)
    abs_actions = torch.zeros((num_eps, max_len, action_dim), dtype=torch.float32)
    rel_actions = torch.zeros((num_eps, max_len, action_dim), dtype=torch.float32)
    velocities = torch.zeros((num_eps, max_len, 0), dtype=torch.float32)

    seq_lengths = []
    cursor = 0
    ep_map = {ep_idx: i for i, ep_idx in enumerate(episode_indices)}
    for ep_idx, length in enumerate(lengths):
        if ep_idx in ep_map:
            out_idx = ep_map[ep_idx]
            states[out_idx, :length] = torch.from_numpy(
                states_np[cursor : cursor + length]
            )
            proprios[out_idx, :length] = torch.from_numpy(
                states_np[cursor : cursor + length]
            )
            abs_actions[out_idx, :length] = torch.from_numpy(
                actions[cursor : cursor + length]
            )
            rel_actions[out_idx, :length] = torch.from_numpy(
                actions[cursor : cursor + length]
            )
            seq_lengths.append(length)
        cursor += length

    return states, proprios, velocities, abs_actions, rel_actions, seq_lengths


def compute_stats(states, proprios, actions, seq_lengths, eps=1e-6):
    all_states = []
    all_proprios = []
    all_actions = []
    for i, length in enumerate(seq_lengths):
        all_states.append(states[i, :length])
        all_proprios.append(proprios[i, :length])
        all_actions.append(actions[i, :length])
    all_states = torch.cat(all_states, dim=0)
    all_proprios = torch.cat(all_proprios, dim=0)
    all_actions = torch.cat(all_actions, dim=0)

    state_mean = all_states.mean(dim=0)
    state_std = torch.clamp(all_states.std(dim=0), min=eps)
    proprio_mean = all_proprios.mean(dim=0)
    proprio_std = torch.clamp(all_proprios.std(dim=0), min=eps)
    action_mean = all_actions.mean(dim=0)
    action_std = torch.clamp(all_actions.std(dim=0), min=eps)

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def load_sprite_sheets(root, task_name, frame_size=224):
    Image.MAX_IMAGE_PIXELS = None
    sheets = []
    idx = 0
    while True:
        p = Path(root) / f"{task_name}-{idx}.png"
        if not p.exists():
            break
        img = Image.open(p)
        width, height = img.size
        if height != frame_size or width % frame_size != 0:
            raise ValueError(f"Unexpected sprite size: {p} {img.size}")
        n_frames = width // frame_size
        sheets.append((img, n_frames))
        idx += 1
    if not sheets:
        raise FileNotFoundError(f"No sprite sheets found for {task_name}")
    return sheets


def get_frame_from_sheets(sheets, frame_idx, frame_size=224):
    remaining = frame_idx
    for img, n_frames in sheets:
        if remaining < n_frames:
            left = remaining * frame_size
            frame = img.crop((left, 0, left + frame_size, frame_size))
            return np.asarray(frame)
        remaining -= n_frames
    raise IndexError(f"Frame index {frame_idx} out of range")


def write_videos(
    sheets,
    lengths,
    episode_indices,
    out_dir,
    fps=30,
    frame_size=224,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    ep_map = {ep_idx: i for i, ep_idx in enumerate(episode_indices)}
    for ep_idx, length in enumerate(lengths):
        if ep_idx in ep_map:
            out_idx = ep_map[ep_idx]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = out_dir / f"episode_{out_idx:05d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path), fourcc, float(fps), (frame_size, frame_size)
            )
            for t in range(length):
                frame = get_frame_from_sheets(sheets, cursor + t, frame_size)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()
        cursor += length


def save_split(out_dir, split_name, arrays, stats, seq_lengths):
    split_dir = Path(out_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "obses").mkdir(exist_ok=True)

    states, proprios, velocities, abs_actions, rel_actions = arrays
    torch.save(states, split_dir / "states.pth")
    torch.save(proprios, split_dir / "proprios.pth")
    torch.save(velocities, split_dir / "velocities.pth")
    torch.save(abs_actions, split_dir / "abs_actions.pth")
    torch.save(rel_actions, split_dir / "rel_actions.pth")

    with open(split_dir / "seq_lengths.pkl", "wb") as f:
        pickle.dump(seq_lengths, f)

    torch.save(stats["state_mean"], split_dir / "state_mean.pth")
    torch.save(stats["state_std"], split_dir / "state_std.pth")
    torch.save(stats["proprio_mean"], split_dir / "proprio_mean.pth")
    torch.save(stats["proprio_std"], split_dir / "proprio_std.pth")
    torch.save(stats["action_mean"], split_dir / "action_mean.pth")
    torch.save(stats["action_std"], split_dir / "action_std.pth")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a single MMBench task to robomimic-style format."
    )
    parser.add_argument("--task_name", default=None)
    parser.add_argument(
        "--pt_root",
        default="/data2/minghao/hf_cache/hub/datasets--nicklashansen--mmbench/snapshots/a59d457df617400d3e45a5158c8deac8a52055b4",
    )
    parser.add_argument(
        "--tasks_json", default="/data2/minghao/newt/tasks.json"
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--out_root",
        default="/data2/minghao/data/mmbench",
        help="Root directory to place <task_name> folder when --out_dir not set.",
    )
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all tasks found in --pt_root",
    )
    args = parser.parse_args()

    pt_root = Path(args.pt_root)
    if args.all:
        task_names = sorted([p.stem for p in pt_root.glob("*.pt")])
    elif args.task_name:
        task_names = [args.task_name]
    else:
        raise ValueError("Provide --task_name or use --all")

    for task_name in task_names:
        pt_path = pt_root / f"{task_name}.pt"
        if not pt_path.exists():
            print(f"missing: {pt_path}")
            continue

        td = torch.load(pt_path, map_location="cpu")
        obs = td["obs"].cpu().numpy().astype(np.float32)
        actions = td["action"].cpu().numpy().astype(np.float32)
        episode_ids = td["episode"].cpu().numpy()

        obs, actions, episode_ids = drop_first_step_per_episode(
            obs, actions, episode_ids
        )

        if args.out_dir and not args.all:
            out_root = Path(args.out_dir)
        else:
            out_root = Path(args.out_root) / task_name
        out_root.mkdir(parents=True, exist_ok=True)

        try:
            instruction = load_instruction(args.tasks_json, task_name)
            (out_root / "task_instruction.txt").write_text(instruction)
        except KeyError:
            print(f"warning: no instruction for {task_name}")

        lengths = split_episodes(episode_ids)
        num_eps = len(lengths)
        split_idx = int(num_eps * args.split_ratio)
        train_eps = list(range(split_idx))
        val_eps = list(range(split_idx, num_eps))

        # Remove padded zeros to recover per-task dims.
        obs_mask = compute_nonzero_mask(obs)
        act_mask = compute_nonzero_mask(actions)
        obs = obs[:, obs_mask]
        actions = actions[:, act_mask]
        states_np = obs

        train_arrays = build_subset_arrays(actions, states_np, lengths, train_eps)
        val_arrays = build_subset_arrays(actions, states_np, lengths, val_eps)

        train_stats = compute_stats(
            train_arrays[0], train_arrays[1], train_arrays[3], train_arrays[5]
        )
        val_stats = compute_stats(
            val_arrays[0], val_arrays[1], val_arrays[3], val_arrays[5]
        )

        save_split(
            out_root, "train", train_arrays[:5], train_stats, train_arrays[5]
        )
        save_split(out_root, "val", val_arrays[:5], val_stats, val_arrays[5])

        sheets = load_sprite_sheets(args.pt_root, task_name)
        write_videos(
            sheets, lengths, train_eps, out_root / "train" / "obses", args.fps
        )
        write_videos(
            sheets, lengths, val_eps, out_root / "val" / "obses", args.fps
        )

        print(f"Converted {task_name} -> {out_root}")


if __name__ == "__main__":
    main()
