#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from datetime import datetime

BASE = Path("/mnt/data_nvme1/minghao.fu/logs")
OUT = Path("/home/minghao.fu/workspace/minimal_wm/experiments/ablation_alignment/report/ablation_alignment.md")
WANDB_DIR = Path("/home/minghao.fu/workspace/minimal_wm/wandb")

ENV_DIRS = {
    "lift": "outputs_robomimic_lift",
    "pusht": "outputs_pusht",
}

DESCS = {
    ("lift", True): "align_true_lift",
    ("lift", False): "align_false_lift",
    ("pusht", True): "align_true_pusht",
    ("pusht", False): "align_false_pusht",
}

METRIC_KEYS = [
    "train_loss",
    "val_loss",
]

VAL_IMG_PREFIX = "val_img_"
ROLL_PREFIX = "z_"


def _find_latest_run(env_name, desc):
    base_dir = BASE / ENV_DIRS[env_name]
    if not base_dir.exists():
        return None
    # Search all dated folders, newest first
    candidates = []
    for date_dir in sorted(base_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            hydra_path = run_dir / "hydra.yaml"
            if not hydra_path.exists():
                continue
            try:
                text = hydra_path.read_text()
            except Exception:
                continue
            if f"description: {desc}" in text or f"description: '{desc}'" in text:
                candidates.append(run_dir)
    return candidates[0] if candidates else None


def _parse_log_metrics(log_path):
    if not log_path.exists():
        return {}
    last_train = None
    last_val = None
    # Match: Epoch X  Training loss: A  Validation loss: B
    pattern = re.compile(r"Epoch\s+\d+\s+Training loss:\s+([0-9.]+)\s+Validation loss:\s+([0-9.]+)")
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            last_train = float(m.group(1))
            last_val = float(m.group(2))
    metrics = {}
    if last_train is not None:
        metrics["train_loss"] = last_train
    if last_val is not None:
        metrics["val_loss"] = last_val
    return metrics


def _find_wandb_summary(run_id):
    if not run_id:
        return None
    for run_dir in WANDB_DIR.glob(f"run-*-{run_id}"):
        summary = run_dir / "files" / "wandb-summary.json"
        if summary.exists():
            return summary
    return None


def _load_wandb_summary(summary_path):
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return {}
    metrics = {}
    for k, v in data.items():
        if k in METRIC_KEYS or k.startswith(VAL_IMG_PREFIX) or k.startswith(ROLL_PREFIX):
            # filter non-scalar
            if isinstance(v, (int, float)):
                metrics[k] = v
    return metrics


def _find_key_images(run_dir):
    # Prefer threeway comparisons if exist
    recon_dir = run_dir / "reconstructions"
    threeways = []
    if recon_dir.exists():
        threeways = sorted(recon_dir.glob("epoch_*_threeway.png"))
    # Fallback: rollout pngs
    rollout_dir = run_dir / "rollout_plots"
    roll_pngs = []
    if rollout_dir.exists():
        for sub in sorted(rollout_dir.glob("e*_rollout")):
            roll_pngs.extend(sorted(sub.glob("*.png")))
    return {
        "threeways": threeways[:3],
        "rollout_pngs": roll_pngs[:3],
        "rollout_videos": sorted(rollout_dir.rglob("*.mp4"))[:3] if rollout_dir.exists() else [],
    }


def main():
    rows = []
    images = {}
    for (env_name, align), desc in DESCS.items():
        run_dir = _find_latest_run(env_name, desc)
        if run_dir is None:
            rows.append((env_name, align, None, {}))
            continue
        # log metrics
        log_path = run_dir / "train_tcwm.log"
        metrics = _parse_log_metrics(log_path)
        # wandb metrics (if available)
        wandb_id = None
        hydra_path = run_dir / "hydra.yaml"
        if hydra_path.exists():
            txt = hydra_path.read_text()
            m = re.search(r"wandb_run_id:\s*(\w+)", txt)
            if m:
                wandb_id = m.group(1)
        summary_path = _find_wandb_summary(wandb_id)
        if summary_path:
            metrics.update(_load_wandb_summary(summary_path))
        rows.append((env_name, align, run_dir, metrics))
        images[(env_name, align)] = _find_key_images(run_dir)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("# Alignment Ablation Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Summary Table\n\n")
        # collect all metric keys
        metric_keys = set()
        for _, _, _, metrics in rows:
            metric_keys.update(metrics.keys())
        metric_keys = ["train_loss", "val_loss"] + sorted(k for k in metric_keys if k not in {"train_loss", "val_loss"})
        # header
        f.write("| env | alignment | run_dir | " + " | ".join(metric_keys) + " |\n")
        f.write("|---|---|---|" + "|".join(["---"] * len(metric_keys)) + "|\n")
        for env_name, align, run_dir, metrics in rows:
            run_str = str(run_dir) if run_dir else "MISSING"
            vals = []
            for k in metric_keys:
                v = metrics.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write(f"| {env_name} | {align} | {run_str} | " + " | ".join(vals) + " |\n")

        f.write("\n## Key Visualizations\n\n")
        f.write("Three-way comparison images (GT / Pred / Recon) if available, else rollout PNGs.\n\n")
        for env_name, align, run_dir, _ in rows:
            f.write(f"### {env_name} | alignment={align}\n\n")
            if run_dir is None:
                f.write("Run not found.\n\n")
                continue
            imgs = images.get((env_name, align), {})
            if imgs.get("threeways"):
                for p in imgs["threeways"]:
                    f.write(f"- threeway: `{p}`\n")
                f.write("\n")
            elif imgs.get("rollout_pngs"):
                for p in imgs["rollout_pngs"]:
                    f.write(f"- rollout png: `{p}`\n")
                f.write("\n")
            if imgs.get("rollout_videos"):
                for p in imgs["rollout_videos"]:
                    f.write(f"- rollout video: `{p}`\n")
                f.write("\n")

    print(f"Report written to: {OUT}")

if __name__ == "__main__":
    main()
