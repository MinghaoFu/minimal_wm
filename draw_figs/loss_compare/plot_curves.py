import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_epoch_aligned_interp(
    items,
    epochs=None,                # 如果 None 会尝试从 run.config 或 run.summary 推断，否则用这个值（例如 100）
    target_points=100,          # 把每条曲线插值到相同的横坐标点数（通常跟 epoch 数相同）
    x_key=None,                 # None -> 用 _step
    run_match="exact",
    samples=200000,
    smooth=None,                # None 默认不平滑；传 int 会做 rolling
    mode="line",                # "line" 或 "scatter"（若想看原始点可用 scatter）
    marker_size=6,
    title=None,
    max_epochs=None,
):
    api = wandb.Api()
    plt.figure(figsize=(6, 4), dpi=150)

    # 共同横坐标：0 ... epochs-1
    # epochs 可能因 run 不同而在各自内推断；但插值需要一个共同的目标 x
    # 如果用户没有传 epochs 参数，我们会尝试从第一个 run 的 config / summary 推断并作为全局值，若失败则默认 100
    global_epochs = epochs

    # First pass: optionally infer global_epochs if not provided
    if global_epochs is None:
        # try to infer from first available run.config or run.summary
        for it in items:
            try:
                proj_runs = api.runs(it["path"])
                matched = [r for r in proj_runs if (r.name or "") == it["run"]] if run_match=="exact" else [r for r in proj_runs if it["run"] in (r.name or "")]
                if matched:
                    r = matched[0]
                    # common places people store epochs
                    cfg = getattr(r, "config", {}) or {}
                    s = getattr(r, "summary", {}) or {}
                    if "epochs" in cfg:
                        global_epochs = int(cfg["epochs"])
                        break
                    if "num_epochs" in cfg:
                        global_epochs = int(cfg["num_epochs"])
                        break
                    if "epochs" in s:
                        global_epochs = int(s["epochs"])
                        break
            except Exception:
                continue
        if global_epochs is None:
            global_epochs = 100  # fallback default
            print(f"[INFO] Unable to infer epochs from runs. Using default epochs={global_epochs}")

    if max_epochs is None:
        max_epochs = global_epochs

    x_common = np.linspace(0, max_epochs - 1, max_epochs)

    for it in items:
        path = it["path"]
        run_name = it["run"]
        metric = it["metric"]
        label = it.get("label", f"{path}/{run_name}:{metric}")

        # 1) get runs
        proj_runs = api.runs(path)
        if run_match == "exact":
            matched = [r for r in proj_runs if r.name == run_name]
        else:
            matched = [r for r in proj_runs if run_name in (r.name or "")]

        if not matched:
            print(f"[WARN] run not found: {path} / {run_name}")
            continue
        if len(matched) > 1:
            print(f"[WARN] multiple runs matched ({len(matched)}). use the first one: {path}/{run_name}")
        run = matched[0]

        x = x_key or "_step"

        hist = run.history(keys=[x, metric], samples=samples)
        if hist is None or len(hist) == 0:
            print(f"[WARN] empty history: {label}")
            continue

        df = hist.dropna(subset=[x, metric]).sort_values(by=x)
        if df.empty:
            print(f"[WARN] no valid points after dropna: {label}")
            continue

        # get max step (注意：_step 可能不是连续 0..N，但取 max 做归一化是合理的近似)
        max_step = float(df[x].max())
        if max_step <= 0:
            print(f"[WARN] max step <= 0 for {label}, skipping")
            continue

        # try to find run-specific epoch count (prefer config if exists)
        run_epochs = None
        try:
            cfg = getattr(run, "config", {}) or {}
            s = getattr(run, "summary", {}) or {}
            if "epochs" in cfg:
                run_epochs = int(cfg["epochs"])
            elif "num_epochs" in cfg:
                run_epochs = int(cfg["num_epochs"])
            elif "epochs" in s:
                run_epochs = int(s["epochs"])
        except Exception:
            run_epochs = None

        # If run-specific epoch exists, use it to scale; otherwise use global_epochs
        use_epochs = run_epochs if run_epochs is not None else global_epochs

        # map step -> fractional epoch in [0, use_epochs-1]
        epoch_float = (df[x].astype(float) / max_step) * (use_epochs - 1)

        y = df[metric].values.astype(float)

        # optional smoothing (rolling on original df) - note this changes y before interp
        if smooth is not None and isinstance(smooth, int) and smooth > 1:
            # do rolling on pandas Series indexed by epoch_float order
            srs = pd.Series(y).rolling(window=smooth, min_periods=1).mean().values
            y = srs

        # Interpolate to common x axis.
        # Need monotonic increasing x for interp; epoch_float should be non-decreasing because df sorted by step.
        x_src = np.array(epoch_float)
        # To avoid issues with duplicated x_src, we will use np.unique to keep last value for duplicates:
        xu, iu = np.unique(x_src, return_index=True)
        yu = y[iu]

        # If unique points less than 2, skip
        if len(xu) < 2:
            print(f"[WARN] not enough unique points to interpolate for {label} (only {len(xu)}). Skipping.")
            continue

        # Now interpolate onto the common grid 0..(global_epochs-1)
        # Note: we interpolate onto x_common scaled to use_epochs range if run_epochs differs,
        # so first compute x_common_for_run mapping
        x_run = x_common * (use_epochs - 1) / (global_epochs - 1)

        # Interpolate
        y_interp = np.interp(x_run, xu, yu)

        plt.plot(x_common, y_interp, label=label)
        print(f"[INFO] plotted {label}: max_step={max_step:.0f}, run_epochs={run_epochs}, used_epochs={use_epochs}")

    plt.xlabel("epoch (aligned)")
    plt.ylabel("metric value")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("aligned_plot.png")
    plt.show()


# ===== 使用示例 =====
items = [
    {"path": "minghao_workaholic/minimal_wm",
     "run": "/mnt/data_nvme1/minghao.fu/logs/outputs_robomimic_lift/2026-01-19/02-56-05_robomimic_lift_f5_h3_p1",
     "metric": "train_z_projected_err_pred", "label": "Ours"},
    {"path": "minghao_workaholic/dino_wm",
     "run": "/workspace/minghao/dino_wm/outputs_robomimic_lift/2026-01-14/07-18-27_robomimic_lift_f5_h3_p1",
     "metric": "train_z_visual_err_pred", "label": "DINO WM"},
]

# 假设它们都是 100 epochs（如果 run.config 有 epochs，会优先使用）
plot_epoch_aligned_interp(
    items,
    epochs=100,          # 全局 epoch 数（若 run 自己有 epochs 会覆盖）
    target_points=100,   # 每条曲线插值为 100 个点
    x_key=None,
    run_match="exact",
    samples=200000,
    smooth=None,
    mode="line",
    title="Epoch-aligned (interpolated) comparison",
    max_epochs=40,
)