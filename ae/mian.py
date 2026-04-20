# main.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import cwru as cwru  # your cwru.py  :contentReference[oaicite:5]{index=5}
from viz import plot_digital_twin_space
from train import (
    set_seed,
    prepare_abcd_from_cwru,
    train_piae_dt,
    build_dt_distributions,
    generate_E_to_dir,
    eval_generated_quality_mmd_rmse,
)

# --- monkey patch for cwru.py missing Path import (safe no-op if already present)
if not hasattr(cwru, "Path"):
    from pathlib import Path as _Path
    cwru.Path = _Path


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--datadir", type=str, default="D:\study\PIAE-4\Data", help="root dir of CWRU data")
    p.add_argument("--load", type=int, default=0, choices=[0, 1, 2, 3], help="working condition index (rpm) in dataname_dict")
    p.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9", help="label set, default 10 classes")
    p.add_argument("--normal_label", type=int, default=0, help="healthy label = 0 (confirmed)")
    p.add_argument("--per_class", type=int, default=600, help="paper uses 600 samples/class")
    p.add_argument("--window", type=int, default=1024, help="paper window=1024")
    p.add_argument("--stride", type=int, default=512, help="paper stride=512")

    # signal preprocessing (must match your cwru.transformation)
    p.add_argument("--normalization", type=str, default="mean-std", choices=["mean-std", "0-1", "none"])
    p.add_argument("--backbone", type=str, default="CNN1D", choices=["CNN1D", "ResNet1D", "ResNet2D"])
    p.add_argument("--fft", action="store_true")

    # paper training setup
    p.add_argument("--epochs", type=int, default=200)          # paper 200 :contentReference[oaicite:6]{index=6}
    p.add_argument("--batch_size", type=int, default=5)        # paper 5 :contentReference[oaicite:7]{index=7}
    p.add_argument("--lr", type=float, default=5e-4)           # paper 0.0005 :contentReference[oaicite:8]{index=8}
    p.add_argument("--weight_lr_scale", type=float, default=5.0, help="multiplier for loss weight learning rate")
    p.add_argument("--step_size", type=int, default=50)        # StepLR step=50 :contentReference[oaicite:9]{index=9}
    p.add_argument("--gamma", type=float, default=0.5)         # StepLR gamma=0.5 :contentReference[oaicite:10]{index=10}
    p.add_argument("--grad_clip", type=float, default=1.0)     # clip=1.0 :contentReference[oaicite:11]{index=11}
    p.add_argument("--repeats", type=int, default=10)          # 10 runs :contentReference[oaicite:12]{index=12}
    p.add_argument("--adaptive_loss", action="store_true", default=True, help="enable adaptive loss weights")
    p.add_argument("--loss_weights", type=str, default="0.2, 0.1, 0.3", help="init alpha,beta,gamma")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1_epochs", type=int, default=200, help="Stage-1 warmup epochs before Stage-2 begins (paper ~50)")
    p.add_argument("--param_reg_lambda_start", type=float, default=100.0, help="initial reg strength for (c,k)")
    p.add_argument("--param_reg_lambda_end", type=float, default=0.2, help="final reg strength (kept >0 to avoid collapse)")
    p.add_argument("--param_reg_lambda_decay", type=int, default=100, help="epochs to decay regularization")

    # physics / dt-loss
    p.add_argument("--fs", type=float, default=12000.0, help="CWRU sampling freq 48kHz")  # :contentReference[oaicite:13]{index=13}
    p.add_argument("--mass", type=float, default= 0.01)
    p.add_argument("--load_force", type=float, default=0)
    p.add_argument("--delta_mode", type=str, default="normalized", choices=["mil", "mm", "normalized"])

    # imbalance ratios
    p.add_argument("--ratios", type=str, default="100:1,50:1,25:1,10:1,5:1,2:1,1:1",
                   help="paper ratios")  # :contentReference[oaicite:14]{index=14}

    # io
    p.add_argument("--outdir", type=str, default="runs_piaedt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # misc
    p.add_argument("--ckpt", type=str, default="", help="(unused legacy arg)")
    p.add_argument("--only_ratio", type=str, default="", help="run only one ratio e.g. 10:1")

    return p.parse_args()


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    args.labels = [int(x) for x in args.labels.split(",")]
    ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]
    if args.only_ratio:
        ratios = [args.only_ratio]

    summary_all = {}

    for rep in range(args.repeats):
        seed = args.seed + rep
        set_seed(seed)
        print(f"\n========== Repeat {rep+1}/{args.repeats} (seed={seed}) ==========")

        # 1) Build Dataset A/B/C/D from raw CWRU (paper split)
        A, B, C, D, meta = prepare_abcd_from_cwru(args, seed=seed)

        # 2) Train generative model on Dataset C
        ckpt_path = train_piae_dt(args, C, meta, seed=seed)

        # 3) Build digital twin distributions
        dist = build_dt_distributions(args, ckpt_path, C, meta)
        z_by_label = {label: data["z_pool"] for label, data in dist.items()}
        dt_plot_path = Path(args.outdir) / f"load{args.load}" / f"seed{seed}" / "dt_space.png"
        plot_digital_twin_space(z_by_label, dt_plot_path)

        rep_result = {}
        for ratio in ratios:
            ratio_dir = Path(args.outdir) / f"load{args.load}" / f"seed{seed}" / f"ratio_{ratio.replace(':','_')}"
            ratio_dir.mkdir(parents=True, exist_ok=True)

            gen_dir = ratio_dir / "E_generated"
            gen_dir.mkdir(parents=True, exist_ok=True)
            generate_E_to_dir(args, ckpt_path, dist, ratio, gen_dir, meta, seed=seed)

            mmd_mean, rmse_mean = eval_generated_quality_mmd_rmse(
                args, ckpt_path, gen_dir, D, meta, ratio=ratio
            )

            rep_result[ratio] = {
                "MMD": float(mmd_mean),
                "RMSE": float(rmse_mean),
            }
            print(f"[ratio {ratio}]  MMD={mmd_mean:.4f}  RMSE={rmse_mean:.4f}")

        summary_all[f"seed{seed}"] = rep_result

        with open(Path(args.outdir) / f"summary_load{args.load}_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(rep_result, f, indent=2)

    agg = {}
    for ratio in ratios:
        mmds, rmses = [], []
        for rep in range(args.repeats):
            seed = args.seed + rep
            r = summary_all[f"seed{seed}"][ratio]
            mmds.append(r["MMD"]); rmses.append(r["RMSE"])
        agg[ratio] = {
            "MMD_mean": float(np.mean(mmds)),
            "RMSE_mean": float(np.mean(rmses)),
        }

    with open(Path(args.outdir) / f"summary_load{args.load}_ALL.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("\n========== Aggregate Reconstruction Metrics ==========")
    for ratio in ratios:
        a = agg[ratio]
        print(f"[ratio {ratio}] MMD={a['MMD_mean']:.4f}  RMSE={a['RMSE_mean']:.4f}")


if __name__ == "__main__":
    main()
