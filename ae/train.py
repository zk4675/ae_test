# train.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cwru  # 确保你使用了包含 align_signal 的新版 cwru.py
from PIAENet import PIAENet
from loss import (
    loss_total,
    loss_rec,
    loss_ddl,
    loss_dt,
    mmd_rbf
)
from viz import plot_reconstruction_comparison, plot_loss_curves


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=num_classes).float()


def dict_to_xy(dict_by_class: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return cwru.stack_xy(dict_by_class)


# -------------------------
# Meta / paper constants
# -------------------------
@dataclass
class Meta:
    num_classes: int = 10
    x_dim: int = 1024
    normal_label: int = 0
    label_set: List[int] = None
    delta_n: torch.Tensor = None
    delta_n_phys: torch.Tensor = None


def build_delta_n(delta_mode: str = "normalized", num_classes: int = 10) -> torch.Tensor:
    mil = torch.tensor([0, 7, 14, 21, 7, 14, 21, 7, 14, 21], dtype=torch.float32)
    if delta_mode == "mil": return mil
    mm = mil * 0.0254
    if delta_mode == "mm": return mm
    return mm / (mm.max().clamp_min(1e-8))


class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def prepare_abcd_from_cwru(args, seed: int) -> Tuple[Dict, Dict, Dict, Dict, Meta]:
    label_set = args.labels
    dataset_by_class = cwru.CWRU(
        datadir=args.datadir,
        load=args.load,
        labels=label_set,
        stride=args.stride,
        normalization=args.normalization if args.normalization != "none" else "mean-std",
        backbone=args.backbone,
        fft=args.fft,
        per_class=args.per_class,
        seed=seed,
    )
    A, B, C, D = cwru.build_ABCD_from_600(dataset_by_class, normal_label=args.normal_label, seed=seed)
    delta_n_user = build_delta_n(args.delta_mode, num_classes=len(label_set))
    delta_n_mm = build_delta_n("mm", num_classes=len(label_set))
    meta = Meta(
        num_classes=len(label_set),
        x_dim=args.window,
        normal_label=args.normal_label,
        label_set=label_set,
        delta_n=delta_n_user,
        delta_n_phys=(delta_n_mm * 1e-3),
    )
    return A, B, C, D, meta


# -------------------------
# Training Logic
# -------------------------
def train_piae_dt(args, C: Dict, meta: Meta, seed: int) -> str:
    device = torch.device(args.device)
    adaptive_loss = bool(getattr(args, "adaptive_loss", False))
    # Zero static load for zero-mean vibration assumption.
    args.load_force = 0.0
    
    # 解析权重初始值
    lw_raw = getattr(args, "loss_weights", [1.0, 0.1, 2.0])
    if isinstance(lw_raw, str):
        lw_vals = [float(x.strip()) for x in lw_raw.split(",") if x.strip()]
    else:
        lw_vals = [float(x) for x in lw_raw]
    while len(lw_vals) < 3: lw_vals.append(1.0)
    loss_weight_vals = lw_vals[:3]

    # 计算全局尺度 (Scale)
    def _compute_scales(acc_np: np.ndarray, dt_phys: float, eps: float = 1e-6):
        a = acc_np.reshape(acc_np.shape[0], -1)
        a0 = float(np.std(a)) + eps
        scale_a = a0
        scale_v = a0 * dt_phys
        scale_s = a0 * (dt_phys ** 2)
        return scale_a, scale_v, scale_s

    dt_phys = 1.0 / float(args.fs)
    C_X, C_y = dict_to_xy(C)
    ds = XYDataset(C_X, C_y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    acc_all = C_X.reshape(C_X.shape[0], -1)
    scale_a, scale_v, scale_s = _compute_scales(acc_all, dt_phys)
    scales = {
        "scale_a": torch.tensor(scale_a, dtype=torch.float32, device=device),
        "scale_v": torch.tensor(scale_v, dtype=torch.float32, device=device),
        "scale_s": torch.tensor(scale_s, dtype=torch.float32, device=device),
    }
    
    mass_M = float(args.mass)
    delta_n_phys = meta.delta_n_phys.to(device)
    dt_train = 1.0

    reg_lambda_start = float(getattr(args, "param_reg_lambda_start", 100.0))
    reg_lambda_end = float(getattr(args, "param_reg_lambda_end", 1.0))
    reg_lambda_decay = int(getattr(args, "param_reg_lambda_decay", max(args.epochs, 1)))

    def get_reg_lambda(epoch_idx: int) -> float:
        if reg_lambda_decay <= 0:
            return reg_lambda_end
        frac = min(max(epoch_idx, 0) / reg_lambda_decay, 1.0)
        return reg_lambda_start + (reg_lambda_end - reg_lambda_start) * frac

    model = PIAENet(x_dim=meta.x_dim, y_dim=meta.num_classes, z_dim=2).to(device)

    # 自适应权重参数
    if adaptive_loss:
        log_alpha = nn.Parameter(torch.tensor(math.log(loss_weight_vals[0]), device=device))
        log_beta = nn.Parameter(torch.tensor(math.log(loss_weight_vals[1]), device=device))
        log_gamma = nn.Parameter(torch.tensor(math.log(loss_weight_vals[2]), device=device))
        loss_weight_params = [log_alpha, log_beta, log_gamma]
    else:
        log_alpha = log_beta = log_gamma = None
        loss_weight_params = []

    # 优化器
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(model.pim.parameters(), lr=args.lr)

    if adaptive_loss:
        weight_lr = args.lr * getattr(args, "weight_lr_scale", 5.0)
        opt_all = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr},
            {"params": loss_weight_params, "lr": weight_lr},
        ])
    else:
        opt_all = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度
    sch_dec = torch.optim.lr_scheduler.StepLR(opt_dec, step_size=args.step_size, gamma=args.gamma)
    sch_enc = torch.optim.lr_scheduler.StepLR(opt_enc, step_size=args.step_size, gamma=args.gamma)
    sch_pi = torch.optim.lr_scheduler.StepLR(opt_pi, step_size=args.step_size, gamma=args.gamma)
    sch_all = torch.optim.lr_scheduler.StepLR(opt_all, step_size=args.step_size, gamma=args.gamma)

    outdir = Path(args.outdir) / f"load{args.load}" / f"seed{seed}" / "piaedt"
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = str(outdir / "piaedt_best.pt")
    final_path = str(outdir / "piaedt_final.pt")
    best_loss = float("inf")

    # ====================================================
    # 核心 Loss 计算函数 (替代原来的 compute_losses_ck_paper)
    # ====================================================
    def compute_loss_terms_manual(out_dict, acc_real_phys, y_onehot, reg_lambda: float):
        acc_dimless = acc_real_phys / scales["scale_a"]

        Lrec = loss_rec(acc_dimless, out_dict["acc_hat"], sigmas=None)

        Lddl, v_hat, _ = loss_ddl(
            s_hat=out_dict["s_hat"],
            a_real=acc_dimless,
            dt=dt_train,
            sigmas=None,
        )

        Ldt = loss_dt(
            a_in=acc_dimless,
            v_derived=v_hat,
            s_hat=out_dict["s_hat"],
            y_onehot=y_onehot,
            delta_n_phys=delta_n_phys,
            mass_M=mass_M,
            scales=scales,
            params_phys=out_dict["params"],
            normal_label=meta.normal_label,
            reg_lambda=reg_lambda,
        )
        return Lrec, Lddl, Ldt

    stage1_epochs = max(0, getattr(args, "stage1_epochs", 0))
    last_log = {}

    # -------- Stage-1 Warmup --------
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        loss_epoch = 0.0
        n_steps = 0

        for acc, y in dl:
            acc = acc.to(device)
            y = y.to(device)
            yoh = one_hot(y, meta.num_classes).to(device)
            
            # 注意：网络输入也要归一化
            acc_in = acc / scales["scale_a"]

            # 1. Train PI
            opt_pi.zero_grad()
            out = model(acc_in, yoh)
            _, Lddl, _ = compute_loss_terms_manual(out, acc, yoh, reg_lambda_start)
            Lddl.backward()
            torch.nn.utils.clip_grad_norm_(model.pim.parameters(), args.grad_clip)
            opt_pi.step()

            # 2. Train Encoder (optimize Ldt)
            opt_enc.zero_grad()
            out = model(acc_in, yoh)
            _, _, Ldt = compute_loss_terms_manual(out, acc, yoh, reg_lambda_start)
            Ldt.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), args.grad_clip)
            opt_enc.step()

            # 3. Train Decoder (optimize Lrec)
            opt_dec.zero_grad()
            out = model(acc_in, yoh)
            Lrec, _, _ = compute_loss_terms_manual(out, acc, yoh, reg_lambda_start)
            Lrec.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_clip)
            opt_dec.step()

            loss_epoch += float(Lddl.item())
            n_steps += 1
            last_log = {
                "Lddl": float(Lddl.item()), 
                "Ldt": float(Ldt.item()), 
                "Lrec": float(Lrec.item())
            }

        sch_pi.step(); sch_enc.step(); sch_dec.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"[PIAE-DT][Stage-1] epoch {epoch:03d} Lddl={last_log['Lddl']:.3e} Ldt={last_log['Ldt']:.3e} Lrec={last_log['Lrec']:.3e}")

    viz_dir = outdir / "viz_curves"
    viz_dir.mkdir(parents=True, exist_ok=True)
    loss_history = {"Ltot": [], "Lddl": [], "Ldt": [], "Lrec": []}

    # -------- Stage-2 Joint Training --------
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_epoch = 0.0
        n_steps = 0
        sum_ddl = sum_dt = sum_rec = 0.0
        reg_lambda_epoch = get_reg_lambda(epoch - 1)

        for acc, y in dl:
            acc = acc.to(device)
            y = y.to(device)
            yoh = one_hot(y, meta.num_classes).to(device)
            acc_in = acc / scales["scale_a"]

            opt_all.zero_grad()
            out = model(acc_in, yoh)
            Lrec, Lddl, Ldt = compute_loss_terms_manual(out, acc, yoh, reg_lambda_epoch)

            if adaptive_loss:
                alpha = torch.exp(log_alpha)
                beta = torch.exp(log_beta)
                gamma = torch.exp(log_gamma)
            else:
                alpha = torch.tensor(loss_weight_vals[0], device=device)
                beta = torch.tensor(loss_weight_vals[1], device=device)
                gamma = torch.tensor(loss_weight_vals[2], device=device)

            Ltot = loss_total(Lddl, Ldt, Lrec, alpha, beta, gamma)
            Ltot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt_all.step()

            loss_epoch += Ltot.item()
            sum_ddl += Lddl.item(); sum_dt += Ldt.item(); sum_rec += Lrec.item()
            n_steps += 1

        sch_all.step()
        
        avg_tot = loss_epoch / n_steps
        avg_ddl = sum_ddl / n_steps
        avg_dt = sum_dt / n_steps
        avg_rec = sum_rec / n_steps

        # Save Best
        if avg_tot < best_loss:
            best_loss = avg_tot
            save_dict = {
                "model": model.state_dict(),
                "meta": meta.__dict__,
                "args": vars(args),
                "best_loss": best_loss,
            }
            torch.save(save_dict, best_path)

        loss_history["Ltot"].append(avg_tot)
        loss_history["Lddl"].append(avg_ddl)
        loss_history["Ldt"].append(avg_dt)
        loss_history["Lrec"].append(avg_rec)

        if epoch % 10 == 0 or epoch == 1:
            a_val = float(alpha.item())
            b_val = float(beta.item())
            g_val = float(gamma.item())
            print(f"[PIAE-DT][Stage-2] epoch {epoch:03d} Ltot={avg_tot:.4f} Lrec={avg_rec:.4f} Lddl={avg_ddl:.4f} Ldt={avg_dt:.4f} "
                  f"alpha={a_val:.2f} beta={b_val:.2f} gamma={g_val:.2f} reg_lambda={reg_lambda_epoch:.2f}")
            
            # Plot check
            plot_reconstruction_comparison(acc, out["acc_hat"], epoch, viz_dir, fs=args.fs)

    # Save Final
    torch.save({"model": model.state_dict(), "meta": meta.__dict__}, final_path)
    return best_path


# --- Distribution & Gen helper ---
def load_piaedt_model(ckpt_path: str, device: torch.device) -> Tuple[PIAENet, Meta]:
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = Meta(**ckpt["meta"])
    model = PIAENet(x_dim=meta.x_dim, y_dim=meta.num_classes, z_dim=2).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, meta

def infer_ck_params(model: PIAENet, X: np.ndarray, y: np.ndarray, device: torch.device, num_classes: int, bs: int = 64):
    ds = XYDataset(X, y)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    params = []
    # 需要 scale_a 来归一化输入
    scale_a = float(np.std(X)) + 1e-6
    
    with torch.no_grad():
        for acc, yy in dl:
            acc = acc.to(device)
            acc_in = acc / scale_a  # 归一化推理
            yoh = one_hot(yy.to(device), num_classes)
            out = model(acc_in, yoh)
            z = torch.stack([out["params"]["c_tilde"], out["params"]["k_tilde"]], dim=-1)
            params.append(z.cpu().numpy())
    return np.concatenate(params, axis=0)

def build_dt_distributions(args, ckpt_path: str, C: Dict, meta: Meta) -> Dict:
    device = torch.device(args.device)
    model, _ = load_piaedt_model(ckpt_path, device)
    dist = {}
    for label, arr in C.items():
        y = np.full((arr.shape[0],), label, dtype=np.int64)
        z = infer_ck_params(model, arr, y, device, meta.num_classes)
        std = np.clip(np.std(z, axis=0), 1e-6, None)
        dist[label] = {"z_pool": z.astype(np.float32), "std": std.astype(np.float32)}
    return dist

def decode_from_params(model, z, y, device, num_classes):
    zt = torch.from_numpy(z).to(device)
    yoh = one_hot(torch.from_numpy(y).to(device), num_classes)
    with torch.no_grad():
        x = model.decoder(zt, yoh)
    return x.cpu().numpy()

def generate_E_to_dir(args, ckpt_path: str, dist: Dict, ratio: str, out_dir: Path, meta: Meta, seed: int):
    # 与原代码逻辑一致，略去 RATIO 表定义
    from cwru import RATIO_TO_GEN_PER_FAULT
    gen_n = RATIO_TO_GEN_PER_FAULT.get(ratio, 0)
    if gen_n == 0: return

    device = torch.device(args.device)
    model, _ = load_piaedt_model(ckpt_path, device)
    
    for label in meta.label_set:
        if label == meta.normal_label: continue
        d = dist[label]
        rng = np.random.default_rng(seed + 1000 + label)
        idx = rng.choice(d["z_pool"].shape[0], size=gen_n, replace=True)
        z_base = d["z_pool"][idx]
        z_noise = rng.normal(0, d["std"] * 0.01, z_base.shape).astype(np.float32)
        z = np.clip(z_base + z_noise, 1e-6, None)
        
        y = np.full((gen_n,), label, dtype=np.int64)
        x_gen = decode_from_params(model, z, y, device, meta.num_classes)
        
        if args.backbone in ("CNN1D", "ResNet1D"):
            x_gen = x_gen[:, np.newaxis, :]
        np.save(out_dir / f"{label}.npy", x_gen)

def eval_generated_quality_mmd_rmse(args, ckpt_path, gen_dir, D, meta, ratio):
    from loss import mmd_rbf
    device = torch.device(args.device)
    mmds, rmses = [], []
    for label, real in D.items():
        if label == meta.normal_label: continue
        try:
            gen = np.load(gen_dir / f"{label}.npy")
        except: continue
        
        if len(gen) == 0: continue
        n = min(len(real), len(gen))
        
        # 统一归一化比较
        g = torch.from_numpy(gen[:n].reshape(n,-1)).to(device)
        r = torch.from_numpy(real[:n].reshape(n,-1)).to(device)
        
        # 使用修复后的 MMD
        with torch.no_grad():
            m = mmd_rbf(g, r).item()
            mse = F.mse_loss(g, r).item()
        mmds.append(m); rmses.append(np.sqrt(mse))
        
    if not mmds: return float("nan"), float("nan")
    return float(np.mean(mmds)), float(np.mean(rmses))
