"""Train/evaluate WDCNN classifiers using generated samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch.utils.data import DataLoader

from train import (
    set_seed,
    prepare_abcd_from_cwru,
    dict_to_xy,
    Meta,
    XYDataset,
)


def parse_args():
    p = argparse.ArgumentParser("WDCNN diagnosis using generated PIAE-DT samples")

    # dataset / preprocessing
    p.add_argument("--datadir", type=str, default="D:\\study\\PIAE-4\\Data", help="root dir of CWRU data")
    p.add_argument("--load", type=int, default=0, choices=[0, 1, 2, 3], help="working condition index")
    p.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9", help="label set, default 10 classes")
    p.add_argument("--normal_label", type=int, default=0, help="healthy label label id")
    p.add_argument("--per_class", type=int, default=600, help="Paper uses 600 samples per class")
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--normalization", type=str, default="mean-std", choices=["mean-std", "0-1", "none"])
    p.add_argument("--backbone", type=str, default="CNN1D", choices=["CNN1D", "ResNet1D", "ResNet2D"])
    p.add_argument("--fft", action="store_true")

    # imbalance ratios
    p.add_argument("--ratios", type=str, default="100:1,50:1,25:1,10:1,5:1,2:1,1:1")
    p.add_argument("--only_ratio", type=str, default="", help="evaluate only a single ratio e.g. 10:1")

    # seeds / repeats
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument("--gen_outdir", type=str, default="runs_piaedt", help="directory where generation outputs live")
    p.add_argument("--outdir", type=str, default="runs_wdcnn", help="directory to save classifier checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # classifier hyper-parameters
    p.add_argument("--clf_epochs", type=int, default=50)
    p.add_argument("--clf_lr", type=float, default=1e-3)
    p.add_argument("--clf_step_size", type=int, default=20)
    p.add_argument("--clf_gamma", type=float, default=0.5)

    return p.parse_args()


def build_trainset_BCE(B, C, gen_dir: Path, ratio: str, meta: Meta, seed: int):
    parts_X, parts_y = [], []

    for label, arr in B.items():
        parts_X.append(arr)
        parts_y.append(np.full((arr.shape[0],), label, dtype=np.int64))

    for label, arr in C.items():
        parts_X.append(arr)
        parts_y.append(np.full((arr.shape[0],), label, dtype=np.int64))

    if ratio != "100:1":
        for label in meta.label_set:
            if label == meta.normal_label:
                continue
            path = gen_dir / f"{label}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Generated data missing: {path}")
            arr = np.load(path)
            if arr.shape[0] > 0:
                parts_X.append(arr)
                parts_y.append(np.full((arr.shape[0],), label, dtype=np.int64))

    X = np.concatenate(parts_X, axis=0).astype(np.float32)
    y = np.concatenate(parts_y, axis=0).astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


class ConvQuadraticOperation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight_r = Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_g = Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_b = Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias_g, 1.0)
            nn.init.constant_(self.bias_b, 0.0)
        nn.init.constant_(self.weight_g, 0.0)
        nn.init.constant_(self.weight_b, 0.0)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(
            self.weight_r,
            mean=0.0,
            std=np.sqrt(0.25 / (self.weight_r.shape[1] * np.prod(self.weight_r.shape[2:]))) * 8,
        )
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def forward(self, x):
        conv_r = F.conv1d(x, self.weight_r, self.bias_r if self.bias else None, self.stride, self.padding)
        conv_g = F.conv1d(x, self.weight_g, self.bias_g if self.bias else None, self.stride, self.padding)
        conv_b = F.conv1d(torch.pow(x, 2), self.weight_b, self.bias_b if self.bias else None, self.stride, self.padding)
        return conv_r * conv_g + conv_b


class WDCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvQuadraticOperation(1, 32, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 1024)
            flat_dim = self.features(dummy).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.features(x)
        feats = feats.flatten(1)
        return self.classifier(feats)


def train_wdcnn_classifier(args, train_X, train_y, outdir: Path):
    device = torch.device(args.device)
    ds = XYDataset(train_X, train_y)
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    model = WDCNN(num_classes=len(args.labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.clf_lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=args.clf_step_size, gamma=args.clf_gamma)

    best_acc = -1.0
    best_path = str(outdir / "wdcnn_best.pt")
    final_path = str(outdir / "wdcnn_final.pt")

    for epoch in range(1, args.clf_epochs + 1):
        model.train()
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        sch.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                y = y.to(device)
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = 100.0 * correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

    torch.save({"model": model.state_dict(), "epoch": args.clf_epochs}, final_path)
    return best_path


def eval_classifier_accuracy(args, clf_ckpt_path: str, A):
    device = torch.device(args.device)
    model = WDCNN(num_classes=len(args.labels)).to(device)
    ckpt = torch.load(clf_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    A_X, A_y = dict_to_xy(A)
    ds = XYDataset(A_X, A_y)
    dl = DataLoader(ds, batch_size=128, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return 100.0 * correct / max(1, total)


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

        A, B, C, _, meta = prepare_abcd_from_cwru(args, seed=seed)

        rep_result = {}
        for ratio in ratios:
            ratio_gen_dir = Path(args.gen_outdir) / f"load{args.load}" / f"seed{seed}" / f"ratio_{ratio.replace(':','_')}" / "E_generated"
            if not ratio_gen_dir.exists():
                raise FileNotFoundError(f"Missing generated data directory: {ratio_gen_dir}")

            ratio_clf_dir = Path(args.outdir) / f"load{args.load}" / f"seed{seed}" / f"ratio_{ratio.replace(':','_')}"
            ratio_clf_dir.mkdir(parents=True, exist_ok=True)

            train_X, train_y = build_trainset_BCE(B, C, ratio_gen_dir, ratio, meta, seed=seed)
            clf_ckpt = train_wdcnn_classifier(args, train_X, train_y, ratio_clf_dir)
            acc = eval_classifier_accuracy(args, clf_ckpt, A)

            rep_result[ratio] = {"ACC": float(acc)}
            print(f"[ratio {ratio}]  ACC={acc:.2f}%")

        summary_all[f"seed{seed}"] = rep_result
        with open(Path(args.outdir) / f"summary_load{args.load}_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(rep_result, f, indent=2)

    agg = {}
    for ratio in ratios:
        accs = []
        for rep in range(args.repeats):
            seed = args.seed + rep
            accs.append(summary_all[f"seed{seed}"][ratio]["ACC"])
        agg[ratio] = {
            "ACC_mean": float(np.mean(accs)),
            "ACC_std": float(np.std(accs)),
        }

    with open(Path(args.outdir) / f"summary_load{args.load}_ALL.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("\n========== Aggregate ACC (mean±std) ==========")
    for ratio in ratios:
        a = agg[ratio]
        print(f"[ratio {ratio}] ACC={a['ACC_mean']:.2f}±{a['ACC_std']:.2f}%")


if __name__ == "__main__":
    main()
