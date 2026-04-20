from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_xy_signals_from_folder(
    data_folder_path: Union[str, Path],
    sample_length: int = 4096,
    conditions: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load two-channel acceleration segments from BJTU CSV files.

    Parameters
    ----------
    data_folder_path:
        Directory containing BJTU CSV files.
    sample_length:
        Segment length in samples.

    Returns
    -------
    x_segments : np.ndarray
        Array shaped [N, 1, sample_length] with CH17 data (in g).
    y_segments : np.ndarray
        Array shaped [N, 1, sample_length] with CH18 data (in g).
    fshaft_series : np.ndarray
        Array shaped [N] with shaft frequency in Hz extracted from file names.
    """
    folder_path = Path(data_folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_folder_path}")

    csv_files = sorted(folder_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_folder_path}")

    x_segments: list[np.ndarray] = []
    y_segments: list[np.ndarray] = []
    fshaft_list: list[float] = []
    load_list: list[float] = []

    def _extract_channel(df: pd.DataFrame, name: str, index: int) -> Optional[np.ndarray]:
        """Map named channel to its CSV column; fallback to fixed CH18/19 indices."""
        if name in df.columns:
            return df[name].to_numpy(dtype=np.float32, copy=False)
        if index < df.shape[1]:
            return df.iloc[:, index].to_numpy(dtype=np.float32, copy=False)
        return None

    for csv_file in csv_files:
        if conditions:
            name = csv_file.stem
            if not any(cond in name for cond in conditions):
                continue
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        sig_ch18 = _extract_channel(df, "CH18", 17)
        sig_ch19 = _extract_channel(df, "CH19", 18)
        if sig_ch18 is None or sig_ch19 is None:
            continue

        signal_x = sig_ch18
        signal_y = sig_ch19
        min_len = min(signal_x.shape[0], signal_y.shape[0])
        if min_len < sample_length:
            continue

        freq_match = re.search(r"_(\d+(?:\.\d+)?)Hz", csv_file.stem)
        fshaft_hz = float(freq_match.group(1)) if freq_match else 40.0
        load_match = re.search(r"_([+-]?\d+(?:\.\d+)?)kN", csv_file.stem)
        load_kn = float(load_match.group(1)) if load_match else 0.0

        num_segments = min_len // sample_length
        for seg_idx in range(num_segments):
            start = seg_idx * sample_length
            end = start + sample_length
            x_segments.append(signal_x[start:end])
            y_segments.append(signal_y[start:end])
            fshaft_list.append(fshaft_hz)
            load_list.append(load_kn)

    if not x_segments:
        raise RuntimeError(f"No valid samples extracted from {data_folder_path}")

    x_array = np.expand_dims(np.stack(x_segments, axis=0), axis=1)
    y_array = np.expand_dims(np.stack(y_segments, axis=0), axis=1)
    fshaft_array = np.asarray(fshaft_list, dtype=np.float32)
    load_array = np.asarray(load_list, dtype=np.float32)
    return x_array, y_array, fshaft_array, load_array


class BJTUDataset(Dataset):
    """Two-channel BJTU dataset returning left axlebox accelerations and shaft speed."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        sequence_length: int,
        split: str = "train",
        fs: Union[int, float] = 64000,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        conditions: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.fs = float(fs)

        x_segments, y_segments, fshaft_series, load_series = load_xy_signals_from_folder(
            data_dir,
            sample_length=self.sequence_length,
            conditions=conditions,
        )

        num_samples = x_segments.shape[0]
        rng = np.random.default_rng(random_seed)
        indices = np.arange(num_samples)
        rng.shuffle(indices)
        if max_samples is not None and max_samples > 0:
            limit = min(num_samples, int(max_samples))
            indices = indices[:limit]

        split_idx = int(num_samples * train_ratio)
        if split.lower() == "train":
            chosen = indices[: max(split_idx, 1)]
        else:
            chosen = indices[max(split_idx, 1) :]
        if chosen.size == 0:
            chosen = indices

        self.x = torch.from_numpy(x_segments[chosen]).float()
        self.y = torch.from_numpy(y_segments[chosen]).float()
        self.fshaft = torch.from_numpy(fshaft_series[chosen]).float()
        self.load = torch.from_numpy(load_series[chosen]).float()
        self.length = self.x.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.cat([self.x[idx], self.y[idx]], dim=0)  # [2, T]
        label = torch.stack([self.fshaft[idx], self.load[idx]], dim=0)
        return sample, label
