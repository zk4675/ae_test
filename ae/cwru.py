# cwru.py
from pathlib import Path
import numpy as np
from scipy.io import loadmat

# ==========================================
# 标准 CWRU 12k Drive End 数据字典
# 对应你截图中的文件编号
# ==========================================
dataname_dict = {
    0: [97, 105, 118, 130, 169, 185, 197, 209, 222, 234],
    1: [98, 106, 119, 131, 170, 186, 198, 210, 223, 235],
    2: [99, 107, 120, 132, 171, 187, 199, 211, 224, 236],
    3: [100, 108, 121, 133, 172, 188, 200, 212, 225, 237]
}

# 你的数据中，Key 可能不统一，这里列出常见的 Key
axis_candidates = ["X{code}_DE_time", "DE_time", "X{code}_DE_time.1"]
data_length = 1024

RATIO_TO_GEN_PER_FAULT = {
    "100:1": 0, "50:1": 5, "25:1": 15, "10:1": 45, "5:1": 95, "2:1": 145, "1:1": 495
}

def align_signal(data, target_pos=256):
    """ 
    相位对齐：将信号中绝对值最大的点移到 target_pos。
    这对于让简单的 Decoder 学会画波形至关重要。
    """
    if len(data) == 0: return data
    idx = np.argmax(np.abs(data))
    shift = target_pos - idx
    return np.roll(data, shift)

def transformation(sub_data, fft, normalization, backbone):
    # 1. 强制相位对齐
    sub_data = align_signal(sub_data, target_pos=256)

    # 2. FFT (可选)
    if fft:
        sub_data = np.fft.fft(sub_data)
        sub_data = np.abs(sub_data) / len(sub_data)
        sub_data = sub_data[:int(sub_data.shape[0] / 2)].reshape(-1,)

    # 3. 归一化
    if normalization == "0-1":
        sub_data = (sub_data - sub_data.min()) / (sub_data.max() - sub_data.min() + 1e-12)
    elif normalization == "mean-std":
        sub_data = (sub_data - sub_data.mean()) / (sub_data.std() + 1e-12)

    # 4. 维度调整
    if backbone in ("ResNet1D", "CNN1D"):
        sub_data = sub_data[np.newaxis, :]
    elif backbone == "ResNet2D":
        n = int(np.sqrt(sub_data.shape[0]))
        if fft:
            sub_data = sub_data[:n*n]
        sub_data = np.reshape(sub_data, (n, n))
        sub_data = sub_data[np.newaxis, :]
        sub_data = np.concatenate((sub_data, sub_data, sub_data), axis=0)
    return sub_data


def CWRU(
    datadir, load, labels,
    stride=512, normalization="mean-std", backbone="CNN1D", fft=False,
    per_class=600, seed=42
):
    rng = np.random.default_rng(seed)
    
    # 根目录: .../Data/CWRU
    base_dir = Path(datadir) / "CWRU" 
    
    if not base_dir.exists():
        # 容错：如果用户直接传了 .../Data/CWRU，就不要再拼一层 CWRU
        if Path(datadir).name == "CWRU":
            base_dir = Path(datadir)
        else:
            raise FileNotFoundError(f"Directory not found: {base_dir}")

    dataset = {label: [] for label in labels}

    for label in labels:
        segments = []
        # 获取该工况(Load)下的标准文件编号 (如 97)
        code = dataname_dict[load][label]
        
        # 构造文件名: 97_0.mat (编号_负载.mat)
        filename = f"{code}_{load}.mat"
        
        # 构造完整路径: .../Drive_end_0/97_0.mat
        folder_name = f"Drive_end_{load}"
        mat_path = base_dir / folder_name / filename
        
        if not mat_path.exists():
            # 尝试容错：有些文件名可能没有 _load 后缀 (如 97.mat)
            mat_path_alt = base_dir / folder_name / f"{code}.mat"
            if mat_path_alt.exists():
                mat_path = mat_path_alt
            else:
                print(f"[Warning] File not found: {mat_path}")
                continue

        try:
            mat = loadmat(mat_path)
        except Exception as e:
            print(f"[Error] Corrupt mat file {mat_path}: {e}")
            continue

        # 查找数据 Key (X097_DE_time 或 DE_time)
        data_key = None
        # 优先尝试带编号的 Key (X097_DE_time)
        possible_keys = [f"X{code:03d}_DE_time", f"X{code}_DE_time", "DE_time"]
        
        for k in mat.keys():
            if k in possible_keys or k.endswith("DE_time"):
                data_key = k
                break
        
        if data_key is None:
            print(f"[Skip] No DE_time data in {filename}")
            continue

        mat_data = mat[data_key].reshape(-1,)
        
        # 切片
        length = mat_data.shape[0]
        if length >= data_length:
            max_n = 1 + (length - data_length) // stride
            if max_n > 0:
                remaining = per_class - len(segments)
                n_take = min(max_n, remaining)
                # 随机采样起点
                starts = rng.choice(max_n, size=n_take, replace=False)

                for i in starts:
                    st = int(i) * stride
                    sub = mat_data[st:st + data_length].reshape(-1,)
                    sub = transformation(sub, fft, normalization, backbone)
                    segments.append(sub.astype("float32"))

        # 补全数据 (Bootstrapping)
        if len(segments) < per_class and len(segments) > 0:
            need = per_class - len(segments)
            indices = rng.choice(len(segments), size=need, replace=True)
            segments.extend([segments[idx].copy() for idx in indices])
        
        if segments:
            dataset[label] = np.stack(segments[:per_class], axis=0)
        else:
            print(f"[Error] Class {label} (File {filename}): Failed to extract segments.")

    return dataset


def build_ABCD_from_600(dataset_by_class, normal_label=0, seed=42):
    rng = np.random.default_rng(seed)
    A, B, C, D = {}, {}, {}, {}
    for label, arr in dataset_by_class.items():
        if isinstance(arr, list): # 空数据保护
            if not arr: continue
            arr = np.array(arr)
            
        n_samples = arr.shape[0]
        if n_samples == 0: continue
        
        perm = rng.permutation(n_samples)
        arr = arr[perm]
        
        # 确保 A集 (测试集) 至少有数据，哪怕不足 100
        n_test = min(100, int(n_samples * 0.2)) 
        # 如果数据足够标准 (600)，则 n_test=100
        if n_samples >= 200: n_test = 100
            
        A[label] = arr[:n_test]
        pool = arr[n_test:]
        
        if label == normal_label:
            B[label] = pool
        else:
            # C集 (Few-shot) 取 5 个
            n_c = min(5, pool.shape[0])
            C[label] = pool[:n_c]
            D[label] = pool[n_c:]
            
    return A, B, C, D


def stack_xy(dict_by_class):
    X_list, y_list = [], []
    for label, arr in dict_by_class.items():
        if len(arr) == 0: continue
        X_list.append(arr)
        y_list.append(np.full((arr.shape[0],), label, dtype=np.int64))
    if not X_list: return np.array([]), np.array([])
    X = np.concatenate(X_list, axis=0).astype("float32")
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return X, y


def add_generated_E(C_fault_seed, ratio, generated_dir, seed=42):
    if ratio not in RATIO_TO_GEN_PER_FAULT: return {}
    gen_n = RATIO_TO_GEN_PER_FAULT[ratio]
    if gen_n == 0: return {}
    rng = np.random.default_rng(seed)
    gen_dir = Path(generated_dir)
    E = {}
    for label in C_fault_seed.keys():
        f = gen_dir / f"{label}.npy"
        if not f.exists(): continue
        arr = np.load(f)
        if arr.shape[0] < gen_n: continue
        idx = rng.choice(arr.shape[0], size=gen_n, replace=False)
        E[label] = arr[idx].astype("float32")
    return E
