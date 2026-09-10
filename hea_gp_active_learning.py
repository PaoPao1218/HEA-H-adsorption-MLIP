# -*- coding: utf-8 -*-
"""GP 主动学习闭环: 用已算成分的 μ 拟合 GP, 推荐"下一个该算的成分"。

候选池 = 随机五元成分(Dirichlet(1) 均匀先验 + min-frac 拒绝), 排除已算成分;
采集函数 LCB(lower confidence bound):  a(c) = μ̂(c) − κ·σ̂(c),
即"预测最负吸附能 + 预测最不确定"的成分。κ 越大越偏探索。

输出 top-K 推荐(比例 + 预测 μ + GP 不确定度), 交给 MACE 去算那几条,
算完把新数据追加进 CSV 再重跑本脚本 → 闭环。

说明: 这里 GP 只用"纯成分物性特征"(计算前即可得), 不含 ΔH; 若要引入 ΔH,
需先给候选成分算 ΔH(廉价 bulk 计算)再并入特征。
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

OUT_DIR = os.environ.get("HEA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CSV = os.path.join(OUT_DIR, "hea_h_adsorption.csv")

ELEMENTS = ["Ru", "Ni", "Co", "Fe", "Cu"]
PROPS = {
    "Ru": {"chi": 2.20, "r": 1.34, "vec": 8},
    "Ni": {"chi": 1.91, "r": 1.24, "vec": 10},
    "Co": {"chi": 1.88, "r": 1.25, "vec": 9},
    "Fe": {"chi": 1.83, "r": 1.26, "vec": 8},
    "Cu": {"chi": 1.90, "r": 1.28, "vec": 11},
}

N_CAND = 200          # 候选池大小
MIN_FRAC = 0.05       # 五元每元素最低摩尔分数(与主脚本一致)
KAPPA = 2.0           # LCB 探索-开发权衡(越大越偏探索)
TOP_K = 5             # 每次推荐几个
SEED = 0


def parse_comp(label):
    d = {}
    for el in ELEMENTS:
        m = re.search(el + r"(\d+)", label)
        d[el] = int(m.group(1)) / 100.0 if m else 0.0
    s = sum(d.values())
    if s > 0:
        d = {k: v / s for k, v in d.items()}
    return d


def comp_features(label):
    x = np.array([parse_comp(label)[el] for el in ELEMENTS])
    chi = np.array([PROPS[el]["chi"] for el in ELEMENTS])
    r = np.array([PROPS[el]["r"] for el in ELEMENTS])
    vec = np.array([PROPS[el]["vec"] for el in ELEMENTS])
    S = -float(np.sum([xi * np.log(xi) for xi in x if xi > 0]))
    chi_mean = float(x @ chi)
    r_mean = float(x @ r)
    vec_mean = float(x @ vec)
    chi_mm = float(np.sqrt(x @ (chi - chi_mean) ** 2))
    r_mm = float(np.sqrt(x @ (1.0 - r / r_mean) ** 2))
    return {**{f"x_{el}": x[i] for i, el in enumerate(ELEMENTS)},
            "S_conf": S, "chi_mean": chi_mean, "chi_mm": chi_mm,
            "r_mean": r_mean, "r_mm": r_mm, "vec_mean": vec_mean}


def label_from_frac(x):
    """分数 -> 'Ru42Ni35Co9Fe10Cu5' 风格标签(修正舍入到 100)"""
    pct = [int(round(float(v) * 100)) for v in x]
    pct[0] += 100 - sum(pct)
    return "".join(f"{el}{pct[i]}" for i, el in enumerate(ELEMENTS))


def random_candidates(n, rng):
    out = []
    while len(out) < n:
        x = rng.dirichlet(np.ones(len(ELEMENTS)))
        if x.min() >= MIN_FRAC:
            out.append(x)
    return np.array(out)


# H 吸附能物理窗口(eV): 之外视为发散弛豫, 剔除
E_LO, E_HI = -2.0, 1.0


def _clean(comp, s):
    bad = (s < E_LO) | (s > E_HI)
    if bad.any():
        print(f"  [清洗] {comp}: 剔除 {int(bad.sum())} 个发散值 "
              f"{[round(float(v), 2) for v in sorted(s[bad])]}")
    return s[~bad]


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    _rows = []
    for comp, s in df.groupby("comp")["E_ads"]:
        c = _clean(comp, s)
        _rows.append([comp, c.mean(), len(c)])
    stat = pd.DataFrame(_rows, columns=["comp", "mu", "n"])
    feat_names = list(comp_features(stat["comp"].iloc[0]).keys())
    X = np.array([list(comp_features(c).values()) for c in stat["comp"]])
    y = stat["mu"].values

    computed = set(stat["comp"])
    print(f"已算成分: {len(computed)} 个  (μ 范围 {y.min():.3f} ~ {y.max():.3f} eV)\n")

    rng = np.random.default_rng(SEED)
    cand_frac = random_candidates(N_CAND, rng)
    cand_labels = [label_from_frac(x) for x in cand_frac]
    keep = [i for i, lb in enumerate(cand_labels) if lb not in computed]
    cand_frac = cand_frac[keep]
    cand_labels = [cand_labels[i] for i in keep]
    Xc = np.array([list(comp_features(lb).values()) for lb in cand_labels])
    print(f"候选池: {len(cand_labels)} 个未算五元成分(Dirichlet, min-frac={MIN_FRAC})\n")

    gp = Pipeline([("s", StandardScaler()),
                   ("m", GaussianProcessRegressor(
                       kernel=C(1.0) * RBF(1.0) + WhiteKernel(1.0),
                       normalize_y=True, random_state=0))])
    gp.fit(X, y)
    mu, sd = gp.predict(Xc, return_std=True)
    acq = mu - KAPPA * sd

    order = np.argsort(acq)
    print(f"===== GP 主动学习 · 推荐下一个计算的成分(LCB, κ={KAPPA}) =====")
    print(f"  {'#':>2}  {'成分':<26}{'预测 μ':>9}{'σ_GP':>8}{'LCB':>8}")
    for rank, i in enumerate(order[:TOP_K], 1):
        print(f"  {rank:>2}  {cand_labels[i]:<26}{mu[i]:>9.3f}{sd[i]:>8.3f}{acq[i]:>8.3f}")

    print("\n用法: 把上面成分加进主脚本的 comps 列表(或另起一批)用 MACE 算,",
          "算完把新行追加进 hea_h_adsorption.csv, 再重跑本脚本即为闭环迭代。")
    print(f"[注] 特征维度: {feat_names}")


if __name__ == "__main__":
    main()
