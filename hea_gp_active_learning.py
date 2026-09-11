# -*- coding: utf-8 -*-
"""GP 主动学习闭环: 用已算成分的 μ 拟合 GP, 推荐"下一个该算的成分"。

候选池 = 随机五元成分(Dirichlet(1) 均匀先验 + min-frac 拒绝), 排除已算成分。
同一个 μ-GP 提供两个采集目标:
  1) 强结合 / 氢储存 —— LCB(lower confidence bound): a(c) = μ̂(c) − κ·σ̂(c),
     即"预测最负吸附能 + 预测最不确定", 适合找结合最强的成分。
  2) HER 催化 —— 距热中性距离: a(c) = |μ̂(c) − μ_opt| − κ·σ̂(c),
     μ_opt = -0.24 eV(ΔG_H* = E_ads + 0.24 eV ≈ 0 的火山顶点)。
     注意: HER 活性最好的是"μ 最接近 -0.24 eV", 不是"最负"; 最负=过结合=脱附慢。

输出两个目标的 top-K 推荐(比例 + 预测 μ + GP 不确定度), 交给 MACE 去算那几条,
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = os.environ.get("HEA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CSV = os.path.join(OUT_DIR, "hea_h_adsorption.csv")
FIG_DIR = os.environ.get("HEA_FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))

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

# HER 热中性目标: ΔG_H* = E_ads + 0.24 eV ≈ 0 时活性最优(火山顶点)
# 即 μ(E_ads) ≈ -0.24 eV 的成分 HER 活性最好, 而非"最负(结合最强)"。
MU_OPT = -0.24        # eV
DG_ZPE = 0.24         # eV, ΔG_H* ≈ E_ads + 0.24


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


def reachability_figure(gp):
    """Fig 11: "min-frac 约束下能否逼近 HER 热中性"分析。

    主脚本五元成分每个元素占比 ≥ MIN_FRAC(=0.05)。问题: 在这个约束下, 用 μ-GP
    在可行域内搜索, 最接近热中性 μ_opt=-0.24 eV(ΔG_H*≈0)的成分能到多近?
    富 Ru/Fe 强结合元素即使只占 5% 也会把 μ 拽到过结合侧。

    (a) 一系列 min-frac 下, 可行域内"最接近 μ_opt 的 μ"曲线;
    (b) min-frac=MIN_FRAC 下 μ 的预测分布, 标出 μ_opt 与最近点。
    """
    rng = np.random.default_rng(1)
    min_fracs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    N = 4000

    def _closest(mf):
        """采样可行域 {x_i >= mf, sum=1}, 返回 (μ, x, 最接近 μ_opt 的下标)。"""
        y = rng.dirichlet(np.ones(len(ELEMENTS)), size=N)
        x = mf + (1.0 - len(ELEMENTS) * mf) * y          # x_i >= mf, sum=1
        Xc = np.array([list(comp_features(label_from_frac(r)).values()) for r in x])
        mu = gp.predict(Xc)
        i = int(np.argmin(np.abs(mu - MU_OPT)))
        return mu, x, i

    close_mu, close_comp = [], []
    mu_minfrac = None
    for mf in min_fracs:
        mu, x, i = _closest(mf)
        close_mu.append(mu[i])
        close_comp.append(label_from_frac(x[i]))
        if mf == MIN_FRAC:               # 记住 min-frac=MIN_FRAC 的整批采样, 供 (b) 直方图
            mu_minfrac, i_minfrac = mu, i

    comp0 = close_comp[min_fracs.index(MIN_FRAC)]
    mu0, i0 = mu_minfrac, i_minfrac

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # (a) 最接近热中性的 μ vs min-frac
    ax1.plot(min_fracs, close_mu, "o-", color="#4C72B0", lw=1.2, ms=5)
    ax1.axhline(MU_OPT, color="0.35", ls=":", lw=0.9)
    ax1.set_xlabel("min. per-element fraction  (min-frac)")
    ax1.set_ylabel("closest achievable  μ (eV)")
    ax1.set_title("(a) best-case μ vs min-frac")
    for mf, mu, c in [(0.0, close_mu[0], close_comp[0]),
                      (MIN_FRAC, mu0[i0], comp0)]:
        ax1.annotate(f"{c}\nμ={mu:.2f}", (mf, mu), xytext=(6, 6),
                     textcoords="offset points", fontsize=6, color="#4C72B0")

    # (b) min-frac=0.05 的 μ 分布
    n, _bins, _patches = ax2.hist(mu0, bins=40, color="0.8", edgecolor="0.5", lw=0.3)
    ymax = float(n.max())
    ax2.axvline(MU_OPT, color="0.35", ls=":", lw=0.9)
    ax2.axvline(mu0[i0], color="#C44E52", ls="--", lw=0.9)
    ax2.annotate(f"closest\n{comp0}\nμ={mu0[i0]:.2f}", (mu0[i0], ymax),
                 xytext=(0, -6), textcoords="offset points", fontsize=6,
                 color="#C44E52", ha="center", va="top")
    ax2.text(MU_OPT - 0.01, ymax * 0.92, "ΔG$_H$*≈0", fontsize=6, color="0.3",
             rotation=90, va="top", ha="right")
    ax2.set_xlabel("predicted μ (eV)  at min-frac=0.05")
    ax2.set_ylabel("count")
    ax2.set_title("(b) distribution, min-frac = 0.05")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, "fig11_thermoneutral_reachability.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"图已保存 -> {out_png}")
    print(f"[Fig 11] min-frac={MIN_FRAC} 约束下最接近热中性的成分: {comp0} "
          f"(μ={mu0[i0]:.3f} eV, 距 μ_opt 还差 {abs(mu0[i0] - MU_OPT):.3f} eV)")


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
    acq = mu - KAPPA * sd                       # 强结合(最小化 μ)
    order = np.argsort(acq)

    acq_her = np.abs(mu - MU_OPT) - KAPPA * sd  # HER 热中性(μ 越接近 MU_OPT 越好)
    order_her = np.argsort(acq_her)

    print(f"===== GP 主动学习 · 推荐下一个计算的成分 (κ={KAPPA}) =====")

    print(f"\n[目标 1] 强结合 / 氢储存 —— 最小化 μ (LCB)")
    print(f"  {'#':>2}  {'成分':<26}{'预测 μ':>9}{'σ_GP':>8}{'LCB':>8}")
    for rank, i in enumerate(order[:TOP_K], 1):
        print(f"  {rank:>2}  {cand_labels[i]:<26}{mu[i]:>9.3f}{sd[i]:>8.3f}{acq[i]:>8.3f}")

    print(f"\n[目标 2] HER 催化 —— μ 最接近热中性 {MU_OPT} eV (ΔG_H*≈0)")
    print(f"  {'#':>2}  {'成分':<26}{'预测 μ':>9}{'ΔG_H*':>8}{'|μ-μ_opt|':>10}{'σ_GP':>8}")
    for rank, i in enumerate(order_her[:TOP_K], 1):
        print(f"  {rank:>2}  {cand_labels[i]:<26}{mu[i]:>9.3f}{mu[i] + DG_ZPE:>8.3f}"
              f"{abs(mu[i] - MU_OPT):>10.3f}{sd[i]:>8.3f}")

    print("\n用法: 把上面成分加进主脚本的 comps 列表(或另起一批)用 MACE 算,",
          "算完把新行追加进 hea_h_adsorption.csv, 再重跑本脚本即为闭环迭代。")
    print(f"[注] 特征维度: {feat_names}")

    reachability_figure(gp)


if __name__ == "__main__":
    main()
