# -*- coding: utf-8 -*-
"""生成论文 Fig 1–8(见 论文思路.md 第 5 节), 300 dpi 存 figures/。

依赖: hea_h_adsorption.csv(2500 行) + hea_formation_enthalpy.csv(25 行)。
发散弛豫已在 [-2, 1] eV 窗口剔除。直接 `python hea_make_figures.py` 即可。
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ---------- 出版级样式 ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = os.environ.get("HEA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
FIG_DIR = os.environ.get("HEA_FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
CSV = os.path.join(OUT_DIR, "hea_h_adsorption.csv")
DH_CSV = os.path.join(OUT_DIR, "hea_formation_enthalpy.csv")

ELEMENTS = ["Ru", "Ni", "Co", "Fe", "Cu"]
EL_COLOR = {"Ru": "#4C72B0", "Ni": "#DD8452", "Co": "#55A868", "Fe": "#C44E52", "Cu": "#8172B3"}
PROPS = {
    "Ru": {"chi": 2.20, "r": 1.34, "vec": 8},
    "Ni": {"chi": 1.91, "r": 1.24, "vec": 10},
    "Co": {"chi": 1.88, "r": 1.25, "vec": 9},
    "Fe": {"chi": 1.83, "r": 1.26, "vec": 8},
    "Cu": {"chi": 1.90, "r": 1.28, "vec": 11},
}
E_LO, E_HI = -2.0, 1.0

# 文献 DFT H 吸附能(近似参考值, 单位 eV; 务必核对并替换为你的引用)
REF_DFT = {"Ru": -0.54, "Ni": -0.28, "Co": -0.50, "Fe": -0.46, "Cu": -0.06}


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
    chi_mean = float(x @ chi); r_mean = float(x @ r); vec_mean = float(x @ vec)
    chi_mm = float(np.sqrt(x @ (chi - chi_mean) ** 2))
    r_mm = float(np.sqrt(x @ (1.0 - r / r_mean) ** 2))
    return {**{f"x_{el}": x[i] for i, el in enumerate(ELEMENTS)},
            "S_conf": S, "chi_mean": chi_mean, "chi_mm": chi_mm,
            "r_mean": r_mean, "r_mm": r_mm, "vec_mean": vec_mean}


def _frac_label(x):
    """分数向量 -> 'Ru42Ni35Co9Fe10Cu5' 风格标签(修正舍入到 100)。"""
    pct = [int(round(float(v) * 100)) for v in x]
    pct[0] += 100 - sum(pct)
    return "".join(f"{el}{pct[i]}" for i, el in enumerate(ELEMENTS))


def _clean(s):
    return s[(s >= E_LO) & (s <= E_HI)]


def _is_pure(label):
    fracs = [parse_comp(label)[el] for el in ELEMENTS]
    return sum(1 for f in fracs if f > 0.99) == 1


def gp_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("m", GaussianProcessRegressor(
                         kernel=C(1.0) * RBF(1.0) + WhiteKernel(1.0),
                         normalize_y=True, random_state=0))])


def rf_pipe():
    return Pipeline([("m", RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0))])


# ---------- 数据装载 ----------
def load():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df = df[(df["E_ads"] >= E_LO) & (df["E_ads"] <= E_HI)].reset_index(drop=True)
    dh = pd.read_csv(DH_CSV, encoding="utf-8-sig")
    dh_map = dict(zip(dh["comp"], dh["dH_form"]))
    _rows = []
    for comp, s in df.groupby("comp")["E_ads"]:
        c = _clean(s)
        _rows.append([comp, c.mean(), c.std(), len(c)])
    stat = pd.DataFrame(_rows, columns=["comp", "mu", "sigma", "n"])
    # 特征(成分物性 + ΔH)
    cf = pd.DataFrame([comp_features(c) for c in stat["comp"]], index=stat.index)
    cf["dH_form"] = stat["comp"].map(dh_map)
    stat["x_Ru"] = cf["x_Ru"].values
    stat["S_conf"] = cf["S_conf"].values
    stat["vec_mean"] = cf["vec_mean"].values
    stat["is_pure"] = stat["comp"].apply(_is_pure)
    return df, stat, cf


# =========================================================
# Fig 1 工作流示意
# =========================================================
def fig1_workflow():
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    ax.axis("off")
    boxes = [
        (0.00, "MLIP screening\nMACE-MP-0\n25 comps × 10 cfgs\n× 10 sites"),
        (0.29, "Site-resolved E_ads\n2500 relaxations\n+ site classification"),
        (0.58, "Composition targets\nμ = mean, σ = spread\n+ formation energy ΔH"),
        (0.87, "GP surrogate\n+ active learning\n→ full comp. map"),
    ]
    for x, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.15), 0.20, 0.70,
                                    boxstyle="round,pad=0.01",
                                    fc="#eef3f8", ec="#4C72B0", lw=1.0))
        ax.text(x + 0.10, 0.50, txt, ha="center", va="center", fontsize=7)
    for x in (0.20, 0.49, 0.78):
        ax.add_patch(FancyArrowPatch((x, 0.50), (x + 0.085, 0.50),
                                     arrowstyle="-|>", mutation_scale=12, color="#555"))
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1)
    fig.savefig(os.path.join(FIG_DIR, "fig1_workflow.png"))
    plt.close(fig)
    print("Fig 1 工作流  -> fig1_workflow.png")


# =========================================================
# Fig 2 位点迁移矩阵
# =========================================================
def fig2_migration(df):
    types = ["top", "bridge", "fcc", "hcp"]
    cm = np.zeros((4, 4))
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            cm[i, j] = ((df["site_type"] == a) & (df["act_site_type"] == b)).sum()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    mig = (df["site_type"] != df["act_site_type"]).mean() * 100

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(types)
    ax.set_yticks(range(4)); ax.set_yticklabels(types)
    ax.set_xlabel("Relaxed site"); ax.set_ylabel("Initial site")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fraction")
    ax.set_title(f"H migration ({mig:.0f}% overall)", fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig2_migration.png"))
    plt.close(fig)
    print(f"Fig 2 位点迁移(总迁移 {mig:.0f}%)  -> fig2_migration.png")


# =========================================================
# Fig 3 纯金属 E_ads 验证
# =========================================================
def fig3_validation(stat):
    pure = stat[stat["is_pure"]].copy()
    pure["el"] = pure["comp"].apply(lambda c: parse_comp(c) and
                                    [e for e in ELEMENTS if parse_comp(c)[e] > 0.99][0])
    pure = pure.sort_values("mu")
    els = pure["el"].tolist()
    mu = pure["mu"].values; sg = pure["sigma"].values
    ref = np.array([REF_DFT[e] for e in els])

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(els))
    ax.bar(x, mu, yerr=sg, capsize=3, color=[EL_COLOR[e] for e in els],
           alpha=0.85, label="MACE-MP-0")
    ax.scatter(x, ref, marker="D", s=36, color="#C44E52", zorder=3,
               label="Ref. DFT (approx.)")
    ax.axhline(-0.24, ls="--", color="gray", lw=0.8)
    ax.text(len(els) - 0.3, -0.24, "ΔG$_H$=0", fontsize=7, va="bottom", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(els)
    ax.set_ylabel("E$_{ads}$ (eV)"); ax.set_xlabel("Element")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig3_validation.png"))
    plt.close(fig)
    print("Fig 3 纯金属验证  -> fig3_validation.png  (REF_DFT 为近似值, 需替换引用)")


# =========================================================
# Fig 4 μ 随成分: VEC 主导 + Ru 分数
# =========================================================
def _simplex_heatmap(stat, ax):
    """PCA 把 5 元成分投到 2D, GP(成分物性特征, 不含 ΔH)预测 μ 填色。"""
    from sklearn.decomposition import PCA
    fracs = np.array([list(parse_comp(c).values()) for c in stat["comp"]])
    pca = PCA(n_components=2).fit(fracs)

    rng = np.random.default_rng(0)
    cand = []
    while len(cand) < 2500:
        x = rng.dirichlet(np.ones(len(ELEMENTS)))
        if x.min() >= 0.05:
            cand.append(x)
    cand = np.array(cand)
    Xc = np.array([list(comp_features(_frac_label(x)).values()) for x in cand])

    X11 = np.array([list(comp_features(c).values()) for c in stat["comp"]])
    y = stat["mu"].values
    mu_c = gp_pipe().fit(X11, y).predict(Xc)

    xy = pca.transform(cand)
    xy_c = pca.transform(fracs)
    vmin, vmax = float(y.min()), float(y.max())

    tcf = ax.tricontourf(xy[:, 0], xy[:, 1], mu_c, levels=24,
                         cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax.scatter(xy_c[:, 0], xy_c[:, 1], c=y, cmap="viridis_r",
               vmin=vmin, vmax=vmax, edgecolor="k", lw=0.5, s=26, zorder=3)
    for el in ELEMENTS:
        f = np.array([[1.0 if e == el else 0.0 for e in ELEMENTS]])
        p = pca.transform(f)[0]
        ax.text(p[0], p[1], el, fontsize=8, ha="center", va="center", fontweight="bold",
                color="white", zorder=4,
                bbox=dict(fc=EL_COLOR[el], ec="none", boxstyle="round,pad=0.12", alpha=0.95))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}%)")
    ax.set_title("PCA projection", fontsize=8)
    return tcf


def _pentagon_vertices(n=5):
    """正五边形顶点(顶部起, 逆时针)。"""
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (n, 2)


def _pentagon_heatmap(stat, ax):
    """正五边形重心投影: 5 元分数 -> 顶点加权平均(数学上严格的单纯形图)。"""
    n = len(ELEMENTS)
    V = _pentagon_vertices(n)
    for i in range(n):                                   # 五边形边框
        ax.plot([V[i, 0], V[(i + 1) % n, 0]], [V[i, 1], V[(i + 1) % n, 1]],
                color="#bbb", lw=1.0, zorder=1)

    rng = np.random.default_rng(0)
    cand = []
    while len(cand) < 2500:
        x = rng.dirichlet(np.ones(n))
        if x.min() >= 0.05:
            cand.append(x)
    cand = np.array(cand)
    Xc = np.array([list(comp_features(_frac_label(x)).values()) for x in cand])
    X11 = np.array([list(comp_features(c).values()) for c in stat["comp"]])
    y = stat["mu"].values
    mu_c = gp_pipe().fit(X11, y).predict(Xc)

    xy = cand @ V
    xy_c = np.array([list(parse_comp(c).values()) for c in stat["comp"]]) @ V
    vmin, vmax = float(y.min()), float(y.max())

    tcf = ax.tricontourf(xy[:, 0], xy[:, 1], mu_c, levels=24,
                         cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax.scatter(xy_c[:, 0], xy_c[:, 1], c=y, cmap="viridis_r",
               vmin=vmin, vmax=vmax, edgecolor="k", lw=0.5, s=26, zorder=3)
    for i, el in enumerate(ELEMENTS):                    # 顶点标注
        ax.text(V[i, 0] * 1.14, V[i, 1] * 1.14, el, fontsize=9, ha="center", va="center",
                fontweight="bold", color=EL_COLOR[el])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("5-component simplex (pentagon)", fontsize=8)
    return tcf


def fig4_mu_drivers(stat):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8))
    # (a) μ vs VEC
    ax = axes[0, 0]
    x = stat["vec_mean"].values; y = stat["mu"].values
    sc = ax.scatter(x, y, c=stat["x_Ru"], cmap="magma", s=30, edgecolor="k", lw=0.4)
    c = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(c, xs), color="gray", lw=0.8, ls="--")
    ax.text(0.05, 0.06, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("VEC"); ax.set_ylabel("μ, mean E$_{ads}$ (eV)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("x$_{Ru}$")
    # (b) μ vs Ru 分数
    ax = axes[0, 1]
    xr = stat["x_Ru"].values
    ax.scatter(xr, y, c=[EL_COLOR["Ru"] if p else "#999" for p in stat["is_pure"]],
               s=30, edgecolor="k", lw=0.4)
    r2 = np.corrcoef(xr, y)[0, 1]
    ax.text(0.05, 0.06, f"r = {r2:+.2f}", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Ru mole fraction x$_{Ru}$"); ax.set_ylabel("μ, mean E$_{ads}$ (eV)")
    # (c) PCA 投影热图
    ax = axes[1, 0]
    tcf = _simplex_heatmap(stat, ax)
    cb = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04); cb.set_label("μ (eV)")
    # (d) 正五边形单纯形热图
    ax = axes[1, 1]
    tcf2 = _pentagon_heatmap(stat, ax)
    cb = fig.colorbar(tcf2, ax=ax, fraction=0.046, pad=0.04); cb.set_label("μ (eV)")
    fig.savefig(os.path.join(FIG_DIR, "fig4_mu_drivers.png"))
    plt.close(fig)
    print("Fig 4 μ 驱动因素 + PCA/五边形单纯形热图  -> fig4_mu_drivers.png")


# =========================================================
# Fig 5 E_ads 分布 + σ vs 构型熵
# =========================================================
def fig5_heterogeneity(df, stat):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    # (a) 纯 Ru vs 一个五元 的 E_ads 分布
    ax = axes[0]
    pure_ru = df[df["comp"] == "Ru100Ni0Co0Fe0Cu0"]["E_ads"]
    hea_c = df[df["comp"] == "Ru13Ni41Co23Fe8Cu15"]["E_ads"]
    bins = np.linspace(-1.4, 0.2, 32)
    ax.hist(pure_ru, bins=bins, density=True, alpha=0.6, color=EL_COLOR["Ru"],
            label="Pure Ru (σ=%.2f)" % pure_ru.std())
    ax.hist(hea_c, bins=bins, density=True, alpha=0.6, color="#555",
            label="HEA (σ=%.2f)" % hea_c.std())
    ax.set_xlabel("E$_{ads}$ (eV)"); ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6.5)
    # (b) σ vs 构型熵
    ax = axes[1]
    ax.scatter(stat["S_conf"], stat["sigma"], s=40,
               c=[EL_COLOR["Ru"] if p else "#555" for p in stat["is_pure"]],
               edgecolor="k", lw=0.4)
    ax.set_xlabel("Configurational entropy S$_{conf}$ (k$_B$)")
    ax.set_ylabel("σ, std of E$_{ads}$ (eV)")
    ax.annotate("pure metals", xy=(0.05, 0.30), fontsize=7, color=EL_COLOR["Ru"])
    ax.annotate("HEAs", xy=(0.9, 0.14), fontsize=7, color="#555")
    fig.savefig(os.path.join(FIG_DIR, "fig5_heterogeneity.png"))
    plt.close(fig)
    print("Fig 5 位点异质性  -> fig5_heterogeneity.png")


# =========================================================
# Fig 6 局域环境: hollow 壳层 Ru 数 vs E_ads
# =========================================================
def fig6_local_env(df):
    h = df[df["act_site_type"].isin(["fcc", "hcp"])].copy()
    h["n_Ru"] = h["act_local_env"].apply(lambda s: s.count("Ru") if isinstance(s, str) else 0)
    grp = h.groupby("n_Ru")["E_ads"].agg(mu="mean", sd="std", n="count").reset_index()

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.errorbar(grp["n_Ru"], grp["mu"], yerr=grp["sd"], capsize=3, marker="o",
                color="#4C72B0", ls="-", lw=1.0)
    for _, r in grp.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["n_Ru"], r["mu"]), textcoords="offset points",
                    xytext=(6, -4), fontsize=6.5, color="gray")
    ax.set_xlabel("Ru atoms in hollow shell"); ax.set_ylabel("E$_{ads}$ (eV)")
    ax.set_xticks(sorted(grp["n_Ru"]))
    fig.savefig(os.path.join(FIG_DIR, "fig6_local_env.png"))
    plt.close(fig)
    print("Fig 6 局域环境分解  -> fig6_local_env.png")


# =========================================================
# Fig 7 LOOCV parity + 特征重要性
# =========================================================
def fig7_parity(stat, X):
    groups = np.arange(len(stat))
    y_mu = stat["mu"].values; y_sg = stat["sigma"].values
    yp_mu = cross_val_predict(gp_pipe(), X.values, y_mu, groups=groups, cv=LeaveOneGroupOut())
    yp_sg = cross_val_predict(rf_pipe(), X.values, y_sg, groups=groups, cv=LeaveOneGroupOut())
    from sklearn.metrics import r2_score
    rf = rf_pipe().named_steps["m"].fit(X.values, y_mu)
    imp = np.argsort(rf.feature_importances_)[::-1][:6]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, y, yp, t in [(axes[0], y_mu, yp_mu, "μ (GP)"),
                         (axes[1], y_sg, yp_sg, "σ (RF)")]:
        lo = min(y.min(), yp.min()); hi = max(y.max(), yp.max())
        ax.scatter(y, yp, s=26, color="#4C72B0", edgecolor="k", lw=0.3)
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=0.8)
        ax.set_xlabel(f"True {t.split()[0]}"); ax.set_ylabel(f"Pred. {t.split()[0]}")
        ax.text(0.05, 0.90, f"R²={r2_score(y, yp):+.2f}", transform=ax.transAxes, fontsize=8)
    ax = axes[2]
    ax.barh(range(6), rf.feature_importances_[imp][::-1],
            color="#DD8452", edgecolor="k", lw=0.4)
    ax.set_yticks(range(6)); ax.set_yticklabels([X.columns[i] for i in imp][::-1], fontsize=7)
    ax.set_xlabel("RF importance (μ)")
    ax.invert_yaxis()
    fig.savefig(os.path.join(FIG_DIR, "fig7_parity.png"))
    plt.close(fig)
    print("Fig 7 LOOCV parity + 重要性  -> fig7_parity.png")


# =========================================================
# Fig 8 主动学习 vs 随机采样
# =========================================================
def fig8_active_learning(stat, X, n_test=5, n_start=3, n_trials=8, seed=0):
    y = stat["mu"].values
    n = len(X)
    rng = np.random.default_rng(seed)
    errs = {"random": [], "active": []}
    for _ in range(n_trials):
        idx = np.arange(n)
        test = rng.choice(idx, n_test, replace=False)
        pool = np.array([i for i in idx if i not in test])
        # 随机
        tr = list(pool[:n_start]); rem = list(pool[n_start:])
        e = []
        for k in range(len(tr), len(pool) + 1):
            g = gp_pipe().fit(X.iloc[tr].values, y[tr])
            e.append(np.sqrt(mean_squared_error(y[test], g.predict(X.iloc[test].values))))
            if rem:
                tr.append(rem.pop(0))
        errs["random"].append(e)
        # 主动(不确定性采样)
        tr = list(pool[:n_start]); rem = list(pool[n_start:])
        e = []
        for k in range(len(tr), len(pool) + 1):
            g = gp_pipe().fit(X.iloc[tr].values, y[tr])
            e.append(np.sqrt(mean_squared_error(y[test], g.predict(X.iloc[test].values))))
            if rem:
                _, sd = g.predict(X.iloc[rem].values, return_std=True)
                pick = rem[int(np.argmax(sd))]
                rem.remove(pick); tr.append(pick)
        errs["active"].append(e)
    xs = np.arange(n_start, n_start + len(errs["random"][0]))
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for key, color, lab in [("random", "#999", "Random"), ("active", "#4C72B0", "Active (GP)")]:
        m = np.array(errs[key]); mm = m.mean(0); ss = m.std(0)
        ax.plot(xs, mm, color=color, lw=1.2, label=lab)
        ax.fill_between(xs, mm - ss, mm + ss, color=color, alpha=0.2)
    ax.set_xlabel("Training compositions"); ax.set_ylabel("Test RMSE (eV)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, "fig8_active_learning.png"))
    plt.close(fig)
    print("Fig 8 主动学习效率  -> fig8_active_learning.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df, stat, X = load()
    print(f"数据: {len(df)} 个位点, {len(stat)} 个成分\n")
    fig1_workflow()
    fig2_migration(df)
    fig3_validation(stat)
    fig4_mu_drivers(stat)
    fig5_heterogeneity(df, stat)
    fig6_local_env(df)
    fig7_parity(stat, X)
    fig8_active_learning(stat, X)
    print("\n全部完成 ->", FIG_DIR)


if __name__ == "__main__":
    main()
