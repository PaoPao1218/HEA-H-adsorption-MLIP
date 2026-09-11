# -*- coding: utf-8 -*-
"""HEA 可合成性 / 固溶体形成能力指标: δ、ΔSmix、Ω、VEC, 并出图。

经验判据(经典 HEA 文献):
  δ      ≤ 6.6%                原子尺寸差小 → 易形成固溶体
  11 ≤ ΔSmix ≤ 19.5 J/(K·mol)  高熵效应足以稳定固溶体(5 元等摩尔 = R·ln5 = 13.4)
  Ω      ≥ 1.1                 Ω = Tm·ΔSmix / |ΔHmix|, 熵贡献超过焓 → 单相固溶体
  VEC    ≥ 8                   FCC 结构(本工作用的就是 fcc(111))

δ / ΔSmix / VEC 是纯成分物性(只需摩尔分数 + 元素常数), 直接算;
Ω 需要 T_m(熔点加权平均) + ΔHmix(混合焓)。

ΔHmix 两种口径:
  · 代理(proxy)  = |dH_form|: 相对元素稳定相基态(hcp-Ru/Co, bcc-Fe), 即 CSV 里的 dH_form。
  · 严格(strict) = dH_form + Σ c_i·ΔE_struct_i: 相对"同 fcc 晶格"的纯元素,
    ΔE_struct_i = E_ground_i − E_fcc_i 来自 data/hea_pure_reference.csv
    (hea_ref_energies.py 用 MACE 算)。严格口径下纯金属 ΔHmix 严格为 0。

注意: 当 ΔHmix → 0(接近理想固溶体)时 Ω 发散, 判据失去判别力; MACE 对混合焓
的绝对精度有限, Ω 的绝对值应谨慎解读(见 README)。
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = os.environ.get(
    "HEA_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
FIG_DIR = os.environ.get(
    "HEA_FIG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
DH_CSV = os.path.join(DATA_DIR, "hea_formation_enthalpy.csv")
REF_CSV = os.path.join(DATA_DIR, "hea_pure_reference.csv")
ADS_CSV = os.path.join(DATA_DIR, "hea_h_adsorption.csv")

ELEMENTS = ["Ru", "Ni", "Co", "Fe", "Cu"]
RADII = {"Ru": 1.34, "Ni": 1.24, "Co": 1.25, "Fe": 1.26, "Cu": 1.28}   # 金属原子半径 Å
VEC   = {"Ru": 8,    "Ni": 10,   "Co": 9,    "Fe": 8,    "Cu": 11}     # 价电子浓度
TMELT = {"Ru": 2607, "Ni": 1728, "Co": 1768, "Fe": 1811, "Cu": 1358}   # 熔点 K

R_GAS = 8.314                                   # J/(K·mol)
EV_ATOM_TO_J_MOL = 96485.3
DZ_THRESH = 0.01                                # eV/atom: |ΔHmix| 低于此视为"接近理想固溶体", Ω 不可靠
E_LO, E_HI = -2.0, 1.0                          # E_ads 物理窗口(eV): 之外视为发散弛豫, 剔除


def parse_comp(label):
    d = {}
    for el in ELEMENTS:
        m = re.search(el + r"(\d+)", label)
        d[el] = int(m.group(1)) / 100.0 if m else 0.0
    s = sum(d.values())
    if s > 0:
        d = {k: v / s for k, v in d.items()}
    return d


def metrics(x):
    r_mean = sum(x[el] * RADII[el] for el in ELEMENTS)
    delta = 100.0 * np.sqrt(sum(x[el] * (1.0 - RADII[el] / r_mean) ** 2
                                for el in ELEMENTS))                    # %
    dS = -R_GAS * sum(x[el] * np.log(x[el]) for el in ELEMENTS if x[el] > 0)
    Tm = sum(x[el] * TMELT[el] for el in ELEMENTS)                       # K
    vec = sum(x[el] * VEC[el] for el in ELEMENTS)
    return delta, dS, Tm, vec


def _omega(Tm, dS, dHmix_J):
    return Tm * dS / dHmix_J if dHmix_J > 1e-6 else np.inf


def load_mu():
    """读 hea_h_adsorption.csv, 按成分聚合 μ=mean(E_ads), 剔除发散弛豫 [-2,1] eV。"""
    if not os.path.isfile(ADS_CSV):
        return {}
    ads = pd.read_csv(ADS_CSV, encoding="utf-8-sig")
    mu = {}
    for comp, s in ads.groupby("comp")["E_ads"]:
        s = s[(s >= E_LO) & (s <= E_HI)]
        if len(s):
            mu[comp] = float(s.mean())
    return mu


def main():
    df = pd.read_csv(DH_CSV, encoding="utf-8-sig")

    # 严格 ΔHmix 的结构项(若存在)
    dE_struct = {el: 0.0 for el in ELEMENTS}
    if os.path.isfile(REF_CSV):
        ref = pd.read_csv(REF_CSV, encoding="utf-8-sig")
        dE_struct = dict(zip(ref["el"], ref["dE_struct"]))
        print(f"[严格口径] 读入 {len(ref)} 个元素的 ΔE_struct = E_ground − E_fcc:\n"
              f"  " + ", ".join(f"{el}:{dE_struct[el]:+.4f}" for el in ELEMENTS) + "\n")
    else:
        print("[提示] 无 hea_pure_reference.csv, Ω 只用代理口径(dH_form); "
              "先跑 hea_ref_energies.py 得到严格 ΔHmix\n")

    rows = []
    for _, r in df.iterrows():
        x = parse_comp(r["comp"])
        delta, dS, Tm, vec = metrics(x)
        dH_form_eV = r["dH_form"]                                   # eV/atom (相对基态)
        dH_strict_eV = dH_form_eV + sum(x[el] * dE_struct[el] for el in ELEMENTS)  # eV/atom
        dH_proxy_J = abs(dH_form_eV) * EV_ATOM_TO_J_MOL
        dH_strict_J = abs(dH_strict_eV) * EV_ATOM_TO_J_MOL
        is_pure = any(v > 0.99 for v in x.values())
        rows.append({
            "comp": r["comp"], "delta_%": delta, "dSmix": dS, "Tm": Tm, "VEC": vec,
            "x_Ru": x["Ru"],
            "dH_proxy_J": dH_proxy_J, "dH_strict_J": dH_strict_J,
            "Omega_proxy": _omega(Tm, dS, dH_proxy_J),
            "Omega_strict": _omega(Tm, dS, dH_strict_J),
            "is_pure": is_pure,
        })
    out = pd.DataFrame(rows)
    # 纯金属无混合(ΔSmix=0), Ω 无意义 → 标 NaN
    out.loc[out["is_pure"], ["Omega_proxy", "Omega_strict"]] = np.nan
    out["mu"] = out["comp"].map(load_mu())

    # 判据(δ / ΔSmix / VEC 用物理值, Ω 用严格口径)
    out["pass_delta"] = out["delta_%"] <= 6.6
    out["pass_dS"] = (out["dSmix"] >= 11.0) & (out["dSmix"] <= 19.5)
    out["pass_Omega"] = out["Omega_strict"] >= 1.1
    out["pass_VEC_fcc"] = out["VEC"] >= 8.0
    out["all_pass"] = out[["pass_delta", "pass_dS", "pass_Omega", "pass_VEC_fcc"]].all(axis=1)

    pd.set_option("display.width", 220)
    show = out[["comp", "delta_%", "dSmix", "Omega_proxy", "Omega_strict", "VEC"]].round(2)
    print("===== 全部 25 成分(Ω_proxy 相对基态, Ω_strict 相对同晶格) =====")
    print(show.to_string(index=False))

    hea = out[~out["is_pure"]]
    print("\n===== 固溶体判据通过情况(仅 20 个五元成分) =====")
    for col, name in [("pass_delta", "δ ≤ 6.6%"),
                      ("pass_dS", "11 ≤ ΔSmix ≤ 19.5 J/(K·mol)"),
                      ("pass_Omega", "Ω_strict ≥ 1.1"),
                      ("pass_VEC_fcc", "VEC ≥ 8 (FCC)"),
                      ("all_pass", "四项全通过")]:
        print(f"  {name:<28} {hea[col].sum():>2}/{len(hea)}")

    print("\n未全通过的五元成分:")
    for _, r in hea[~hea["all_pass"]].iterrows():
        fails = []
        if not r["pass_delta"]:
            fails.append(f"δ={r['delta_%']:.1f}%")
        if not r["pass_dS"]:
            fails.append(f"ΔSmix={r['dSmix']:.1f}")
        if not r["pass_Omega"]:
            fails.append(f"Ω={r['Omega_strict']:.2f}")
        if not r["pass_VEC_fcc"]:
            fails.append(f"VEC={r['VEC']:.1f}")
        print(f"  {r['comp']:<26} 失败项: {', '.join(fails)}")

    near_zero = hea[hea["dH_strict_J"] < DZ_THRESH * EV_ATOM_TO_J_MOL]
    if len(near_zero):
        print(f"\n[注意] {len(near_zero)} 个成分 |ΔHmix_strict| < {DZ_THRESH} eV/atom"
              f"(接近理想固溶体), Ω_strict 发散、数值不可靠, 判据失去判别力。")

    make_figure(hea)
    make_tradeoff_figure(hea)


def make_figure(hea):
    """经典可合成性图: δ–ΔSmix 散点, 颜色标 Ω(严格口径)。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    # 固溶体形成窗口(δ ≤ 6.6 且 11 ≤ ΔSmix ≤ 19.5)
    ax.add_patch(plt.Rectangle((0, 11.0), 6.6, 8.5, color="0.93", zorder=0))
    ax.axvline(6.6, color="0.4", ls="--", lw=0.8)
    ax.axhline(11.0, color="0.4", ls="--", lw=0.8)
    ax.axhline(19.5, color="0.4", ls="--", lw=0.8)

    omega = hea["Omega_strict"].values
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    sc = ax.scatter(
        hea["delta_%"], hea["dSmix"], c=omega, s=64, zorder=3,
        cmap=cmap, norm=LogNorm(vmin=1.0, vmax=max(omega)),
        edgecolors="black", linewidths=0.6)

    # 未全通过的成分用空心 + 红圈标出
    fail = hea[~hea["all_pass"]]
    if len(fail):
        ax.scatter(fail["delta_%"], fail["dSmix"], s=110, facecolors="none",
                   edgecolors="crimson", linewidths=1.4, zorder=4)
        for _, r in fail.iterrows():
            ax.annotate(r["comp"], (r["delta_%"], r["dSmix"]),
                        xytext=(5, 6), textcoords="offset points",
                        fontsize=6.5, color="crimson")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("$\Omega = T_m\,\Delta S_{mix}\,/\,|\Delta H_{mix}|$  (strict)")
    ax.set_xlabel("Atomic size mismatch  δ (%)")
    ax.set_ylabel("Configurational entropy  ΔSmix (J·K$^{-1}$·mol$^{-1}$)")
    ax.set_title("Solid-solution formability criteria")
    ax.text(0.03, 0.03,
            "shaded = δ ≤ 6.6% & 11 ≤ ΔSmix ≤ 19.5\n"
            "all points: VEC ≥ 8 (FCC) & Ω ≥ 1.1\n"
            "red rings = fail (ΔSmix < 11)",
            transform=ax.transAxes, fontsize=6.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))
    ax.set_xlim(0, 7.5)
    ax.set_ylim(7, 21)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, "fig9_formability.png")
    fig.savefig(out_png)
    print(f"\n图已保存 -> {out_png}")


def make_tradeoff_figure(hea):
    """Fig 10: 活性(μ)–可合成性(ΔSmix)权衡。(a) Ω–Ru 含量; (b) μ–ΔSmix。"""
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

    # (a) Ω_strict vs x_Ru, 颜色 = μ(活性)
    sc1 = ax1.scatter(hea["x_Ru"], hea["Omega_strict"], c=hea["mu"],
                      s=60, zorder=3, cmap="viridis",
                      edgecolors="black", linewidths=0.5)
    ax1.axhline(1.1, color="0.4", ls="--", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Ru molar fraction  x(Ru)")
    ax1.set_ylabel("Ω (strict, log scale)")
    ax1.set_title("(a) Ω vs Ru content")
    cb1 = fig.colorbar(sc1, ax=ax1)
    cb1.set_label("μ = mean E_ads (eV)")

    # (b) μ vs ΔSmix, 颜色 = x_Ru; 标出 ΔSmix 判据边界
    sc2 = ax2.scatter(hea["dSmix"], hea["mu"], c=hea["x_Ru"],
                      s=60, zorder=3, cmap="viridis",
                      edgecolors="black", linewidths=0.5)
    ax2.axvline(11.0, color="0.4", ls="--", lw=0.8)
    ax2.axvline(19.5, color="0.4", ls="--", lw=0.8)
    ax2.set_xlabel("Configurational entropy  ΔSmix (J·K$^{-1}$·mol$^{-1}$)")
    ax2.set_ylabel("μ = mean E_ads (eV)")
    ax2.set_title("(b) activity–formability trade-off")
    cb2 = fig.colorbar(sc2, ax=ax2)
    cb2.set_label("x(Ru)")

    # 两个 ΔSmix 不过的成分用红圈 + 标注
    fail = hea[~hea["all_pass"]]
    for _, r in fail.iterrows():
        ax1.scatter([r["x_Ru"]], [r["Omega_strict"]], s=130, facecolors="none",
                    edgecolors="crimson", linewidths=1.4, zorder=4)
        ax2.scatter([r["dSmix"]], [r["mu"]], s=130, facecolors="none",
                    edgecolors="crimson", linewidths=1.4, zorder=4)
        ax1.annotate(r["comp"], (r["x_Ru"], r["Omega_strict"]),
                     xytext=(5, 5), textcoords="offset points",
                     fontsize=6, color="crimson")
        ax2.annotate(r["comp"], (r["dSmix"], r["mu"]),
                     xytext=(5, 5), textcoords="offset points",
                     fontsize=6, color="crimson")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, "fig10_activity_formability.png")
    fig.savefig(out_png)
    print(f"图已保存 -> {out_png}")


if __name__ == "__main__":
    main()
