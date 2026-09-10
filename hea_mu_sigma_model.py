# -*- coding: utf-8 -*-
"""HEA 成分 → {μ, σ} 双目标代理模型(成分级)。

读 hea_h_adsorption.csv(逐位点) → 按成分聚合出
  μ  = E_ads 的均值  (吸附强度, 筛选主目标)
  σ  = E_ads 的标准差(同一成分下 位点/构型 的异质性, 反映"位点多样性")
用成分物性特征(χ / 半径 / VEC / 构型熵 / 失配, 可选 ΔH_form)预测 μ 与 σ。

评估: 留一成分 LOOCV(成分数少, 这才是"未见成分"的泛化)。
μ 用 GP(顺带拿到预测方差, 供主动学习用); σ 非负, 用 RF 直预测 + GP 预测 log σ 对比。
数据补齐后(五元比例算完)直接重跑本脚本即可, 无需改动。
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

OUT_DIR = os.environ.get("HEA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CSV = os.path.join(OUT_DIR, "hea_h_adsorption.csv")
DH_CSV = os.path.join(OUT_DIR, "hea_formation_enthalpy.csv")

ELEMENTS = ["Ru", "Ni", "Co", "Fe", "Cu"]
PROPS = {
    "Ru": {"chi": 2.20, "r": 1.34, "vec": 8},
    "Ni": {"chi": 1.91, "r": 1.24, "vec": 10},
    "Co": {"chi": 1.88, "r": 1.25, "vec": 9},
    "Fe": {"chi": 1.83, "r": 1.26, "vec": 8},
    "Cu": {"chi": 1.90, "r": 1.28, "vec": 11},
}


def parse_comp(label):
    """'Ru42Ni35Co9Fe10Cu5' -> {el: fraction}(归一化, 处理 99/101 舍入)"""
    d = {}
    for el in ELEMENTS:
        m = re.search(el + r"(\d+)", label)
        d[el] = int(m.group(1)) / 100.0 if m else 0.0
    s = sum(d.values())
    if s > 0:
        d = {k: v / s for k, v in d.items()}
    return d


def comp_features(label):
    """成分物性特征: 摩尔分数 + 构型熵 + 平均/失配(χ, r, VEC)。纯成分特征, 不含 ΔH。"""
    x = np.array([parse_comp(label)[el] for el in ELEMENTS])
    chi = np.array([PROPS[el]["chi"] for el in ELEMENTS])
    r = np.array([PROPS[el]["r"] for el in ELEMENTS])
    vec = np.array([PROPS[el]["vec"] for el in ELEMENTS])
    S = -float(np.sum([xi * np.log(xi) for xi in x if xi > 0]))   # 构型熵
    chi_mean = float(x @ chi)
    r_mean = float(x @ r)
    vec_mean = float(x @ vec)
    chi_mm = float(np.sqrt(x @ (chi - chi_mean) ** 2))            # 电负性 mismatch
    r_mm = float(np.sqrt(x @ (1.0 - r / r_mean) ** 2))            # 半径 mismatch
    return {**{f"x_{el}": x[i] for i, el in enumerate(ELEMENTS)},
            "S_conf": S, "chi_mean": chi_mean, "chi_mm": chi_mm,
            "r_mean": r_mean, "r_mm": r_mm, "vec_mean": vec_mean}


def _report(name, y, yp):
    r2 = r2_score(y, yp)
    rmse = float(np.sqrt(mean_squared_error(y, yp)))
    mae = float(mean_absolute_error(y, yp))
    return f"  {name:<16} R²={r2:+.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"


# H 吸附能物理窗口(eV): 之外视为发散弛豫(如 H 掉进表层 E_ads≈-12 eV), 剔除
E_LO, E_HI = -2.0, 1.0


def _clean(comp, s):
    """剔除发散弛豫的 E_ads(物理上 H 吸附能不会超出 [E_LO, E_HI])。"""
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
        _rows.append([comp, c.mean(), c.std(), len(c)])
    stat = pd.DataFrame(_rows, columns=["comp", "mu", "sigma", "n"])
    print(f"===== 成分级 μ/σ 快照({len(stat)} 个成分, 已剔除发散弛豫) =====")
    print(stat.to_string(index=False), "\n")

    X = pd.DataFrame([comp_features(c) for c in stat["comp"]], index=stat.index)
    if os.path.isfile(DH_CSV):
        dh = pd.read_csv(DH_CSV, encoding="utf-8-sig")
        X["dH_form"] = stat["comp"].map(dict(zip(dh["comp"], dh["dH_form"])))
    else:
        print("[提示] 无 hea_formation_enthalpy.csv(跑完全部比例才会写), 本轮不含 ΔH 特征\n")

    y_mu = stat["mu"].values
    y_sig = stat["sigma"].values
    n = len(stat)
    groups = np.arange(n)                      # 每个成分各成一组 = 留一
    logo = LeaveOneGroupOut()

    gp = Pipeline([("s", StandardScaler()),
                   ("m", GaussianProcessRegressor(
                       kernel=C(1.0) * RBF(1.0) + WhiteKernel(1.0),
                       normalize_y=True, random_state=0))])
    rf = Pipeline([("m", RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0))])

    print(f"===== 目标 μ = mean(E_ads) · 留一成分 LOOCV({n} fold) =====")
    print(_report("GP", y_mu, cross_val_predict(gp, X.values, y_mu, groups=groups, cv=logo)))
    print(_report("RF", y_mu, cross_val_predict(rf, X.values, y_mu, groups=groups, cv=logo)))
    print(_report("(均值基线)", y_mu, np.full_like(y_mu, y_mu.mean())))

    print(f"\n===== 目标 σ = std(E_ads) · 留一成分 LOOCV =====")
    print(_report("RF(直预测σ)", y_sig,
                  cross_val_predict(rf, X.values, y_sig, groups=groups, cv=logo)))
    yp_ls = cross_val_predict(gp, X.values, np.log(y_sig + 1e-6), groups=groups, cv=logo)
    print(_report("GP(log σ)", y_sig, np.exp(yp_ls)))
    print(_report("(均值基线)", y_sig, np.full_like(y_sig, y_sig.mean())))

    print("\n[注] 成分数少时 LOOCV 数值会抖; GP 的真正价值在其预测方差(供主动学习), 见 "
          "hea_gp_active_learning.py")


if __name__ == "__main__":
    main()
