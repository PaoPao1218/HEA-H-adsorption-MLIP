# -*- coding: utf-8 -*-
"""
HEA 表面 H 吸附能筛选 —— 三层嵌套循环
========================================
第1层(外循环): 元素比例 Composition   —— 网格自动生成, 探索成分空间
第2层(中循环): 随机排列 Configuration —— 覆盖无序度, 消除"运气"偏差
第3层(内循环): 吸附位点 Site          —— top / fcc / hcp / bridge

目标: 对每个比例, 得到 H 吸附能 E_ads 的分布 μ(均值) 和 σ(标准差),
      用于筛选 HEA 组分。

E_ads = E(slab+H) − E(slab) − 0.5·E(H2)
      (H2 用同一个 MACE 模型算, 参考零点抵消, E_ads 可直接比较)

用法:
    python hea_screening.py
    # 或在 notebook 里:  %run hea_screening.py
"""

import os
import sys
import time
import numpy as np
from itertools import product
from ase import Atoms, Atom
from ase.build import bulk, surface
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.io import write
from mace.calculators import MACECalculator

# Windows 下默认 GBK 控制台无法编码 'μ/σ/Å' 等字符, 统一改成 UTF-8 避免崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================ 配置区 ============================
ELEMENTS = ["Ru", "Ni", "Co", "Fe", "Cu"]

# ---- 第1层: 比例生成 ----
# COMP_MODE="pure+random_quinary": 只算纯金属 + 随机五元(所有元素都>0), 跳过二元/三元/四元
# COMP_MODE="grid": 原来的全网格(按 COMP_STEP 步长)
COMP_STEP = 1 / 5                     # 网格步长(仅 COMP_MODE="grid" 时使用)
COMP_MODE = "pure+random_quinary"
N_QUINARY = 20                        # 随机五元比例个数
MIN_QUINARY_FRAC = 0.05               # 随机五元里每个元素的最低占比(保证>0)
QUINARY_SEED = 42                     # 随机种子(可复现)


# ---- 第2层: 随机排列 ----
N_CONFIGS = 10

# ---- 第3层: 位点 ----
N_SITES = 10                     # 每个排列采样 10 个位点
SITE_TYPES = ["top", "fcc", "hcp", "bridge"]
# 每个类型各采几个; None = 均分(N_SITES 会被平均分到各类型)
N_PER_TYPE = None

# ---- slab 结构 ----
A_LATTICE = 3.52                  # fcc 晶格常数 Å
FACET = (1, 1, 1)
N_LAYERS = 5                      # slab 层数
VACUUM = 18.0                     # 真空层(两侧各加, 即总 2×VACUUM)
SUPERCELL = (2, 2, 1)
N_FIX_LAYERS = 3                  # 固定底层 3 层(用于干净 slab 弛豫; H 吸附时见 relax_h)

# ---- 弛豫 ----
FMAX = 0.02                       # 力收敛判据 eV/Å
MAX_STEPS = 300

# ---- 模型 ----
MODEL_PATH = os.environ.get("MACE_MODEL_PATH", "2023-12-03-mace-128-L1_epoch-199.model")
DEVICE = "cuda"
DEFAULT_DTYPE = "float64"

# H 吸附自由能近似(Nørskov 约定): ΔG_H* ≈ E_ads + 0.24 eV(ZPE + 熵修正)。
# 热中性 ΔG_H* = 0 对应 E_ads ≈ -0.24 eV, 即 HER 火山图顶点。
# 注意: E_ads 是"结合强度"描述符, 不是活性本身; 最负 E_ads = 过结合, HER 反而差。
DG_ZPE = 0.24                       # eV

# ---- 输出 ----
OUT_DIR = os.environ.get("HEA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT_CSV = os.path.join(OUT_DIR, "hea_h_adsorption.csv")
OUT_STATS = os.path.join(OUT_DIR, "hea_h_stats.csv")

# ---- 结构保存(供 hea_postprocess.py 后处理) ----
SAVE_STRUCTURES = True              # 是否保存弛豫后的 VASP 结构
STRUCT_DIR = os.path.join(OUT_DIR, "structures")
SAVE_MAX_CONFIGS = None             # 每个比例保存几个排列; None = 全部(N_CONFIGS 个)
SAVE_SLABH = True                  # 是否保存弛豫后的 slab+H(吸附模型)
SLABH_DIR = os.path.join(OUT_DIR, "slabH_structures")
# ================================================================


# ============================ 工具函数 ============================
def comp_label(comp):
    """把比例 dict 变成紧凑标签, 如 {Ru:1/3,Ni:1/3,Co:1/3,Fe:0} -> Ru33Ni33Co33Fe0"""
    parts = []
    for el in ELEMENTS:
        pct = int(round(comp.get(el, 0.0) * 100))
        parts.append(f"{el}{pct}")
    return "".join(parts)


def grid_compositions(elements, step):
    """在单纯形上按步长生成所有比例(非负整数划分), 返回 list[dict]"""
    k = int(round(1.0 / step))
    comps = []
    for xs in product(range(k + 1), repeat=len(elements)):
        if sum(xs) == k:
            comps.append({el: x / k for el, x in zip(elements, xs)})
    return comps


def random_quinary_compositions(elements, n, min_frac, seed):
    """随机生成 n 个"所有元素都>0"的五元比例。

    用 Dirichlet(α=1) 在单纯形上均匀采样, 再拒绝任一元素占比 < min_frac 的样本,
    保证每个元素都"有意义地"存在(避免 80 原子 slab 里某元素被四舍五入成 0)。
    """
    rng = np.random.default_rng(seed)
    comps = []
    guard = 0
    while len(comps) < n and guard < 20000:
        guard += 1
        x = rng.dirichlet(np.ones(len(elements)))
        if float(x.min()) >= min_frac:
            comps.append({el: float(v) for el, v in zip(elements, x)})
    return comps


def build_slab(composition, seed):
    """构建 (2,2,1) fcc(111) slab 并随机占位; 复用 High_Alloy_construct 的修复逻辑"""
    base = bulk("Ni", "fcc", a=A_LATTICE, cubic=True)
    slab = surface(base, indices=FACET, layers=N_LAYERS, vacuum=None)
    slab = slab * SUPERCELL
    slab.wrap()                       # 收拢跨边界原子
    slab.center(vacuum=VACUUM, axis=2)  # 扩胞/wrap 之后再加真空

    rng = np.random.default_rng(seed)
    n_total = len(slab)
    n_each = [int(round(n_total * composition[el])) for el in ELEMENTS]
    diff = sum(n_each) - n_total
    if diff != 0:
        n_each[np.argmax(n_each)] -= diff   # 修正取整误差

    pool = []
    for el, n in zip(ELEMENTS, n_each):
        pool += [el] * n
    rng.shuffle(pool)
    slab.set_chemical_symbols(pool)
    return slab


def fix_layers(atoms, n_fix):
    """固定底部 n_fix 层(按 z 高度分层)"""
    if n_fix <= 0:
        return
    z = atoms.positions[:, 2]
    levels = np.sort(np.unique(np.round(z, 3)))
    bottom = levels[:n_fix]
    mask = np.isin(np.round(z, 3), bottom)
    atoms.set_constraint(FixAtoms(mask=mask))


def relax(atoms, calc, fmax=FMAX, steps=MAX_STEPS):
    """BFGS 弛豫, 返回弛豫后的 atoms(带计算器)"""
    atoms.calc = calc
    opt = BFGS(atoms, trajectory=None, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms


def relax_h(slab, calc, h_position, fmax=FMAX, steps=MAX_STEPS):
    """吸附 H 弛豫: 固定整个 slab, 只弛豫 H 原子。

    MACE-MP-0 对金属原子(尤其受 H 扰动时)的力不够稳: 若放开顶层金属,
    会诱发虚假表面重建(顶层 Ru 被拉出 1.2 Å, E_ads 掉到 -5.7 eV)。
    单点高度扫描证实体相刚性吸附(min ~1.0 Å, E_ads≈-0.97 eV)是对的,
    因此这里采用"刚性 slab + 只弛豫 H"的标准近似, 对 μ/σ 筛选足够。
    """
    sh = slab.copy()
    sh.append(Atom("H", position=h_position))
    sh.calc = calc
    mask = [True] * len(slab) + [False]      # 金属全固定, H 自由
    sh.set_constraint(FixAtoms(mask=mask))
    opt = BFGS(sh, trajectory=None, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return sh


def half_h2_energy(calc, box=10.0):
    """用同一个 MACE 模型算 H2 分子能量, 返回 0.5·E(H2) 作为 H 的参考"""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.cell = [box, box, box]
    h2.center()
    h2.calc = calc
    opt = BFGS(h2, logfile=None)
    opt.run(fmax=0.01, steps=100)
    return h2.get_potential_energy() / 2.0


# ---- 纯元素稳定相参考(形成焓用): (晶体结构, 初始 a, 初始 c; fcc/bcc 的 c 为 None) ----
STABLE_PHASE = {
    "Ru": ("hcp", 2.706, 4.282),
    "Ni": ("fcc", 3.52,  None),
    "Co": ("hcp", 2.507, 4.069),
    "Fe": ("bcc", 2.866, None),
    "Cu": ("fcc", 3.61,  None),
}


def pure_ground_energy(calc, el):
    """纯元素稳定相 bulk 的单位原子能量(各自稳定相 + 各自体积弛豫)。"""
    phase, a0, c0 = STABLE_PHASE[el]
    if phase == "hcp":
        at = bulk(el, "hcp", a=a0, c=c0)
    else:
        at = bulk(el, phase, a=a0)            # fcc/bcc 原胞(1 原子)
    at.calc = calc
    opt = BFGS(FrechetCellFilter(at, hydrostatic_strain=True), logfile=None)
    opt.run(fmax=FMAX)
    return at.get_potential_energy() / len(at)


def build_bulk_alloy(composition, seed):
    """构建与合金同组分的 bulk fcc 超胞(80 原子, 随机占位, 无表面)。"""
    base = bulk("Ni", "fcc", a=A_LATTICE, cubic=True)   # 4 原子惯用胞
    at = base * (2, 2, 5)                                 # 4×2×2×5 = 80 原子
    n_total = len(at)
    n_each = [int(round(n_total * composition[el])) for el in ELEMENTS]
    diff = sum(n_each) - n_total
    if diff != 0:
        n_each[np.argmax(n_each)] -= diff
    pool = [el for el, n in zip(ELEMENTS, n_each) for _ in range(n)]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    at.set_chemical_symbols(pool)
    return at


def relax_bulk(at, calc):
    """bulk 体积 + 原子位置弛豫(各向同性缩放晶胞)。"""
    at.calc = calc
    opt = BFGS(FrechetCellFilter(at, hydrostatic_strain=True), logfile=None)
    opt.run(fmax=FMAX)
    return at


def _min_image_2d(p, q, cell_xy):
    """2D 最小镜像位移向量 (p - q, 周期化到单元胞内)"""
    d = p - q
    frac = d @ np.linalg.inv(cell_xy)
    frac -= np.round(frac)
    return frac @ cell_xy


def enumerate_sites(slab):
    """
    枚举 fcc(111) 表面高对称位点, 返回 list[(type, position, local_env)]。
    local_env 是配位金属原子元素的有序字符串, 如 'Ru-Ru-Ni'(hollow) 或 'Co'(top)。
    """
    cell_xy = slab.cell[:2, :2].copy()
    pos = slab.positions
    syms = slab.get_chemical_symbols()
    z = pos[:, 2]

    # 顶层表面原子: 按"每层原子数"取 z 最高的一层(抗表面粗糙)。
    # 无序弛豫后顶层高低差可能 >0.5 Å, 固定 Å 窗口会漏掉顶层原子 → 位点数不足。
    n_surf = len(slab) // N_LAYERS            # 每层原子数(2×2 超胞 = 16)
    order = np.argsort(z)
    surf_idx = order[-n_surf:]
    z_surf = float(z[surf_idx].mean())         # 顶层平均高度(比 max 更稳)
    surf_xy = pos[surf_idx][:, :2]

    # 第二层原子(区分 fcc/hcp hollow): 次高的 n_surf 个
    sub_idx = order[-2 * n_surf:-n_surf]
    sub_xy = pos[sub_idx][:, :2]

    n = len(surf_idx)

    # 2D PBC 距离矩阵 + 最近邻图
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(_min_image_2d(surf_xy[i], surf_xy[j], cell_xy))
            dist[i, j] = dist[j, i] = d
    nn_d = dist[dist > 0].min()          # 最近邻距离 ≈ a/√2 ≈ 2.49 Å
    cutoff = 1.3 * nn_d
    neighbors = [[j for j in range(n) if i != j and dist[i, j] < cutoff]
                 for i in range(n)]

    sites = []

    # top: 每个表面原子正上方
    for i in range(n):
        x, y = surf_xy[i]
        env = syms[surf_idx[i]]
        sites.append(("top", np.array([x, y, z_surf + 1.5]), env))

    def hollow_env(tri):
        return "-".join(sorted(syms[surf_idx[k]] for k in tri))

    # bridge + hollow: 基于最近邻方向
    for i in range(n):
        v = {j: _min_image_2d(surf_xy[j], surf_xy[i], cell_xy)
             for j in neighbors[i]}
        # bridge: 与每个最近邻的中点
        for j in neighbors[i]:
            mid = surf_xy[i] + v[j] / 2.0
            env = "-".join(sorted([syms[surf_idx[i]], syms[surf_idx[j]]]))
            sites.append(("bridge", np.array([mid[0], mid[1], z_surf + 1.2]), env))

        # hollow: 相邻最近邻对构成的三角形质心
        nb_sorted = sorted(neighbors[i],
                           key=lambda j: np.arctan2(v[j][1], v[j][0]))
        for a, b in zip(nb_sorted, nb_sorted[1:] + nb_sorted[:1]):
            cent = surf_xy[i] + (v[a] + v[b]) / 3.0
            tri = (i, a, b)
            # 区分 fcc/hcp: 质心正下方是否有第二层原子
            is_hcp = any(np.linalg.norm(_min_image_2d(s, cent[:2], cell_xy)) < 0.6
                         for s in sub_xy)
            stype = "hcp" if is_hcp else "fcc"
            sites.append((stype, np.array([cent[0], cent[1], z_surf + 0.9]),
                          hollow_env(tri)))

    # 去重: 先把 (x, y) 收拢进周期单元胞, 再按 0.01 Å 舍入去重
    # (不 wrap 会把跨边界位点的周期镜像当成不同位点, 导致 127 个而非 96 个)
    seen = {}
    out = []
    inv = np.linalg.inv(cell_xy)
    for stype, p, env in sites:
        frac = p[:2] @ inv
        frac -= np.floor(frac)
        wxy = frac @ cell_xy
        key = (round(wxy[0], 2), round(wxy[1], 2))
        if key not in seen:
            seen[key] = True
            out.append((stype, np.array([wxy[0], wxy[1], p[2]]), env))
    return out


def _site_key(s):
    """位点的可哈希 key(用于去重/成员判断, 避免 numpy 数组真值歧义)"""
    stype, p, env = s
    return (stype, round(p[0], 3), round(p[1], 3), env)


def sample_sites(sites, n_sites, rng, per_type=None):
    """从枚举出的位点里采样 n_sites 个; per_type 可指定每类型数量"""
    by_type = {}
    for s in sites:
        by_type.setdefault(s[0], []).append(s)

    if per_type is not None:
        chosen = []
        for stype in SITE_TYPES:
            pool = by_type.get(stype, [])
            k = min(per_type.get(stype, 0), len(pool))
            if k:
                idx = rng.choice(len(pool), k, replace=False)
                chosen += [pool[t] for t in idx]
        return chosen

    # 默认: 按各类型数量比例分层采样
    chosen = []
    for stype, pool in by_type.items():
        k = int(round(n_sites * len(pool) / len(sites)))
        k = min(k, len(pool))
        if k:
            idx = rng.choice(len(pool), k, replace=False)
            chosen += [pool[t] for t in idx]
    # 补齐/裁剪到 n_sites
    if len(chosen) < n_sites:
        chosen_keys = {_site_key(s) for s in chosen}
        rest = [s for s in sites if _site_key(s) not in chosen_keys]
        need = n_sites - len(chosen)
        if len(rest) >= need:
            extra = [rest[t] for t in rng.choice(len(rest), need, replace=False)]
        else:
            extra = rest[:]                 # 位点数不足 n_sites 时有多少取多少(不再崩)
        chosen += extra
    chosen = chosen[:n_sites]
    rng.shuffle(chosen)
    return chosen


def classify_h_site(slab, sites, h_xyz):
    """弛豫后判定 H 的实际吸附位点。

    slab 在 relax_h 里刚性固定, 金属位置不变, 直接复用 enumerate_sites
    枚举出的全部位点; 用 2D 最小镜像距离把弛豫后的 H 分到最近的那个位点。
    返回 (act_type, act_env, h_height, d_xy):
        act_type  — 实际位点类型 top/bridge/fcc/hcp
        act_env   — 实际位点的局域化学环境(如 'Ru-Ni-Co')
        h_height  — H 相对表层高度 (Å)
        d_xy      — H 到该位点在 xy 面的最小镜像距离 (Å, 诊断量, 应≈0)
    """
    z_surf = slab.positions[:, 2].max()
    h_xy = h_xyz[:2]
    cell_xy = slab.cell[:2, :2].copy()

    best_type, best_env, best_d = None, None, np.inf
    for stype, p, env in sites:
        d = float(np.linalg.norm(_min_image_2d(h_xy, p[:2], cell_xy)))
        if d < best_d:
            best_d, best_type, best_env = d, stype, env
    return best_type, best_env, float(h_xyz[2] - z_surf), best_d


# ============================ 主流程 ============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if COMP_MODE == "pure+random_quinary":
        # 纯金属(5 个端点) + 随机五元比例(所有元素都>0)
        comps = [{e: (1.0 if el == e else 0.0) for e in ELEMENTS} for el in ELEMENTS]
        comps += random_quinary_compositions(ELEMENTS, N_QUINARY,
                                             MIN_QUINARY_FRAC, QUINARY_SEED)
    else:
        comps = grid_compositions(ELEMENTS, COMP_STEP)
    print(f"比例数量: {len(comps)}")
    print(f"总任务量: {len(comps)} 比例 × {N_CONFIGS} 排列 × {N_SITES} 位点 = "
          f"{len(comps) * N_CONFIGS * N_SITES} 个 H 弛豫\n")

    # 计算器只初始化一次
    calc = MACECalculator(model_paths=[MODEL_PATH], device=DEVICE,
                          default_dtype=DEFAULT_DTYPE)
    e_ref_h = half_h2_energy(calc)
    print(f"H2 参考能量 0.5·E(H2) = {e_ref_h:.4f} eV\n")

    # 位点采样方案
    if N_PER_TYPE is None:
        per_type = None
    else:
        per_type = N_PER_TYPE

    # 纯元素稳定相参考能量(各自稳定相 + 各自体积弛豫, 单位 eV/atom)
    e_ground = {}
    print("---- 纯元素稳定相参考能量(单位 eV/atom) ----")
    for el in ELEMENTS:
        e_ground[el] = pure_ground_energy(calc, el)
        print(f"  {el} ({STABLE_PHASE[el][0]}): {e_ground[el]:.4f} eV/atom")

    rows = []
    dh_rows = []          # 每个排列的形成焓: [comp, E_alloy_per_atom, dH_form]
    import csv
    header = ["comp", "comp_idx", "cfg", "site_idx", "site_type", "local_env",
              "act_site_type", "act_local_env", "h_height", "d_xy",
              "E_slab", "E_slabH", "E_ads", "dG_H"]

    # ---- 断点续跑: 读旧 CSV, 恢复已完成成分的行, 并跳过其 H 位点计算 -------
    done_comps = set()
    _str_cols = {"comp", "site_type", "local_env", "act_site_type", "act_local_env"}
    _int_cols = {"comp_idx", "cfg", "site_idx"}
    _full = N_CONFIGS * N_SITES                  # 每个成分应有的行数
    ci_col = header.index("comp_idx")
    if os.path.isfile(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8-sig") as f:
            rd = csv.reader(f)
            old_hdr = next(rd, None)
            if old_hdr == header:                # header 一致才续(防结构变更错位)
                loaded = list(rd)
                comp_count = {}
                for r in loaded:
                    c = int(r[ci_col])
                    comp_count[c] = comp_count.get(c, 0) + 1
                done_comps = {c for c, n in comp_count.items() if n == _full}
                # 只恢复"已完成"成分的行(不完整成分丢弃, 交主循环重算, 避免重复)
                for r in loaded:
                    if int(r[ci_col]) in done_comps:
                        rows.append([int(v) if h in _int_cols else
                                     (float(v) if h not in _str_cols else v)
                                     for h, v in zip(header, r)])
        if done_comps:
            print(f"断点续跑: 已有 {len(done_comps)} 个成分完成 {sorted(done_comps)}, "
                  f"将跳过其 H 位点(ΔH 仍重算)\n")

    t0 = time.time()
    for ci, comp in enumerate(comps):
        label = comp_label(comp)
        t_comp = time.time()
        print(f"\n>>> 比例 {ci + 1}/{len(comps)}: {label} 开始 "
              f"(累计 {time.time() - t0:.0f}s)")
        # 形成焓: 该比例的 bulk fcc 合金(体积弛豫) − 纯元素稳定相加权和
        at_bulk = build_bulk_alloy(comp, seed=ci * 100000)
        relax_bulk(at_bulk, calc)
        e_alloy_atom = at_bulk.get_potential_energy() / len(at_bulk)
        dH_form = e_alloy_atom - sum(comp[el] * e_ground[el] for el in ELEMENTS)
        dh_rows.append([label, e_alloy_atom, dH_form])

        if ci in done_comps:
            print(f">>> 比例 {ci + 1}/{len(comps)}: {label} 已完成, 跳过 H 位点")
            continue

        for cfg in range(N_CONFIGS):
            seed = ci * 100000 + cfg
            slab = build_slab(comp, seed=seed)
            fix_layers(slab, N_FIX_LAYERS)
            relax(slab, calc)
            E_slab = slab.get_potential_energy()

            # 保存弛豫后的 VASP 结构, 供 hea_postprocess.py 后处理
            # (晶格畸变 / 局域化学环境 / 表面粗糙度)
            if SAVE_STRUCTURES and (SAVE_MAX_CONFIGS is None or cfg < SAVE_MAX_CONFIGS):
                d = os.path.join(STRUCT_DIR, label)
                os.makedirs(d, exist_ok=True)
                write(os.path.join(d, f"cfg{cfg:03d}.vasp"), slab,
                      format="vasp", sort=True)

            rng = np.random.default_rng(seed + 999)
            sites = enumerate_sites(slab)
            chosen = sample_sites(sites, N_SITES, rng, per_type)

            for si, (stype, p, env) in enumerate(chosen):
                sh = relax_h(slab, calc, p)      # 刚性 slab + 只弛豫 H
                E_sh = sh.get_potential_energy()
                E_ads = E_sh - E_slab - e_ref_h
                # 判定弛豫后 H 的实际位点(可能从 bridge 漂到 fcc/hcp 等)
                act_type, act_env, h_height, d_xy = classify_h_site(
                    slab, sites, sh.positions[-1])
                # 保存弛豫后的吸附模型 slab+H, 供人工检查 H 位置是否合理
                if SAVE_SLABH:
                    dh = os.path.join(SLABH_DIR, label)
                    os.makedirs(dh, exist_ok=True)
                    fname = f"cfg{cfg:03d}_site{si:02d}_{act_type}.vasp"
                    if act_type != stype:          # H 发生位点迁移, 文件名标出
                        fname = f"cfg{cfg:03d}_site{si:02d}_{stype}_to_{act_type}.vasp"
                    write(os.path.join(dh, fname), sh, format="vasp", sort=True)
                rows.append([label, ci, cfg, si, stype, env,
                             act_type, act_env, round(h_height, 3), round(d_xy, 3),
                             round(E_slab, 4), round(E_sh, 4), round(E_ads, 4),
                             round(E_ads + DG_ZPE, 4)])

            print(f"  [{label}] 排列 {cfg + 1}/{N_CONFIGS} 完成 "
                  f"({time.time() - t_comp:.0f}s)")
        print(f">>> 比例 {ci + 1}/{len(comps)}: {label} 完成 "
              f"(本比例 {time.time() - t_comp:.0f}s, "
              f"预计还剩约 {(time.time() - t_comp) * (len(comps) - ci - 1):.0f}s)")

        # 每完成一个比例, 增量写 CSV(防中途崩溃丢数据)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    # ---- 汇总 μ / σ ----
    import pandas as pd
    df = pd.DataFrame(rows, columns=header)
    stats = df.groupby("comp")["E_ads"].agg(
        n="count", mu="mean", sigma="std", vmin="min", vmax="max").reset_index()
    stats_by_type = df.groupby(["comp", "site_type"])["E_ads"].agg(
        n="count", mu="mean", sigma="std").reset_index()
    stats_by_act_type = df.groupby(["comp", "act_site_type"])["E_ads"].agg(
        n="count", mu="mean", sigma="std").reset_index()

    stats.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
    stats_by_type.to_csv(os.path.join(OUT_DIR, "hea_h_stats_by_type.csv"),
                         index=False, encoding="utf-8-sig")
    stats_by_act_type.to_csv(os.path.join(OUT_DIR, "hea_h_stats_by_act_type.csv"),
                             index=False, encoding="utf-8-sig")

    # ---- 形成焓汇总(每种比例一个值, 对多个排列取平均) ----
    dh_df = pd.DataFrame(dh_rows, columns=["comp", "E_alloy_per_atom", "dH_form"])
    dh_stats = dh_df.groupby("comp")[["E_alloy_per_atom", "dH_form"]].mean().reset_index()
    dh_stats.to_csv(os.path.join(OUT_DIR, "hea_formation_enthalpy.csv"),
                    index=False, encoding="utf-8-sig")
    print("\n各比例形成焓 ΔH (eV/atom, 稳定相参考):")
    print(dh_stats.to_string(index=False))

    print("\n===== 完成 =====")
    print(f"原始数据: {OUT_CSV}")
    print(f"每个比例的 μ/σ: {OUT_STATS}")
    print("\n各比例 H 吸附能统计(μ 均值 / σ 标准差):")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
