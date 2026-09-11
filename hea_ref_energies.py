# -*- coding: utf-8 -*-
"""计算 5 个元素的"纯元素参考能", 用于严格 ΔHmix(同晶格混合焓)。

dH_form(hea_formation_enthalpy.csv) 相对的是元素稳定相基态:
  Ru/Co → hcp, Fe → bcc, Ni/Cu → fcc(即 STABLE_PHASE)。
严格 ΔHmix 相对的是"同 fcc 晶格"的纯元素(即假设每个元素都被放进 fcc 晶格),
二者差一个"结构项" ΔE_struct_i = E_ground_i − E_fcc_i ≤ 0(基态比 fcc 稳多少):

    ΔHmix = dH_form + Σ c_i·ΔE_struct_i

本脚本用 MACE 算 E_ground(复用 hea_screening.pure_ground_energy, 与已算 dH_form 同源)
与 E_fcc(纯元素 fcc bulk 各向同性弛豫), 存 data/hea_pure_reference.csv,
供 hea_formability.py 读入后算严格 Ω。

用法:
    export MACE_MODEL_PATH=/path/to/2023-12-03-mace-128-L1_epoch-199.model
    python hea_ref_energies.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

from hea_screening import (ELEMENTS, A_LATTICE, FMAX, pure_ground_energy,
                           MODEL_PATH, DEVICE, DEFAULT_DTYPE)
from mace.calculators import MACECalculator

DATA_DIR = os.environ.get(
    "HEA_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT_CSV = os.path.join(DATA_DIR, "hea_pure_reference.csv")


def fcc_energy(calc, el):
    """纯元素放进 fcc 晶格(从合金晶格常数 A_LATTICE 出发, 体积+原子弛豫)的每原子能量。"""
    at = bulk(el, "fcc", a=A_LATTICE)
    at.calc = calc
    opt = BFGS(FrechetCellFilter(at, hydrostatic_strain=True), logfile=None)
    opt.run(fmax=FMAX)
    return at.get_potential_energy() / len(at)


def main():
    calc = MACECalculator(model_paths=[MODEL_PATH], device=DEVICE,
                          default_dtype=DEFAULT_DTYPE)
    rows = []
    print(f"模型: {MODEL_PATH}  (device={DEVICE}, dtype={DEFAULT_DTYPE})")
    print(f"FMAX={FMAX}, fcc 起始晶格常数 A_LATTICE={A_LATTICE}\n")
    for el in ELEMENTS:
        eg = pure_ground_energy(calc, el)          # 稳定相基态(与 dH_form 同源)
        ef = fcc_energy(calc, el)                  # 同 fcc 晶格参考
        rows.append([el, eg, ef, eg - ef])
        print(f"{el:>3}  E_ground={eg:+.4f}  E_fcc={ef:+.4f}  "
              f"ΔE_struct={eg - ef:+.4f} eV/atom")
    import pandas as pd
    df = pd.DataFrame(rows, columns=["el", "E_ground", "E_fcc", "dE_struct"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n写入", OUT_CSV)


if __name__ == "__main__":
    main()
