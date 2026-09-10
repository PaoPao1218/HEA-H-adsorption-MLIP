# data/

Output of `hea_screening.py` (Part 1) on 25 compositions (5 pure metals + 20 random quinary).

- `hea_h_adsorption.csv` — per-site H adsorption energies (2500 rows).
- `hea_formation_enthalpy.csv` — per-composition formation enthalpy (25 rows).

Shipped so Parts 2–4 run out of the box. To regenerate or extend them, run Part 1 (`hea_screening.py`); to point the scripts at your own directory instead, set `HEA_DATA_DIR`:

```bash
export HEA_DATA_DIR=/path/to/your/data
```
