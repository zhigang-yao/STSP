"""Final manuscript settings for the STSP/MS-STSP experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

CASES = (
    ("curve_circle", "circle", 5000, 0.08, 1),
    ("curve_tennis", "tennis", 5000, 0.08, 1),
    ("curve_trefoil", "trefoil", 5000, 0.08, 1),
    ("curve_sixfolds", "sixfolds", 5000, 0.08, 1),
    ("surface_sphere", "sphere", 20000, 0.08, 2),
    ("surface_torus", "torus", 20000, 0.08, 2),
    ("high_circle_s0p10", "circle", 5000, 0.10, 1),
    ("high_circle_s0p20", "circle", 5000, 0.20, 1),
    ("high_circle_s0p30", "circle", 5000, 0.30, 1),
)
CASE_BY_NAME = {entry[0]: entry for entry in CASES}
TMAX = 30
TOLERANCE = 1e-3
DENTATE_P = (20, 30, 40, 50)

# Recorded final h/sigma values and iteration counts, in replicate order 1--10.
FROZEN_SYNTHETIC = {
    "curve_circle": dict(st_h=[2.5]*10, st_t=[8,7,8,7,8,8,8,8,7,7],
                         ms_h=[(4,5,6)]*3+[(5,6,8)]*7, ms_t=[5]*3+[4]*7),
    "curve_tennis": dict(st_h=[2.5]*10, st_t=[8,8,8,7,8,7,7,8,8,8],
                         ms_h=[(4,5,6)]*10, ms_t=[5]*10),
    "curve_trefoil": dict(st_h=[4,3.5,3.5,3.5,3.5,4,3.5,4,3.5,4],
                          st_t=[5]*10, ms_h=[(5,6,8)]*10, ms_t=[4]*10),
    "curve_sixfolds": dict(st_h=[2,2,2,2,2.5,2,2,2,2,2],
                           st_t=[12,11,12,11,8,12,11,11,11,11],
                           ms_h=[(3,3.5,4)]*7+[(3.5,4,5)]+[(3,3.5,4)]*2,
                           ms_t=[6,7,7,7,6,6,7,6,6,7]),
    "surface_sphere": dict(st_h=[2]*10, st_t=[11]*8+[12,11],
                           ms_h=[(5,6,8)]*10, ms_t=[5]*10),
    "surface_torus": dict(st_h=[2]*10, st_t=[12]*10,
                          ms_h=[(3,3.5,4)]*10, ms_t=[8]*10),
    "high_circle_s0p10": dict(st_h=[2.5,2.5,2.5,2,2,2.5,2,2.5,2.5,2.5],
                              st_t=[8,7,8,10,11,8,11,7,7,8],
                              ms_h=[(3,3.5,4)]*10, ms_t=[6]*10),
    "high_circle_s0p20": dict(st_h=[1.5,2,1.5,1.5,1.5,2,2,2,2,1.5],
                              st_t=[18,11,16,17,18,10,11,11,11,15],
                              ms_h=[(3,3.5,4)]*5+[(2.5,3,3.5)]*2
                                   +[(3,3.5,4),(2.5,3,3.5),(3,3.5,4)],
                              ms_t=[6]*5+[8,8,6,8,6]),
    "high_circle_s0p30": dict(st_h=[1.5]*10,
                              st_t=[17,16,17,19,17,17,16,17,18,17],
                              ms_h=[(2.5,3,3.5)]+[(2,2.5,3)]
                                   +[(2.5,3,3.5)]*2+[(2,2.5,3)]*4
                                   +[(2.5,3,3.5),(2,2.5,3)],
                              ms_t=[9,11,9,9,11,11,11,11,10,11]),
}

# The Dentate Gyrus data are the 2,930-cell mouse single-cell dataset.
# Entries are (STSP radius, iterations, MS-STSP radii, iterations).
FROZEN_DENTATE = {
    20: (0.6944859228759096, 10,
         (0.6944859228759096, 1.478367512855324, 1.7195159277281062), 23),
    30: (0.6944859228759096, 11,
         (0.6944859228759096, 1.271038242695257, 1.7195159277281062), 16),
    40: (0.8077691071969223, 9,
         (0.8077691071969223, 1.271038242695257, 1.7195159277281062), 14),
    50: (0.8077691071969223, 11,
         (0.8077691071969223, 1.271038242695257, 1.7195159277281062), 19),
}
