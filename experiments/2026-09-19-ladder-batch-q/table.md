# Ladder tables (results/*.json; `python plot.py` redraws curves.png)

## 6 skills, easy (noise .1, budget 600, 100 seeds)

| arm | final (true score of best) | reach 0.9 | median rounds to 0.9 | median rounds total | mean pool | sibling pairs |
|---|---|---|---|---|---|---|
| gepa q=1 | 0.871 ± 0.014 | 52% | 21.0 | 30 | 13 | 0 |
| gepa q=2 | 0.826 ± 0.015 | 33% | 13 | 17 | 13 | 206 |
| gepa-ei q=1 | 0.847 ± 0.016 | 48% | 23.5 | 33 | 14 | 0 |
| gepa-ei q=2 q-EI | 0.856 ± 0.014 | 44% | 13.0 | 17 | 14 | 201 |
| gepa-ei q=2 top-2 | 0.865 ± 0.014 | 47% | 13 | 17 | 14 | 376 |

Paired: gepa q2 − q1 = -0.046 ± 0.020; gepa-ei q2 − q1 = +0.009 ± 0.023; gepa-ei q2 − gepa q2 = +0.030 ± 0.017; top-2 − q-EI = +0.009 ± 0.020.

## 6 skills, easy, budget 2000 (100 seeds) — saturates; rounds only

| arm | final (true score of best) | reach 0.9 | median rounds to 0.9 | median rounds total | mean pool | sibling pairs |
|---|---|---|---|---|---|---|
| gepa q=1 | 1.000 ± 0.000 | 100% | 27.0 | 332 | 25 | 0 |
| gepa q=2 | 1.000 ± 0.000 | 100% | 17.0 | 257 | 28 | 417 |
| gepa-ei q=1 | 0.998 ± 0.001 | 100% | 36.0 | 254 | 45 | 0 |
| gepa-ei q=2 q-EI | 1.000 ± 0.000 | 100% | 17.0 | 110 | 43 | 2571 |
| gepa-ei q=2 top-2 | 1.000 ± 0.000 | 100% | 16.0 | 124 | 35 | 5943 |

## 6 skills, hard (noise .3, p_informed .25, budget 2000, 100 seeds)

| arm | final (true score of best) | reach 0.9 | median rounds to 0.9 | median rounds total | mean pool | sibling pairs |
|---|---|---|---|---|---|---|
| gepa q=1 | 0.957 ± 0.008 | 79% | 61 | 116 | 43 | 0 |
| gepa q=2 | 0.939 ± 0.009 | 68% | 35.5 | 59 | 43 | 454 |
| gepa-ei q=1 | 0.915 ± 0.014 | 68% | 71.0 | 155 | 52 | 0 |
| gepa-ei q=2 q-EI | 0.952 ± 0.013 | 85% | 32 | 73 | 44 | 965 |
| gepa-ei q=2 top-2 | 0.974 ± 0.007 | 88% | 29.5 | 78 | 44 | 2104 |

Paired: gepa q2 − q1 = -0.018 ± 0.012; gepa-ei q2 − q1 = +0.036 ± 0.018; gepa-ei q2 − gepa q2 = +0.013 ± 0.016; top-2 − q-EI = +0.022 ± 0.014.

## 12 skills (noise .1, budget 2000, 30 seeds) — q = 1, 2, 4

| arm | final (true score of best) | reach 0.9 | median rounds to 0.9 | median rounds total | mean pool | sibling pairs / q-rounds |
|---|---|---|---|---|---|---|
| gepa q=1 | 0.833 ± 0.023 | 37% | 103 | 119 | 42 | 0/0 |
| gepa q=2 | 0.744 ± 0.022 | 13% | 40.0 | 58 | 41 | 133/1695 |
| gepa q=4 | 0.660 ± 0.018 | 0% | — | 31 | 42 | 306/890 |
| gepa-ei q=1 | 0.729 ± 0.030 | 20% | 119.5 | 144 | 50 | 0/0 |
| gepa-ei q=2 q-EI (local) | 0.864 ± 0.025 | 53% | 54.0 | 66 | 43 | 290/1923 |
| gepa-ei q=2 q-EI (hosted) | 0.900 ± 0.027 | 67% | 53.5 | 66 | 43 | 241/1994 |
| gepa-ei q=4 q-EI (local) | 0.845 ± 0.021 | 37% | 31 | 35 | 43 | 468/917 |
| gepa-ei q=4 top-4 | 0.831 ± 0.023 | 27% | 27.0 | 34 | 42 | 603/924 |
| gepa-ei q=4 q-EI (hosted) | 0.775 ± 0.023 | 20% | 24.5 | 31 | 43 | 475/875 |

Paired differences (30 seeds):

- gepa q2 − gepa q1: -0.090 ± 0.032
- gepa q4 − gepa q1: -0.173 ± 0.029
- gepa-ei q1 − gepa q1: -0.105 ± 0.033
- gepa-ei q2 q-EI − gepa-ei q1: +0.135 ± 0.038
- gepa-ei q4 q-EI − gepa-ei q1: +0.117 ± 0.034
- gepa-ei q4 q-EI − gepa-ei q2 q-EI: -0.018 ± 0.037
- gepa-ei q2 q-EI − gepa q2: +0.120 ± 0.034
- gepa-ei q4 q-EI − gepa q4: +0.185 ± 0.027
- gepa-ei q4 q-EI − gepa q1: +0.012 ± 0.031
- gepa-ei q4 q-EI − top-4: +0.015 ± 0.031
- hosted q2 − local q2: +0.036 ± 0.041
- hosted q4 − local q4: -0.071 ± 0.031
- hosted q4 − hosted q2: -0.125 ± 0.040
