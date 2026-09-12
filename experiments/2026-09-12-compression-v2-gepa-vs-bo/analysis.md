# runs/compress_v2: 12 complete seeds x ['gepa', 'bo']

## Threshold table - shortest candidate with train F1 >= root F1 - gap (tokens @ rollout found)

| gap | gepa tokens | gepa found@ | bo tokens | bo found@ | paired gepa-bo tokens | BO shorter / ties / n | union-front regret gepa / bo |
|---|---|---|---|---|---|---|---|
| 0.00 | 26.3 ± 6.6 | 1183 | 52.7 ± 11.5 | 649 | -26.3 ± 13.8 | 3 / 0 / 12 | 1.6 / 27.9 |
| 0.02 | 18.8 ± 1.3 | 1351 | 19.8 ± 3.4 | 1127 | -1.1 ± 3.6 | 4 / 0 / 12 | 4.7 / 5.8 |
| 0.05 | 15.8 ± 1.3 | 1254 | 12.0 ± 1.8 | 1245 | +3.8 ± 2.0 | 8 / 0 / 12 | 6.5 / 2.7 |
| 0.10 | 12.3 ± 1.4 | 1138 | 8.9 ± 1.6 | 1307 | +3.4 ± 2.2 | 9 / 1 / 12 | 5.2 / 1.8 |

## Rollout accounting (mean per run)

| arm | candidates | root | minibatch | full evals | of which advanced front | share of rollouts on non-advancing full evals |
|---|---|---|---|---|---|---|
| gepa | 21.1 | 100 | 178 | 2008 | 10.1 of 20.1 | 49% |
| bo | 19.8 | 100 | 184 | 1883 | 11.7 of 18.8 | 35% |

## Per seed, gap 0.05: shortest candidate

| seed | root F1 | gepa | bo |
|---|---|---|---|
| 0 | 0.967 | 19 tok, F1 0.967 @1616: `List all unique full names of people in: {text}, ignoring titles and o` | 6 tok, F1 0.948 @673: `Extract full names from: {text}` |
| 1 | 0.978 | 14 tok, F1 0.933 @935: `List unique full names from this text, ignoring titles: {text}` | 12 tok, F1 0.957 @813: `Extract full names from text, excluding titles: {text}` |
| 2 | 0.990 | 6 tok, F1 0.965 @507: `Extract human names from: {text}` | 12 tok, F1 0.948 @601: `Extract only personal names from this passage: {text}` |
| 3 | 0.990 | 18 tok, F1 0.978 @1010: `Extract all unique full names from: {text}, ignoring titles and organi` | 15 tok, F1 0.943 @1260: `Extract personal names from {text}, avoid titles and organizations.` |
| 4 | 0.987 | 14 tok, F1 0.960 @1107: `Extract names from: {text}. List only people, no titles or orgs.` | 8 tok, F1 0.947 @1824: `Extract full human names from: {text}` |
| 5 | 0.983 | 15 tok, F1 0.987 @1170: `List all names of people in this passage, excluding titles: {text}` | 8 tok, F1 0.965 @1533: `Extract all human names from: {text}` |
| 6 | 1.000 | 22 tok, F1 0.965 @1898: `Identify all people's full names mentioned in: {text}. List them once,` | 8 tok, F1 0.960 @1820: `Extract only person names from: {text}` |
| 7 | 0.983 | 16 tok, F1 0.966 @1096: `Extract full names from this text, omitting titles and entities: {text` | 6 tok, F1 0.965 @1190: `Identify human names in: {text}.` |
| 8 | 0.972 | 11 tok, F1 0.943 @1226: `Extract all full names of individuals from: {text}` | 15 tok, F1 0.955 @1035: `List all full person names from the following text.
Passage:
{text}` |
| 9 | 0.967 | 14 tok, F1 0.978 @896: `Extract full names from: {text}. Ignore titles and locations.` | 15 tok, F1 0.978 @1336: `List names in this text, excluding titles and organizations: {text}` |
| 10 | 0.995 | 23 tok, F1 0.997 @1521: `Identify all full names of individuals in this passage, avoiding title` | 29 tok, F1 0.960 @1648: `Identify and list all names of people mentioned in the text below, ign` |
| 11 | 0.995 | 18 tok, F1 0.958 @2060: `Identify all full names of people in this passage and list them in ord` | 10 tok, F1 0.990 @1212: `Extract full names from: {text}. Exclude titles.` |
