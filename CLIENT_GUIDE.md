# Using the batch-suggestion API from another project

This is the only document another project needs. It describes the HTTP
contract; how the scores are computed is not part of the contract and is not
described here.

## What it does

You have a fixed, finite set of things you *could* try next (the candidate
pool), and a handful of things you have already tried with a numeric result.
One POST returns the `q` candidates from the pool that are the best joint
next batch to evaluate. Send the results back next round; repeat.

## Endpoint

- `POST <MODAL_BO_API_URL>` with a JSON body. No authentication.
- Put the URL in an environment variable (`MODAL_BO_API_URL`); it changes
  whenever the service is redeployed.
- First call after idle takes 20–30 s (container start). Warm calls are
  1–4 s. Set a client timeout of at least 120 s and treat one retry as normal.

## Request

```json
{
  "X":          [[0.1, 0.4], [0.7, 0.2], ...],   // evaluated points, one row each
  "y":          [0.83, 0.61, ...],               // their results, HIGHER = BETTER
  "candidates": [[0.3, 0.3], [0.9, 0.1], ...],   // the pool to choose from
  "q":          4                                // how many to return
}
```

Rules:

1. `X` and `candidates` must be in the **same coordinate space** (same
   columns, same meaning). Any numeric embedding works; the service rescales
   internally.
2. `y` is maximised. For a loss / error / cost, **negate it** before sending.
3. `y` needs no scaling or transform of any kind. Send raw values.
4. `candidates` is the complete menu: the response only ever contains members
   of it. Include already-evaluated points only if re-evaluating them is
   acceptable.
5. Around 5 rows of `X` is the practical minimum for useful suggestions;
   fewer is allowed.

Optional fields, all with sensible defaults — leave them out unless told
otherwise: `n_batches` (512), `n_prefilter` (10000), `train_steps` (60),
`lr` (0.1), `mode` (`"production"`; `"debug"` adds diagnostics).

## Response

```json
{
  "candidates": [
    {"index": 17, "x": [0.9, 0.1], "mu": 0.42, "sigma": 0.31},
    ...
  ]
}
```

- `index` is the row of `candidates` in *your* request — use it to look up
  whatever the row stands for. `x` echoes the row.
- `mu` / `sigma` are the model's mean and uncertainty for that candidate on an
  internal scale. Higher `mu` is better; they are for display, not for
  decisions.
- Exactly `q` entries (fewer only if the pool has fewer than `q` rows).

## Minimal client (no dependencies)

```python
import json, os, urllib.request

def suggest(X, y, candidates, q=4, timeout=180):
    body = json.dumps({"X": X, "y": y, "candidates": candidates, "q": q}).encode()
    req = urllib.request.Request(os.environ["MODAL_BO_API_URL"], data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return [c["index"] for c in json.load(r)["candidates"]]

# minimisation example: negate the loss
idx = suggest(X_done, [-loss for loss in losses], pool, q=4)
next_batch = [pool[i] for i in idx]
```

## Loop shape

```
repeat:
    idx  = suggest(X, y, pool, q)
    evaluate pool[idx] (in parallel if you like)
    append the new rows to X and the new results to y
```

One call per round, not one per candidate. Nothing is stored server-side;
every request is self-contained.

## Multi-platform / multi-modal requests

If your candidates come from two related sources that should share
information (e.g. two ad platforms), or each candidate carries separate text
and image vectors, there are additional fields for that. Ask; do not guess
at them.
