"""Parent-offspring regression of full-evaluation scores over existing trees."""
import json, glob, math, sys, collections
import numpy as np

FAMILIES = {
    'hotpot_program (gepa+bo, gated)':  ('runs/hotpot_program/*/tree.json',  200, ['f1', 'sel_recall']),
    'hotpot_program3 (bo pure, no gate)': ('runs/hotpot_program3/*/tree.json', 200, ['f1', 'sel_recall']),
    'compress_v2 (gepa+bo, gated)':     ('runs/compress_v2/*/tree.json',     100, ['f1', 'template_tokens']),
}

def load(pat, nfull):
    runs = {}
    for f in sorted(glob.glob(pat)):
        t = json.load(open(f))
        nodes = {n['id']: n for n in t['nodes']}
        full = {i: n for i, n in nodes.items() if n.get('evaluation') and n['evaluation']['n'] >= nfull}
        root = next(n for n in nodes.values() if n['parent_id'] is None)
        runs[f.split('/')[-2]] = (nodes, full, root)
    return runs

def ols(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float('nan'), float('nan'), float('nan')
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    # SE of slope
    resid = y - (a + b * x)
    se = math.sqrt((resid @ resid) / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
    return b, se, r

def analyse(name, pat, nfull, metrics):
    runs = load(pat, nfull)
    print(f"\n=== {name}: {len(runs)} runs ===")
    # noise floor: same root prompt, same seed rows, evaluated independently in different run dirs
    seed_of = lambda rn: rn.split('-s')[-1]
    for m in metrics:
        by_seed = collections.defaultdict(list)
        rows_by_seed = collections.defaultdict(list)
        for rn, (nodes, full, root) in runs.items():
            by_seed[seed_of(rn)].append(root['evaluation']['metrics'][m])
            rows_by_seed[seed_of(rn)].append({r['example_id']: (r['metrics'] or {}).get(m, 0.0) for r in root['evaluation']['per_example']})
        sds = [np.std(v, ddof=1) for v in by_seed.values() if len(v) > 1]
        # per-row disagreement between the two/three repeats
        diffs = []
        for rows in rows_by_seed.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    ks = set(a) & set(b)
                    diffs.append(np.mean([abs(a[k] - b[k]) > 1e-9 for k in ks]))
        if sds:
            noise_sd = math.sqrt(np.mean(np.square(sds)))
            print(f"[{m}] noise floor from repeated root evals: {len(by_seed)} seeds x {[len(v) for v in by_seed.values()]} repeats; "
                  f"pooled SD = {noise_sd:.4f}  (per-seed SDs {', '.join(f'{s:.4f}' for s in sds)}); "
                  f"rows differing between identical evals: {np.mean(diffs):.1%}")
        else:
            noise_sd = float('nan'); print(f"[{m}] no repeated root evals")

        # parent-child pairs on full evals, centered on each run's root
        px, cy, run_of, dep = [], [], [], []
        best_pairs = []
        sib_groups = []
        n_full_total = 0
        for rn, (nodes, full, root) in runs.items():
            r0 = root['evaluation']['metrics'][m]
            n_full_total += len(full) - 1
            kids = collections.defaultdict(list)
            for i, n in full.items():
                p = n['parent_id']
                if p is not None and p in full:
                    ps = full[p]['evaluation']['metrics'][m] - r0
                    cs = n['evaluation']['metrics'][m] - r0
                    px.append(ps); cy.append(cs); run_of.append(rn); dep.append(n['depth'])
                    kids[p].append(cs)
            for p, cs in kids.items():
                ps = full[p]['evaluation']['metrics'][m] - r0
                best_pairs.append((ps, max(cs), len(cs)))
                if len(cs) > 1: sib_groups.append(cs)
        px, cy = np.array(px), np.array(cy)
        b, se, r = ols(px, cy)
        d = cy - px
        print(f"[{m}] {len(px)} parent->child full-eval pairs (of {n_full_total} non-root full evals); "
              f"parents: {len(best_pairs)}; parent-score SD {px.std(ddof=1):.4f}; child-score SD {cy.std(ddof=1):.4f}")
        from scipy.stats import spearmanr
        rho = spearmanr(px, cy).correlation
        sh = [i for i, dd in enumerate(dep) if dd <= 2]
        bs, ses, rs = ols(px[sh], cy[sh])
        print(f"[{m}]   child on parent (root-centered):  slope {b:+.3f} ± {se:.3f}   r = {r:+.3f}   Spearman ρ = {rho:+.3f}")
        print(f"[{m}]   depth ≤ 2 only ({len(sh)} pairs):       slope {bs:+.3f} ± {ses:.3f}   r = {rs:+.3f}")
        if not math.isnan(noise_sd) and px.std(ddof=1) > 0:
            rel = 1 - noise_sd**2 / px.var(ddof=1)
            print(f"[{m}]   reliability of parent score 1 - σ²_noise/σ²_parent = {rel:.2f}  -> disattenuated slope {b/rel if rel>0 else float('nan'):+.3f}")
        print(f"[{m}]   child - parent: mean {d.mean():+.4f}, SD {d.std(ddof=1):.4f}  (pure noise would give SD ≈ √2·σ_noise = {math.sqrt(2)*noise_sd:.4f}); "
              f"P(child > parent) = {np.mean(d > 0):.2f}; P(child > parent + 2σ_noise) = {np.mean(d > 2*noise_sd):.2f}")
        # sibling ICC (one-way ANOVA): variance between parents / total
        if len(sib_groups) >= 3:
            allv = np.concatenate(sib_groups); k = len(sib_groups)
            n0 = (len(allv) - sum(len(g)**2 for g in sib_groups) / len(allv)) / (k - 1)
            msb = sum(len(g) * (np.mean(g) - allv.mean())**2 for g in sib_groups) / (k - 1)
            msw = sum(((np.array(g) - np.mean(g))**2).sum() for g in sib_groups) / (len(allv) - k)
            icc = (msb - msw) / (msb + (n0 - 1) * msw)
            print(f"[{m}]   sibling ICC over {k} parents with ≥2 full-eval children ({len(allv)} children): {icc:+.3f}  "
                  f"(within-family SD {math.sqrt(msw):.4f})")
        bx = np.array([p for p, _, _ in best_pairs]); by = np.array([c for _, c, _ in best_pairs])
        b2, se2, r2 = ols(bx, by)
        print(f"[{m}]   best-of-children on parent: slope {b2:+.3f} ± {se2:.3f}  r = {r2:+.3f}; "
              f"mean(best child - parent) {np.mean(by-bx):+.4f}; children/parent median {np.median([k for *_, k in best_pairs]):.0f}")
        # by depth
        byd = collections.defaultdict(list)
        for p, c, dd in zip(px, cy, dep): byd[dd].append((p, c))
        print(f"[{m}]   pairs by child depth: " + ", ".join(f"d{dd}:{len(v)}" for dd, v in sorted(byd.items())))
        # per-run slopes
        pr = []
        for rn in runs:
            idx = [i for i, x in enumerate(run_of) if x == rn]
            if len(idx) >= 4:
                bb, _, rr = ols(px[idx], cy[idx]); pr.append(f"{rn}: n={len(idx)} b={bb:+.2f} r={rr:+.2f}")
        print(f"[{m}]   per run: " + " | ".join(pr))

for name, (pat, nfull, metrics) in FAMILIES.items():
    analyse(name, pat, nfull, metrics)
