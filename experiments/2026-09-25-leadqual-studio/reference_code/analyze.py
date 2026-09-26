"""Ceiling checks and trap checks for the lead-qualification dataset -> report.md.

Reads the published files plus sealed/ (for the true labels). Every model is seeded; CV is 5-fold stratified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

OUT = Path(__file__).resolve().parent
LABELS = ["accepted", "rejected_fit", "rejected_no_intent", "existing_customer", "partner_route"]
SEED = 7
SKF = StratifiedKFold(5, shuffle=True, random_state=SEED)
STAGES = ["Bootstrapped", "Seed", "Series A", "Series B", "Series C", "Series D+", "PE-backed", "Public"]


def structured(df, partners, with_source_title=False):
    """Dense numeric matrix from the enrichment columns (+ optionally source, title words, partner flag)."""
    X = pd.DataFrame(index=df.index)
    hc = df.snap_headcount.astype(float)
    X["log_hc"] = np.log(hc)
    for lo, hi in [(0, 20), (20, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 1e9)]:
        X[f"hc_{lo}"] = ((hc >= lo) & (hc < hi)).astype(float)
    X = X.join(pd.get_dummies(df.snap_industry, prefix="ind", dtype=float))
    X = X.join(pd.get_dummies(df.snap_funding_stage.fillna("none"), prefix="stage", dtype=float))
    m = df.snap_months_since_funding.astype(float)
    X["months_missing"] = m.isna().astype(float)
    m = m.fillna(m.median())
    for lo, hi in [(0, 6), (6, 12), (12, 24), (24, 1e9)]:
        X[f"m_{lo}"] = ((m > lo - 1) & (m <= hi)).astype(float) * (1 - X.months_missing)
    X["stack_missing"] = df.snap_tech_stack.isna().astype(float)
    X = X.join(df.snap_tech_stack.fillna("").str.get_dummies(sep=";").add_prefix("tool_").astype(float))
    X["hiring"] = df.snap_hiring_signal.astype(float)
    X["existing"] = df.is_existing_customer.astype(float)
    if with_source_title:
        X = X.join(pd.get_dummies(df.source, prefix="src", dtype=float))
        X["partner"] = df.is_partner.astype(float)
    return X


def title_tfidf():
    return TfidfVectorizer(token_pattern=r"[A-Za-z&]+", lowercase=True, ngram_range=(1, 2), min_df=3)


def cv_eval(make_X, y, model="lr", y_true=None):
    """make_X(train_idx, test_idx) -> (Xtr, Xte). Returns out-of-fold predictions and scores."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in SKF.split(np.zeros(len(y)), y):
        Xtr, Xte = make_X(tr, te)
        if model == "lr":
            clf = LogisticRegression(C=1.0, max_iter=4000)
        else:
            clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=.05)
            Xtr, Xte = (x.toarray() if sparse.issparse(x) else x for x in (Xtr, Xte))
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
    r = {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, average="macro", labels=LABELS)}
    if y_true is not None:
        r["acc_true"] = accuracy_score(y_true, pred)
    return r, pred


def scaled(X):
    X = X.to_numpy(float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def main():
    df = pd.read_csv(OUT / "leads.csv", keep_default_na=True)
    df["message_text"] = df.message_text.fillna("")
    sealed = pd.read_csv(OUT / "sealed" / "true_labels.csv")
    live = pd.read_csv(OUT / "live_enrichment.csv")
    routed = pd.read_csv(OUT / "labels_routed.csv")
    meta = json.loads((OUT / "sealed" / "generation_meta.json").read_text())
    partners = None
    assert (df.lead_id == sealed.lead_id).all() and (df.lead_id == live.lead_id).all()
    y = df.sdr_disposition.to_numpy()
    yt = sealed.true_label.to_numpy()
    n = len(df)
    lines = []
    w = lines.append

    w("# Lead-qualification synthetic dataset: report\n")
    gs = meta["gen_stats"]
    w(f"N = {n} leads, created {df.created_at.min()[:10]} to {df.created_at.max()[:10]}. Seed {meta['seed']}. "
      f"Message text: **{meta['text']}**" + (f" ({meta['model']} via {gs.get('backend')}, "
      f"{gs['calls_new'] + gs['calls_cached']} batched calls, {gs['in_all']:,} input / {gs['out_all']:,} output tokens, "
      f"total generation cost ${meta['cost_total_usd']:.2f}; this run spent ${meta['cost_new_usd']:.2f}, the rest came from cache)."
                                                   if meta["text"] == "llm" else " (template stub: NOT publishable)."))
    w("")

    # ---------------- class balance + rater noise
    w("## 1. Class balance and injected rater noise\n")
    w("| label | target | true | SDR (published) |\n|---|---:|---:|---:|")
    tgt = {"accepted": 18, "rejected_fit": 40, "rejected_no_intent": 30, "existing_customer": 7, "partner_route": 5}
    for l in LABELS:
        w(f"| {l} | {tgt[l]}% | {(yt == l).mean():.1%} | {(y == l).mean():.1%} |")
    flip = (yt != y).mean()
    w(f"\nRater flips: **{flip:.1%}** of leads. Confusion matrix, rows = true disposition, columns = SDR label (counts):\n")
    cm = confusion_matrix(yt, y, labels=LABELS)
    short = ["acc", "rej_fit", "rej_no_int", "existing", "partner"]
    w("| true \\ SDR | " + " | ".join(short) + " |\n|---|" + "---:|" * 5)
    for l, row in zip(short, cm):
        w(f"| {l} | " + " | ".join(str(v) for v in row) + " |")
    ai = cm[0, 2] + cm[2, 0]
    w(f"\naccepted <-> rejected_no_intent accounts for {ai} of {(yt != y).sum()} flips ({ai / (yt != y).sum():.0%}).\n")

    # ---------------- ceilings and baselines
    w("## 2. Ceilings and baselines (5-fold stratified CV against the published SDR labels)\n")
    rows = []

    def add(name, r, note=""):
        rows.append((name, r["acc"], r["f1"], r.get("acc_true", np.nan), note))

    nm = (sealed.msg_kind == "normal").to_numpy()
    lp = np.empty(nm.sum(), dtype=int)
    lv = sealed.msg_level.to_numpy()[nm]
    tm = df.message_text[nm].reset_index(drop=True)
    for tr, te in SKF.split(np.zeros(nm.sum()), lv):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tm.iloc[tr])
        lp[te] = LogisticRegression(max_iter=3000).fit(vec.transform(tm.iloc[tr]), lv[tr]).predict(vec.transform(tm.iloc[te]))
    level_r = np.corrcoef(lp, lv)[0, 1]

    add("True generating rule, with its noise (sealed true_label)",
        {"acc": accuracy_score(y, yt), "f1": f1_score(y, yt, average="macro"), "acc_true": 1.0},
        "ceiling: only rater noise stands between it and the labels")
    rn = sealed.rule_noiseless.to_numpy()
    add("Generating rule on the latents, no logistic noise",
        {"acc": accuracy_score(y, rn), "f1": f1_score(y, rn, average="macro"), "acc_true": accuracy_score(yt, rn)},
        "Bayes-ish ceiling for any reader of the inputs")

    base_counts = pd.Series(y).value_counts()
    add("Majority class", {"acc": base_counts.iloc[0] / n, "f1": np.nan, "acc_true": (yt == base_counts.index[0]).mean()})

    Xs = scaled(structured(df, partners))
    r_snap, _ = cv_eval(lambda tr, te: (Xs[tr], Xs[te]), y, y_true=yt)
    add("Structured only: LR on snap_* + is_existing_customer", r_snap)

    Xa = scaled(structured(df, partners, with_source_title=True))

    def with_title(tr, te, text=False, extra=None, base=None):
        B = Xa if base is None else base
        tv = title_tfidf().fit(df.title.iloc[tr])
        parts_tr = [sparse.csr_matrix(B[tr]), tv.transform(df.title.iloc[tr])]
        parts_te = [sparse.csr_matrix(B[te]), tv.transform(df.title.iloc[te])]
        if text:
            mv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
            mv.fit(df.message_text.iloc[tr])
            parts_tr.append(mv.transform(df.message_text.iloc[tr]) * 1.5)
            parts_te.append(mv.transform(df.message_text.iloc[te]) * 1.5)
            parts_tr.append(sparse.csr_matrix((df.message_text.iloc[tr].str.len() == 0).to_numpy(float)[:, None]))
            parts_te.append(sparse.csr_matrix((df.message_text.iloc[te].str.len() == 0).to_numpy(float)[:, None]))
        if extra is not None:
            parts_tr.append(sparse.csr_matrix(extra[tr])); parts_te.append(sparse.csr_matrix(extra[te]))
        return sparse.hstack(parts_tr).tocsr(), sparse.hstack(parts_te).tocsr()

    r_all, _ = cv_eval(with_title, y, y_true=yt)
    add("All structured: + source, title TF-IDF, is_partner", r_all)
    r_txt, _ = cv_eval(lambda tr, te: with_title(tr, te, text=True), y, y_true=yt)
    add("Text + structured: + TF-IDF on message_text", r_txt)

    w("Two ceilings. The true label includes logistic noise no input reveals, so only an oracle reaches the first row; "
      "the second applies the same rule to the noise-free latents, the best a reader of every input could hope for "
      "(and still above reach: headcount, stack and title are only partly visible in the columns).\n")
    w("| model | acc vs SDR | macro-F1 vs SDR | acc vs true label | |\n|---|---:|---:|---:|---|")
    for name, a, f, at, note in rows:
        w(f"| {name} | {a:.3f} | {'' if np.isnan(f) else f'{f:.3f}'} | {'' if np.isnan(at) else f'{at:.3f}'} | {note} |")
    ceil = rows[0][1]
    bayes = rows[1][1]
    w("")
    w(f"- Gap, structured-only (snap_*) to the true-rule ceiling: **{(ceil - r_snap['acc']) * 100:.1f} points** "
      f"(requirement: more than 2). To the no-noise rule: {(bayes - r_snap['acc']) * 100:.1f} points.")
    w(f"- Gap, all structured to text+structured: **{(r_txt['acc'] - r_all['acc']) * 100:.1f} points** against SDR labels, "
      f"**{(r_txt['acc_true'] - r_all['acc_true']) * 100:.1f} points** against the true labels. The text's job is to "
      f"separate accepted from rejected_no_intent, which is exactly where the rater noise sits, so part of what it "
      f"gets right is scored wrong. TF-IDF is a floor for the text: it recovers each message's latent intent level "
      f"with r = {level_r:.2f}, and a reader that understands the text should do better.")
    w(f"- Headroom above the TF-IDF baseline to the ceiling: {(ceil - r_txt['acc']) * 100:.1f} points.")
    se = lambda p, m: np.sqrt(p * (1 - p) / m)
    m = int(round(n * .2))
    w(f"- Standard error of accuracy on a 20% hold-out (n = {m}): {se(r_txt['acc'], m):.4f} at the text+structured "
      f"accuracy, {se(0.5, m):.4f} worst case (p = 0.5). Requirement < 0.03: "
      f"**{'met' if se(0.5, m) < .03 else 'NOT met'}**. A difference between two prompts scored on the same "
      f"hold-out needs roughly 2 x {se(0.5, m):.3f} x sqrt(2) = {2 * se(0.5, m) * np.sqrt(2):.3f} to clear two "
      f"standard errors if the errors were independent; paired comparison on the same leads is tighter.\n")

    # ---------------- traps
    w("## 3. Trap checks\n")
    w("### 3.1 Redundant field: email_domain\n")
    d1 = df.groupby("company").email_domain.nunique().max()
    d2 = df.groupby("email_domain").company.nunique().max()
    w(f"Max distinct domains per company: {d1}; max distinct companies per domain: {d2}. "
      f"{'Company and email_domain are a 1:1 mapping' if d1 == d2 == 1 else 'NOT 1:1'} "
      f"({df.company.nunique()} companies across {n} leads).\n")

    w("### 3.2 Leakage field: sdr_notes_len\n")
    z = df.sdr_notes_len
    w(f"Zero for {(z == 0).mean():.0%} of leads. Median by SDR label: " +
      ", ".join(f"{l} {int(z[y == l].median())}" for l in LABELS) + ".\n")
    Xn = np.c_[np.log1p(z.to_numpy()), (z == 0).to_numpy(float)]
    Xn = (Xn - Xn.mean(0)) / Xn.std(0)
    r_n, _ = cv_eval(lambda tr, te: (Xn[tr], Xn[te]), y, y_true=yt)
    r_leak, _ = cv_eval(lambda tr, te: with_title(tr, te, text=True, extra=Xn), y, y_true=yt)
    w(f"sdr_notes_len alone: accuracy {r_n['acc']:.3f} (majority class {base_counts.iloc[0] / n:.3f}). "
      f"Adding it to text+structured: {r_txt['acc']:.3f} -> **{r_leak['acc']:.3f}** "
      f"({(r_leak['acc'] - r_txt['acc']) * 100:+.1f} points from one field that does not exist at routing time). "
      f"A validator should flag it: it is written after the label is decided, by the person who decides it.\n")

    w("### 3.3 Frozen vs live enrichment\n")
    L2 = df.copy()
    for c in [c for c in live.columns if c.startswith("snap_")] + ["is_existing_customer"]:
        L2[c] = live[c].to_numpy()
    acc_now = (live.is_existing_customer & ~df.is_existing_customer)
    w(f"Live values as of {live.enriched_at.iloc[0]}. Leads newly shown as existing customers: {acc_now.sum()}, of which "
      f"{(acc_now & (y == 'accepted')).sum()} were SDR-accepted "
      f"({(acc_now & (y == 'accepted')).sum() / (y == 'accepted').sum():.1%} of accepted leads). "
      f"Median headcount change: {np.median(L2.snap_headcount / df.snap_headcount - 1):+.1%} overall, "
      f"{np.median((L2.snap_headcount / df.snap_headcount - 1)[y == 'accepted']):+.1%} for accepted leads. "
      f"`{'Tallyforge'}` appears in the live tech stack of {L2.snap_tech_stack.fillna('').str.contains('Tallyforge').sum()} leads.\n")
    w("| features | frozen: acc / macro-F1 | live re-fetch: acc / macro-F1 | inflation (acc) |\n|---|---:|---:|---:|")
    snap_cols = [c for c in live.columns if c.startswith("snap_")]
    for text in (False, True):
        for label, cols in [("snap_* re-fetched", snap_cols),
                            ("snap_* + is_existing_customer re-fetched", snap_cols + ["is_existing_customer"])]:
            L3 = df.copy()
            for c in cols:
                L3[c] = live[c].to_numpy()
            Xf, Xl = structured(df, partners, True).align(structured(L3, partners, True), join="outer", axis=1,
                                                         fill_value=0)
            Xf, Xl = scaled(Xf), scaled(Xl)
            rf, _ = cv_eval(lambda tr, te: with_title(tr, te, text=text, base=Xf), y)
            rl, _ = cv_eval(lambda tr, te: with_title(tr, te, text=text, base=Xl), y)
            w(f"| {'text + structured' if text else 'all structured'}, {label} | {rf['acc']:.3f} / {rf['f1']:.3f} | "
              f"{rl['acc']:.3f} / {rl['f1']:.3f} | **{(rl['acc'] - rf['acc']) * 100:+.1f} pts** |")
    w("\nThe live values were fetched after the SDR decided, so they encode the outcome (accounts that bought grew, "
      "raised, started hiring, and show the vendor in their stack). Scoring a prompt against re-fetched values "
      "overstates what it can do at routing time.\n")

    w("### 3.4 Selection bias: labels only where a naive router passed\n")
    lab = routed.sdr_disposition.notna()
    w(f"Router rule: (source is demo_request or referral, or seniority Head/Director/VP/C-level) and headcount >= 50. "
      f"Routed: {routed.routed.mean():.0%} of leads; exploration slice: {routed.exploration.sum()} leads "
      f"({routed.exploration.sum() / (~routed.routed).sum():.1%} of the rest); labelled in total: {lab.mean():.0%}.\n")
    w("| label | all leads (SDR) | labelled subset | labelled subset, IPW |\n|---|---:|---:|---:|")
    wts = 1 / routed.label_propensity[lab]
    for l in LABELS:
        s = (routed.sdr_disposition[lab] == l)
        w(f"| {l} | {(y == l).mean():.1%} | {s.mean():.1%} | {(s * wts).sum() / wts.sum():.1%} |")
    # train on the labelled subset, score on (a) labelled subset CV, (b) everything unlabelled
    idx_lab = np.where(lab)[0]; idx_un = np.where(~lab)[0]
    Xtr, Xte = with_title(idx_lab, idx_un, text=True)
    clf = LogisticRegression(C=1.0, max_iter=4000).fit(Xtr, y[idx_lab])
    acc_un = accuracy_score(y[idx_un], clf.predict(Xte))
    sub = df.iloc[idx_lab].reset_index(drop=True)
    y_sub = y[idx_lab]
    pred_sub = np.empty(len(sub), dtype=object)
    for tr, te in SKF.split(np.zeros(len(sub)), y_sub):
        a_tr, a_te = with_title(idx_lab[tr], idx_lab[te], text=True)
        pred_sub[te] = LogisticRegression(C=1.0, max_iter=4000).fit(a_tr, y_sub[tr]).predict(a_te)
    acc_sub = accuracy_score(y_sub, pred_sub)
    w(f"\nText+structured LR trained on the labelled subset: CV accuracy **within** the labelled subset {acc_sub:.3f}; "
      f"accuracy on the {len(idx_un)} unlabelled leads (true SDR labels, which the file hides) **{acc_un:.3f}**. "
      f"The labelled subset over-represents demo requests and referrals, so an evaluation that uses only it "
      f"measures the wrong population. `label_propensity` (1.0 routed, 0.05 exploration) supports inverse-propensity "
      f"weighting of the exploration slice.\n")

    # ---------------- message text sanity
    w("## 4. Message text\n")
    t = df.message_text
    wc = t.str.split().str.len().fillna(0)
    w(f"Empty: {(t == '').mean():.0%}. Words among non-empty: median {int(wc[t != ''].median())}, "
      f"5th pct {int(wc[t != ''].quantile(.05))}, 95th pct {int(wc[t != ''].quantile(.95))}, max {int(wc.max())}. "
      f"Messages naming a competitor: {t.str.contains('|'.join(['Forecastly', 'PipeRadar', 'Quotient IQ', 'Closewise', 'Revhawk']), case=False).mean():.0%}.\n")
    w("Mean text_score (sealed) by SDR label, the signal the text is meant to carry:\n")
    w("| SDR label | mean text_score | share empty |\n|---|---:|---:|")
    for l in LABELS:
        w(f"| {l} | {sealed.text_score[y == l].mean():+.2f} | {(t[y == l] == '').mean():.0%} |")
    w("\nRandom sample (seeded):\n")
    samp = df[t != ""].sample(8, random_state=SEED)
    for _, r in samp.iterrows():
        msg = r.message_text.replace("\n", " ").replace("|", "/")
        w(f"- *{r.source}, {r.title}, {r.snap_industry}, {r.snap_headcount} employees*: \"{msg}\"")
    w("")
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
