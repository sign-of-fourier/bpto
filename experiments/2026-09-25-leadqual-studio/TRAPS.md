# Traps

Four deliberate defects, each placed so a validator (or a careful person) can find it. The measured size of each
is in `report.md` section 3; the mechanism is in `generator.py`.

## 1. Redundant field: `email_domain`

`email_domain` is a deterministic function of `company` (and the reverse): every company has exactly one domain and
no domain is shared. It adds no information a model could not get from `company`, and a validator that checks
functional dependencies between columns should report it.

**Detect:** `df.groupby("company").email_domain.nunique().max() == 1` and the same the other way round.

Partner routing uses the `is_partner` column (the CRM's flag), not the domain, so dropping either of the pair
loses nothing.

## 2. Leakage field: `sdr_notes_len`

The length of the SDR's notes on the lead. Notes are written only after contact, by the person who assigns the
label: 0 for almost every lead rejected on fit (never contacted), long for accepted leads, short for customer
and partner hand-offs. It correlates strongly with `sdr_disposition`, and it does not exist when a lead arrives,
so a prompt or model that uses it can't be deployed.

**Detect:** the field is populated *after* the target is decided (timing), it is zero for about half the rows,
and on its own it predicts the label far above majority class. Adding it to any baseline gives a jump that no
routing-time input could explain (report §3.2).

It is present in `leads.csv` and `impromptune_ready.csv` on purpose: that is where a validator gets to find it.

## 3. Frozen vs live enrichment: `live_enrichment.csv`

`leads.csv` holds enrichment **as of each lead's `created_at`** (the `snap_` prefix). `live_enrichment.csv` has
the same `lead_id`s and the same column names, re-fetched on 2026-09-15:

- headcount has drifted, and grown more for companies whose leads were accepted;
- some companies raised a new round since (stage bumped, months since funding reset), more often if accepted;
- the hiring signal was re-read, and is on far more often for accepted companies;
- about 15% of accepted leads' companies became customers: `is_existing_customer = true`, and `Tallyforge`
  now shows in their tech stack.

Because the names are identical, a naive join silently overwrites the frozen values. The live values carry the
outcome, so every baseline scores higher on them (report §3.3). That is apparent accuracy a deployed prompt
will never see: at routing time only the frozen values exist.

**Detect:** `enriched_at` is after every `created_at`; values differ for the same `lead_id`; the difference is
correlated with the label.

## 4. Selection bias: `labels_routed.csv`

What the label file would look like if SDRs only worked leads a naive router passed:
`(source in {demo_request, referral} or seniority Head/Director/VP/C-level) and headcount >= 50`. Leads the
router rejected are labelled only in a random 5% exploration slice; every other row has `sdr_disposition`
empty (NULL). Columns: `routed`, `exploration`, `label_propensity` (1.0 if routed, 0.05 if not), `sdr_disposition`.

The labelled rows over-represent hand-raisers and senior titles and under-represent tiny companies, so class
mix and accuracy measured on them differ from the full population (report §3.4). Weighting the labelled rows
by `1 / label_propensity` recovers the population mix approximately; the exploration slice is small, so those
estimates are noisy.

**Detect:** labels are missing not at random (missingness predicted by `source`, title and headcount); a
`label_propensity` column exists; the class mix of labelled rows differs from any external prior.

## Not traps, but worth knowing

- `sdr_disposition` carries about 8% rater noise, mostly accepted ↔ rejected_no_intent (report §1). The
  published labels are what an optimizer trains and scores against; `sealed/true_labels.csv` is for analysis only.
- Tech stack is missing for about 7% of companies, while the latent fit still uses the true stack.
- About 21% of messages are empty (event scans and content downloads most often), and there's some junk
  (`test`, `asdf`) and some spam (SEO and lead-list pitches).
