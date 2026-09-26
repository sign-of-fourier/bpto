# Lead-qualification synthetic dataset: report

N = 3000 leads, created 2026-01-01 to 2026-06-30. Seed 20260925. Message text: **llm** (claude-haiku-4-5-20251001 via bedrock, 117 batched calls, 373,519 input / 153,597 output tokens, total generation cost $1.14; this run spent $0.00, the rest came from cache).

## 1. Class balance and injected rater noise

| label | target | true | SDR (published) |
|---|---:|---:|---:|
| accepted | 18% | 18.1% | 17.4% |
| rejected_fit | 40% | 40.3% | 41.1% |
| rejected_no_intent | 30% | 30.2% | 30.6% |
| existing_customer | 7% | 6.7% | 6.3% |
| partner_route | 5% | 4.7% | 4.7% |

Rater flips: **8.3%** of leads. Confusion matrix, rows = true disposition, columns = SDR label (counts):

| true \ SDR | acc | rej_fit | rej_no_int | existing | partner |
|---|---:|---:|---:|---:|---:|
| acc | 444 | 30 | 70 | 0 | 0 |
| rej_fit | 11 | 1168 | 29 | 0 | 0 |
| rej_no_int | 62 | 32 | 812 | 0 | 0 |
| existing | 6 | 0 | 6 | 188 | 0 |
| partner | 0 | 2 | 0 | 0 | 140 |

accepted <-> rejected_no_intent accounts for 132 of 248 flips (53%).

## 2. Ceilings and baselines (5-fold stratified CV against the published SDR labels)

Two ceilings. The true label includes logistic noise no input reveals, so only an oracle reaches the first row; the second applies the same rule to the noise-free latents, the best a reader of every input could hope for (and still above reach: headcount, stack and title are only partly visible in the columns).

| model | acc vs SDR | macro-F1 vs SDR | acc vs true label | |
|---|---:|---:|---:|---|
| True generating rule, with its noise (sealed true_label) | 0.917 | 0.928 | 1.000 | ceiling: only rater noise stands between it and the labels |
| Generating rule on the latents, no logistic noise | 0.794 | 0.842 | 0.857 | Bayes-ish ceiling for any reader of the inputs |
| Majority class | 0.411 |  | 0.403 |  |
| Structured only: LR on snap_* + is_existing_customer | 0.616 | 0.625 | 0.637 |  |
| All structured: + source, title TF-IDF, is_partner | 0.697 | 0.742 | 0.726 |  |
| Text + structured: + TF-IDF on message_text | 0.725 | 0.781 | 0.774 |  |

- Gap, structured-only (snap_*) to the true-rule ceiling: **30.2 points** (requirement: more than 2). To the no-noise rule: 17.9 points.
- Gap, all structured to text+structured: **2.8 points** against SDR labels, **4.7 points** against the true labels. The text's job is to separate accepted from rejected_no_intent, which is exactly where the rater noise sits, so part of what it gets right is scored wrong. TF-IDF is a floor for the text: it recovers each message's latent intent level with r = 0.89, and a reader that understands the text should do better.
- Headroom above the TF-IDF baseline to the ceiling: 19.2 points.
- Standard error of accuracy on a 20% hold-out (n = 600): 0.0182 at the text+structured accuracy, 0.0204 worst case (p = 0.5). Requirement < 0.03: **met**. A difference between two prompts scored on the same hold-out needs roughly 2 x 0.020 x sqrt(2) = 0.058 to clear two standard errors if the errors were independent; paired comparison on the same leads is tighter.

## 3. Trap checks

### 3.1 Redundant field: email_domain

Max distinct domains per company: 1; max distinct companies per domain: 1. Company and email_domain are a 1:1 mapping (1293 companies across 3000 leads).

### 3.2 Leakage field: sdr_notes_len

Zero for 55% of leads. Median by SDR label: accepted 379, rejected_fit 0, rejected_no_intent 0, existing_customer 47, partner_route 61.

sdr_notes_len alone: accuracy 0.635 (majority class 0.411). Adding it to text+structured: 0.725 -> **0.825** (+10.0 points from one field that does not exist at routing time). A validator should flag it: it is written after the label is decided, by the person who decides it.

### 3.3 Frozen vs live enrichment

Live values as of 2026-09-15. Leads newly shown as existing customers: 157, of which 74 were SDR-accepted (14.1% of accepted leads). Median headcount change: +9.5% overall, +37.5% for accepted leads. `Tallyforge` appears in the live tech stack of 157 leads.

| features | frozen: acc / macro-F1 | live re-fetch: acc / macro-F1 | inflation (acc) |
|---|---:|---:|---:|
| all structured, snap_* re-fetched | 0.697 / 0.742 | 0.721 / 0.779 | **+2.3 pts** |
| all structured, snap_* + is_existing_customer re-fetched | 0.697 / 0.742 | 0.721 / 0.778 | **+2.3 pts** |
| text + structured, snap_* re-fetched | 0.725 / 0.781 | 0.745 / 0.805 | **+2.0 pts** |
| text + structured, snap_* + is_existing_customer re-fetched | 0.725 / 0.781 | 0.745 / 0.804 | **+2.0 pts** |

The live values were fetched after the SDR decided, so they encode the outcome (accounts that bought grew, raised, started hiring, and show the vendor in their stack). Scoring a prompt against re-fetched values overstates what it can do at routing time.

### 3.4 Selection bias: labels only where a naive router passed

Router rule: (source is demo_request or referral, or seniority Head/Director/VP/C-level) and headcount >= 50. Routed: 53% of leads; exploration slice: 89 leads (6.4% of the rest); labelled in total: 56%.

| label | all leads (SDR) | labelled subset | labelled subset, IPW |
|---|---:|---:|---:|
| accepted | 17.4% | 22.6% | 16.9% |
| rejected_fit | 41.1% | 34.9% | 42.2% |
| rejected_no_intent | 30.6% | 33.0% | 27.7% |
| existing_customer | 6.3% | 6.9% | 6.8% |
| partner_route | 4.7% | 2.6% | 6.4% |

Text+structured LR trained on the labelled subset: CV accuracy **within** the labelled subset 0.668; accuracy on the 1308 unlabelled leads (true SDR labels, which the file hides) **0.752**. The labelled subset over-represents demo requests and referrals, so an evaluation that uses only it measures the wrong population. `label_propensity` (1.0 routed, 0.05 exploration) supports inverse-propensity weighting of the exploration slice.

## 4. Message text

Empty: 21%. Words among non-empty: median 26, 5th pct 5, 95th pct 76, max 120. Messages naming a competitor: 13%.

Mean text_score (sealed) by SDR label, the signal the text is meant to carry:

| SDR label | mean text_score | share empty |
|---|---:|---:|
| accepted | +0.95 | 12% |
| rejected_fit | +0.05 | 20% |
| rejected_no_intent | -0.56 | 23% |
| existing_customer | +0.08 | 26% |
| partner_route | -0.05 | 39% |

Random sample (seeded):

- *demo_request, Revenue Ops Specialist, Fintech, 610 employees*: "Just learning about what's available in the forecast space. Nice to see an alternative to the big players. What's your take on implementation complexity?"
- *chat, Vice President, Revenue Operations, Manufacturing, 11 employees*: "Hey there! We've got an incredible SEO and link building service that can really boost your organic search visibility and drive more qualified leads to your platform. We work with tons of SaaS companies and have proven ROI. Let's chat!"
- *demo_request, Graduate Student, Higher Education, 1110 employees*: "I'm writing a thesis on revenue forecasting in SaaS companies and I'm looking for real-world examples and case studies. Would Tallyforge be open to sharing insights on how your customers approach forecasting accuracy and pipeline management?"
- *chat, Graduate Student, B2B SaaS, 27 employees*: "Need evalation urgently"
- *referral, Vice President, Analytics, Logistics, 221 employees*: "We operate a large logistics network and our current forecasting process is scattered across multiple spreadsheets and systems. We have sales, operations, and finance all using different numbers and nobody trusts the forecast. We need a centralized system that can pull data from our CRM and warehouse and give everyone a single source of truth for pipeline visibility. The implementation should work with our existing Salesforce and Databricks environment. Our team is skeptical of new tools but if this solves our alignment problem it could be game-changing for us."
- *referral, Senior IT Manager, B2B SaaS, 48 employees*: "We're evaluating solutions to fix our forecast accuracy problem and keep hearing great things about Tallyforge. With $80k budgeted, we'd love to set up a demo and see if this is the right fit."
- *referral, Senior Finance Manager, B2B SaaS, 45 employees*: "We need to add seats to our current Tallyforge subscription immediately for our expanded revenue team. Additionally, we're evaluating a new integration with our data warehouse and require expedited implementation support. Timeline is critical as we're onboarding stakeholders next week and need everone configured by then."
- *event_scan, Founder & CEO, Retail, 178 employees*: "Really interested in seeing what Tallyforge can do! Would love to get a sense of the platform and how it might work for a retail operation."
