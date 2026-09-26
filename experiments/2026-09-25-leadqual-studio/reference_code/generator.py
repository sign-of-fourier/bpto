"""Synthetic B2B inbound-lead qualification dataset (Impromptune worked example).

The latent model lives here and only here: no output file except sealed/ carries fit, intent, or the
generating rule. Everything is seeded; message text is cached per batch under .cache/, so a rerun
reproduces the same files at no cost.

    python generator.py                      # text from Claude Haiku 4.5 (Anthropic API, or Bedrock fallback)
    python generator.py --text stub          # template text, no API calls (pipeline smoke test only)
    python analyze.py                        # -> report.md

The vendor, its competitors, and every company are fictional.

Latent model
------------
Each lead has two latent scores (logits; fit = sigmoid(fit_logit), intent = sigmoid(intent_logit), both in [0,1]):

  fit_logit    = IND_FIT[industry] + HC_FIT[headcount band] + SEN_FIT[seniority] + DEPT_FIT[department]
                 + STACK_W * |true stack ∩ TARGET_STACK| - STACK_W * 2
  intent_logit = SRC_INTENT[source] + funding_term(months_since_funding) + HIRING_W * hiring
                 + TEXT_W * text_score

text_score is the part of intent only the free text carries. Each message is written from its own latent
level (0-4, from an independent normal) plus three flags drawn with probabilities that rise with the level:

  text_score = LEVEL_W * (level - 2) + COMP_W * competitor + BUDGET_W * budget + TIMELINE_W * timeline
             = 0 when the message is empty, SPAM_SCORE for spam/junk

The generator sees the level description, the flags, title and company context; never a label, fit, or the
overall intent. Enrichment can be missing (stack blank) while fit still uses the true stack, so the structured
columns do not recover fit exactly.

True disposition (deterministic given latents + logistic noise):
  partner domain                     -> partner_route
  elif is_existing_customer          -> existing_customer
  elif fit_logit + e1 < T_FIT        -> rejected_fit
  elif intent_logit + e2 < T_INTENT  -> rejected_no_intent
  else                               -> accepted
  e1, e2 ~ Logistic(0, NOISE_S). T_FIT and T_INTENT are the quantiles of the noisy scores among prospects that
  hit the target priors (accepted 18, rejected_fit 40, rejected_no_intent 30, existing 7, partner 5).

SDR label = true disposition through RATER_CONFUSION (~8% flips, mostly accepted <-> rejected_no_intent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CACHE = OUT / ".cache" / "messages"
SEALED = OUT / "sealed"

SEED = 20260925
N_LEADS = 3000
N_PROSPECT_COS = 1500
N_CUSTOMER_COS = 110
N_PARTNER_COS = 36
P_CUSTOMER, P_PARTNER = 0.07, 0.05

MODEL = "claude-haiku-4-5-20251001"
BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
PRICE_IN, PRICE_OUT = 1.00 / 1e6, 5.00 / 1e6        # Haiku 4.5, $/token, same on both backends
BATCH = 20

VENDOR = "Tallyforge"
COMPETITORS = ["Forecastly", "PipeRadar", "Quotient IQ", "Closewise", "Revhawk"]
LABELS = ["accepted", "rejected_fit", "rejected_no_intent", "existing_customer", "partner_route"]
PRIORS = {"accepted": .18, "rejected_fit": .40, "rejected_no_intent": .30, "existing_customer": .07, "partner_route": .05}

SNAPSHOT_REF = pd.Timestamp("2026-04-01")
LIVE_AT = pd.Timestamp("2026-09-15")

# ---------------------------------------------------------------- latent weights
#            name                        share  fit    modernity
INDUSTRIES = [("B2B SaaS",                 .22,  1.2,   0.8),
              ("Fintech",                  .09,  0.9,   0.6),
              ("Healthcare IT",            .07,  0.5,   0.2),
              ("Financial Services",       .07,  0.2,  -0.1),
              ("E-commerce",               .08,  0.2,   0.4),
              ("Marketing & Advertising",  .07, -0.3,   0.3),
              ("Manufacturing",            .09, -0.4,  -0.6),
              ("Logistics",                .06, -0.2,  -0.4),
              ("Retail",                   .06, -0.6,  -0.3),
              ("IT Services & Consulting", .06, -0.1,   0.3),
              ("Higher Education",         .05, -1.3,  -0.5),
              ("Nonprofit",                .04, -1.4,  -0.6),
              ("Government",               .04, -1.6,  -0.9)]
IND_FIT = {n: f for n, _, f, _ in INDUSTRIES}
HC_BANDS = [0, 20, 100, 500, 2000, 10000]
HC_FIT = [-1.6, -0.3, 0.7, 1.0, 0.3, -0.4]
SEN_FIT = {"C-level": 0.5, "VP": 0.9, "Director": 0.8, "Head": 0.7, "Manager": 0.3, "IC": -0.3, "Junior": -0.8,
           "Student": -2.5}
DEPT_FIT = {"Revenue Operations": 1.1, "Sales Operations": 1.0, "Sales": 0.4, "Marketing Operations": 0.4,
            "Marketing": -0.1, "Finance": 0.2, "Data & Analytics": 0.3, "Engineering": -0.6, "Customer Success": 0.0,
            "People": -1.4, "IT": -0.5, "Executive": 0.3, "None": -1.0}
TARGET_STACK = ["Salesforce", "HubSpot", "Snowflake", "dbt", "Segment", "Fivetran", "Gong", "Outreach", "Salesloft"]
STACK_W = 0.4
SRC_INTENT = {"demo_request": 1.2, "referral": 0.9, "chat": 0.2, "content_download": -0.7, "event_scan": -0.9}
HIRING_W = 0.5
TEXT_W = 1.6
LEVEL_W, COMP_W, BUDGET_W, TIMELINE_W, SPAM_SCORE = 0.9, 0.5, 0.4, 0.5, -3.0
NOISE_S = 0.4


def funding_term(m):
    m = np.asarray(m, float)
    return np.where(np.isnan(m), -0.1, np.select([m <= 6, m <= 12, m <= 24], [0.8, 0.4, 0.0], -0.3))


# Rater noise: row = true disposition, columns = SDR label. Mostly accepted <-> rejected_no_intent.
RATER_CONFUSION = {
    "accepted":           {"accepted": .84, "rejected_no_intent": .12, "rejected_fit": .04},
    "rejected_no_intent": {"rejected_no_intent": .88, "accepted": .09, "rejected_fit": .03},
    "rejected_fit":       {"rejected_fit": .96, "rejected_no_intent": .03, "accepted": .01},
    "existing_customer":  {"existing_customer": .97, "accepted": .02, "rejected_no_intent": .01},
    "partner_route":      {"partner_route": .97, "rejected_fit": .02, "accepted": .01},
}

# ---------------------------------------------------------------- vocabularies (not latent)
NAME_A = ["North", "Blue", "Bright", "Iron", "Silver", "Cedar", "Harbor", "Summit", "Maple", "Granite", "Lumen",
          "Nimbus", "Clear", "Red", "Oak", "Pine", "Stone", "Swift", "True", "Polar", "Echo", "Atlas", "Copper",
          "Birch", "Coral", "Falcon", "Juniper", "Kestrel", "Lark", "Meridian", "Onyx", "Quarry", "Rowan", "Sable",
          "Tidal", "Umber", "Vale", "Willow", "Zephyr", "Amber", "Brook", "Crest", "Delta", "Ember", "Fern",
          "Glen", "Hollow", "Indigo", "Jade", "Keel", "Linden", "Moss", "Noble", "Orchid", "Prairie", "Quill",
          "Ridge", "Slate", "Thistle", "Upland", "Verdant", "Wren", "Yarrow", "Alder", "Basalt", "Cobalt"]
NAME_B = ["lane", "path", "field", "stack", "loop", "grid", "wave", "point", "bridge", "line", "works", "scale",
          "craft", "leaf", "peak", "gate", "shift", "spring", "mark", "base", "yard", "port", "wise", "ly", "view",
          "rock", "stream", "light", "haven", "fold"]
IND_SUFFIX = {"B2B SaaS": ["", " Labs", " Software", " Cloud", " HQ", " Technologies"],
              "Fintech": [" Pay", " Capital", " Finance", " Money", ""],
              "Healthcare IT": [" Health", " Medical Systems", " Care", " Clinical"],
              "Financial Services": [" Financial", " Bancorp", " Insurance", " Credit Union", " Wealth"],
              "E-commerce": [" Goods", " Shop", " Supply Co", " Direct", ""],
              "Marketing & Advertising": [" Media", " Creative", " Digital", " Agency"],
              "Manufacturing": [" Industries", " Manufacturing", " Components", " Fabrication"],
              "Logistics": [" Logistics", " Freight", " Transport", " Shipping"],
              "Retail": [" Outfitters", " Stores", " Market", " Home"],
              "IT Services & Consulting": [" Consulting", " Partners", " Advisory", " Solutions Group",
                                           " RevOps Group", " Digital Partners"],
              "Higher Education": [" University", " College", " Institute of Technology"],
              "Nonprofit": [" Foundation", " Alliance", " Trust", " Fund"],
              "Government": [" County", " City of", " Department of Commerce", " Transit Authority"]}
TLD = {"B2B SaaS": [".io", ".com", ".com", ".ai"], "Higher Education": [".edu"], "Nonprofit": [".org"],
       "Government": [".gov"], "Fintech": [".com", ".io"]}
FIRST_NAMES = ["Aisha", "Alejandro", "Amara", "Ananya", "Andre", "Arjun", "Beatriz", "Ben", "Camila", "Chen",
               "Chloe", "Daniel", "Deepa", "Diego", "Elena", "Emeka", "Emily", "Farah", "Fatima", "Gabriel",
               "Grace", "Hana", "Hassan", "Ines", "Isaac", "Jamal", "James", "Jasmine", "Javier", "Jin", "Jordan",
               "Julia", "Kai", "Karan", "Kate", "Kenji", "Kofi", "Laura", "Leila", "Liam", "Lucas", "Maya", "Mei",
               "Michael", "Mohammed", "Nadia", "Naomi", "Nikhil", "Noah", "Olivia", "Omar", "Priya", "Rachel",
               "Rahul", "Ravi", "Rosa", "Ryan", "Sakura", "Samuel", "Sara", "Sean", "Sofia", "Sung", "Tariq",
               "Thomas", "Tomas", "Valentina", "Wei", "Yara", "Yusuf", "Zoe", "Zainab", "Matt", "Chris", "Pat",
               "Alex", "Sam", "Taylor", "Morgan", "Jessica", "Brian", "Megan", "Kevin", "Lauren", "Eric", "Dana"]
OTHER_TOOLS = {  # name: (intercept, modernity, size)
    "Snowflake": (-1.0, 1.0, .5), "BigQuery": (-1.5, .6, .1), "Redshift": (-1.6, .3, .3), "dbt": (-1.4, 1.1, .2),
    "Segment": (-1.3, .9, -.1), "Fivetran": (-1.5, .9, .2), "Gong": (-1.2, .8, .4), "Outreach": (-1.4, .6, .4),
    "Salesloft": (-1.6, .5, .3), "Marketo": (-1.3, -.1, .5), "Pardot": (-1.6, -.2, .3), "Zendesk": (-.8, .2, .2),
    "Intercom": (-1.2, .6, -.3), "Shopify": (-2.2, .3, -.2), "NetSuite": (-1.3, -.2, .4), "Workday": (-1.8, -.2, 1.0),
    "Tableau": (-1.2, -.1, .5), "Looker": (-1.6, .6, .1), "Power BI": (-1.0, -.5, .3), "Jira": (-.5, .5, .2),
    "Slack": (.3, .8, 0), "Mailchimp": (-1.2, -.3, -.8)}
CRMS = {"Salesforce": (.3, .3, .8), "HubSpot": (.3, .6, -.5), "Microsoft Dynamics": (-.8, -.6, .4),
        "Zoho CRM": (-1.2, 0, -.5), "Pipedrive": (-1.2, .2, -.8), None: (-.8, -.5, -.8)}
STAGES = ["Bootstrapped", "Seed", "Series A", "Series B", "Series C", "Series D+", "PE-backed", "Public"]
STAGE_BY_BAND = [[.45, .40, .15, 0, 0, 0, 0, 0], [.30, .20, .35, .15, 0, 0, 0, 0],
                 [.20, 0, .20, .35, .20, 0, .05, 0], [.10, 0, 0, .15, .30, .25, .15, .05],
                 [0, 0, 0, 0, 0, .20, .30, .50], [0, 0, 0, 0, 0, 0, .30, .70]]
PUBLIC_SECTOR = {"Higher Education", "Nonprofit", "Government"}

INTENT_LEVELS = [
    "no buying intent: a student or researcher, a job seeker, someone lost, or idle curiosity with no project",
    "low intent: early learning, wants the content or a general idea, no project and no urgency",
    "moderate intent: has a real problem in the area, exploring options, vague or no timing",
    "high intent: actively evaluating tools for a specific pain, wants a demo or a real conversation soon",
    "very high intent: urgent, a decision process is under way, other stakeholders involved, asks for pricing or next steps now",
]
SOURCE_BOX = {
    "demo_request": "the 'Anything else we should know?' box on the demo request form",
    "content_download": "the optional 'What are you working on?' box on a gated ebook form",
    "event_scan": "the note field of a conference booth badge-scan tablet, typed by the attendee",
    "chat": "their first message in the website chat widget",
    "referral": "the note on the referral form, written by the person being referred",
}
TONES = ["formal", "casual", "terse", "rambling", "frustrated", "enthusiastic", "matter-of-fact"]
JUNK = ["test", "asdf", ".", "n/a", "na", "?", "hi", "123", "xx", "none", "qwerty", "ok", "-", "testing 123"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def band(hc):
    return np.searchsorted(HC_BANDS, hc, side="right") - 1


def months_between(a, b):
    return int(round((b - a).days / 30.44))


# ---------------------------------------------------------------- companies
def make_companies(rng):
    n = N_PROSPECT_COS + N_CUSTOMER_COS * 3 + N_PARTNER_COS  # customers drawn from a larger pool (below)
    names, domains = set(), set()
    rows = []
    ind_names = [i[0] for i in INDUSTRIES]
    ind_p = np.array([i[1] for i in INDUSTRIES])
    kinds = ["prospect"] * (n - N_PARTNER_COS) + ["partner"] * N_PARTNER_COS
    for kind in kinds:
        ind = "IT Services & Consulting" if kind == "partner" else ind_names[rng.choice(len(ind_names), p=ind_p)]
        while True:
            stem = rng.choice(NAME_A) + (rng.choice(NAME_B) if rng.random() < .7 else " " + rng.choice(NAME_A))
            suf = rng.choice(IND_SUFFIX[ind])
            name = f"City of {stem}" if suf == " City of" else stem + suf
            dom = re.sub(r"[^a-z0-9]", "", stem.lower()) + rng.choice(TLD.get(ind, [".com"]))
            if ind == "Government":
                dom = re.sub(r"[^a-z0-9]", "", stem.lower()) + ("county" if "County" in suf else "") + ".gov"
            if name not in names and dom not in domains:
                break
        names.add(name); domains.add(dom)
        mu = {"Higher Education": 7.0, "Government": 6.8, "IT Services & Consulting": 3.8}.get(ind, 5.0)
        hc = int(np.clip(np.round(np.exp(rng.normal(3.6 if kind == "partner" else mu, 1.4))), 2, 60000))
        size_z = (np.log(hc) - 5.0) / 1.4
        m = rng.normal([i[3] for i in INDUSTRIES if i[0] == ind][0], 1.0)
        # stack
        crm_names = list(CRMS)
        sc = np.array([a + b * m + c * size_z for a, b, c in CRMS.values()])
        pr = np.exp(sc) / np.exp(sc).sum()
        crm = crm_names[rng.choice(len(crm_names), p=pr)]
        stack = [crm] if crm else []
        for tool, (a, b, c) in OTHER_TOOLS.items():
            a2 = a + (2.5 if tool == "Shopify" and ind in ("E-commerce", "Retail") else 0)
            if rng.random() < sigmoid(a2 + b * m + c * size_z):
                stack.append(tool)
        # funding
        if ind in PUBLIC_SECTOR:
            stage, fdate = None, None
        else:
            stage = STAGES[rng.choice(8, p=STAGE_BY_BAND[band(hc)])]
            fdate = None if stage in ("Bootstrapped", "Public") else \
                SNAPSHOT_REF - pd.Timedelta(days=int(rng.exponential(16) * 30.44))
        recent = fdate is not None and (SNAPSHOT_REF - fdate).days < 270
        hiring = bool(rng.random() < sigmoid(-0.9 + 0.8 * recent + 0.3 * m + 0.2 * size_z))
        rows.append(dict(company=name, email_domain=dom, industry=ind, headcount=hc, modernity=m, tools=stack,
                         stack_missing=bool(rng.random() < .07), stage=stage, funding_date=fdate, hiring=hiring,
                         kind=kind))
    co = pd.DataFrame(rows)
    # existing customers: the best-fitting prospect companies (plus noise) already bought
    pros = co.index[co.kind == "prospect"]
    cf = np.array([IND_FIT[co.industry[i]] + HC_FIT[band(co.headcount[i])]
                   + STACK_W * len(set(co.tools[i]) & set(TARGET_STACK)) for i in pros]) + rng.normal(0, 1, len(pros))
    cust = pros[np.argsort(-cf)[:N_CUSTOMER_COS * 3]]
    cust = rng.choice(cust, N_CUSTOMER_COS, replace=False)
    co.loc[cust, "kind"] = "customer"
    # the unused part of the customer pool stays prospect; trim prospects to N_PROSPECT_COS
    extra = co.index[co.kind == "prospect"][N_PROSPECT_COS:]
    return co.drop(extra).reset_index(drop=True)


# ---------------------------------------------------------------- leads
def make_title(rng, sen, dept):
    abbr = {"Revenue Operations": ["Revenue Operations", "RevOps", "Revenue Ops"],
            "Sales Operations": ["Sales Operations", "Sales Ops"],
            "Marketing Operations": ["Marketing Operations", "Marketing Ops", "MOPs"],
            "People": ["People", "HR", "Talent"], "Data & Analytics": ["Data & Analytics", "Analytics", "BI"]}
    d = rng.choice(abbr.get(dept, [dept]))
    if sen == "Student":
        return rng.choice(["Student", "MBA Candidate", "Graduate Student", "Research Assistant"])
    if sen == "C-level":
        return {"Revenue Operations": "Chief Revenue Officer", "Sales": rng.choice(["CRO", "Chief Revenue Officer"]),
                "Finance": rng.choice(["CFO", "Chief Financial Officer"]), "Engineering": "CTO",
                "Marketing": "CMO", "Executive": rng.choice(["CEO", "Founder & CEO", "Co-founder", "COO", "President"]),
                }.get(dept, "COO")
    t = {"VP": [f"VP of {d}", f"VP, {d}", f"Vice President, {d}", f"SVP {d}"],
         "Director": [f"Director of {d}", f"{d} Director", f"Sr. Director, {d}"],
         "Head": [f"Head of {d}"],
         "Manager": [f"{d} Manager", f"Manager, {d}", f"Senior {d} Manager"],
         "IC": [f"{d} Analyst", f"{d} Specialist", f"Senior {d} Analyst"],
         "Junior": [f"{d} Coordinator", f"{d} Associate", f"{d} Intern"]}[sen]
    if sen == "IC" and dept == "Sales":
        t = ["Account Executive", "SDR", "Business Development Rep", "Senior Account Executive"]
    if sen == "IC" and dept == "Engineering":
        t = ["Software Engineer", "Senior Software Engineer", "Data Engineer"]
    return rng.choice(t)


SEN_P = ["C-level", "VP", "Director", "Head", "Manager", "IC", "Junior", "Student"]
DEPTS = ["Revenue Operations", "Sales Operations", "Sales", "Marketing Operations", "Marketing", "Finance",
         "Data & Analytics", "Engineering", "Customer Success", "People", "IT"]
DEPT_P = np.array([.13, .10, .17, .06, .16, .07, .08, .08, .06, .05, .04])


def make_leads(rng, co):
    kind = rng.choice(["customer", "partner", "prospect"], N_LEADS, p=[P_CUSTOMER, P_PARTNER, 1 - P_CUSTOMER - P_PARTNER])
    idx = np.empty(N_LEADS, int)
    for k in ("customer", "partner", "prospect"):
        pool = co.index[co.kind == k].to_numpy()
        w = 1 / (np.arange(len(pool)) + 3) ** (0.35 if k == "prospect" else 0.6)
        idx[kind == k] = rng.choice(pool, (kind == k).sum(), p=w / w.sum())
    # created_at: Jan-Jun 2026, weekdays heavier, business hours
    days = pd.date_range("2026-01-01", "2026-06-30", freq="D")
    dw = np.where(days.dayofweek < 5, 1.0, 0.3)
    d = days[rng.choice(len(days), N_LEADS, p=dw / dw.sum())]
    secs = np.clip(rng.normal(13.5 * 3600, 3.2 * 3600, N_LEADS), 0, 86399).astype(int)
    created = (d + pd.to_timedelta(secs, unit="s")).sort_values()
    src_p = {"prospect": [.20, .34, .18, .17, .11], "customer": [.25, .15, .10, .45, .05],
             "partner": [.30, .05, .10, .10, .45]}
    srcs = list(SRC_INTENT)
    rows = []
    for i in range(N_LEADS):
        c = co.loc[idx[i]]
        k = kind[i]
        small = c.headcount < 50
        sp = np.array([.10 if small else .03, .10, .16, .08, .20, .25, .09, .02 if c.industry != "Higher Education" else .15])
        sen = SEN_P[rng.choice(8, p=sp / sp.sum())]
        dept = "None" if sen == "Student" else (
            rng.choice(["Revenue Operations", "Sales", "Finance", "Engineering", "Marketing", "Executive"],
                       p=[.08, .15, .12, .10, .10, .45]) if sen == "C-level" else DEPTS[rng.choice(len(DEPTS), p=DEPT_P)])
        if sen == "C-level" and dept == "Revenue Operations":
            dept = "Sales"
        rows.append(dict(lead_kind=k, company_idx=idx[i], created_at=created[i],
                         source=srcs[rng.choice(5, p=src_p[k])], first_name=rng.choice(FIRST_NAMES),
                         seniority=sen, department=dept, title=make_title(rng, sen, dept)))
    L = pd.DataFrame(rows)
    L.insert(0, "lead_id", [f"L{26000 + i:05d}" for i in range(N_LEADS)])
    return L


def snapshot(L, co):
    """Frozen enrichment as of each lead's created_at."""
    c = co.loc[L.company_idx].reset_index(drop=True)
    out = pd.DataFrame({"snap_headcount": c.headcount.to_numpy(), "snap_industry": c.industry.to_numpy()})
    stage, months = [], []
    for fd, st, t in zip(c.funding_date, c.stage, L.created_at):
        fd = None if pd.isna(fd) else fd
        if fd is not None and fd > t:        # this round hadn't happened yet: show the previous one
            prev = STAGES[max(STAGES.index(st) - 1, 0)]
            st, fd = prev, (None if prev == "Bootstrapped" else fd - pd.Timedelta(days=456))
        stage.append(st)
        months.append(np.nan if fd is None else months_between(fd, t))
    out["snap_funding_stage"] = stage
    out["snap_months_since_funding"] = pd.array(months, dtype="Int64")
    out["snap_tech_stack"] = [None if miss else ";".join(s) for s, miss in zip(c.tools, c.stack_missing)]
    out["snap_hiring_signal"] = c.hiring.to_numpy()
    return out


# ---------------------------------------------------------------- message directives
def message_directives(rng, L, co):
    empty_p = {"demo_request": .08, "content_download": .45, "event_scan": .60, "chat": .0, "referral": .15}
    med_words = {"demo_request": 35, "content_download": 18, "event_scan": 14, "chat": 16, "referral": 40}
    u = rng.normal(size=len(L))
    level = np.digitize(u, [-1.04, -0.39, 0.39, 1.04])
    rows = []
    for i, r in L.iterrows():
        c = co.loc[r.company_idx]
        x = rng.random()
        kind = "empty" if x < empty_p[r.source] else ("spam" if x < empty_p[r.source] + .03 else
                                                     ("junk" if x < empty_p[r.source] + .05 else "normal"))
        lv = int(level[i])
        comp = rng.random() < [.03, .08, .15, .30, .40][lv]
        budget = rng.random() < [0, .03, .10, .25, .45][lv]
        timeline = rng.random() < [0, .05, .15, .40, .60][lv]
        rows.append(dict(kind=kind, level=lv,
                         competitor=str(rng.choice(COMPETITORS)) if comp else None,
                         budget=bool(budget), timeline=bool(timeline),
                         words=int(np.clip(round(np.exp(rng.normal(np.log(med_words[r.source]), .6))), 5, 120)),
                         tone=str(rng.choice(TONES)),
                         typos=str(rng.choice(["none", "light", "heavy"], p=[.55, .33, .12])),
                         junk=str(rng.choice(JUNK))))
    D = pd.DataFrame(rows)
    ts = LEVEL_W * (D.level - 2) + COMP_W * D.competitor.notna() + BUDGET_W * D.budget + TIMELINE_W * D.timeline
    D["text_score"] = np.select([D.kind == "empty", D.kind.isin(["spam", "junk"])], [0.0, SPAM_SCORE], ts)
    return D


SYSTEM = f"""You write synthetic inbound messages for a B2B software demo dataset. The vendor is {VENDOR}, a revenue forecasting and pipeline analytics platform that sits on top of a company's CRM and data warehouse. Its competitors include {", ".join(COMPETITORS)}. All companies are fictional.

Each item describes one person who contacted {VENDOR}. Write exactly what that person typed into the free-text box for their channel, in their own voice. The channel says which box it was.

Rules:
- Follow each item's intent, tone, approximate length in words, and typo level.
- If "competitor" names a product, mention it naturally (currently using it, evaluating it, contract ending, frustrated with it, or comparing - whatever fits the intent).
- If "budget" is true, reference budget in a way consistent with the intent (a figure, a range, approved, or none yet). If false, say nothing about budget.
- If "timeline" is true, reference timing consistent with the intent. If false, say nothing about timing.
- "existing_customer": true means they already pay for {VENDOR}; they write about their account (more seats, a new team, a support problem, renewal, an integration) regardless of intent.
- kind "spam": not a buyer at all - they pitch their own services (SEO, lead lists, offshore dev, link building, guest posts) or it is promotional spam. Ignore intent, competitor, budget and timeline.
- Never state a verdict about themselves ("I'm a qualified lead", "I'm not a fit"). Don't recite their own title and company as a signature.
- Vary the openings; most real messages don't start with a greeting. Typos "heavy" means several misspellings, missing capitals, run-ons; "light" means one or two slips.

Return only JSON: {{"messages": [{{"id": <id>, "text": "<message>"}}, ...]}} with exactly one entry per item, same ids, nothing else."""


def item_for(i, r, c, d):
    it = {"id": int(i), "channel": SOURCE_BOX[r.source], "first_name": r.first_name, "title": r.title,
          "company": c.company, "industry": c.industry, "employees": int(c.headcount)}
    if c.kind == "customer":
        it["existing_customer"] = True
    if d.kind == "spam":
        it["kind"] = "spam"
    else:
        it.update(intent=INTENT_LEVELS[int(d.level)], competitor=d.competitor if isinstance(d.competitor, str) else None,
                  budget=bool(d.budget), timeline=bool(d.timeline))
    it.update(words=int(d.words), tone=str(d.tone), typos=str(d.typos))
    return it


# ---------------------------------------------------------------- text backends
def load_env_file(path):
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def make_caller(backend):
    if backend == "auto":
        backend = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "bedrock"
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=6)

        def call(user):
            r = client.messages.create(model=MODEL, max_tokens=4000, temperature=1.0, system=SYSTEM,
                                       messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in r.content if b.type == "text"), r.usage.input_tokens, r.usage.output_tokens
    else:
        import boto3
        from botocore.config import Config
        rt = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                          config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=180))

        def call(user):
            r = rt.converse(modelId=BEDROCK_MODEL, system=[{"text": SYSTEM}],
                            messages=[{"role": "user", "content": [{"text": user}]}],
                            inferenceConfig={"maxTokens": 4000, "temperature": 1.0})
            u = r["usage"]
            return r["output"]["message"]["content"][0]["text"], u["inputTokens"], u["outputTokens"]
    return backend, call


def parse_messages(text, ids):
    m = re.search(r"\{.*\}", text, re.S)
    got = {int(x["id"]): str(x["text"]) for x in json.loads(m.group(0))["messages"]}
    if set(got) != set(ids):
        raise ValueError(f"ids mismatch: missing {set(ids) - set(got)}")
    return got


def generate_texts(items, mode, backend, workers):
    """items: list of dicts with 'id'. Returns ({id: text}, stats). Cached per batch."""
    CACHE.mkdir(parents=True, exist_ok=True)
    batches = [items[k:k + BATCH] for k in range(0, len(items), BATCH)]
    out, stats = {}, {"calls_new": 0, "calls_cached": 0, "in_new": 0, "out_new": 0, "in_all": 0, "out_all": 0}
    if mode == "stub":
        for it in items:
            out[it["id"]] = stub_text(it)
        return out, stats | {"backend": "stub"}
    todo, backends = [], set()
    for b in batches:
        user = json.dumps({"items": b}, sort_keys=True)
        key = hashlib.sha256((MODEL + SYSTEM + user).encode()).hexdigest()[:24]
        f = CACHE / f"{key}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            out.update({int(k): v for k, v in rec["texts"].items()})
            stats["calls_cached"] += 1; stats["in_all"] += rec["in"]; stats["out_all"] += rec["out"]
            backends.add(rec.get("backend", "unknown"))
        else:
            todo.append((b, user, f))
    if todo:
        name, caller = make_caller(backend)
        backends.add(name)
        print(f"generating {len(todo)} batches via {name} ({len(batches) - len(todo)} cached)", file=sys.stderr)

        def run(b, user, f):
            ids = [it["id"] for it in b]
            tin = tout = 0
            for attempt in range(4):
                text, i_, o_ = caller(user)
                tin += i_; tout += o_
                try:
                    got = parse_messages(text, ids)
                    break
                except Exception as e:  # malformed JSON or a dropped id: ask again, still pay for it
                    print(f"  retry {attempt + 1} ({e})", file=sys.stderr)
            else:
                raise RuntimeError(f"batch failed 4 times: {f.name}")
            f.write_text(json.dumps({"texts": got, "in": tin, "out": tout, "model": MODEL, "backend": name}))
            return got, tin, tout

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run, *t) for t in todo]
            for n, fu in enumerate(as_completed(futs), 1):
                got, tin, tout = fu.result()
                out.update(got)
                stats["calls_new"] += 1
                for k, v in (("in_new", tin), ("out_new", tout), ("in_all", tin), ("out_all", tout)):
                    stats[k] += v
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}  ${cost(stats['in_new'], stats['out_new']):.3f} so far", file=sys.stderr)
    return out, stats | {"backend": "+".join(sorted(backends))}


def cap_words(t, n=120):
    """The form's text box stops at n words; keep whole sentences where one ends inside the cap."""
    w = t.split()
    if len(w) <= n:
        return t
    cut = " ".join(w[:n])
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > len(cut) // 2 else cut


def cost(i, o):
    return i * PRICE_IN + o * PRICE_OUT


def stub_text(it):
    if it.get("kind") == "spam":
        return "We help companies rank #1 on Google. Can I send pricing for our SEO packages?"
    if it.get("existing_customer"):
        return "We need a few more seats for the new team, who do I talk to?"
    lv = INTENT_LEVELS.index(it["intent"])
    s = ["just researching for a class project", "saw the ebook, curious", "looking at options for forecasting",
         "evaluating tools for pipeline visibility, want a demo", "need pricing asap, decision this month"][lv]
    if it["competitor"]:
        s += f", we use {it['competitor']} today"
    if it["budget"]:
        s += ", budget is set aside"
    if it["timeline"]:
        s += ", want something live next quarter"
    return s


# ---------------------------------------------------------------- disposition, rater, traps
def dispositions(rng, L, co, S, D):
    c = co.loc[L.company_idx].reset_index(drop=True)
    overlap = np.array([len(set(s) & set(TARGET_STACK)) for s in c.tools])
    fit_logit = (c.industry.map(IND_FIT).to_numpy() + np.array(HC_FIT)[band(c.headcount.to_numpy())]
                 + L.seniority.map(SEN_FIT).to_numpy() + L.department.map(DEPT_FIT).to_numpy()
                 + STACK_W * overlap - STACK_W * 2)
    months = S.snap_months_since_funding.astype(float).to_numpy()
    intent_logit = (L.source.map(SRC_INTENT).to_numpy() + funding_term(months)
                    + HIRING_W * S.snap_hiring_signal.to_numpy() + TEXT_W * D.text_score.to_numpy())
    f_n = fit_logit + rng.logistic(0, NOISE_S, len(L))
    i_n = intent_logit + rng.logistic(0, NOISE_S, len(L))
    partner = (c.kind == "partner").to_numpy()
    customer = (c.kind == "customer").to_numpy()
    pros = ~partner & ~customer
    t_fit = np.quantile(f_n[pros], PRIORS["rejected_fit"] / (1 - P_CUSTOMER - P_PARTNER))
    ok = pros & (f_n >= t_fit)
    t_int = np.quantile(i_n[ok], PRIORS["rejected_no_intent"] / (PRIORS["rejected_no_intent"] + PRIORS["accepted"]))

    def rule(f, i):
        return np.select([partner, customer, f < t_fit, i < t_int],
                         ["partner_route", "existing_customer", "rejected_fit", "rejected_no_intent"], "accepted")
    return dict(fit=sigmoid(fit_logit), intent=sigmoid(intent_logit), fit_logit=fit_logit, intent_logit=intent_logit,
                true_label=rule(f_n, i_n), rule_noiseless=rule(fit_logit, intent_logit),
                is_partner_domain=partner, is_existing_customer=customer, t_fit=t_fit, t_int=t_int)


def rater(rng, true):
    out = []
    for t in true:
        row = RATER_CONFUSION[t]
        out.append(rng.choice(list(row), p=np.array(list(row.values())) / sum(row.values())))
    return np.array(out)


def notes_len(rng, sdr):
    """Length of SDR notes: written only after contact, so it follows the SDR's own call."""
    spec = {"accepted": (0.02, 5.9, .5), "rejected_no_intent": (.55, 4.9, .6), "existing_customer": (.2, 4.0, .5),
            "partner_route": (.15, 4.3, .5), "rejected_fit": (.85, 3.6, .6)}
    out = []
    for s in sdr:
        p0, mu, sd = spec[s]
        out.append(0 if rng.random() < p0 else int(np.clip(np.exp(rng.normal(mu, sd)), 8, 4000)))
    return np.array(out)


def live_enrichment(rng, L, co, sdr):
    """Values as of Sep 2026. Companies that went on to buy grew, raised, hired and show up as customers."""
    L = L.assign(sdr=sdr)
    acc_co = set(L.company_idx[L.sdr == "accepted"])
    converted = {c for c in sorted(acc_co) if co.kind[c] == "prospect" and rng.random() < .14}
    rec = {}
    for ci in sorted(set(L.company_idx)):
        c = co.loc[ci]
        a = ci in acc_co
        new_round = c.industry not in PUBLIC_SECTOR and c.stage != "Public" and rng.random() < (.60 if a else .08)
        stage, fd = c.stage, (None if pd.isna(c.funding_date) else c.funding_date)
        if new_round:
            stage = STAGES[min(STAGES.index(stage) + 1, 5)] if stage not in ("PE-backed",) else stage
            fd = pd.Timestamp("2026-06-01") + pd.Timedelta(days=int(rng.integers(0, 90)))
        g = rng.normal(.03, .07) + (.20 if a else 0) + (.15 if new_round else 0)
        stack = list(c.tools) + ([VENDOR] if ci in converted else [])
        rec[ci] = dict(snap_headcount=int(max(2, round(c.headcount * np.exp(g)))), snap_industry=c.industry,
                       snap_funding_stage=stage,
                       snap_months_since_funding=pd.NA if fd is None else months_between(fd, LIVE_AT),
                       snap_tech_stack=None if c.stack_missing and ci not in converted else ";".join(stack),
                       snap_hiring_signal=bool(rng.random() < (.20 + .65 * a + .15 * new_round)),
                       is_existing_customer=bool(c.kind == "customer" or ci in converted))
    live = pd.DataFrame([rec[ci] for ci in L.company_idx])
    live.insert(0, "lead_id", L.lead_id.to_numpy())
    live["snap_months_since_funding"] = live.snap_months_since_funding.astype("Int64")
    live["enriched_at"] = LIVE_AT.date().isoformat()
    return live


def naive_router(L, S):
    """What a RevOps team might ship first: hand-raisers and senior people at mid-size+ companies, nobody tiny."""
    senior = L.seniority.isin(["C-level", "VP", "Director", "Head"])
    return ((L.source.isin(["demo_request", "referral"]) | senior) & (S.snap_headcount >= 50)).to_numpy()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", choices=["llm", "stub"], default="llm")
    ap.add_argument("--backend", choices=["auto", "anthropic", "bedrock"], default="auto")
    ap.add_argument("--env-file", default=None, help="KEY=value file to load credentials from (never printed)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    load_env_file(a.env_file)

    ss = np.random.SeedSequence(SEED)
    r_co, r_lead, r_msg, r_disp, r_rater, r_notes, r_live, r_route = [np.random.default_rng(s) for s in ss.spawn(8)]

    co = make_companies(r_co)
    L = make_leads(r_lead, co)
    S = snapshot(L, co)
    D = message_directives(r_msg, L, co)
    C = co.loc[L.company_idx].reset_index(drop=True)

    items = [item_for(i, L.loc[i], C.loc[i], D.loc[i]) for i in range(len(L)) if D.kind[i] in ("normal", "spam")]
    texts, st = generate_texts(items, a.text, a.backend, a.workers)
    msg = [cap_words(texts.get(i, "")) if D.kind[i] in ("normal", "spam") else (D.junk[i] if D.kind[i] == "junk" else "")
           for i in range(len(L))]

    Z = dispositions(r_disp, L, co, S, D)
    sdr = rater(r_rater, Z["true_label"])

    leads = pd.DataFrame({"lead_id": L.lead_id, "created_at": L.created_at.dt.strftime("%Y-%m-%dT%H:%M:%S"),
                          "source": L.source, "first_name": L.first_name, "title": L.title, "company": C.company,
                          "email_domain": C.email_domain, "message_text": msg})
    leads = pd.concat([leads, S], axis=1)
    leads["is_existing_customer"] = Z["is_existing_customer"]
    leads["is_partner"] = Z["is_partner_domain"]              # the CRM's partner flag, known at routing time
    leads["sdr_notes_len"] = notes_len(r_notes, sdr)
    leads["sdr_disposition"] = sdr
    leads.to_csv(OUT / "leads.csv", index=False)

    live_enrichment(r_live, L, co, sdr).to_csv(OUT / "live_enrichment.csv", index=False)

    routed = naive_router(L, S)
    explore = ~routed & (r_route.random(len(L)) < .05)
    pd.DataFrame({"lead_id": L.lead_id, "routed": routed, "exploration": explore,
                  "label_propensity": np.where(routed, 1.0, .05),
                  "sdr_disposition": np.where(routed | explore, sdr, None)}).to_csv(OUT / "labels_routed.csv", index=False)

    leads.to_csv(OUT / "impromptune_ready.csv", index=False)   # same columns: no sealed, no live values

    SEALED.mkdir(exist_ok=True)
    pd.DataFrame({"lead_id": L.lead_id, "true_label": Z["true_label"], "rule_noiseless": Z["rule_noiseless"],
                  "fit": Z["fit"].round(4), "intent": Z["intent"].round(4), "fit_logit": Z["fit_logit"].round(4),
                  "intent_logit": Z["intent_logit"].round(4), "is_partner_domain": Z["is_partner_domain"],
                  "seniority": L.seniority, "department": L.department, "msg_kind": D.kind, "msg_level": D.level,
                  "msg_competitor": D.competitor, "msg_budget": D.budget, "msg_timeline": D.timeline,
                  "text_score": D.text_score, "true_stack": [";".join(s) for s in C.tools],
                  "rater_flipped": Z["true_label"] != sdr}).to_csv(SEALED / "true_labels.csv", index=False)
    meta = {"seed": SEED, "text": a.text, "model": MODEL, "t_fit": float(Z["t_fit"]), "t_int": float(Z["t_int"]),
            "gen_stats": st, "cost_new_usd": round(cost(st["in_new"], st["out_new"]), 4),
            "cost_total_usd": round(cost(st["in_all"], st["out_all"]), 4)}
    (SEALED / "generation_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
