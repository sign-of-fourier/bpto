"""OSHA Severe Injury Reports: 13-way event classification (top-12 two-digit OIICS event groups + Other).

The optimizer-vs-optimizer benchmark: official GEPA (`gepa` on PyPI) against bpto's q-EI at an equal task-call
budget, same task model, same reflector. `official_gepa.py` routes the official engine through bpto's
`ModelClient`s so both arms share cache, budget, retries and concurrency; `pretest.py` is the offline check that
it does (phase 0a).
"""
