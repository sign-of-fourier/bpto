import json, statistics as st, difflib
items={i["key"]:i for i in json.load(open("pilot_items.json"))}
jev={(r["key"],r["rep"]):r for r in map(json.loads,open("jev_log.jsonl"))}
nova={(r["key"],r["rep"]):r for r in map(json.loads,open("nova_log.jsonl"))}
def auc(s,y):
    P=[a for a,b in zip(s,y) if b]; N=[a for a,b in zip(s,y) if not b]
    return sum((p>n)+0.5*(p==n) for p in P for n in N)/(len(P)*len(N))
def report(name,f,kind,rep_ok=True):
    ks=[k for k in items if items[k]["kind"]==kind]; y=[items[k]["label"] for k in ks]
    s0=[f(k,0) for k in ks]; s1=[f(k,1) for k in ks] if rep_ok else s0
    acc=sum((a>=.5)==b for a,b in zip(s0,y))/len(y)
    brier=st.mean((a-b)**2 for a,b in zip(s0,y))
    flips=sum((a>=.5)!=(b>=.5) for a,b in zip(s0,s1)); diff=st.mean(abs(a-b) for a,b in zip(s0,s1))
    print(f"{name:28s} n={len(y):2d} AUC {auc(s0,y):.2f}  acc@.5 {acc:.2f}  Brier {brier:.3f}  distinct {len(set(s0)):2d}  rep|Δ| {diff:.3f} flips {flips}")
for kind in ["qa","name"]:
    print(f"--- {kind}")
    if kind=="qa":
        report("token F1 (current metric)",lambda k,r:items[k]["f1"],kind,False)
    else:
        report("difflib ratio (string fuzzy)",lambda k,r:difflib.SequenceMatcher(None,items[k]["gold"].lower(),items[k]["pred"].lower()).ratio(),kind,False)
    report("Nova Micro llm_judge",lambda k,r:nova[(k,r)]["score"],kind)
    nq="correct" if kind=="qa" else "same"
    report("Jev noul",lambda k,r:jev[(k,r)]["answers"][nq]["noul"],kind)
    if kind=="qa": report("Jev score/3",lambda k,r:jev[(k,r)]["answers"]["grade"]["score"]/3,kind)
print("latency median s: jev %.2f  nova %.2f"%(st.median(r["latency"] for r in jev.values()),st.median(r["latency"] for r in nova.values())))
print("jev tokens in/out per call: %.0f/%.0f"%(st.mean(r["usage"]["input_tokens"] for r in jev.values()),st.mean(r["usage"]["output_tokens"] for r in jev.values())))
print("\nerrors at .5 (label, F1, nova, jev noul, jev score/3):")
for k,i in items.items():
    nq="correct" if i["kind"]=="qa" else "same"
    j=jev[(k,0)]["answers"][nq]["noul"]; n=nova[(k,0)]["score"]
    g=jev[(k,0)]["answers"]["grade"]["score"]/3 if i["kind"]=="qa" else None
    if (j>=.5)!=i["label"] or (n>=.5)!=i["label"]:
        print(k,i["label"],i.get("f1"),n,j,g,"|",i["gold"][:40],"||",i["pred"][:70])
