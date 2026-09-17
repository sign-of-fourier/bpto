import json, glob, collections, numpy as np, math
def pairs(pat, nfull, m):
    out=[]
    for f in sorted(glob.glob(pat)):
        t=json.load(open(f)); nodes={n['id']:n for n in t['nodes']}
        full={i:n for i,n in nodes.items() if n.get('evaluation') and n['evaluation']['n']>=nfull}
        root=next(n for n in nodes.values() if n['parent_id'] is None); r0=root['evaluation']['metrics'][m]
        for i,n in full.items():
            p=n['parent_id']
            if p in full: out.append((full[p]['evaluation']['metrics'][m]-r0, n['evaluation']['metrics'][m]-r0, f.split('/')[-2]))
    return np.array([o[:2] for o in out]), [o[2] for o in out]
for name,pat,nf,m,noise in [('hotpot_program f1','runs/hotpot_program/*/tree.json',200,'f1',0.0098),
                            ('hotpot_program recall','runs/hotpot_program/*/tree.json',200,'sel_recall',0.0046),
                            ('hotpot_program3 f1','runs/hotpot_program3/*/tree.json',200,'f1',0.0098),
                            ('hotpot_program3 recall','runs/hotpot_program3/*/tree.json',200,'sel_recall',0.0046),
                            ('compress_v2 f1','runs/compress_v2/*/tree.json',100,'f1',0.0091)]:
    P,_=pairs(pat,nf,m); px,cy=P[:,0],P[:,1]
    print(f"\n{name}: {len(px)} pairs, noise σ={noise}")
    for lab,mask in [('parent ≥ root (Δ≥0)',px>=0),('parent < root',px<0),('parent ≥ root+σ',px>=noise),('parent ≥ root+2σ',px>=2*noise)]:
        if mask.sum()>=4:
            b,a=np.polyfit(px[mask],cy[mask],1); r=np.corrcoef(px[mask],cy[mask])[0,1]
            print(f"  {lab:22s} n={mask.sum():3d}  parent mean Δ {px[mask].mean():+.4f}  child mean Δ {cy[mask].mean():+.4f}  "
                  f"slope {b:+.2f} r {r:+.2f}  P(child ≥ root) {np.mean(cy[mask]>=0):.2f}  P(child ≥ parent) {np.mean(cy[mask]>=px[mask]):.2f}")
    # children of the root itself (not in pairs above? they are: parent=root, Δ=0) vs children of improved parents
