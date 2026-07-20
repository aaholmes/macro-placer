"""Soft-core RADIUS density: enlarge each macro's density footprint (splat sigma)
by an annealed d_min, so macros within ~d_min smear into a broad low bump (no
sharp repulsion -> a 'swapping zone') while farther macros still separate. d_min
shrinks to 0 -> recovers the true (hard-core) density. Grid-native (wider Gaussian
splat), no pairwise / ANN. At d_min=0 it equals the baseline density (checked).

Paired vs baseline, checkpointed. Usage:
  python radius_experiment.py [ibm01 ibm09 ibm17] [--n 5] | [--check] | [--report]
"""
import os, sys, json, glob, argparse
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="2"
sys.path.insert(0,"/home/laz/partcl/my-macro-placer"); sys.path.insert(0,"/home/laz/partcl/my-macro-placer/scripts")
import numpy as np, torch
from collections import defaultdict
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from placers.fast_eval import FastEval
from placers.legalizer import legalize
from placers.local_search import optimize_fast
from placers.analytical import DifferentiablePlacer, anneal, lapsum_topk_mean
from congestion_experiment import hp_from_params
CHAL="/home/laz/partcl/macro-place-challenge-2026"; ROOT="/home/laz/partcl/my-macro-placer"; CK=f"{ROOT}/notes/radius_ck"
# (d0 = initial extra sigma as frac of min(W,H); frac = when d_min hits 0)
SETTINGS=[("baseline",None),("rad_.03_.6",(0.03,0.6)),("rad_.06_.6",(0.06,0.6))]
ap=argparse.ArgumentParser(); ap.add_argument("benches",nargs="*",default=["ibm01","ibm09","ibm17"])
ap.add_argument("--n",type=int,default=5); ap.add_argument("--iters",type=int,default=15000)
ap.add_argument("--check",action="store_true"); ap.add_argument("--report",action="store_true"); a=ap.parse_args()
os.makedirs(CK,exist_ok=True)
hp=hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
LD0=hp["lam_d_end"]*hp["lam_d_ratio"]; LD1=hp["lam_d_end"]; TAU=hp["tau_d"]; TGT=hp["target"]

def report():
    cells=defaultdict(dict)
    for f in glob.glob(f"{CK}/*.json"):
        d=json.load(open(f)); cells[(d["bench"],d["seed"])][d["arm"]]=d["proxy"]
    labels=[l for l,_ in SETTINGS if l!="baseline"]; per=defaultdict(lambda:defaultdict(list))
    for (bench,seed),arms in cells.items():
        if "baseline" not in arms: continue
        for l in labels:
            if l in arms: per[bench][l].append(arms[l]-arms["baseline"]); per["ALL"][l].append(arms[l]-arms["baseline"])
    print("\n=== soft-core radius density vs baseline (paired; negative = better) ===",flush=True)
    for bench in sorted(per):
        parts=[f"{l} d={np.mean(per[bench][l]):+.4f} (win {int((np.array(per[bench][l])<0).sum())}/{len(per[bench][l])})" for l in labels if per[bench][l]]
        if parts: print(f"  {bench:5s}: "+" | ".join(parts),flush=True)
if a.report: report(); sys.exit(0)

def dfield_soft(P,coord,extra):
    sx=torch.clamp(P.hw,min=P.gw)+extra; sy=torch.clamp(P.hh,min=P.gh)+extra
    gx=torch.exp(-((coord[:,0:1]-P.cxs)**2)/(2*sx[:,None]**2)); gy=torch.exp(-((coord[:,1:2]-P.cys)**2)/(2*sy[:,None]**2))
    gx=gx/(gx.sum(1,keepdim=True)+1e-9); gy=gy/(gy.sum(1,keepdim=True)+1e-9)
    return torch.einsum("m,mr,mc->rc",P.area,gy,gx)/(P.gw*P.gh)
def density(P,coord,t,sc):
    if sc is None: return P.density(coord,tau=TAU,target=TGT,topk="lapsum",mode="overflow")
    d0,frac=sc; extra=d0*min(P.W,P.H)*(1.0-min(1.0,t/max(1e-9,frac)))
    return lapsum_topk_mean(torch.clamp(dfield_soft(P,coord,extra)-TGT,min=0.0),0.10,TAU)
def place(P,seed,sc,iters):
    g=torch.Generator(device="cpu").manual_seed(seed); c0=torch.rand(P.b.num_macros,2,generator=g)
    c0[:,0]=c0[:,0]*(P.W-2)+1; c0[:,1]=c0[:,1]*(P.H-2)+1; coord=c0.to(P.dev).requires_grad_(True)
    opt=torch.optim.Adam([coord],lr=hp["lr"])
    for it in range(iters):
        t=it/iters; te=t**hp["ramp_p"]
        gamma=anneal(te,P.gw*hp["gamma"],P.gw,geometric=True); lam_d=anneal(te,LD0,LD1); lam_wl=anneal(te,max(hp["lam_wl0"],1e-9),1.0,geometric=True)
        opt.zero_grad(); (lam_wl*P.wirelength(coord,gamma)+lam_d*density(P,coord,t,sc)).backward(); opt.step()
        with torch.no_grad(): coord[:,0].clamp_(P.hw,P.W-P.hw); coord[:,1].clamp_(P.hh,P.H-P.hh)
    return coord.detach().cpu().numpy()
if a.check:
    b,plc=load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/ibm01"); P=DifferentiablePlacer(FastEval(b,plc))
    g=torch.Generator().manual_seed(0); c=torch.rand(b.num_macros,2,generator=g); c[:,0]=c[:,0]*(P.W-2)+1; c[:,1]=c[:,1]*(P.H-2)+1; x=c.to(P.dev)
    d0=float(density(P,x,1.0,None)); d1=float(density(P,x,1.0,(0.03,0.6)))  # extra=0 at t=1
    print(f"radius(d_min=0) vs baseline: {d0:.6f} vs {d1:.6f} match={abs(d0-d1)<1e-5}"); sys.exit(0)
for nm in a.benches:
    b,plc=load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe=FastEval(b,plc); P=DifferentiablePlacer(fe); sz=b.macro_sizes.numpy()
    for seed in range(a.n):
        for arm,sc in SETTINGS:
            ck=f"{CK}/{nm}_s{seed}_{arm}.json"
            if os.path.exists(ck): continue
            raw=place(P,seed,sc,a.iters); lg,_,_=legalize(raw,sz,b.num_hard_macros,b.canvas_width,b.canvas_height,gap=0.01)
            pol,_=optimize_fast(fe,lg,iters=100000,seed=seed,T0=0.0,move_hard=False,move_soft=True,refresh=20000,log_every=10**9,logf=lambda *x:None)
            c=compute_proxy_cost(torch.tensor(pol,dtype=torch.float32),b,plc)
            json.dump({"bench":nm,"seed":seed,"arm":arm,"proxy":float(c["proxy_cost"]),"overlaps":int(c["overlap_count"])},open(ck,"w"))
            print(f"  {nm} s{seed} {arm:11s}: proxy={float(c['proxy_cost']):.4f} ov={int(c['overlap_count'])}",flush=True)
    report()
print("DONE",flush=True)
