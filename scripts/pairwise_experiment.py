"""True pairwise zero-force-within-d_min soft-core, grid-bucketed (no full O(N^2)).

Pairwise box-overlap repulsion U_ij = mask * relu(hw_i+hw_j+margin-|dx|)*relu(hh_i+hh_j+margin-|dy|),
with a distance DEADZONE mask = sigmoid((r_ij - d_min)/tau): ~0 within d_min (macros
slide/pass freely -> sort ordering), ~1 beyond (normal repulsion). d_min anneals to 0.
Neighbor pairs found by a grid cell-list (cutoff = box+margin, 3x3 gather) -> ~O(N).

Arms (paired, shared seed): grid_baseline (config 221, reference) | pw_nodz (pairwise,
no deadzone) | pw_dz (pairwise + annealed deadzone). pw_dz vs pw_nodz isolates the
swapping-zone effect. Usage: python pairwise_experiment.py [ibm01 ibm09 ibm17] [--n 3] | --check | --report
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
from placers.analytical import DifferentiablePlacer, anneal, proxy_loss
from congestion_experiment import hp_from_params
CHAL="/home/laz/partcl/macro-place-challenge-2026"; ROOT="/home/laz/partcl/my-macro-placer"; CK=f"{ROOT}/notes/pairwise_ck"
LAM_PW=8e-4; MARGIN_FRAC=0.04; DMIN0_FRAC=0.06; DZ_FRAC=0.6; TAU_DZ_FRAC=0.01   # rough, un-tuned
SETTINGS=[("grid_base","grid"),("pw_nodz",("pw",False)),("pw_dz",("pw",True))]
ap=argparse.ArgumentParser(); ap.add_argument("benches",nargs="*",default=["ibm01","ibm09","ibm17"])
ap.add_argument("--n",type=int,default=3); ap.add_argument("--iters",type=int,default=15000)
ap.add_argument("--check",action="store_true"); ap.add_argument("--report",action="store_true"); a=ap.parse_args()
os.makedirs(CK,exist_ok=True)
hp=hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
LD0=hp["lam_d_end"]*hp["lam_d_ratio"]; LD1=hp["lam_d_end"]

def cell_pairs(coord, Rc, W, H):
    """i<j pairs within Chebyshev cutoff Rc via a grid cell-list (~O(N))."""
    N=coord.shape[0]; dev=coord.device
    nc=max(1,int(W//Rc)); nr=max(1,int(H//Rc)); cw=W/nc; ch=H/nr
    col=(coord[:,0]/cw).long().clamp(0,nc-1); row=(coord[:,1]/ch).long().clamp(0,nr-1)
    cell=row*nc+col
    order=torch.argsort(cell); scell=cell[order]
    ii=[]; jj=[]
    ar=torch.arange(N,device=dev)
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            qc=col+dc; qr=row+dr; valid=(qc>=0)&(qc<nc)&(qr>=0)&(qr<nr)
            q=(qr.clamp(0,nr-1)*nc+qc.clamp(0,nc-1))
            lo=torch.searchsorted(scell,q,right=False); hi=torch.searchsorted(scell,q,right=True)
            cnt=((hi-lo)*valid).clamp(min=0)
            tot=int(cnt.sum())
            if tot==0: continue
            i_rep=torch.repeat_interleave(ar,cnt)
            base=torch.cumsum(cnt,0)-cnt
            pos=torch.arange(tot,device=dev)-torch.repeat_interleave(base,cnt)
            j_rep=order[torch.repeat_interleave(lo,cnt)+pos]
            ii.append(i_rep); jj.append(j_rep)
    i=torch.cat(ii); j=torch.cat(jj); m=i<j
    return i[m], j[m]

def pw_energy(P, coord, i, j, margin, d_min, tau_dz):
    dx=coord[i,0]-coord[j,0]; dy=coord[i,1]-coord[j,1]
    ox=torch.relu(P.hw[i]+P.hw[j]+margin-dx.abs()); oy=torch.relu(P.hh[i]+P.hh[j]+margin-dy.abs())
    ov=ox*oy
    if d_min>0:
        r=torch.sqrt(dx*dx+dy*dy+1e-9); ov=ov*torch.sigmoid((r-d_min)/tau_dz)
    return ov.sum()

def report():
    cells=defaultdict(dict)
    for f in glob.glob(f"{CK}/*.json"):
        d=json.load(open(f)); cells[(d["bench"],d["seed"])][d["arm"]]=d["proxy"]
    print("\n=== pairwise soft-core: arms (abs proxy) & deadzone effect (pw_dz - pw_nodz) ===",flush=True)
    agg=defaultdict(lambda:defaultdict(list)); dz=defaultdict(list)
    for (bench,seed),arms in cells.items():
        for k,v in arms.items(): agg[bench][k].append(v); agg["ALL"][k].append(v)
        if "pw_dz" in arms and "pw_nodz" in arms:
            dz[bench].append(arms["pw_dz"]-arms["pw_nodz"]); dz["ALL"].append(arms["pw_dz"]-arms["pw_nodz"])
    for bench in sorted(agg):
        means={k:np.mean(v) for k,v in agg[bench].items()}
        d=np.array(dz[bench]); dzs=f"deadzone d={d.mean():+.4f} ({int((d<0).sum())}/{len(d)})" if len(d) else ""
        print(f"  {bench:5s}: "+" ".join(f"{k}={means[k]:.4f}" for k in ['grid_base','pw_nodz','pw_dz'] if k in means)+f"  | {dzs}",flush=True)
if a.report: report(); sys.exit(0)

def place(P, seed, mode, iters):
    g=torch.Generator(device="cpu").manual_seed(seed); c0=torch.rand(P.b.num_macros,2,generator=g)
    c0[:,0]=c0[:,0]*(P.W-2)+1; c0[:,1]=c0[:,1]*(P.H-2)+1; coord=c0.to(P.dev).requires_grad_(True)
    opt=torch.optim.Adam([coord],lr=hp["lr"])
    gloss=proxy_loss(gamma_hi_mult=hp["gamma"],lam_d0=LD0,lam_d1=LD1,lam_c0=0.0,lam_c1=0.0,tau_d=hp["tau_d"],tau_c=hp["tau_c"],target=hp["target"],topk="lapsum",dmode="overflow",lam_wl0=hp["lam_wl0"],ramp_p=hp["ramp_p"])
    margin=MARGIN_FRAC*min(P.W,P.H); Rc=float(2*max(P.hw.max().item(),P.hh.max().item())+margin+DMIN0_FRAC*min(P.W,P.H))
    tau_dz=TAU_DZ_FRAC*min(P.W,P.H); ipair=jpair=None; lam_pw_base=1.0
    if mode!="grid":                                       # calibrate pairwise weight to the grid-density force at init
        x0=coord.detach().clone().requires_grad_(True); P.density(x0,tau=hp["tau_d"],target=hp["target"],topk="lapsum",mode="overflow").backward(); gd=x0.grad.norm().item()
        ip0,jp0=cell_pairs(coord.detach(),Rc,P.W,P.H)
        x0=coord.detach().clone().requires_grad_(True); pw_energy(P,x0,ip0,jp0,margin,0.0,tau_dz).backward(); gp=x0.grad.norm().item()
        lam_pw_base=gd/(gp+1e-12)
    for it in range(iters):
        t=it/iters; te=t**hp["ramp_p"]
        opt.zero_grad()
        if mode=="grid":
            gloss(P,coord,t).backward()
        else:
            gamma=anneal(te,P.gw*hp["gamma"],P.gw,geometric=True); lam_wl=anneal(te,max(hp["lam_wl0"],1e-9),1.0,geometric=True); lam_d=anneal(te,LD0,LD1)
            if it%20==0:                                   # rebuild neighbor list periodically
                ipair,jpair=cell_pairs(coord.detach(),Rc,P.W,P.H)
            d_min=(DMIN0_FRAC*min(P.W,P.H))*(1.0-min(1.0,t/DZ_FRAC)) if mode[1] else 0.0
            (lam_wl*P.wirelength(coord,gamma)+lam_d*lam_pw_base*pw_energy(P,coord,ipair,jpair,margin,d_min,tau_dz)).backward()
        opt.step()
        with torch.no_grad(): coord[:,0].clamp_(P.hw,P.W-P.hw); coord[:,1].clamp_(P.hh,P.H-P.hh)
    return coord.detach().cpu().numpy()

if a.check:
    b,plc=load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/ibm01"); P=DifferentiablePlacer(FastEval(b,plc))
    g=torch.Generator().manual_seed(0); c=torch.rand(b.num_macros,2,generator=g); c[:,0]=c[:,0]*(P.W-2)+1; c[:,1]=c[:,1]*(P.H-2)+1; x=c.to(P.dev)
    margin=MARGIN_FRAC*min(P.W,P.H); Rc=float(2*max(P.hw.max().item(),P.hh.max().item())+margin+DMIN0_FRAC*min(P.W,P.H))
    ib,jb=cell_pairs(x,Rc,P.W,P.H)
    # brute force pairs within Chebyshev Rc
    dxm=(x[:,0][:,None]-x[:,0][None,:]).abs(); dym=(x[:,1][:,None]-x[:,1][None,:]).abs()
    within=(dxm<Rc)&(dym<Rc); iu,ju=torch.triu_indices(x.shape[0],x.shape[0],1,device=P.dev)
    brute=set(map(tuple,torch.stack([iu[within[iu,ju]],ju[within[iu,ju]]],1).tolist()))
    got=set(map(tuple,torch.stack([ib,jb],1).tolist()))
    print(f"cell-list vs brute: got={len(got)} brute={len(brute)} missing={len(brute-got)} extra_outside_Rc={len([p for p in got-brute])}")
    e1=float(pw_energy(P,x,ib,jb,margin,0.0,1.0)); ib2,jb2=iu[within[iu,ju]],ju[within[iu,ju]]; e2=float(pw_energy(P,x,ib2,jb2,margin,0.0,1.0))
    print(f"energy cell-list={e1:.4f} brute={e2:.4f} match={abs(e1-e2)<1e-3*max(1,abs(e2))}"); sys.exit(0)

for nm in a.benches:
    b,plc=load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
    fe=FastEval(b,plc); P=DifferentiablePlacer(fe); sz=b.macro_sizes.numpy()
    for seed in range(a.n):
        for arm,mode in SETTINGS:
            ck=f"{CK}/{nm}_s{seed}_{arm}.json"
            if os.path.exists(ck): continue
            raw=place(P,seed,mode,a.iters); lg,_,_=legalize(raw,sz,b.num_hard_macros,b.canvas_width,b.canvas_height,gap=0.01)
            pol,_=optimize_fast(fe,lg,iters=100000,seed=seed,T0=0.0,move_hard=False,move_soft=True,refresh=20000,log_every=10**9,logf=lambda *x:None)
            c=compute_proxy_cost(torch.tensor(pol,dtype=torch.float32),b,plc)
            json.dump({"bench":nm,"seed":seed,"arm":arm,"proxy":float(c["proxy_cost"]),"overlaps":int(c["overlap_count"])},open(ck,"w"))
            print(f"  {nm} s{seed} {arm:9s}: proxy={float(c['proxy_cost']):.4f} ov={int(c['overlap_count'])}",flush=True)
    report()
print("DONE",flush=True)
