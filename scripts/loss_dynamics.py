import os, sys, json
os.environ["OMP_NUM_THREADS"]="2"
sys.path.insert(0,"/home/laz/partcl/my-macro-placer"); sys.path.insert(0,"/home/laz/partcl/my-macro-placer/scripts")
import torch
from macro_place.loader import load_benchmark_from_dir
from placers.fast_eval import FastEval
from placers.analytical import DifferentiablePlacer, anneal
from congestion_experiment import hp_from_params
CHAL="/home/laz/partcl/macro-place-challenge-2026"; ROOT="/home/laz/partcl/my-macro-placer"
hp=hp_from_params(json.load(open(f"{ROOT}/notes/bayes_final.json"))["params"])
LD0=hp["lam_d_end"]*hp["lam_d_ratio"]; LD1=hp["lam_d_end"]; ITERS=15000
nm=sys.argv[1] if len(sys.argv)>1 else "ibm01"
b,plc=load_benchmark_from_dir(f"{CHAL}/external/MacroPlacement/Testcases/ICCAD04/{nm}")
P=DifferentiablePlacer(FastEval(b,plc))
g=torch.Generator().manual_seed(0); c=torch.rand(b.num_macros,2,generator=g); c[:,0]=c[:,0]*(P.W-2)+1; c[:,1]=c[:,1]*(P.H-2)+1
coord=c.to(P.dev).requires_grad_(True); opt=torch.optim.Adam([coord],lr=hp["lr"])
print(f"{nm}: iter   t   gamma  lam_wl  lam_d |  WL_term  D_term | |gradWL| |gradD|  (D_term = weighted density loss)")
for it in range(ITERS):
    t=it/ITERS; te=t**hp["ramp_p"]
    gamma=anneal(te,P.gw*hp["gamma"],P.gw,geometric=True); lam_d=anneal(te,LD0,LD1); lam_wl=anneal(te,max(hp["lam_wl0"],1e-9),1.0,geometric=True)
    if it%1500==0 or it==ITERS-1:
        x=coord.detach().clone().requires_grad_(True); (lam_wl*P.wirelength(x,gamma)).backward(); gwl=x.grad.norm().item(); wlv=(lam_wl*P.wirelength(x.detach(),gamma)).item()
        x=coord.detach().clone().requires_grad_(True); (lam_d*P.density(x,tau=hp["tau_d"],target=hp["target"],topk="lapsum",mode="overflow")).backward(); gd=x.grad.norm().item(); dv=(lam_d*P.density(x.detach(),tau=hp["tau_d"],target=hp["target"],topk="lapsum",mode="overflow")).item()
        print(f"  {it:5d} {t:.2f} {gamma:6.1f} {lam_wl:6.3f} {lam_d:.4f} | {wlv:8.4f} {dv:.4f} | {gwl:.2e} {gd:.2e}")
    opt.zero_grad(); (lam_wl*P.wirelength(coord,gamma)+lam_d*P.density(coord,tau=hp["tau_d"],target=hp["target"],topk="lapsum",mode="overflow")).backward(); opt.step()
    with torch.no_grad(): coord[:,0].clamp_(P.hw,P.W-P.hw); coord[:,1].clamp_(P.hh,P.H-P.hh)
