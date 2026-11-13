"""Run every experiment reported in the paper and write results/*.json.
Usage: python scripts/run_experiments.py [main ablation sketch noise scale stream] [--quick]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dyfhl.policy import World, get_source
from dyfhl.model import Config, TASKS
from dyfhl.fed import run, make_clients, evaluate, exchange
from dyfhl.baselines import VARIANTS, rf_baseline
from dyfhl.dynamic import run_stream

torch.set_num_threads(int(os.environ.get('DYFHL_THREADS', 2)))
QUICK = "--quick" in sys.argv
WHAT = [a for a in sys.argv[1:] if not a.startswith("--")] or ["main", "ablation", "sketch", "noise", "scale", "stream"]
DATASETS = ["agentic", "xu_stoller", "amazon"]
SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
SEEDS3 = SEEDS[:3]
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def slim(r):
    return {k: v for k, v in r.items()}


def save(name, rows):
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("saved", name, len(rows), flush=True)


def world(ds, seed, **kw):
    return World(get_source(ds), seed=seed, **kw)


def main():
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            w = world(ds, seed); lab = w.labels()
            def add(method, r):
                rows.append(dict(dataset=ds, seed=seed, method=method, **slim(r)))
                print(ds, seed, method, round(r["macro_f1"], 3), round(r["macro_ap"], 3), flush=True)
            add("RF (20% audited labels)", rf_baseline(w, lab, seed, context=False))
            add("RF + count context (20% audited labels)", rf_baseline(w, lab, seed, context=True))
            for name, ov in VARIANTS.items():
                add(name, run(w, Config(seed=seed, **ov), labels=lab))
            add("DyFHL-PA-R", run(w, Config(seed=seed, h=0), labels=lab, rule=True))
            save("main", rows)


def ablation():
    rows = []
    struct = {"full": {}, "- scope partition": dict(scope_part=False), "- conjunctive pooling": dict(conj=False),
              "- wildcard vertices": dict(wild=False), "- signature sketch (counts)": dict(sketch=False)}
    learned = {"full (learned decoder)": {}, "+ learned message channel (h=16)": dict(h=16),
               "+ hyperedge priors": dict(priors=True), "- FedAvg (local training)": dict(fedavg=False),
               "- pseudo-anomaly augmentation": dict(aug=0.0), "+ density matching (recomb=1.5)": dict(recomb=1.5)}
    for ds in DATASETS:
        for seed in SEEDS3:
            w = world(ds, seed); lab = w.labels()
            for name, ov in struct.items():
                r = run(w, Config(seed=seed, h=0, **ov), labels=lab, rule=True)
                rows.append(dict(dataset=ds, seed=seed, group="structure", variant=name, **slim(r)))
            for name, ov in learned.items():
                if ds == "xu_stoller":
                    continue
                r = run(w, Config(seed=seed, **ov), labels=lab)
                rows.append(dict(dataset=ds, seed=seed, group="learning", variant=name, **slim(r)))
                print(ds, seed, name, round(r["macro_f1"], 3), flush=True)
            save("ablation", rows)


def sketch():
    rows = []
    for ds in ["agentic", "amazon"]:
        for seed in SEEDS3:
            w = world(ds, seed); lab = w.labels()
            for B in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                r = run(w, Config(seed=seed, h=0, B=B), labels=lab, rule=True)
                rows.append(dict(dataset=ds, seed=seed, B=B, decoder="rule", **slim(r)))
                if B <= 1024:
                    r = run(w, Config(seed=seed, B=B), labels=lab)
                    rows.append(dict(dataset=ds, seed=seed, B=B, decoder="learned", **slim(r)))
                print(ds, seed, B, flush=True)
            save("sketch", rows)


def noise():
    rows = []
    for ds in ["agentic", "amazon"]:
        for seed in SEEDS3:
            w = world(ds, seed); lab = w.labels()
            for sg in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
                r = run(w, Config(seed=seed, h=0, sigma=sg), labels=lab, rule=True)
                rows.append(dict(dataset=ds, seed=seed, sigma=sg, decoder="rule", A=w.A, **slim(r)))
                r = run(w, Config(seed=seed, sigma=sg), labels=lab)
                rows.append(dict(dataset=ds, seed=seed, sigma=sg, decoder="learned (noise-aware)", A=w.A, **slim(r)))
                print(ds, seed, sg, flush=True)
            save("noise", rows)


def scale():
    rows = []
    for seed in SEEDS3:
        for n in [100, 200, 400, 800, 1600]:
            w = world("agentic", seed, n_rules=n)
            t0 = time.perf_counter(); lab = w.labels(); t_or = time.perf_counter() - t0
            for B in [512, 4 * n]:
                cfg = Config(seed=seed, h=0, B=B)
                clients, _ = make_clients(w, cfg, lab)
                from dyfhl.model import DyFHL
                m = DyFHL(clients[0].geo, cfg)
                t0 = time.perf_counter(); r, _ = evaluate(clients, m, lab, cfg, rule=True); t_inf = time.perf_counter() - t0
                u, _ = clients[0].upload(m)
                rows.append(dict(seed=seed, n_rules=n, K=w.K, B=B, B_mode="fixed" if B == 512 else "scaled",
                                 oracle_s=t_or, dyfhl_s=t_inf, macro_f1=r["macro_f1"],
                                 nnz_bytes=int((u != 0).sum()) * 8, **{t: r[t]["f1"] for t in TASKS}))
            print("scale n", seed, n, flush=True)
        for K in [3, 6, 9, 12]:
            w = world("agentic", seed, K=K, S=K)
            lab = w.labels()
            r = run(w, Config(seed=seed, h=0), labels=lab, rule=True)
            rows.append(dict(seed=seed, n_rules=400, K=K, B=512, B_mode="domains", macro_f1=r["macro_f1"],
                             **{t: r[t]["f1"] for t in TASKS}))
            print("scale K", seed, K, flush=True)
        save("scale", rows)


def stream():
    rows = []
    for ds in ["agentic", "amazon"]:
        for seed in SEEDS3:
            for churn in [0.05, 0.1, 0.2]:
                w = world(ds, seed)
                for r in run_stream(w, Config(seed=seed, h=0), T=10, churn=churn):
                    rows.append(dict(dataset=ds, seed=seed, churn=churn, **r))
                print("stream", ds, seed, churn, flush=True)
            save("stream", rows)


if __name__ == "__main__":
    t = time.time()
    for name in WHAT:
        globals()[name]()
        print("done", name, round(time.time() - t), "s", flush=True)
