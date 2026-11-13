"""DyFHL-PA explorer: pick a dataset, run detection or a policy stream, inspect the hypergraph.
Run with:  streamlit run app/streamlit_app.py"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, streamlit as st, torch
from dyfhl.policy import World, get_source
from dyfhl.model import Config, TASKS
from dyfhl.fed import run
from dyfhl.baselines import VARIANTS, rf_baseline
from dyfhl.dynamic import run_stream
from dyfhl import viz

st.set_page_config(page_title="DyFHL-PA", layout="wide")
st.title("DyFHL-PA — cross-domain policy anomaly analysis")
st.caption("Federated hypergraph analysis of access-control policies. Labels come from an exact oracle "
           "that pools all domains; the federated methods never see other domains' rules.")

with st.sidebar:
    st.header("Dataset")
    ds = st.selectbox("Rule source", ["agentic", "xu_stoller", "amazon"],
                      format_func={"agentic": "Synthetic agentic (agents, MCP servers, tools…)",
                                   "xu_stoller": "Xu–Stoller ABAC case studies (derived)",
                                   "amazon": "Amazon employee access (derived)"}.get)
    seed = st.number_input("Seed", 0, 999, 0)
    K = st.slider("Domains", 3, 12, 6)
    n_rules = st.select_slider("Shared-scope rules per domain", [100, 200, 400, 800], 400)
    st.header("Method")
    B = st.select_slider("Signature sketch bins B", [32, 64, 128, 256, 512, 1024, 2048, 4096], 512)
    sigma = st.slider("Context noise σ", 0.0, 1.0, 0.0, 0.05)
    decoder = st.radio("Decoder", ["Threshold rule (DyFHL-PA-R)", "Learned, label-free (DyFHL-PA-L)"])
    go_btn = st.button("Run detection", type="primary")


@st.cache_resource(show_spinner="Building policies and running the oracle…")
def build(ds, seed, K, n_rules):
    w = World(get_source(ds), seed=seed, K=K, S=K, n_rules=n_rules)
    return w, w.labels()


w, lab = build(ds, seed, K, n_rules)
R, Y, Q, gap = lab
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rules", len(R["eff"])); c2.metric("Attribute-value vertices", w.V)
c3.metric("x-conflict", f"{Y[:, 0].mean():.1%}"); c4.metric("x-redundancy", f"{Y[:, 1].mean():.1%}")
c5.metric("x-irrelevancy", f"{Y[:, 2].mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Detection", "Hypergraph projection", "Baselines", "Policy stream"])

if go_btn:
    rule = decoder.startswith("Threshold")
    cfg = Config(seed=seed, B=B, sigma=sigma, h=0)
    with st.spinner("Exchanging contexts and decoding…" if rule else "Federated label-free training…"):
        res, clients, model, _, (P, Pq) = run(w, cfg, labels=lab, rule=rule, return_all=True)
    st.session_state["res"] = (res, P, Pq, decoder)

with tab1:
    if "res" not in st.session_state:
        st.info("Choose settings in the sidebar and press **Run detection**.")
    else:
        res, P, Pq, dec = st.session_state["res"]
        st.subheader(dec)
        df = pd.DataFrame({t: {"F1": res[t]["f1"], "Precision": res[t]["prec"], "Recall": res[t]["rec"],
                               "AUPRC": res[t]["ap"], "Prevalence": res[t]["pos"], "n": res[t]["n"]} for t in TASKS}).T
        st.dataframe(df.style.format({"F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "AUPRC": "{:.3f}",
                                      "Prevalence": "{:.1%}", "n": "{:.0f}"}), width='stretch')
        a, b, c = st.columns(3)
        a.metric("Macro-F1", f"{res['macro_f1']:.3f}")
        b.metric("Context upload per domain (non-zeros)", f"{res['context_nnz_bytes'] / 1024:.0f} kB")
        c.metric("Train + inference time", f"{res['train_s'] + res['infer_s']:.1f} s")

with tab2:
    method = st.radio("Projection", ["svd", "tsne"], horizontal=True,
                      format_func={"svd": "Truncated SVD (fast)", "tsne": "t-SNE (slower)"}.get)
    pred = None
    if "res" in st.session_state:
        pred = st.session_state["res"][1] >= 0.5
        st.caption("Crosses mark rules where the last run disagrees with the oracle.")
    st.plotly_chart(viz.embedding_projection(R, Y, pred, method=method, seed=seed), width='stretch')
    sc = st.selectbox("Collaboration scope", list(range(w.S)),
                      format_func=lambda s: f"scope {s} — domains {w.members[s]}")
    st.plotly_chart(viz.bipartite_view(w, R, Y, scope=sc), width='stretch')

with tab3:
    st.write("Baselines trained with the same label-free protocol, plus random forests that receive 20% audited labels.")
    if st.button("Run baselines (about a minute)"):
        rows = {}
        with st.spinner("Running…"):
            rows["RF (20% audited labels)"] = rf_baseline(w, lab, seed, False)
            rows["RF + count context"] = rf_baseline(w, lab, seed, True)
            for name, ov in VARIANTS.items():
                if name != "DyFHL-PA-L":
                    rows[name] = run(w, Config(seed=seed, **ov), labels=lab)
            rows["DyFHL-PA-R"] = run(w, Config(seed=seed, h=0, B=B, sigma=sigma), labels=lab, rule=True)
        df = pd.DataFrame({k: {**{t: v[t]["f1"] for t in TASKS}, "macro-F1": v["macro_f1"], "macro-AUPRC": v["macro_ap"]}
                           for k, v in rows.items()}).T
        st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0), width='stretch')

with tab4:
    churn = st.slider("Rules replaced per step", 0.02, 0.3, 0.1, 0.01)
    T = st.slider("Steps", 3, 20, 10)
    if st.button("Run policy stream"):
        w2 = World(get_source(ds), seed=seed, K=K, S=K, n_rules=n_rules)
        with st.spinner("Streaming…"):
            rows = pd.DataFrame(run_stream(w2, Config(seed=seed, h=0, B=B, sigma=sigma), T=T, churn=churn))
        st.line_chart(rows.set_index("t")[["f1_delta", "f1_stale"]].rename(
            columns={"f1_delta": "delta-maintained context", "f1_stale": "stale context"}))
        st.line_chart(rows.set_index("t")[["bytes_delta", "bytes_full"]].rename(
            columns={"bytes_delta": "delta upload (bytes)", "bytes_full": "full re-upload (bytes)"}))
        st.caption(f"Largest difference between delta-maintained and recomputed aggregates: {rows.max_abs_err.max():.1e}")
