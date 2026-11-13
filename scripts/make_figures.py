"""Turn results/*.json into the paper's tables (LaTeX), figures (PDF) and number macros.
Every number in the manuscript is produced here from the result files."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES, OUT = "results", "paper/generated"
os.makedirs(OUT, exist_ok=True)
TASKS = ["x-conflict", "x-redundancy", "x-irrelevancy", "x-gap"]
DS = {"agentic": "Agentic", "xu_stoller": "XS-ABAC", "amazon": "AMZ-Access"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "pdf.fonttype": 42})
C = {"rule": "#1f4e79", "learned": "#c0504d", "base": "#7f7f7f", "alt": "#e0a100", "g": "#3a7d44"}
macros = {}


def load(name):
    p = f"{RES}/{name}.json"
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    for r in rows:
        for t in TASKS:
            if isinstance(r.get(t), dict):
                r[t + "_f1"], r[t + "_ap"], r[t + "_prec"], r[t + "_rec"] = r[t]["f1"], r[t]["ap"], r[t]["prec"], r[t]["rec"]
                r[t + "_pos"], r[t + "_n"] = r[t]["pos"], r[t]["n"]
                del r[t]
            elif t in r:
                r[t + "_f1"] = r.pop(t)
        # a task without any positive instance in a run has no defined F1: exclude it from the averages
        if all((t + "_pos") in r for t in TASKS):
            for t in TASKS:
                if r[t + "_pos"] == 0:
                    r[t + "_f1"] = r[t + "_ap"] = r[t + "_prec"] = r[t + "_rec"] = float("nan")
            r["macro_f1"] = float(np.nanmean([r[t + "_f1"] for t in TASKS]))
            r["macro_ap"] = float(np.nanmean([r[t + "_ap"] for t in TASKS]))
    return pd.DataFrame(rows)


def ms(x, d=3):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "--", "--"
    return f"{x.mean():.{d}f}", f"{x.std(ddof=1) if len(x) > 1 else 0:.{d}f}"


def cell(x, bold=False):
    m, s = ms(x)
    body = f"{m}\\,{{\\scriptsize$\\pm${s}}}"
    return f"\\textbf{{{m}}}\\,{{\\scriptsize$\\pm${s}}}" if bold else body


def mac(name, val):
    macros[name] = val


# ----------------------------------------------------------------------------- main table
def main_table():
    df = load("main")
    if df is None:
        return
    order = ["RF (20% audited labels)", "RF + count context (20% audited labels)", "Local-only", "FedGCN-clique",
             "FedHGNN-count", "DyFHL-PA-L", "DyFHL-PA-R"]
    short = {"RF (20% audited labels)": "RF, local features$^\\dagger$", "RF + count context (20% audited labels)":
             "RF + count context$^\\dagger$", "Local-only": "Local-only", "FedGCN-clique": "FedGCN (clique exp.)",
             "FedHGNN-count": "FedHGNN (count completion)", "DyFHL-PA-L": "\\textbf{DyFHL-PA-L} (learned decoder)",
             "DyFHL-PA-R": "\\textbf{DyFHL-PA-R} (rule decoder)"}
    cols = [t + "_f1" for t in TASKS] + ["macro_f1", "macro_ap"]
    lines = []
    for ds, dname in DS.items():
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        nseeds = sub.seed.nunique()
        best = {c: max(sub[sub.method == m][c].mean() for m in order if (sub.method == m).any()) for c in cols}
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{dname}}} ({nseeds} seeds)}}\\\\")
        for m in order:
            x = sub[sub.method == m]
            if x.empty:
                continue
            lines.append(short[m] + " & " + " & ".join(cell(x[c], abs(x[c].mean() - best[c]) < 1e-9) for c in cols) + "\\\\")
        lines.append("\\midrule")
        # numbers for the text
        key = ds.replace("_", "")
        for m, tag in [("DyFHL-PA-R", "R"), ("DyFHL-PA-L", "L"), ("FedHGNN-count", "H"), ("Local-only", "Loc"),
                       ("RF + count context (20% audited labels)", "RFc"), ("FedGCN-clique", "G")]:
            x = sub[sub.method == m]
            if not x.empty:
                mac(f"main{key}{tag}", ms(x.macro_f1)[0]); mac(f"main{key}{tag}ap", ms(x.macro_ap)[0])
                for t in TASKS:
                    mac(f"main{key}{tag}{t[2:5]}", ms(x[t + '_f1'])[0])
                    mac(f"main{key}{tag}{t[2:5]}rec", ms(x[t + '_rec'])[0]); mac(f"main{key}{tag}{t[2:5]}prec", ms(x[t + '_prec'])[0])
        # paired test: rule decoder vs best baseline
        base = [m for m in order if not m.startswith("DyFHL")]
        bb = max(base, key=lambda m: sub[sub.method == m].macro_f1.mean())
        a = sub[sub.method == "DyFHL-PA-R"].sort_values("seed").macro_f1.values
        b = sub[sub.method == bb].sort_values("seed").macro_f1.values
        mac(f"main{key}best", f"{b.mean():.3f}")
        if len(a) == len(b) and len(a) > 2:
            p = stats.ttest_rel(a, b).pvalue
            mac(f"main{key}p", (f"{p:.2f}" if p >= 0.01 else f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"))
            mac(f"main{key}gain", f"{a.mean() - b.mean():+.3f}")
            print(key, "best baseline:", bb)
        a2 = sub[sub.method == "DyFHL-PA-L"].sort_values("seed").macro_f1.values
        if len(a2) == len(a) and len(a) > 2:
            mac(f"main{key}pRL", f"{stats.ttest_rel(a, a2).pvalue:.3f}")
        st = sub[sub.method == "DyFHL-PA-R"]
        for t in TASKS:
            mac(f"prev{key}{t[2:5]}", f"{100 * st[t + '_pos'].mean():.1f}")
        mac(f"n{key}rules", f"{int(st['x-conflict_n'].mean())}"); mac(f"n{key}req", f"{int(st['x-gap_n'].mean())}")
        mac(f"cost{key}ctx", f"{st.context_nnz_bytes.mean() / 1024:.0f}")
        ll = sub[sub.method == "DyFHL-PA-L"]
        if not ll.empty:
            mac(f"cost{key}model", f"{ll.model_bytes.mean() / 1024:.0f}"); mac(f"time{key}L", f"{ll.train_s.mean():.0f}")
            mac(f"time{key}R", f"{st.infer_s.mean():.1f}")
    open(f"{OUT}/tab_main.tex", "w").write("\n".join(lines[:-1]) + "\n")


# ----------------------------------------------------------------------------- ablations
def ablation_tables():
    df = load("ablation")
    if df is None:
        return
    for group, fn in [("structure", "tab_abl_structure.tex"), ("learning", "tab_abl_learning.tex")]:
        sub = df[df.group == group]
        dss = [d for d in DS if (sub.dataset == d).any()]
        lines = []
        for v in sub.variant.unique():
            row = [v.replace("- ", "$-$ ").replace("+ ", "$+$ ").replace("%", "\\%")]
            for d in dss:
                x = sub[(sub.variant == v) & (sub.dataset == d)]
                row.append(cell(x.macro_f1) if len(x) else "--")
            x = sub[sub.variant == v]
            row += [f"{x[t + '_f1'].mean():.3f}" for t in TASKS]
            lines.append(" & ".join(row) + "\\\\")
            tag = "".join(ch for ch in v if ch.isalpha())[:14]
            mac(f"abl{group[:3]}{tag}", f"{x.macro_f1.mean():.3f}")
            for d in dss:
                xx = x[x.dataset == d]
                mac(f"abl{group[:3]}{tag}{d.replace('_', '')}", f"{xx.macro_f1.mean():.3f}")
            for t in TASKS:
                mac(f"abl{group[:3]}{tag}{t[2:5]}", f"{x[t + '_f1'].mean():.3f}")
        open(f"{OUT}/{fn}", "w").write("\n".join(lines) + "\n")
        mac(f"abl{group[:3]}ncols", str(len(dss)))


# ----------------------------------------------------------------------------- sketch size
def sketch_fig():
    df = load("sketch")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    for ax, ds in zip(axes, ["agentic", "amazon"]):
        for dec, col, mk in [("rule", C["rule"], "o"), ("learned", C["learned"], "s")]:
            g = df[(df.dataset == ds) & (df.decoder == dec)].groupby("B").macro_f1.agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], g["std"].fillna(0), color=col, marker=mk, ms=3.5, lw=1.2, capsize=2,
                        label=f"DyFHL-PA-{'R' if dec == 'rule' else 'L'}")
        ax.set_xscale("log", base=2); ax.set_xlabel("sketch bins $B$"); ax.set_title(DS[ds], fontsize=9)
        ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    axes[0].set_ylabel("macro-F1"); axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax = axes[2]
    for ds, ls in [("agentic", "-"), ("amazon", "--")]:
        g = df[(df.dataset == ds) & (df.decoder == "rule")].groupby("B")
        ax.plot(g.context_nnz_bytes.mean().index, g.context_nnz_bytes.mean() / 1024, ls, color=C["rule"], marker="o", ms=3,
                label=f"non-zeros, {DS[ds]}")
        ax.plot(g.context_bytes.mean().index, g.context_bytes.mean() / 1024, ls, color=C["base"], marker="^", ms=3,
                label=f"dense, {DS[ds]}")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sketch bins $B$"); ax.set_ylabel("upload per domain (kB)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=1, handlelength=1.5); ax.grid(alpha=.25); ax.set_title("Communication", fontsize=9)
    ax.set_ylim(top=ax.get_ylim()[1] * 300)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sketch.pdf"); plt.close(fig)
    for ds in ["agentic", "amazon", "xu_stoller"]:
        for dec in ["rule", "learned"]:
            g = df[(df.dataset == ds) & (df.decoder == dec)].groupby("B").macro_f1.mean()
            for B in g.index:
                mac(f"sk{ds}{dec}{'B'}{num(B)}", f"{g[B]:.3f}")
        g = df[(df.dataset == ds) & (df.decoder == "rule")].groupby("B")
        for B in g.context_nnz_bytes.mean().index:
            mac(f"sk{ds}bytesB{num(B)}", f"{g.context_nnz_bytes.mean()[B] / 1024:.0f}")
        for t in TASKS:
            gg = df[(df.dataset == ds) & (df.decoder == "rule")].groupby("B")
            for B in [512, 4096]:
                if B in gg[t + "_prec"].mean().index:
                    mac(f"sk{ds}{t[2:5]}fB{num(B)}", f"{gg[t + '_f1'].mean()[B]:.3f}")
                    mac(f"sk{ds}{t[2:5]}precB{num(B)}", f"{gg[t + '_prec'].mean()[B]:.3f}")
                    mac(f"sk{ds}{t[2:5]}recB{num(B)}", f"{gg[t + '_rec'].mean()[B]:.3f}")


def num(n):
    words = {32: "xxxii", 64: "lxiv", 128: "cxxviii", 256: "cclvi", 512: "dxii", 1024: "mxxiv", 2048: "mmxlviii", 4096: "mmmmxcvi"}
    return words.get(int(n), str(n))


# ----------------------------------------------------------------------------- noise
def noise_fig():
    df = load("noise")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
    style = {"rule": (C["rule"], "o", "DyFHL-PA-R"), "learned (noise-aware)": (C["learned"], "s", "DyFHL-PA-L, noise-aware")}
    for ax, ds in zip(axes, ["agentic", "amazon"]):
        for dec, (col, mk, lab) in style.items():
            g = df[(df.dataset == ds) & (df.decoder == dec)].groupby("sigma").macro_f1.agg(["mean", "std"])
            if len(g):
                ax.errorbar(g.index, g["mean"], g["std"].fillna(0), color=col, marker=mk, ms=3.5, lw=1.2, capsize=2, label=lab)
        ax.set_xlabel("context noise $\\sigma$"); ax.set_title(DS[ds], fontsize=9); ax.grid(alpha=.25); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("macro-F1"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_noise.pdf"); plt.close(fig)
    delta = 1e-5
    lines = []
    for sg in sorted(df.sigma.unique()):
        row = [f"{sg:g}"]
        for ds in ["agentic", "amazon"]:
            A = int(df[df.dataset == ds].A.iloc[0])
            eps = "$\\infty$" if sg == 0 else f"{np.sqrt(2 * A) * np.sqrt(2 * np.log(1.25 / delta)) / sg:.0f}"
            row.append(eps)
            for dec in ["rule", "learned (noise-aware)"]:
                x = df[(df.dataset == ds) & (df.decoder == dec) & (df.sigma == sg)]
                row.append(cell(x.macro_f1))
                mac(f"nz{ds}{'R' if dec == 'rule' else 'L'}s{str(sg).replace('.', 'p')}", ms(x.macro_f1)[0])
        lines.append(" & ".join(row) + "\\\\")
    open(f"{OUT}/tab_noise.tex", "w").write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- scalability
def scale_fig():
    df = load("scale")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    a = df[df.B_mode.isin(["fixed", "scaled"])].copy()
    a["total"] = a.n_rules * a.K
    g = a[a.B_mode == "fixed"].groupby("total")
    axes[0].plot(g.oracle_s.mean().index, g.oracle_s.mean(), color=C["base"], marker="^", ms=3.5, label="pooled exact oracle")
    for mode, col, lab in [("fixed", C["rule"], "DyFHL-PA-R, $B{=}512$"), ("scaled", C["g"], "DyFHL-PA-R, $B{=}4n$")]:
        g = a[a.B_mode == mode].groupby("total")
        axes[0].plot(g.dyfhl_s.mean().index, g.dyfhl_s.mean(), color=col, marker="o", ms=3.5, label=lab)
        axes[1].errorbar(g.macro_f1.mean().index, g.macro_f1.mean(), g.macro_f1.std().fillna(0), color=col, marker="o", ms=3.5,
                         capsize=2, label=lab)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("rules in shared scopes"); axes[0].set_ylabel("seconds")
    axes[0].legend(frameon=False, fontsize=6.5); axes[0].grid(alpha=.25)
    axes[1].set_xscale("log"); axes[1].set_xlabel("rules in shared scopes"); axes[1].set_ylabel("macro-F1"); axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=6.5); axes[1].grid(alpha=.25)
    k = df[df.B_mode == "domains"].groupby("K").macro_f1.agg(["mean", "std"])
    axes[2].errorbar(k.index, k["mean"], k["std"].fillna(0), color=C["rule"], marker="o", ms=3.5, capsize=2)
    axes[2].set_xlabel("domains $K$"); axes[2].set_ylabel("macro-F1"); axes[2].set_ylim(0, 1.02); axes[2].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_scale.pdf"); plt.close(fig)
    for mode in ["fixed", "scaled"]:
        g = a[a.B_mode == mode].groupby("total")
        for tot in g.macro_f1.mean().index:
            mac(f"sc{mode}f{num2(tot)}", f"{g.macro_f1.mean()[tot]:.3f}")
            mac(f"sc{mode}t{num2(tot)}", f"{g.dyfhl_s.mean()[tot]:.2f}")
            mac(f"sc{mode}o{num2(tot)}", f"{g.oracle_s.mean()[tot]:.2f}")
            mac(f"sc{mode}b{num2(tot)}", f"{g.nnz_bytes.mean()[tot] / 1024:.0f}")
    for K in k.index:
        mac(f"scK{num2(K)}", f"{k['mean'][K]:.3f}")


def num2(n):
    return "".join("abcdefghij"[int(c)] for c in str(int(n)))


# ----------------------------------------------------------------------------- stream
def stream_fig():
    df = load("stream")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    for ax, ds in zip(axes, ["agentic", "amazon"]):
        for churn, ls in [(0.05, ":"), (0.1, "--"), (0.2, "-")]:
            g = df[(df.dataset == ds) & (df.churn == churn)].groupby("t")
            ax.plot(g.f1_delta.mean().index, g.f1_delta.mean(), ls, color=C["rule"], lw=1.3, label=f"delta, churn {churn:g}")
            ax.plot(g.f1_stale.mean().index, g.f1_stale.mean(), ls, color=C["learned"], lw=1.3, label=f"stale, churn {churn:g}")
        ax.set_xlabel("time step"); ax.set_title(DS[ds], fontsize=9); ax.grid(alpha=.25); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("macro-F1"); axes[0].legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
    d = df[df.t > 0].groupby(["dataset", "churn"])[["bytes_delta", "bytes_full"]].mean()
    x = np.arange(3); wdt = 0.38
    for i, ds in enumerate(["agentic", "amazon"]):
        r = (d.loc[ds].bytes_delta / d.loc[ds].bytes_full).values
        axes[2].bar(x + (i - .5) * wdt, r, wdt, color=[C["rule"], C["g"]][i], label=DS[ds])
        for c, v in zip([0.05, 0.1, 0.2], r):
            mac(f"st{ds}ratio{str(c).replace('.', 'p')}", f"{100 * v:.0f}")
    axes[2].set_xticks(x); axes[2].set_xticklabels(["5%", "10%", "20%"]); axes[2].set_xlabel("churn per step")
    axes[2].set_ylabel("delta / full upload"); axes[2].legend(frameon=False, fontsize=7); axes[2].grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_stream.pdf"); plt.close(fig)
    mac("stmaxerr", f"{df.max_abs_err.max():.0e}" if df.max_abs_err.max() > 0 else "0")
    last = df[df.t == df.t.max()]
    for ds in ["agentic", "amazon"]:
        for c in [0.05, 0.1, 0.2]:
            x = last[(last.dataset == ds) & (last.churn == c)]
            mac(f"st{ds}delta{str(c).replace('.', 'p')}", f"{x.f1_delta.mean():.3f}")
            mac(f"st{ds}stale{str(c).replace('.', 'p')}", f"{x.f1_stale.mean():.3f}")
    mac("sttdelta", f"{1000 * df[df.t > 0].time_delta.mean():.1f}"); mac("sttfull", f"{1000 * df[df.t > 0].time_full.mean():.1f}")


if __name__ == "__main__":
    for f in (main_table, ablation_tables, sketch_fig, noise_fig, scale_fig, stream_fig):
        f()
    with open(f"{OUT}/numbers.tex", "w") as fh:
        for k, v in sorted(macros.items()):
            name = "".join(ch if ch.isalpha() else "ABCDEFGHIJ"[int(ch)] if ch.isdigit() else "" for ch in k)
            fh.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
    print("wrote", len(macros), "macros to", OUT)
