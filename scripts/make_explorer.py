"""Build ``docs/explorer.html``: an interactive reliability-diagram explorer for Exp A (arm x answer type x split x
temperature scaling), from the committed predictions. One self-contained file: open it in a browser.

    python scripts/make_explorer.py [--runs results/expA/runs] [--out docs/explorer.html]

Reliability curves pool the three seeds of an arm (15 equal-width confidence bins, top-label confidence); the ECE,
accuracy and log loss in the table are the seed means, as in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcd import evaluate as EV  # noqa: E402
from rlcd import scoring as S  # noqa: E402
from rlcd.plots import ARM_STYLES, FAMILIES  # noqa: E402

ARMS = [
    ("a0", "A0", "A0 start model", [("a0", 0)]),
    ("a1", "A1", "A1 supervised (A2 = A1 + TS)", [(f"a1_lr1e-4_s{s}", s) for s in (0, 1, 2)]),
    ("a3", "A3", "A3 RLVR", [(f"a3_lr1e-4_s{s}", s) for s in (0, 1, 2)]),
    ("a4", "A4", "A4 RLCD-bandit", [(f"a4_lr1e-4_s{s}", s) for s in (0, 1, 2)]),
    ("a5", "A5", "A5 RLCD-full", [("a5_lr1e-4_s0", 0)]),
    ("a6", "A6", "A6 mined", [(f"a6_lr1e-4_s{s}", s) for s in (0, 1, 2)]),
    ("a6r", "A6r", "A6r random", [(f"a6r_lr1e-4_s{s}", s) for s in (0, 1, 2)]),
    ("a3b0", "A3b0", "A3 β = 0 (exploratory)", [("a3beta0_lr1e-4_s0", 0)]),
]
SCOPES = {"all": None, "noul": 0, "choice": 1, "score": 2}
BINS = 15


def _run_stats(
    dists: list[np.ndarray], gold: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray | float]:
    idx = np.flatnonzero(mask)
    conf = np.array([dists[i].max() for i in idx])
    correct = np.array([int(dists[i].argmax()) == int(gold[i]) for i in idx], dtype=float)
    nll = np.array([-np.log(max(dists[i][int(gold[i])], 1e-12)) for i in idx])
    return {"conf": conf, "correct": correct, "nll": nll}


def build(runs_dir: Path) -> dict:
    out: dict = {"bins": BINS, "arms": {}}
    for key, short, label, runs in ARMS:
        fam = "ablation" if key == "a3b0" else ARM_STYLES[short].family
        out["arms"][key] = {"short": short, "label": label, "family": fam, "data": {}}
        pooled: dict = {}
        for name, _seed in runs:
            rd = runs_dir / name
            temps = EV.fit_temperatures(EV.load_preds(rd / "preds_cal.npz"))
            for split in ("dev", "select"):
                preds = EV.load_preds(rd / f"preds_{split}.npz")
                for row, t in (("raw", 1.0), ("ts", temps)):
                    dists = EV.probabilities(preds, t)
                    for scope, code in SCOPES.items():
                        mask = np.ones(len(dists), bool) if code is None else preds["qtype"] == code
                        st = _run_stats(dists, preds["gold"], mask)
                        d = pooled.setdefault(
                            (split, scope, row),
                            {"conf": [], "correct": [], "ece": [], "acc": [], "nll": []},
                        )
                        d["conf"].append(st["conf"])
                        d["correct"].append(st["correct"])
                        d["ece"].append(
                            S.ece(st["conf"], st["correct"].astype(bool), BINS, "width")
                        )
                        d["acc"].append(float(st["correct"].mean()))
                        d["nll"].append(float(st["nll"].mean()))
        for (split, scope, row), d in pooled.items():
            conf, correct = np.concatenate(d["conf"]), np.concatenate(d["correct"])
            r = S.reliability_bins(conf, correct, BINS, "width")
            out["arms"][key]["data"][f"{split}|{scope}|{row}"] = {
                "n": int(len(d["conf"][0])),
                "count": [int(c) for c in r["count"]],
                "conf": [None if np.isnan(x) else round(float(x), 4) for x in r["conf"]],
                "acc": [None if np.isnan(x) else round(float(x), 4) for x in r["acc"]],
                "ece": round(float(np.mean(d["ece"])), 4),
                "acc_mean": round(float(np.mean(d["acc"])), 4),
                "log_loss": round(float(np.mean(d["nll"])), 4),
                "seeds": len(d["conf"]),
            }
    out["colors"] = {k: {"light": v[0], "dark": v[1]} for k, v in FAMILIES.items()}
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RLCD reliability explorer</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--panel:#f4f3ee}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--panel:#232321}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--panel:#232321}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,-apple-system,Helvetica,Arial,sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 18px;max-width:760px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:24px}
@media (max-width:820px){.layout{grid-template-columns:1fr}}
fieldset{border:0;margin:0 0 14px;padding:0}legend{font-weight:600;font-size:13px;color:var(--ink2);margin-bottom:6px;padding:0}
label.opt{display:flex;align-items:center;gap:8px;margin:3px 0;cursor:pointer;font-size:14px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block;border:2px solid;flex:none}
.seg{display:flex;flex-wrap:wrap;gap:6px}.seg label{border:1px solid var(--axis);border-radius:6px;padding:3px 10px;cursor:pointer;font-size:13px;color:var(--ink2)}
.seg input{position:absolute;opacity:0;pointer-events:none}.seg input:checked+span{color:var(--ink);font-weight:600}
.seg label:has(input:checked){background:var(--panel);border-color:var(--ink2)}
svg{width:100%;height:auto;display:block}svg text{fill:var(--ink2);font-size:11px}
table{border-collapse:collapse;width:100%;margin-top:16px;font-size:13px}th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
.note{color:var(--muted);font-size:12.5px;margin-top:14px;max-width:760px}
button{background:var(--panel);color:var(--ink2);border:1px solid var(--axis);border-radius:6px;padding:3px 10px;font-size:12.5px;cursor:pointer}
</style></head><body><main>
<h1>Reliability explorer</h1>
<p class="sub">Exp A of open-rlcd: does the model's stated confidence match how often it is right? Pick an answer type, a split and arms. Points on the diagonal are perfectly calibrated; above it the model is under-confident, below it over-confident. Point size is the number of questions in the bin.</p>
<div class="layout"><div>
<svg id="chart" viewBox="0 0 640 520" role="img" aria-label="Reliability diagram"></svg>
<table id="tbl"></table>
<p class="note" id="note"></p></div>
<div id="controls">
<fieldset><legend>Split</legend><div class="seg" id="split"></div></fieldset>
<fieldset><legend>Answer type</legend><div class="seg" id="scope"></div></fieldset>
<fieldset><legend>Temperature scaling</legend><div class="seg" id="row"></div></fieldset>
<fieldset><legend>Arms <button id="all">all</button> <button id="none">none</button></legend><div id="arms"></div></fieldset>
</div></div>
<script>
const D = __DATA__;
const SPLITS=[["dev","Dev set (pre-registered)"],["select","Selection split (exploratory)"]];
const SCOPES=[["all","All"],["noul","NOUL"],["choice","CHOICE"],["score","SCORE"]];
const ROWS=[["raw","Raw (T = 1)"],["ts","After TS"],["both","Both"]];
const st={split:"dev",scope:"all",row:"raw",arms:new Set(["a1","a3","a4"])};
const dark=()=>document.documentElement.dataset.theme==="dark"||(document.documentElement.dataset.theme!=="light"&&matchMedia("(prefers-color-scheme: dark)").matches);
const col=a=>D.colors[D.arms[a].family][dark()?"dark":"light"];
function seg(id,list,key){const el=document.getElementById(id);el.innerHTML=list.map(([v,l])=>`<label><input type="radio" name="${id}" value="${v}" ${st[key]===v?"checked":""}><span>${l}</span></label>`).join("");
 el.onchange=e=>{st[key]=e.target.value;draw()}}
function arms(){const el=document.getElementById("arms");el.innerHTML=Object.entries(D.arms).map(([k,a])=>`<label class="opt"><input type="checkbox" value="${k}" ${st.arms.has(k)?"checked":""}><i class="sw" style="border-color:${col(k)};background:${col(k)}"></i>${a.label}</label>`).join("");
 el.onchange=e=>{e.target.checked?st.arms.add(e.target.value):st.arms.delete(e.target.value);draw()}}
document.getElementById("all").onclick=()=>{st.arms=new Set(Object.keys(D.arms));arms();draw()};
document.getElementById("none").onclick=()=>{st.arms=new Set();arms();draw()};
function draw(){
 const W=640,H=520,L=58,R=18,T=14,B=48,pw=W-L-R,ph=H-T-B,X=v=>L+v*pw,Y=v=>T+(1-v)*ph;let s="";
 for(let i=0;i<=5;i++){const v=i/5;s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><line y1="${T}" y2="${T+ph}" x1="${X(v)}" x2="${X(v)}" stroke="var(--grid)"/><text x="${L-8}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(1)}</text><text x="${X(v)}" y="${T+ph+16}" text-anchor="middle">${v.toFixed(1)}</text>`}
 s+=`<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--muted)" stroke-dasharray="4 4"/><text x="${W/2}" y="${H-8}" text-anchor="middle">confidence (probability of the top answer)</text><text transform="translate(14 ${T+ph/2}) rotate(-90)" text-anchor="middle">accuracy</text>`;
 const rows=st.row==="both"?["raw","ts"]:[st.row];let tbl="";
 for(const k of st.arms){const c=col(k);for(const r of rows){const d=D.arms[k].data[`${st.split}|${st.scope}|${r}`];if(!d)continue;const pts=[];
   d.count.forEach((n,i)=>{if(n>0&&d.conf[i]!=null)pts.push([d.conf[i],d.acc[i],n])});
   const dash=r==="ts"?'stroke-dasharray="5 3"':"";
   s+=`<polyline points="${pts.map(p=>X(p[0])+","+Y(p[1])).join(" ")}" fill="none" stroke="${c}" stroke-width="1.8" ${dash} opacity=".85"/>`;
   for(const p of pts){s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${Math.min(9,Math.max(3,3+0.045*Math.sqrt(p[2])))}" fill="${r==="ts"?"var(--bg)":c}" stroke="${c}" stroke-width="1.8"><title>${D.arms[k].short} ${r}: confidence ${p[0].toFixed(2)}, accuracy ${p[1].toFixed(2)}, n=${p[2]}</title></circle>`}
   tbl+=`<tr><td><i class="sw" style="border-color:${c};background:${r==="ts"?"transparent":c}"></i> ${D.arms[k].short} ${r==="ts"?"after TS":"raw"}</td><td>${d.ece.toFixed(3)}</td><td>${d.acc_mean.toFixed(3)}</td><td>${d.log_loss.toFixed(3)}</td><td>${d.n}</td></tr>`}}
 document.getElementById("chart").innerHTML=s;
 document.getElementById("tbl").innerHTML=`<tr><th>arm</th><th>ECE</th><th>accuracy</th><th>log loss</th><th>questions</th></tr>`+tbl;
 document.getElementById("note").textContent="Curves pool the seeds of an arm (A0, A5 and the β = 0 run are single runs); ECE, accuracy and log loss are seed means. 15 equal-width bins. Filled marker: raw (T = 1). Hollow, dashed: after one temperature per answer type fitted on the calibration split. "+(st.split==="select"?"The selection split is harder than the dev set and was used to choose learning rates and checkpoints, so it is not a test set.":"The dev set is the pre-registered test set.");
}
seg("split",SPLITS,"split");seg("scope",SCOPES,"scope");seg("row",ROWS,"row");arms();draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>{arms();draw()});
</script></main></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/expA/runs")
    ap.add_argument("--out", default="docs/explorer.html")
    a = ap.parse_args()
    data = build(Path(a.runs))
    Path(a.out).write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
