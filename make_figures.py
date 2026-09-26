"""Every figure in the v2 paper, from the two results files and nothing else.

    .venv/bin/python make_figures.py --open results/v2_openloop.json \
        --closed results/v2_closed.json --out blog/figures_v2
"""
import argparse
import json
import pathlib

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEADLINE_MS = 100.0
FONT = dict(family="Helvetica Neue, Arial, sans-serif", size=15, color="#222")
COLORS = {"reference": "#222222", "verifier": "#1f77b4", "draft": "#ff7f0e", "gate": "#d62728"}


def style(fig, height):
    fig.update_layout(template="simple_white", font=FONT, height=height, width=1100,
                      margin=dict(l=80, r=30, t=50, b=70), legend=dict(font=dict(size=13)))
    return fig


def kind(name: str) -> str:
    if "(reference)" in name:
        return "reference"
    if name.startswith("verifier"):
        return "verifier"
    if name.startswith("draft"):
        return "draft"
    return "gate"


def short(name: str) -> str:
    q = name.split("(val q=")[1].rstrip(")") if "(val q=" in name else None
    name = name.split(" @ ")[0] + (f" (q={q})" if q else "")
    for old, new in (("gate: needs verifier (notebook rule)", "notebook rule"),
                     ("gate: previous agreement", "previous agreement"),
                     ("gate: draft ensemble", "draft ensemble"),
                     ("verifier ", ""), ("/chunk8", ""), ("/chunk", ", chunk ")):
        name = name.replace(old, new)
    return name


def twin_positions(points, close):
    """Label side for each point: where two points share an x, the lower one's label
    goes below-right and the higher one's above-right."""
    pos = {}
    for key, x, y in points:
        twins = sorted((yy for kk, xx, yy in points if abs(xx - x) < close), key=float)
        pos[key] = "middle right" if len(twins) == 1 else ("bottom right" if y == twins[0] else "top right")
    return pos


def frontier(open_res: dict, out: pathlib.Path):
    """Open-loop MSE against the demonstrator vs time per frame for the verifier samplers.

    The draft (0.1 ms, MSE ~380) is off this scale and is in Table 1. Gates at their
    chosen thresholds fall back to the reference and are shown in Figure 2 instead.
    """
    rows = [r for r in open_res["summary"] if r["strategy"].startswith("verifier")]
    side = twin_positions([(r["strategy"], r["ms_per_frame"], r["mse"]) for r in rows], close=3)
    for r in rows:
        if "(reference)" in r["strategy"]:
            side[r["strategy"]] = "middle left"
    fig = go.Figure()
    for r in rows:
        k = kind(r["strategy"])
        ok = r["pipelined_10hz_ok"]  # filled: the slowest call leaves time to plan ahead at 10 Hz
        fig.add_trace(go.Scatter(
            x=[r["ms_per_frame"]], y=[r["mse"]], mode="markers+text", text=[f" {short(r['strategy'])} "],
            textposition=side[r["strategy"]], textfont=dict(size=12), showlegend=False,
            marker=dict(size=12, color=COLORS[k] if ok else "white", line=dict(color=COLORS[k], width=2)),
            error_y=dict(type="data", symmetric=False, array=[r["mse_ci"][1] - r["mse"]],
                         arrayminus=[r["mse"] - r["mse_ci"][0]], color=COLORS[k], thickness=1.2)))
    for label, filled in (("fits a pipelined 10 Hz loop", True), ("doesn't", False)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(size=11, color=COLORS["verifier"] if filled else "white",
                                             line=dict(color=COLORS["verifier"], width=2))))
    top = max(r["mse_ci"][1] for r in rows) * 1.1
    device = open_res.get("provenance", {}).get("device", "").upper() or "this machine"
    fig.update_xaxes(title=f"model time per frame (ms, median benchmarked call on {device})",
                     range=[0, max(r["ms_per_frame"] for r in rows) * 1.18])
    fig.update_yaxes(title="action MSE vs demonstrator (px², 95% CI)", range=[0, top * 1.08])
    fig.update_layout(legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"))
    style(fig, 540).write_image(out, scale=2)


def gate_curves(open_res: dict, out: pathlib.Path):
    """Quality cost vs compute: each gate across its validation-derived thresholds,
    next to the verifier samplers, all on test and paired against the reference."""
    fig = go.Figure()
    palette = {"gate: needs verifier (notebook rule)": "#d62728", "gate: previous agreement": "#9467bd",
               "gate: draft ensemble": "#2ca02c"}
    for name, pts in open_res["test_curves"].items():
        pts = sorted(pts, key=lambda p: -p["speedup_vs_ref"])
        fig.add_trace(go.Scatter(
            x=[1 / p["speedup_vs_ref"] for p in pts], y=[p["mse_vs_ref"] for p in pts],
            mode="lines+markers", name=short(name), line=dict(color=palette[name]), marker=dict(size=7),
            error_y=dict(type="data", symmetric=False, thickness=1,
                         array=[p["mse_vs_ref_ci"][1] - p["mse_vs_ref"] for p in pts],
                         arrayminus=[p["mse_vs_ref"] - p["mse_vs_ref_ci"][0] for p in pts])))
    samplers = [r for r in open_res["summary"] if r["strategy"].startswith("verifier") and "mse_vs_ref" in r]
    side = twin_positions([(r["strategy"], 1 / r["speedup_vs_ref"], r["mse_vs_ref"]) for r in samplers], close=0.02)
    fig.add_trace(go.Scatter(
        x=[1 / r["speedup_vs_ref"] for r in samplers], y=[r["mse_vs_ref"] for r in samplers],
        mode="markers+text", name="fewer steps / longer chunks", text=[f" {short(r['strategy'])}" for r in samplers],
        textposition=[side[r["strategy"]] for r in samplers], textfont=dict(size=12),
        marker=dict(size=11, color=COLORS["verifier"]),
        error_y=dict(type="data", symmetric=False, thickness=1,
                     array=[r["mse_vs_ref_ci"][1] - r["mse_vs_ref"] for r in samplers],
                     arrayminus=[r["mse_vs_ref"] - r["mse_vs_ref_ci"][0] for r in samplers])))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.update_xaxes(title="model time per frame, relative to the reference", range=[0, 1.08])
    fig.update_yaxes(title="extra MSE vs reference (px², paired, 95% CI)")
    fig.update_layout(legend=dict(x=0.45, y=0.98, bgcolor="rgba(255,255,255,0.85)"))
    style(fig, 560).write_image(out, scale=2)


def closed_loop(closed: dict, out: pathlib.Path):
    rows = closed["summary"]
    names = [short(r["strategy"]) for r in rows]
    colors = [COLORS[kind(r["strategy"])] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("success rate", "best coverage reached"),
                        horizontal_spacing=0.12)
    for col, key in ((1, "success"), (2, "max_coverage")):
        fig.add_trace(go.Bar(
            x=names, y=[r[key] for r in rows], marker_color=colors, showlegend=False,
            error_y=dict(type="data", symmetric=False, thickness=1.2,
                         array=[r[f"{key}_ci"][1] - r[key] for r in rows],
                         arrayminus=[r[key] - r[f"{key}_ci"][0] for r in rows])), row=1, col=col)
        fig.update_yaxes(range=[0, 1.05], row=1, col=col)
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=12))
    style(fig, 560).write_image(out, scale=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", required=True)
    ap.add_argument("--closed", default=None)
    ap.add_argument("--out", default="blog/figures_v2")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    open_res = json.loads(pathlib.Path(args.open).read_text())
    frontier(open_res, out / "fig1_frontier.png")
    gate_curves(open_res, out / "fig2_gates.png")
    if args.closed:
        closed_loop(json.loads(pathlib.Path(args.closed).read_text()), out / "fig3_closed_loop.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
