"""Interactive transfer-matrix explorer (plotly, self-contained HTML for
the repo): hover shows the exact mean Spearman for every train->eval pair.
"""
import json

import numpy as np
import plotly.graph_objects as go

d = json.load(open("results/crossdata_ridge_14ds.json"))["summary"]["ridge"]
names = list(d.keys())
Z = np.full((len(names), len(names)), np.nan)
for i, a in enumerate(names):
    Z[i, i] = d[a]["within_mean"]
    for j, b in enumerate(names):
        if a != b:
            Z[i, j] = d[a]["cross_mean"][b]
fig = go.Figure(go.Heatmap(
    z=Z, x=names, y=names, zmid=0, colorscale="RdBu_r",
    hovertemplate="train %{y} -> eval %{x}: %{z:.3f}<extra></extra>"))
fig.update_layout(title="Cross-dataset gRNA efficacy transfer (ridge, 3-seed mean Spearman)",
                  width=760, height=680, yaxis_autorange="reversed")
fig.write_html("docs/transfer_matrix_interactive.html", include_plotlyjs=True)
print("wrote docs/transfer_matrix_interactive.html")
