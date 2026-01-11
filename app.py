import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------- LOAD DATA ----------
def load_data():
    info = pd.read_csv("data/UCSD-P1_node_info_cleaned.csv")
    graphs = np.load("data/graphs_by_time_UCSD-P1.npy", allow_pickle=True).item()
    return info, graphs

info, graphs_by_time = load_data()

# ---------- NODE → PHYLUM ----------
node_to_phylum = dict(
    zip(info["target (driver)"], info["Phylum"])
)

palette = {
    "p__Proteobacteria": "#0077b6",
    "p__Bacteroidota": "#fff3b0",
    "p__Verrucomicrobiota": "#e09f3e",
    "p__Bdellovibrionota": "#9e2a2b",
    "p__Cyanobacteria": "#9370dc",
    "Algae": "#6a994e",
}

#------ HELPER------#
def safe_float(x):
    try:
        # If it's a list/array with one element, take that element
        if isinstance(x, (list, np.ndarray)):
            x = x[0]
        return float(x)
    except Exception:
        return 0.0


# ---------- FIXED LAYOUT ----------
def compute_layout(graphs):
    G_all = nx.DiGraph()
    for G in graphs.values():
        G_all.add_nodes_from(G.nodes)
    return nx.spring_layout(G_all, seed=42)

fixed_pos = compute_layout(graphs_by_time)

# ---------- UI ----------
st.title("Microbial Network Structure")

times = sorted(graphs_by_time.keys())
t = st.slider("Time", min(times), max(times), step=1)

G = graphs_by_time[t]

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(7, 8))
ax.axis("off")

pos = {n: fixed_pos[n] for n in G.nodes}

# Node colors
node_colors = [
    palette.get(node_to_phylum.get(n, "Algae"), "#6a994e")
    for n in G.nodes
]

# Node sizes (outdegree)
out_deg = np.array([G.out_degree(n) for n in G.nodes])
node_sizes = np.interp(out_deg, (out_deg.min(), out_deg.max()), (200, 1200))

# Edge weights
weights = np.array([
    safe_float(d.get("weight", 0))
    for _, _, d in G.edges(data=True)
])


alphas = np.interp(np.abs(weights), (np.abs(weights).min(), np.abs(weights).max()), (0.2, 1.0))
widths = np.interp(np.abs(weights), (np.abs(weights).min(), np.abs(weights).max()), (0.5, 3))

edge_colors = [
    (0, 0, 1, a) if w > 0 else (1, 0, 0, a)
    for w, a in zip(weights, alphas)
]

nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    edge_color=edge_colors,
    width=widths,
    arrows=True,
    arrowsize=10
)

nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors="black"
)

nx.draw_networkx_labels(
    G,
    pos,
    ax=ax,
    font_size=8
)

ax.set_title(f"Time {t}")
st.pyplot(fig)