# app.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

# -----------------------------
# 1️⃣ Load data safely
# -----------------------------
@st.cache_data
def load_data():
    info = pd.read_csv("data/UCSD-P1_node_info_cleaned.csv")
    graphs_by_time = np.load("data/graphs_by_time_UCSD-P1.npy", allow_pickle=True).item()
    return info, graphs_by_time

info, graphs_by_time = load_data()

# -----------------------------
# 2️⃣ Node color palette
# -----------------------------
palette = {
    "p__Proteobacteria": "#0077b6",
    "p__Bacteroidota": "#fff3b0",
    "p__Verrucomicrobiota": "#e09f3e",
    "p__Bdellovibrionota": "#9e2a2b",
    "p__Cyanobacteria": "#9370dc",
    "Algae": "#6a994e",
}

# Map nodes → phylum (default "Algae")
node_to_phylum = dict(zip(info["target (driver)"], info["Phylum"]))

# -----------------------------
# 3️⃣ Compute fixed layout once
# -----------------------------
def compute_layout(graphs):
    G_all = nx.DiGraph()
    for G in graphs.values():
        G_all.add_nodes_from(G.nodes)
    return nx.spring_layout(G_all, seed=42)

fixed_pos = compute_layout(graphs_by_time)

# -----------------------------
# 4️⃣ Safe float helper
# -----------------------------
def safe_float(x):
    try:
        if isinstance(x, (list, np.ndarray)):
            x = x[0]
        return float(x)
    except Exception:
        return 0.0

# -----------------------------
# 5️⃣ Streamlit UI
# -----------------------------
st.title("Microbial Network Evolution")

# Animation controls
play = st.button("Play Animation")
speed = st.slider("Animation speed (seconds per frame)", 0.1, 2.0, 0.6)
manual_time = st.slider(
    "Select time step manually", 
    min_value=min(graphs_by_time.keys()),
    max_value=max(graphs_by_time.keys()),
    value=min(graphs_by_time.keys()),
    step=1
)

placeholder = st.empty()  # reserve plot space

# -----------------------------
# 6️⃣ Function to draw a network
# -----------------------------
def draw_network(G, time_label):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")

    pos = {n: fixed_pos[n] for n in G.nodes}

    # Node colors
    node_colors = [
        palette.get(node_to_phylum.get(n, "Algae"), "#6a994e")
        for n in G.nodes
    ]

    # Node sizes (scaled by out-degree)
    out_deg = np.array([G.out_degree(n) for n in G.nodes])
    if len(out_deg) > 0:
        node_sizes = np.interp(out_deg, (out_deg.min(), out_deg.max()), (200, 1200))
    else:
        node_sizes = np.array([])

    # Edge weights and colors
    weights = np.array([safe_float(d.get("weight", 0)) for _, _, d in G.edges(data=True)])
    if len(weights) > 0:
        widths = np.interp(np.abs(weights), (np.abs(weights).min(), np.abs(weights).max()), (0.5, 3))
        edge_colors = [(0, 0, 1, 0.5) if w > 0 else (1, 0, 0, 0.5) for w in weights]
    else:
        widths = []
        edge_colors = []

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=widths, arrows=True)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title(f"UCSD-Pond 1: Time {time_label}")
    return fig

# -----------------------------
# 7️⃣ Show network
# -----------------------------
if play:
    # Animate all time steps
    for t in sorted(graphs_by_time.keys()):
        G = graphs_by_time[t]
        if G.number_of_nodes() == 0:
            continue  # skip empty graphs
        fig = draw_network(G, t)
        placeholder.pyplot(fig)
        time.sleep(speed)
else:
    # Manual single time step
    G = graphs_by_time[manual_time]
    if G.number_of_nodes() > 0:
        fig = draw_network(G, manual_time)
        placeholder.pyplot(fig)
