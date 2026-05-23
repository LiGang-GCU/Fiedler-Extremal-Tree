#!/usr/bin/env python3
"""
plot_counterexamples.py
=======================
Generate publication-quality figure showing T1 and T2 OLEP counterexamples.
Shows tree layout with BFS layers, node colors, Fiedler values,
highlighting F- (deepest V- leaves) vs actual global minimum.
Saves olep_counterexamples.pdf
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

def laplacian(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        row += [i, j, i, j]; col += [j, i, i, j]; data += [-1,-1,1,1]
    return csr_matrix((data, (row, col)), shape=(n, n)), nodes

def fiedler_vector(G):
    L, nodes = laplacian(G)
    vals, vecs = eigh(L.toarray(), subset_by_index=[1,1])
    f = vecs[:,0]
    if np.max(f) < -np.min(f): f = -f
    return float(vals[0]), dict(zip(nodes, f))

def make_T1():
    G = nx.Graph(); G.add_nodes_from(range(13))
    G.add_edges_from([(0,1),(0,5),(0,8),(1,2),(2,3),(3,4),(5,6),(6,7),(8,9),(8,10),(8,11),(8,12)])
    return G

def make_T2():
    G = nx.Graph(); G.add_nodes_from(range(13))
    G.add_edges_from([(0,1),(0,5),(0,8),(1,2),(2,3),(2,4),(5,6),(6,7),(8,9),(8,10),(8,11),(8,12)])
    return G

def tree_layout(G, root, spacing_x=1.2, spacing_y=1.0):
    """Custom hierarchical layout for a tree rooted at 'root'."""
    # BFS from root to get parent/children structure
    bfs = dict(nx.bfs_predecessors(G, root))
    children = {v: [] for v in G.nodes()}
    for child, parent in bfs.items():
        children[parent].append(child)
    bfs_depth = nx.single_source_shortest_path_length(G, root)

    # Assign x positions using leaf count
    def count_leaves(v):
        if not children[v]: return 1
        return sum(count_leaves(c) for c in children[v])

    pos = {}
    def assign_x(v, x_start, width):
        pos[v] = (x_start + width/2, -bfs_depth[v] * spacing_y)
        if not children[v]: return
        w = width / len(children[v])
        for i, c in enumerate(children[v]):
            assign_x(c, x_start + i*w, w)

    # Use leaf-count based width
    total_leaves = count_leaves(root)
    def assign_x_leaf(v, x_start):
        if not children[v]:
            pos[v] = (x_start, -bfs_depth[v] * spacing_y)
            return 1
        total = 0
        for c in children[v]:
            w = count_leaves(c)
            assign_x_leaf(c, x_start + total + w/2 - 0.5)
            total += w
        pos[v] = (x_start + total/2 - 0.5, -bfs_depth[v] * spacing_y)
        return total

    assign_x_leaf(root, 0)
    return pos

def draw_tree_panel(ax, G, title, root, f_dict, F_minus, F_plus):
    """Draw a single tree panel with colored nodes and Fiedler values."""
    bfs_depth = nx.single_source_shortest_path_length(G, root)

    # Positions: custom hierarchical
    pos = tree_layout(G, root, spacing_x=1.0, spacing_y=1.4)

    # Scale to fit nicely
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xrange = max(xmax - xmin, 0.1)
    yrange = max(ymax - ymin, 0.1)
    pos = {v: ((pos[v][0]-xmin)/xrange * 7, (pos[v][1]-ymin)/yrange * 5) for v in pos}

    # Node colors
    global_min_val = min(f_dict.values())
    global_max_val = max(f_dict.values())

    node_colors = []
    node_sizes = []
    for v in sorted(G.nodes()):
        fv = f_dict[v]
        if v == root:
            node_colors.append('#9b59b6')  # purple: spectral center
            node_sizes.append(400)
        elif v in F_minus:
            node_colors.append('#e67e22')  # orange: outermost negative leaves (F-)
            node_sizes.append(380)
        elif abs(fv - global_min_val) < 1e-8:
            node_colors.append('#c0392b')  # dark red: actual global minimum (OLEP violator)
            node_sizes.append(420)
        elif fv < 0:
            node_colors.append('#3498db')  # blue: V-
            node_sizes.append(300)
        else:
            node_colors.append('#e74c3c')  # red: V+
            node_sizes.append(300)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color='#555555', alpha=0.8)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           linewidths=1.5,
                           edgecolors='white')

    # Labels: node id + f value
    labels = {v: f'{v}\n{f_dict[v]:.3f}' for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5,
                            font_color='white', font_weight='bold')

    # Mark F- with dashed box annotation
    for v in F_minus:
        px, py = pos[v]
        circ = plt.Circle((px, py), 0.35, fill=False, color='#27ae60',
                           linewidth=2.5, linestyle='--', zorder=5)
        ax.add_patch(circ)

    # Mark global minimum nodes with bold ring
    for v in G.nodes():
        if abs(f_dict[v] - global_min_val) < 1e-8 and v not in F_minus:
            px, py = pos[v]
            circ = plt.Circle((px, py), 0.38, fill=False, color='#c0392b',
                               linewidth=2.5, linestyle='-', zorder=5)
            ax.add_patch(circ)

    # BFS depth labels on right
    max_depth = max(bfs_depth.values())
    depth_ys = {}
    for v, d in bfs_depth.items():
        if d not in depth_ys:
            depth_ys[d] = []
        depth_ys[d].append(pos[v][1])

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-0.8, 5.8)
    ax.axis('off')

# ── Main ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, (name, G) in zip(axes, [("$T_1$: $n=13$, $D=7$", make_T1()),
                                   ("$T_2$: $n=13$, $D=6$", make_T2())]):
    lam2, f = fiedler_vector(G)
    root = min(G.nodes(), key=lambda v: abs(f[v]))  # spectral center
    bfs_depth = nx.single_source_shortest_path_length(G, root)

    leaves_minus = [v for v in G.nodes() if G.degree(v)==1 and f[v] < 0]
    leaves_plus  = [v for v in G.nodes() if G.degree(v)==1 and f[v] > 0]

    max_dm = max(bfs_depth[v] for v in leaves_minus)
    max_dp = max(bfs_depth[v] for v in leaves_plus)
    F_minus = [v for v in leaves_minus if bfs_depth[v] == max_dm]
    F_plus  = [v for v in leaves_plus  if bfs_depth[v] == max_dp]

    draw_tree_panel(ax, G, name, root, f, F_minus, F_plus)

# ── Legend ──────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color='#9b59b6', label='Spectral center $c$ (root)'),
    mpatches.Patch(color='#3498db', label='$V_-$ (negative Fiedler)'),
    mpatches.Patch(color='#e74c3c', label='$V_+$ (positive Fiedler)'),
    mpatches.Patch(color='#e67e22', label='$\\mathcal{F}_-$: outermost neg. leaves (deepest in $V_-$)'),
    mpatches.Patch(color='#c0392b', label='Actual global min (not in $\\mathcal{F}_-$ — OLEP violated!)'),
    mpatches.Patch(facecolor='none', edgecolor='#27ae60', linewidth=2, linestyle='--',
                   label='Dashed circle: $\\mathcal{F}_-$ membership'),
    mpatches.Patch(facecolor='none', edgecolor='#c0392b', linewidth=2, linestyle='-',
                   label='Solid circle: global min not in $\\mathcal{F}_-$'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02),
           fancybox=True, shadow=False)

fig.suptitle('Two Minimal OLEP Counterexamples ($n=13$ vertices each)\n'
             'The global minimum of $f$ occurs at depth-2 leaves $\\{9,10,11,12\\}$, '
             'not at the deepest $V_-$ leaf $\\{7\\}$ ($\\mathcal{F}_-$)',
             fontsize=10.5, y=1.01)

# Add annotation explaining OLEP violation
for ax in axes:
    ax.text(0.5, -0.08,
            'OLEP violated: $\\min_V f \\notin \\mathcal{F}_-$\n'
            '(collective pull from 4 leaf children at hub 8)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8, style='italic', color='#7f8c8d')

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('olep_counterexamples.pdf', dpi=300, bbox_inches='tight')
plt.savefig('olep_counterexamples.png', dpi=150, bbox_inches='tight')
print("Saved olep_counterexamples.pdf and .png")
