#!/usr/bin/env python3
"""
run_experiments_v2.py
=====================
CORRECTED version: uses the INTENDED OLEP definition:
  F- = leaves in V- at the GLOBALLY MAXIMUM BFS depth in V-
  F+ = leaves in V+ at the GLOBALLY MAXIMUM BFS depth in V+
  OLEP: global min f ∈ F-  AND  global max f ∈ F+

This matches the paper's intent (makes T1, T2 genuine counterexamples)
but NOT the flawed formal definition in the paper (which is vacuous for all leaves).

Outputs:
  results/exhaustive_counts_v2.json
  results/T1_fiedler.json
  results/T2_fiedler.json
  results/random_models_v2.json
  results/algorithm_accuracy_v2.json
"""

import json, os, time, random
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

os.makedirs("results", exist_ok=True)
random.seed(42)
np.random.seed(42)

# ── helpers ──────────────────────────────────────────────────────────────────

def laplacian_sparse(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    row, col, data = [], [], []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        row += [i, j, i, j]
        col += [j, i, i, j]
        data += [-1, -1, 1, 1]
    return csr_matrix((data, (row, col)), shape=(n, n)), nodes

def fiedler_vector(G):
    """Return (lambda2, f_array) indexed by sorted(G.nodes()), max(f)>0."""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, np.array([0.0])
    L, nodes = laplacian_sparse(G)
    from scipy.linalg import eigh
    Ld = L.toarray()
    vals, vecs = eigh(Ld, subset_by_index=[1, 1])
    lam2 = float(vals[0])
    f = vecs[:, 0]
    if np.max(f) < -np.min(f):
        f = -f
    return lam2, f

def check_olep_correct(G):
    """
    CORRECTED OLEP check using the INTENDED definition:
      F- = leaves in V- at the globally maximum BFS depth in V-
      F+ = leaves in V+ at the globally maximum BFS depth in V+
    Returns (olep_satisfied, debug_dict)
    """
    n = G.number_of_nodes()
    if n <= 2:
        return True, {}
    lam2, f = fiedler_vector(G)
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    # Spectral center: argmin |f(v)| with tie-break by node label
    absf = np.abs(f)
    min_val = np.min(absf)
    c_idx = min([i for i in range(n) if absf[i] == min_val],
                key=lambda i: nodes[i])
    c = nodes[c_idx]

    # BFS depths from c
    bfs_depth = nx.single_source_shortest_path_length(G, c)

    # Sign sets (leaves only)
    leaves_plus  = [v for v in nodes if G.degree(v) == 1 and f[node_idx[v]] > 0]
    leaves_minus = [v for v in nodes if G.degree(v) == 1 and f[node_idx[v]] < 0]

    if not leaves_plus or not leaves_minus:
        return True, {}

    # GLOBALLY MAXIMUM BFS depth in each sign component's leaves
    max_depth_plus  = max(bfs_depth[v] for v in leaves_plus)
    max_depth_minus = max(bfs_depth[v] for v in leaves_minus)

    # F+ and F-: leaves at the globally maximum BFS depth in their sign component
    F_plus  = [v for v in leaves_plus  if bfs_depth[v] == max_depth_plus]
    F_minus = [v for v in leaves_minus if bfs_depth[v] == max_depth_minus]

    global_max = np.max(f)
    global_min = np.min(f)

    max_in_F_plus  = max(f[node_idx[v]] for v in F_plus)
    min_in_F_minus = min(f[node_idx[v]] for v in F_minus)

    tol = 1e-8
    olep = (abs(max_in_F_plus - global_max) < tol and
            abs(min_in_F_minus - global_min) < tol)

    debug = {
        "c": int(c),
        "f_c": float(f[c_idx]),
        "max_depth_plus": int(max_depth_plus),
        "max_depth_minus": int(max_depth_minus),
        "F_plus": [int(v) for v in F_plus],
        "F_minus": [int(v) for v in F_minus],
        "global_max": float(global_max),
        "max_in_F_plus": float(max_in_F_plus),
        "global_min": float(global_min),
        "min_in_F_minus": float(min_in_F_minus),
        "olep": bool(olep),
    }
    return olep, debug

# ── TASK 1: Verify T1 and T2 ─────────────────────────────────────────────────
print("=== Task 1: Verifying T1 and T2 ===")

def make_T1():
    G = nx.Graph()
    G.add_nodes_from(range(13))
    edges = [(0,1),(0,5),(0,8),(1,2),(2,3),(3,4),(5,6),(6,7),(8,9),(8,10),(8,11),(8,12)]
    G.add_edges_from(edges)
    return G

def make_T2():
    G = nx.Graph()
    G.add_nodes_from(range(13))
    edges = [(0,1),(0,5),(0,8),(1,2),(2,3),(2,4),(5,6),(6,7),(8,9),(8,10),(8,11),(8,12)]
    G.add_edges_from(edges)
    return G

for name, Gx in [("T1", make_T1()), ("T2", make_T2())]:
    lam2, f = fiedler_vector(Gx)
    nodes = sorted(Gx.nodes())
    olep, dbg = check_olep_correct(Gx)
    diam = nx.diameter(Gx)
    print(f"\n{name}: lambda2={lam2:.6f}, diameter={diam}, OLEP={olep}")
    print(f"  Spectral center: {dbg['c']} (f={dbg['f_c']:.6f})")
    print(f"  F-: {dbg['F_minus']}  (max_depth_minus={dbg['max_depth_minus']})")
    print(f"  global_min={dbg['global_min']:.6f}, min_in_F-={dbg['min_in_F_minus']:.6f}")
    print(f"  F+: {dbg['F_plus']}  (max_depth_plus={dbg['max_depth_plus']})")
    print(f"  global_max={dbg['global_max']:.6f}, max_in_F+={dbg['max_in_F_plus']:.6f}")
    for i, v in enumerate(nodes):
        print(f"    f({v}) = {f[i]:.6f}  depth={nx.shortest_path_length(Gx, dbg['c'], v)}")

    result = {
        "name": name,
        "lambda2": float(lam2),
        "diameter": int(diam),
        "olep": bool(olep),
        "debug": dbg,
        "fiedler": {str(nodes[i]): float(f[i]) for i in range(len(nodes))}
    }
    with open(f"results/{name}_fiedler.json", "w") as fp:
        json.dump(result, fp, indent=2)

# ── TASK 2: Exhaustive enumeration n=5..17 ───────────────────────────────────
print("\n=== Task 2: Exhaustive enumeration n=5..17 ===")

OEIS = {5:3,6:6,7:11,8:23,9:47,10:106,11:235,12:551,
        13:1301,14:3159,15:7741,16:19320,17:48629}

exhaustive = {}
for n in range(5, 18):
    t0 = time.time()
    total = counterex = 0
    for G in nx.nonisomorphic_trees(n):
        total += 1
        olep, _ = check_olep_correct(G)
        if not olep:
            counterex += 1
    elapsed = time.time() - t0
    rate = counterex / total if total else 0
    match_str = "OK" if total == OEIS[n] else f"MISMATCH(exp {OEIS[n]})"
    print(f"  n={n:2d}: {total:6d} trees {match_str}, {counterex:4d} counterex ({100*rate:.2f}%) [{elapsed:.1f}s]")
    exhaustive[str(n)] = {
        "n": n, "total": total, "oeis": OEIS[n],
        "counterex": counterex, "rate": float(rate), "t": float(elapsed)
    }

with open("results/exhaustive_counts_v2.json", "w") as fp:
    json.dump(exhaustive, fp, indent=2)

total_all = sum(v["total"] for v in exhaustive.values())
total_cx  = sum(v["counterex"] for v in exhaustive.values())
print(f"Total n=5..17: {total_all} trees, {total_cx} counterexamples ({100*total_cx/total_all:.3f}%)")

# ── TASK 3: Random tree models ───────────────────────────────────────────────
print("\n=== Task 3: Random tree models (2500 each) ===")

def gen_ba_tree(n, seed=None):
    rng = random.Random(seed)
    G = nx.Graph(); G.add_node(0)
    if n == 1: return G
    G.add_node(1); G.add_edge(0, 1)
    degrees = [1, 1]
    for new_node in range(2, n):
        total_deg = sum(degrees)
        r = rng.random() * total_deg
        cumsum = 0; target = 0
        for i, d in enumerate(degrees):
            cumsum += d
            if cumsum >= r: target = i; break
        G.add_node(new_node); G.add_edge(new_node, target)
        degrees.append(1); degrees[target] += 1
    return G

def gen_er_mst(n, seed=None):
    rng = np.random.RandomState(seed)
    w = rng.random((n, n)); w = (w + w.T) / 2
    Gc = nx.complete_graph(n)
    for u, v in Gc.edges(): Gc[u][v]['weight'] = w[u][v]
    return nx.minimum_spanning_tree(Gc)

def gen_prufer_tree(n, seed=None):
    rng = random.Random(seed)
    if n <= 1: G = nx.Graph(); G.add_node(0); return G
    if n == 2: G = nx.Graph(); G.add_edge(0,1); return G
    seq = [rng.randint(0, n-1) for _ in range(n-2)]
    return nx.from_prufer_sequence(seq)

def gen_caterpillar(n, seed=None):
    rng = random.Random(seed)
    if n <= 2:
        G = nx.Graph()
        if n == 1: G.add_node(0)
        else: G.add_edge(0,1)
        return G
    k = rng.randint(2, max(2, n-1))
    G = nx.path_graph(k)
    node_id = k
    spine = list(range(k))
    for _ in range(n - k):
        p = rng.choice(spine); G.add_edge(p, node_id); node_id += 1
    return G

def gen_lobster(n, seed=None):
    rng = random.Random(seed)
    if n <= 3: return gen_caterpillar(n, seed)
    k = max(2, rng.randint(2, max(2, n//3)))
    G = nx.path_graph(k); node_id = k; spine = list(range(k))
    first_level = []
    pending = n - k
    for sv in spine:
        if pending <= 0: break
        nc = rng.randint(0, min(2, pending))
        for _ in range(nc):
            G.add_edge(sv, node_id); first_level.append(node_id); node_id += 1; pending -= 1
    remaining2 = n - node_id
    if remaining2 > 0 and first_level:
        for _ in range(remaining2):
            p = rng.choice(first_level); G.add_edge(p, node_id); node_id += 1
    return G

def gen_binary_tree(n, seed=None):
    rng = random.Random(seed)
    G = nx.Graph(); G.add_node(0); leaves_list = [0]
    for new_node in range(1, n):
        parent = rng.choice(leaves_list); G.add_edge(parent, new_node)
        children = [v for v in G.neighbors(parent) if v > parent]
        if len(children) >= 2 and parent in leaves_list: leaves_list.remove(parent)
        leaves_list.append(new_node)
    return G

MODELS = {
    "BA": gen_ba_tree,
    "ER-MST": gen_er_mst,
    "Binary": gen_binary_tree,
    "Prufer": gen_prufer_tree,
    "Caterpillar": gen_caterpillar,
    "Lobster": gen_lobster,
}
N_PER = 2500

random_results = {}
for model_name, gen_fn in MODELS.items():
    total = counterex = 0
    t0 = time.time()
    for i in range(N_PER):
        nt = random.randint(20, 100)
        G = gen_fn(nt, seed=i*1000 + hash(model_name) % 1000)
        if not nx.is_tree(G): G = nx.minimum_spanning_tree(G)
        total += 1
        olep, _ = check_olep_correct(G)
        if not olep: counterex += 1
    elapsed = time.time() - t0
    rate = counterex / total
    print(f"  {model_name:12s}: {total} trees, {counterex:4d} counterex ({100*rate:.2f}%) [{elapsed:.1f}s]")
    random_results[model_name] = {"total": total, "counterex": counterex, "rate": float(rate), "t": float(elapsed)}

with open("results/random_models_v2.json", "w") as fp:
    json.dump(random_results, fp, indent=2)

# ── TASK 4: Algorithm accuracy on 5000 trees ─────────────────────────────────
print("\n=== Task 4: Algorithm accuracy (5000 trees) ===")

def spectral_center_exact(G, f, nodes):
    absf = np.abs(f)
    min_v = np.min(absf)
    c_idx = min([i for i in range(len(nodes)) if absf[i] == min_v], key=lambda i: nodes[i])
    return nodes[c_idx]

def approx_spectral_center(G):
    """O(n) approximation: two-BFS diameter path + branch-balance score."""
    nodes = sorted(G.nodes())
    n = G.number_of_nodes()
    if n == 1: return nodes[0]
    # BFS1
    far1 = max(nx.single_source_shortest_path_length(G, nodes[0]).items(), key=lambda x: x[1])[0]
    # BFS2
    dist2 = nx.single_source_shortest_path_length(G, far1)
    far2 = max(dist2.items(), key=lambda x: x[1])[0]
    D = dist2[far2]
    path = nx.shortest_path(G, far1, far2)
    mid_idx = len(path) // 2

    def branch_balance(v):
        G2 = G.copy(); G2.remove_node(v)
        comps = list(nx.connected_components(G2))
        sizes = sorted(len(c) for c in comps)
        return abs(sizes[-1] - (n-1)/2) if sizes else 0

    best = None
    best_score = float('inf')
    for pi, pv in enumerate(path):
        sc = branch_balance(pv) - 0.1 * abs(pi - mid_idx)
        if sc < best_score:
            best_score = sc; best = pv

    # refinement
    for w in G.neighbors(best):
        sc = branch_balance(w) - 0.1 * 0
        if sc < best_score:
            best_score = sc; best = w
    return best

N_ALG = 5000
exact = 0; total_err = 0.0; t0 = time.time()
for i in range(N_ALG):
    nt = random.randint(10, 50)
    G = gen_prufer_tree(nt, seed=i*7+13)
    if not nx.is_tree(G): continue
    lam2, f = fiedler_vector(G)
    nodes = sorted(G.nodes())
    c_true = spectral_center_exact(G, f, nodes)
    c_approx = approx_spectral_center(G)
    err = nx.shortest_path_length(G, c_true, c_approx)
    if err == 0: exact += 1
    total_err += err
elapsed = time.time() - t0
acc = exact / N_ALG
avg_err = total_err / N_ALG
print(f"  {N_ALG} trees: exact={exact} ({100*acc:.1f}%), avg_error={avg_err:.3f} [{elapsed:.1f}s]")

alg_result = {"n_trees": N_ALG, "exact": exact, "accuracy": float(acc),
              "avg_error": float(avg_err), "t": float(elapsed)}
with open("results/algorithm_accuracy_v2.json", "w") as fp:
    json.dump(alg_result, fp, indent=2)

print("\n=== Done. Results in results/ ===")
