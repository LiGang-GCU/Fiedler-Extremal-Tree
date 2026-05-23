#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OLEP 验证程序 — 定义B最终版（全局最深，v17正文 §2 形式定义）

定义B（全局最深）：
  谱中心 c = argmin |f(v)|（顶点编号打破平局）
  V+ = {f(v)>0}, V- = {f(v)<0}
  K+ = max dist(c,v) over ALL V+ 叶子（全局最大深度）
  K- = max dist(c,v) over ALL V- 叶子
  F+ = {v ∈ V+ 叶子 : dist(c,v) = K+}
  F- = {v ∈ V- 叶子 : dist(c,v) = K-}
  OLEP: global_max f 在 F+ 中取到 AND global_min f 在 F- 中取到

实验内容（与论文表1、表2对应）：
  表1：穷举枚举 n=5..17 所有非同构树
  表2：6种随机树模型各 2500 棵，n∈[20,100]，seed=42

输出（results/final_defB/）：
  table1_summary.txt          — 可读摘要
  table1_by_n.json            — 按n统计
  table2_random.json          — 表2统计
  counterex_exhaustive.json   — 表1所有反例（完整图+Fiedler向量）
  counterex_metadata.json     — 反例元信息汇总
"""

import os, json, time, random
import numpy as np
import networkx as nx
from scipy.linalg import eigh
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'final_defB')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-9

# ─────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────

def compute_fiedler(G):
    """Return (f, nodes, lambda2, lambda3); sign convention: max(f)>0."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    L = np.zeros((n, n))
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        L[i,i]+=1; L[j,j]+=1; L[i,j]-=1; L[j,i]-=1
    vals, vecs = eigh(L)
    f = vecs[:, 1].copy()
    if f.max() < -f.min():
        f = -f
    lam2 = float(vals[1])
    lam3 = float(vals[2]) if n > 2 else float('inf')
    return f, nodes, lam2, lam3


def check_olep_B(G, f, nodes):
    """
    定义B（全局最深）OLEP 检查。
    返回 (satisfies: bool, info: dict)
    """
    n = len(nodes)
    node_f = {nodes[i]: f[i] for i in range(n)}

    # spectral center: argmin |f(v)|, ties by vertex index
    arr = np.abs(f)
    min_val = arr.min()
    c = min(nodes[i] for i in range(n) if arr[i] <= min_val + EPS)

    char_case = 'vertex' if abs(node_f[c]) < EPS else 'edge'

    depth = nx.single_source_shortest_path_length(G, c)

    # leaves only
    def is_leaf(v): return G.degree(v) == 1

    Vp_leaves = [v for v in nodes if node_f[v] >  EPS and is_leaf(v)]
    Vm_leaves = [v for v in nodes if node_f[v] < -EPS and is_leaf(v)]

    if not Vp_leaves or not Vm_leaves:
        return True, {'note': 'degenerate', 'satisfies': True}

    # globally deepest BFS depth
    K_plus  = max(depth[v] for v in Vp_leaves)
    K_minus = max(depth[v] for v in Vm_leaves)

    # outermost leaf sets F+, F-
    F_plus  = [v for v in Vp_leaves if depth[v] == K_plus]
    F_minus = [v for v in Vm_leaves if depth[v] == K_minus]

    global_max = float(np.max(f))
    global_min = float(np.min(f))

    # global extremal vertices
    max_nodes = [nodes[i] for i in range(n) if abs(f[i] - global_max) < EPS]
    min_nodes = [nodes[i] for i in range(n) if abs(f[i] - global_min) < EPS]

    max_in_Fp = any(v in F_plus  for v in max_nodes)
    min_in_Fm = any(v in F_minus for v in min_nodes)
    satisfies = max_in_Fp and min_in_Fm

    info = {
        'c': int(c),
        'char_case': char_case,
        'K_plus': int(K_plus),
        'K_minus': int(K_minus),
        'F_plus': sorted(int(v) for v in F_plus),
        'F_minus': sorted(int(v) for v in F_minus),
        'V_plus_leaves': sorted(int(v) for v in Vp_leaves),
        'V_minus_leaves': sorted(int(v) for v in Vm_leaves),
        'global_max': global_max,
        'global_min': global_min,
        'max_nodes': sorted(int(v) for v in max_nodes),
        'min_nodes': sorted(int(v) for v in min_nodes),
        'max_in_Fp': bool(max_in_Fp),
        'min_in_Fm': bool(min_in_Fm),
        'satisfies': bool(satisfies),
    }
    return satisfies, info


def graph_to_record(G, f, nodes, lam2, lam3, info, source='exhaustive'):
    """Serialize graph and Fiedler data to a JSON-serialisable dict."""
    return {
        'source': source,
        'n': int(G.number_of_nodes()),
        'edges': sorted([sorted([int(u), int(v)]) for u, v in G.edges()]),
        'diameter': int(nx.diameter(G)),
        'lambda2': float(lam2),
        'lambda3': float(lam3),
        'spectral_gap': float(lam3 - lam2),
        'fiedler_vector': {str(nodes[i]): float(f[i]) for i in range(len(nodes))},
        'olep_info': info,
    }


# ─────────────────────────────────────────────
# Table 3: exhaustive enumeration n=5..17
# ─────────────────────────────────────────────

OEIS_A000055 = {5:3,6:6,7:11,8:23,9:47,10:106,11:235,12:551,
                13:1301,14:3159,15:7741,16:19320,17:48629}

def run_table1(n_max=17):
    print(f"\n{'='*60}")
    print(f"表1：穷举枚举 n=5..{n_max}（定义B）")
    print(f"{'='*60}")

    rows = []
    all_counterex = []

    for n in range(5, n_max + 1):
        t0 = time.time()
        trees = list(nx.nonisomorphic_trees(n))
        total = len(trees)
        oeis_ok = (total == OEIS_A000055.get(n, -1))

        cx_list = []
        for G in trees:
            f, nodes, lam2, lam3 = compute_fiedler(G)
            ok, info = check_olep_B(G, f, nodes)
            if not ok:
                rec = graph_to_record(G, f, nodes, lam2, lam3, info)
                cx_list.append(rec)

        nc = len(cx_list)
        rate = nc / total * 100
        elapsed = time.time() - t0
        row = {
            'n': n, 'total_trees': int(total),
            'oeis_ok': bool(oeis_ok),
            'counterexamples': int(nc),
            'rate_pct': round(rate, 4),
            'time_s': round(elapsed, 1),
        }
        rows.append(row)
        all_counterex.extend(cx_list)

        print(f"  n={n:2d}: {total:6d}棵 (OEIS {'OK' if oeis_ok else 'FAIL'}), "
              f"反例 {nc:4d} ({rate:.4f}%), 用时 {elapsed:.1f}s")

    return rows, all_counterex


# ─────────────────────────────────────────────
# Table 4: six random tree models
# ─────────────────────────────────────────────

def gen_ba(n, seed=None):
    rng = random.Random(seed)
    return nx.barabasi_albert_graph(n, 1, seed=rng.randint(0, 999999))

def gen_prufer(n, seed=None):
    rng = np.random.default_rng(seed)
    return nx.from_prufer_sequence(rng.integers(0, n, size=n-2).tolist())

def gen_er_mst(n, seed=None):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        G_er = nx.erdos_renyi_graph(n, 2*np.log(n)/n, seed=int(rng.integers(0, 999999)))
        if nx.is_connected(G_er): break
    for u, v in G_er.edges():
        G_er[u][v]['weight'] = float(rng.random())
    return nx.minimum_spanning_tree(G_er)

def gen_binary(n, seed=None):
    rng = random.Random(seed)
    G = nx.Graph(); G.add_node(0)
    for i in range(1, n):
        cands = [v for v in G.nodes() if G.degree(v) < 3]
        G.add_edge(rng.choice(cands) if cands else rng.choice(list(G.nodes())), i)
    return G

def gen_caterpillar(n, seed=None):
    rng = random.Random(seed)
    sl = rng.randint(2, max(2, n-1))
    G = nx.path_graph(sl)
    nn, spine = sl, list(range(sl))
    while nn < n:
        G.add_edge(rng.choice(spine), nn); nn += 1
    return G

def gen_lobster(n, seed=None):
    rng = random.Random(seed)
    sl = rng.randint(2, max(2, n//3))
    G = nx.path_graph(sl)
    nn, spine, brs = sl, list(range(sl)), []
    while nn < n and len(brs) < n//2:
        G.add_edge(rng.choice(spine), nn); brs.append(nn); nn += 1
    if brs:
        while nn < n:
            G.add_edge(rng.choice(brs), nn); nn += 1
    return G

def run_table2(n_per=2500, n_range=(20, 100), seed=42):
    print(f"\n{'='*60}")
    print(f"表2：6种随机树各{n_per}棵，n∈{n_range}（定义B，seed={seed}）")
    print(f"{'='*60}")

    models = {
        'BA':          gen_ba,
        'ER-MST':      gen_er_mst,
        'Binary':      gen_binary,
        'Prufer':      gen_prufer,
        'Caterpillar': gen_caterpillar,
        'Lobster':     gen_lobster,
    }
    rng = np.random.default_rng(seed)
    rows = []

    for name, fn in models.items():
        cx = 0; t0 = time.time()
        for _ in range(n_per):
            n = int(rng.integers(n_range[0], n_range[1]+1))
            s = int(rng.integers(0, 999999))
            try:
                G = fn(n, seed=s)
                if not nx.is_tree(G) or G.number_of_nodes() < 3: continue
                f, nodes, lam2, _ = compute_fiedler(G)
                ok, _ = check_olep_B(G, f, nodes)
                if not ok: cx += 1
            except Exception: continue

        rate = cx / n_per * 100
        elapsed = time.time() - t0
        row = {
            'model': name, 'total': int(n_per),
            'counterexamples': int(cx),
            'rate_pct': round(rate, 2),
            'time_s': round(elapsed, 1),
        }
        rows.append(row)
        print(f"  {name:<12}: 反例 {cx:4d}/{n_per} ({rate:.2f}%),  用时 {elapsed:.1f}s")

    return rows


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OLEP Verification -- global-deepest definition")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Table 3
    t1_rows, cx_all = run_table1(n_max=17)

    # save outputs
    with open(os.path.join(OUTPUT_DIR, 'table1_by_n.json'), 'w', encoding='utf-8') as fp:
        json.dump({'table1': t1_rows, 'definition': 'B_global_deepest'}, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, 'counterex_exhaustive.json'), 'w', encoding='utf-8') as fp:
        json.dump(cx_all, fp, indent=2, ensure_ascii=False)

    # human-readable summary
    summary_path = os.path.join(OUTPUT_DIR, 'table1_summary.txt')
    total_trees_all = sum(r['total_trees'] for r in t1_rows)
    total_cx_all    = sum(r['counterexamples'] for r in t1_rows)
    with open(summary_path, 'w', encoding='utf-8') as fp:
        fp.write("OLEP 反例统计（定义B：全局最深，v17正文定义）\n")
        fp.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fp.write(f"{'n':>4}  {'总树数':>8}  {'反例数':>7}  {'比例':>9}  {'OEIS':>5}\n")
        fp.write('-' * 48 + '\n')
        for r in t1_rows:
            fp.write(f"{r['n']:>4}  {r['total_trees']:>8}  {r['counterexamples']:>7}  "
                     f"{r['rate_pct']:>8.4f}%  {'OK' if r['oeis_ok'] else 'FAIL':>5}\n")
        fp.write('-' * 48 + '\n')
        fp.write(f"{'合计':>4}  {total_trees_all:>8}  {total_cx_all:>7}\n")
    print(f"\n表1摘要已保存: {summary_path}")

    # Table 4
    t2_rows = run_table2()

    with open(os.path.join(OUTPUT_DIR, 'table2_random.json'), 'w', encoding='utf-8') as fp:
        json.dump({'table2': t2_rows, 'definition': 'B_global_deepest',
                   'n_per_model': 2500, 'n_range': [20, 100], 'seed': 42},
                  fp, indent=2, ensure_ascii=False)

    # metadata
    meta = {
        'definition': 'B_global_deepest',
        'timestamp': datetime.now().isoformat(),
        'table1_total_trees': int(total_trees_all),
        'table1_total_counterex': int(total_cx_all),
        'table1_min_n_with_counterex': min((r['n'] for r in t1_rows if r['counterexamples']>0), default=None),
        'table1_by_n': {str(r['n']): r['counterexamples'] for r in t1_rows},
        'table2_by_model': {r['model']: {'counterexamples': r['counterexamples'],
                                          'rate_pct': r['rate_pct']} for r in t2_rows},
    }
    with open(os.path.join(OUTPUT_DIR, 'counterex_metadata.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    # final summary
    min_n = meta['table1_min_n_with_counterex']
    print(f"\n{'='*60}")
    print(f"【表1最终结果（定义B）】")
    print(f"  {'n':>4}  {'总数':>7}  {'反例':>6}  {'比例':>9}")
    for r in t1_rows:
        print(f"  {r['n']:>4}  {r['total_trees']:>7}  {r['counterexamples']:>6}  {r['rate_pct']:>8.4f}%")
    print(f"  合计  {total_trees_all:>7}  {total_cx_all:>6}")
    print(f"  Smallest counterexample: n = {min_n}")
    print(f"\n【表2最终结果（定义B）】")
    print(f"  {'模型':<12}  {'反例数':>6}  {'比例':>8}")
    for r in t2_rows:
        print(f"  {r['model']:<12}  {r['counterexamples']:>6}  {r['rate_pct']:>7.2f}%")
    print(f"\n所有结果已保存至: {OUTPUT_DIR}")
    print(f"  表1穷举反例: counterex_exhaustive.json ({total_cx_all} 个)")
    print(f"  元信息汇总:  counterex_metadata.json")

    return t1_rows, t2_rows, total_cx_all, min_n


if __name__ == '__main__':
    main()
