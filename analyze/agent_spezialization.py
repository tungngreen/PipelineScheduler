#!/usr/bin/env python3
"""
multiagent_specialization.py

Compute inter-agent specialization/divergence metrics from rollout logs.
Supports either:
 - multiple CSV files named agent_<id>.csv, each with columns [rollout_id, obj1, obj2, obj3]
 - OR a single CSV with column 'agent_id' plus 'rollout_id' and objectives.

Outputs per-agent HV and SP, pairwise Avg-Min, Hausdorff, centroid and Wasserstein distances,
and a composite specialization score.

Dependencies: numpy, pandas, scipy. Optional: pymoo for exact HV (if available).
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from itertools import combinations
import math
from pymoo.indicators.hv import HV as PymooHV

# -------------------------
# Utilities
# -------------------------
def load_agents_from_folder(pattern="agent_*.csv", n_rollouts=30, obj_cols=None):
    agents = {}
    for path in sorted(glob.glob(pattern)):
        aid = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        # If rollout_id present, aggregate per rollout_id (mean)
        if 'rollout_id' in df.columns:
            dfg = df.groupby('rollout_id').mean()
            arr = dfg.iloc[:n_rollouts].to_numpy()
        else:
            arr = df.iloc[:n_rollouts].to_numpy()
        if obj_cols:
            arr = df[obj_cols].head(n_rollouts).to_numpy()
        agents[aid] = arr
    return agents


def dominates_max(a, b):
    return np.all(a >= b) and np.any(a > b)


def pareto_filter(points):
    pts = np.asarray(points)
    if len(pts) == 0:
        return pts
    mask = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not mask[i]:
            continue
        for j in range(len(pts)):
            if i == j or not mask[j]:
                continue
            if dominates_max(pts[j], pts[i]):
                mask[i] = False
                break
    return pts[mask]


def mc_hypervolume(front, ref, n_samples=200000):
    front = np.asarray(front)
    if len(front) == 0:
        return 0.0
    d = front.shape[1]
    upper = np.max(front, axis=0)
    if np.any(upper <= ref):
        return 0.0
    box_size = upper - ref
    box_vol = float(np.prod(box_size))
    count_in = 0
    # vectorized-ish sampling in batches to speed up
    batch = 5000
    samples_needed = n_samples
    while samples_needed > 0:
        b = min(batch, samples_needed)
        samp = np.random.rand(b, d) * box_size + ref
        # check dominance per sample
        # for each sample, see if any front point dominates it
        # compute comparisons in vectorized manner
        dominated = np.any(np.all(front[:, None, :] >= samp[None, :, :], axis=2) & np.any(front[:, None, :] > samp[None, :, :], axis=2), axis=0)
        count_in += np.count_nonzero(dominated)
        samples_needed -= b
    return box_vol * (count_in / float(n_samples))


def compute_hv(front, ref, mc_samples=200000):
    front = np.array(front)
    # filter dominated and points <= ref
    front = front[np.all(front > ref, axis=1)]
    front = pareto_filter(front)
    if front.shape[0] == 0:
        return 0.0
    try:
        hv = PymooHV(ref_point=np.array(ref))
        return float(hv(front))
    except Exception:
        return mc_hypervolume(front, ref, n_samples=mc_samples)


def sparsity(front):
    front = np.asarray(front)
    if len(front) < 2:
        return 0.0
    dists = cdist(front, front)
    # take upper triangle
    iu = np.triu_indices_from(dists, k=1)
    return float(np.mean(dists[iu]))


def hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return float('inf')
    D = cdist(A, B)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def avg_min(A, B):
    if len(A) == 0 or len(B) == 0:
        return float('inf')
    D = cdist(A, B)
    return float(np.mean(D.min(axis=1)))


def centroid_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return float('inf')
    return float(np.linalg.norm(np.mean(A, axis=0) - np.mean(B, axis=0)))


def per_object_wasserstein(A, B):
    # average 1D Wasserstein over objectives
    A = np.array(A)
    B = np.array(B)
    if A.size == 0 or B.size == 0:
        return float('inf')
    d = A.shape[1]
    ws = [wasserstein_distance(A[:, i], B[:, i]) for i in range(d)]
    return float(np.mean(ws))


def composite_specialization(agents_arrs):
    """
    Compute symmetric mean pairwise avg-min normalized by mean within-agent spread.
    Higher => more between-agent separation relative to within-agent variability.
    """
    keys = list(agents_arrs.keys())
    m = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            A = agents_arrs[keys[i]]
            B = agents_arrs[keys[j]]
            v = 0.5 * (avg_min(A, B) + avg_min(B, A))
            m.append(v)
    mean_between = np.mean(m) if len(m) > 0 else 0.0
    within = [sparsity(agents_arrs[k]) for k in keys]
    mean_within = np.mean(within) if len(within) > 0 else 1.0
    return float(mean_between / (mean_within + 1e-12)), mean_between, mean_within


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_pattern", type=str, default="agent_*.csv",
                        help="pattern for per-agent CSV files (default agent_*.csv)")
    parser.add_argument("--agent_col", type=str, default="agent_id",
                        help="column name for agent id in single-csv mode")
    parser.add_argument("--obj_cols", type=str, default=None,
                        help="comma separated objective column names (e.g. obj1,obj2,obj3). If omitted tries to infer.")
    parser.add_argument("--rollouts", type=int, default=30)
    parser.add_argument("--mc_samples", type=int, default=100000)
    args = parser.parse_args()

    agents = load_agents_from_folder(pattern=args.folder_pattern, n_rollouts=args.rollouts,
                                         obj_cols=(args.obj_cols.split(",") if args.obj_cols else None))
    if len(agents) == 0:
        print("No agents found. Check paths or CSV format.")
        return

    # infer reference point across all agents (5% below worst)
    all_pts = np.vstack([v for v in agents.values() if len(v) > 0])
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    ref = mins - 0.05 * (maxs - mins + 1e-12)

    print("Agents found:", list(agents.keys()))
    print("Reference point (shared):", ref)

    per_agent = {}
    for aid, arr in agents.items():
        arr = np.array(arr)
        front = pareto_filter(arr)
        hv = compute_hv(front, ref, mc_samples=args.mc_samples)
        sp = sparsity(front)
        centroid = np.mean(arr, axis=0) if arr.size else np.zeros(arr.shape[1])
        per_agent[aid] = {"front": front, "hv": hv, "sp": sp, "centroid": centroid}
        print(f"\nAgent {aid}:")
        print(f"  rollouts: {arr.shape[0]}, non-dominated: {front.shape[0]}")
        print(f"  HV: {hv:.6f}")
        print(f"  Sparsity: {sp:.6f}")
        print(f"  Centroid: {centroid}")

    print("\nPairwise metrics (rows -> cols):")
    keys = list(per_agent.keys())
    K = len(keys)
    haus = np.zeros((K, K))
    avgmin = np.zeros((K, K))
    cent = np.zeros((K, K))
    wdist = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A = per_agent[keys[i]]["front"]
            B = per_agent[keys[j]]["front"]
            haus[i, j] = hausdorff(A, B)
            avgmin[i, j] = avg_min(A, B)
            cent[i, j] = centroid_distance(per_agent[keys[i]]["centroid"], per_agent[keys[j]]["centroid"])
            wdist[i, j] = per_object_wasserstein(per_agent[keys[i]]["front"], per_agent[keys[j]]["front"])

    print("\nHausdorff distance matrix:")
    print(pd.DataFrame(haus, index=keys, columns=keys).round(6))

    print("\nAvg-Min distance matrix (A -> B):")
    print(pd.DataFrame(avgmin, index=keys, columns=keys).round(6))

    print("\nCentroid distance matrix:")
    print(pd.DataFrame(cent, index=keys, columns=keys).round(6))

    print("\nPer-object Wasserstein distance matrix (avg over objectives):")
    print(pd.DataFrame(wdist, index=keys, columns=keys).round(6))

    spec_score, mean_between, mean_within = composite_specialization({k: per_agent[k]["front"] for k in keys})
    print("\nComposite Specialization Score (mean pairwise avg-min / mean within-agent sparsity):")
    print(f"  specialization = {spec_score:.6f}")
    print(f"  mean_between = {mean_between:.6f}, mean_within = {mean_within:.6f}")

if __name__ == "__main__":
    main()
