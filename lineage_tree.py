from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from modular_network import ActivationPattern, HyperParams


@dataclass
class TaskNode:
    task_id: str
    pattern: ActivationPattern
    hyper_params: HyperParams
    performance: float = float("-inf")
    causal_effects: Dict[str, float] = field(default_factory=dict)
    visit_count: int = 0
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    creation_step: int = 0
    merged_from: List[str] = field(default_factory=list)

    def feature_vector(self, all_module_ids: List[str]) -> np.ndarray:
        pattern_vec = self.pattern.to_vector(all_module_ids)
        hp_vec = self.hyper_params.to_tensor().numpy()
        causal_vec = np.array(
            [self.causal_effects.get(mid, 0.0) for mid in all_module_ids],
            dtype=np.float32,
        )
        return np.concatenate([pattern_vec * 2.0, hp_vec * 0.5, causal_vec * 1.5])

    def composite_affinity(
        self,
        other: "TaskNode",
        all_module_ids: List[str],
        jaccard_weight: float = 0.4,
        causal_weight: float = 0.35,
        hp_weight: float = 0.25,
    ) -> float:
        jaccard = self.pattern.jaccard_similarity(other.pattern)

        common_modules = self.pattern.active_modules & other.pattern.active_modules
        if common_modules:
            causal_sim = 0.0
            for m in common_modules:
                ce_a = self.causal_effects.get(m, 0.0)
                ce_b = other.causal_effects.get(m, 0.0)
                if ce_a + ce_b > 0:
                    causal_sim += 2.0 * min(ce_a, ce_b) / (ce_a + ce_b)
            causal_sim /= len(common_modules)
        else:
            causal_sim = 0.0

        hp_a = self.hyper_params.to_tensor().numpy()
        hp_b = other.hyper_params.to_tensor().numpy()
        hp_dist = np.linalg.norm(hp_a - hp_b)
        hp_sim = float(np.exp(-hp_dist * 5.0))

        return (
            jaccard_weight * jaccard
            + causal_weight * causal_sim
            + hp_weight * hp_sim
        )


class LineageTree:
    def __init__(
        self,
        all_module_ids: List[str],
        merge_threshold: float = 0.85,
        prune_threshold: float = 0.1,
        max_depth: int = 10,
        max_nodes: int = 200,
    ):
        self.all_module_ids = all_module_ids
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        self.nodes: Dict[str, TaskNode] = {}
        self.root_ids: List[str] = []
        self.step_counter: int = 0
        self.performance_history: Dict[str, List[float]] = {}

    def _generate_id(self) -> str:
        return f"task_{uuid.uuid4().hex[:8]}"

    def add_node(
        self,
        pattern: ActivationPattern,
        hyper_params: HyperParams,
        performance: Optional[float] = None,
        causal_effects: Optional[Dict[str, float]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        node_id = self._generate_id()
        depth = 0
        if parent_id and parent_id in self.nodes:
            depth = self.nodes[parent_id].depth + 1
            self.nodes[parent_id].children_ids.append(node_id)

        perf = performance if performance is not None else float("-inf")

        node = TaskNode(
            task_id=node_id,
            pattern=pattern,
            hyper_params=hyper_params,
            performance=perf,
            causal_effects=causal_effects or {},
            depth=depth,
            parent_id=parent_id,
            creation_step=self.step_counter,
        )
        self.nodes[node_id] = node
        self.performance_history[node_id] = [perf] if performance is not None else []

        if parent_id is None:
            self.root_ids.append(node_id)

        self.step_counter += 1

        if len(self.nodes) > self.max_nodes * 1.5:
            self.prune()

        return node_id

    def update_node(
        self,
        node_id: str,
        performance: Optional[float] = None,
        causal_effects: Optional[Dict[str, float]] = None,
        pattern: Optional[ActivationPattern] = None,
        hyper_params: Optional[HyperParams] = None,
    ):
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.visit_count += 1
        if performance is not None:
            if node.performance == float("-inf"):
                node.performance = performance
            else:
                node.performance = max(node.performance, performance)
            self.performance_history[node_id].append(performance)
        if causal_effects is not None:
            for m, v in causal_effects.items():
                node.causal_effects[m] = 0.7 * node.causal_effects.get(m, 0.0) + 0.3 * v
        if pattern is not None:
            node.pattern = pattern
        if hyper_params is not None:
            node.hyper_params = hyper_params

    def _valid_performances(self) -> Dict[str, float]:
        return {
            nid: node.performance
            for nid, node in self.nodes.items()
            if node.performance > float("-inf")
        }

    def _perf_rank_bonus(self, node_perf: float, valid_perfs: Dict[str, float]) -> float:
        if not valid_perfs or node_perf == float("-inf"):
            return 0.0
        perfs = list(valid_perfs.values())
        if len(perfs) < 2:
            return 0.05
        min_p = min(perfs)
        max_p = max(perfs)
        if max_p - min_p < 1e-8:
            return 0.05
        normalized = (node_perf - min_p) / (max_p - min_p)
        return 0.1 * float(normalized)

    def find_nearest_ancestor(
        self,
        pattern: ActivationPattern,
        causal_effects: Optional[Dict[str, float]] = None,
        candidate_hp: Optional[HyperParams] = None,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        if not self.nodes:
            return []

        tmp_causal = causal_effects or {}
        tmp_hp = candidate_hp or HyperParams()

        query_node = TaskNode(
            task_id="_query_",
            pattern=pattern,
            hyper_params=tmp_hp,
            causal_effects=tmp_causal,
        )

        valid_perfs = self._valid_performances()

        scores: List[Tuple[str, float]] = []
        for nid, node in self.nodes.items():
            affinity = query_node.composite_affinity(node, self.all_module_ids)
            depth_bonus = 1.0 / (1.0 + node.depth * 0.05)
            perf_bonus = self._perf_rank_bonus(node.performance, valid_perfs)
            freq_bonus = 0.05 * min(node.visit_count, 20) / 20.0
            final_score = affinity * depth_bonus + perf_bonus + freq_bonus
            scores.append((nid, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_path_to_root(self, node_id: str, max_depth: int = 200) -> List[str]:
        path = []
        current = node_id
        visited = set()
        while (
            current is not None
            and current in self.nodes
            and len(path) < max_depth
            and current not in visited
        ):
            visited.add(current)
            path.append(current)
            current = self.nodes[current].parent_id
        return path

    def get_siblings(self, node_id: str) -> List[str]:
        if node_id not in self.nodes:
            return []
        pid = self.nodes[node_id].parent_id
        if pid is None or pid not in self.nodes:
            return [n for n in self.root_ids if n != node_id]
        return [c for c in self.nodes[pid].children_ids if c != node_id]

    def cluster_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        num_clusters: Optional[int] = None,
    ) -> Dict[int, List[str]]:
        target_ids = node_ids or list(self.nodes.keys())
        if len(target_ids) < 2:
            return {0: target_ids}

        features = []
        for nid in target_ids:
            node = self.nodes[nid]
            features.append(node.feature_vector(self.all_module_ids))

        features_np = np.vstack(features)
        dist = pdist(features_np, metric="cosine")

        if num_clusters is None:
            num_clusters = max(2, min(len(target_ids) // 4, 10))
        num_clusters = min(num_clusters, len(target_ids))

        if len(target_ids) <= 2:
            return {i: [target_ids[i]] for i in range(len(target_ids))}

        try:
            z = linkage(dist, method="ward")
            labels = fcluster(z, t=num_clusters, criterion="maxclust")
        except Exception:
            labels = np.array([(i % num_clusters) + 1 for i in range(len(target_ids))])

        clusters: Dict[int, List[str]] = {}
        for nid, label in zip(target_ids, labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(nid)
        return clusters

    def merge_similar_nodes(self, force: bool = False) -> List[Tuple[str, List[str]]]:
        if len(self.nodes) < 3:
            return []

        merges: List[Tuple[str, List[str]]] = []
        clusters = self.cluster_nodes()

        for cluster_id, member_ids in clusters.items():
            if len(member_ids) < 2:
                continue

            members = [(nid, self.nodes[nid]) for nid in member_ids]

            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    nid_a, node_a = members[i]
                    nid_b, node_b = members[j]

                    if nid_a not in self.nodes or nid_b not in self.nodes:
                        continue
                    if nid_a == nid_b:
                        continue

                    path_a = self.find_path_to_root(nid_a)
                    path_b = self.find_path_to_root(nid_b)

                    a_is_ancestor_of_b = nid_a in path_b
                    b_is_ancestor_of_a = nid_b in path_a
                    if a_is_ancestor_of_b or b_is_ancestor_of_a:
                        continue

                    affinity = node_a.composite_affinity(node_b, self.all_module_ids)

                    if affinity >= self.merge_threshold or force:
                        surviving_id, removed_id = (
                            (nid_a, nid_b)
                            if node_a.performance >= node_b.performance
                            else (nid_b, nid_a)
                        )
                        survivor = self.nodes[surviving_id]
                        removed = self.nodes[removed_id]

                        survivor.performance = max(survivor.performance, removed.performance)
                        survivor.visit_count += removed.visit_count

                        new_active = survivor.pattern.active_modules | removed.pattern.active_modules
                        new_weights = {}
                        for m in new_active:
                            wa = survivor.pattern.weights.get(m, 0.0)
                            wb = removed.pattern.weights.get(m, 0.0)
                            new_weights[m] = (wa + wb) / 2.0
                        total_w = sum(new_weights.values())
                        new_weights = {k: v / total_w for k, v in new_weights.items()}
                        survivor.pattern = ActivationPattern(new_active, new_weights)

                        for m, v in removed.causal_effects.items():
                            survivor.causal_effects[m] = 0.5 * survivor.causal_effects.get(m, 0.0) + 0.5 * v

                        survivor.hyper_params = HyperParams.crossover(
                            survivor.hyper_params,
                            removed.hyper_params,
                            alpha=0.6,
                        )

                        for cid in removed.children_ids:
                            if cid in self.nodes:
                                self.nodes[cid].parent_id = surviving_id
                                survivor.children_ids.append(cid)

                        pid = removed.parent_id
                        if pid is not None and pid in self.nodes:
                            if removed_id in self.nodes[pid].children_ids:
                                self.nodes[pid].children_ids.remove(removed_id)
                            if surviving_id not in self.nodes[pid].children_ids:
                                self.nodes[pid].children_ids.append(surviving_id)
                        else:
                            if removed_id in self.root_ids:
                                self.root_ids.remove(removed_id)
                            if surviving_id not in self.root_ids and survivor.parent_id is None:
                                self.root_ids.append(surviving_id)

                        survivor.merged_from.append(removed_id)
                        survivor.merged_from.extend(removed.merged_from)

                        del self.nodes[removed_id]
                        if removed_id in self.performance_history:
                            del self.performance_history[removed_id]

                        merges.append((surviving_id, [removed_id] + removed.merged_from))
                        break
        return merges

    def prune(self, min_performance: Optional[float] = None) -> List[str]:
        if len(self.nodes) <= self.max_nodes:
            return []

        threshold = min_performance if min_performance is not None else float("-inf")
        pruned: List[str] = []

        def sort_key(item):
            nid, node = item
            has_perf = node.performance > float("-inf")
            perf_score = node.performance if has_perf else float("-inf")
            return (
                0 if has_perf else 1,
                perf_score - threshold,
                -node.visit_count,
                -node.creation_step,
            )

        sorted_nodes = sorted(self.nodes.items(), key=sort_key)

        to_remove = max(0, len(self.nodes) - self.max_nodes)
        candidates = [nid for nid, node in sorted_nodes[: to_remove + len(sorted_nodes) // 4]]

        for nid in candidates:
            if len(self.nodes) <= self.max_nodes:
                break
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.visit_count > 10 and node.performance > threshold and node.performance > float("-inf"):
                continue
            if node.children_ids:
                continue

            pid = node.parent_id
            if pid is not None and pid in self.nodes:
                if nid in self.nodes[pid].children_ids:
                    self.nodes[pid].children_ids.remove(nid)
            if nid in self.root_ids:
                self.root_ids.remove(nid)

            del self.nodes[nid]
            if nid in self.performance_history:
                del self.performance_history[nid]
            pruned.append(nid)

        return pruned

    def evolve(self) -> Dict[str, object]:
        merges = self.merge_similar_nodes()
        pruned = self.prune()
        return {"merges": merges, "pruned": pruned}

    def get_statistics(self) -> Dict[str, object]:
        if not self.nodes:
            return {
                "num_nodes": 0,
                "num_roots": 0,
                "avg_depth": 0.0,
                "avg_perf": float("-inf"),
                "best_perf": float("-inf"),
                "valid_perf_count": 0,
            }
        depths = [n.depth for n in self.nodes.values()]
        valid_perfs = [n.performance for n in self.nodes.values() if n.performance > float("-inf")]
        avg_perf = float(np.mean(valid_perfs)) if valid_perfs else float("-inf")
        best_perf = float(max(valid_perfs)) if valid_perfs else float("-inf")
        return {
            "num_nodes": len(self.nodes),
            "num_roots": len(self.root_ids),
            "max_depth": max(depths),
            "avg_depth": float(np.mean(depths)),
            "avg_perf": avg_perf,
            "best_perf": best_perf,
            "valid_perf_count": len(valid_perfs),
            "total_visits": sum(n.visit_count for n in self.nodes.values()),
        }

    def generate_report(self, top_k: int = 10, recent_history: int = 5) -> Dict[str, object]:
        valid_items = [
            (nid, node)
            for nid, node in self.nodes.items()
            if node.performance > float("-inf")
        ]
        sorted_desc = sorted(valid_items, key=lambda x: x[1].performance, reverse=True)
        sorted_asc = sorted(valid_items, key=lambda x: x[1].performance, reverse=False)

        def build_node_detail(nid: str, node: TaskNode) -> Dict[str, object]:
            history = self.performance_history.get(nid, [])
            recent = history[-recent_history:] if history else []
            parent_perf = None
            if node.parent_id and node.parent_id in self.nodes:
                pnode = self.nodes[node.parent_id]
                parent_perf = pnode.performance if pnode.performance > float("-inf") else None
            return {
                "task_id": nid,
                "performance": float(node.performance) if node.performance > float("-inf") else None,
                "depth": node.depth,
                "visit_count": node.visit_count,
                "creation_step": node.creation_step,
                "parent_id": node.parent_id,
                "parent_performance": float(parent_perf) if parent_perf is not None else None,
                "num_children": len(node.children_ids),
                "active_modules": sorted(list(node.pattern.active_modules)),
                "module_weights": {
                    k: float(v) for k, v in node.pattern.weights.items()
                },
                "causal_effects": {
                    k: float(v) for k, v in node.causal_effects.items()
                },
                "hyper_params": {
                    "learning_rate": float(node.hyper_params.learning_rate),
                    "momentum": float(node.hyper_params.momentum),
                    "dropout_rate": float(node.hyper_params.dropout_rate),
                    "weight_decay": float(node.hyper_params.weight_decay),
                    "hidden_dim": int(node.hyper_params.hidden_dim),
                },
                "recent_performances": [float(v) for v in recent],
                "merged_from": list(node.merged_from),
            }

        top_details = [build_node_detail(nid, n) for nid, n in sorted_desc[:top_k]]
        bottom_details = [build_node_detail(nid, n) for nid, n in sorted_asc[:top_k]]

        module_usage: Dict[str, int] = {}
        module_best_perf: Dict[str, float] = {}
        for nid, node in valid_items:
            for m in node.pattern.active_modules:
                module_usage[m] = module_usage.get(m, 0) + 1
                cur = module_best_perf.get(m, float("-inf"))
                if node.performance > cur:
                    module_best_perf[m] = node.performance

        depth_stats: Dict[int, Dict[str, float]] = {}
        for nid, node in valid_items:
            d = node.depth
            if d not in depth_stats:
                depth_stats[d] = {"count": 0, "perf_sum": 0.0}
            depth_stats[d]["count"] += 1
            depth_stats[d]["perf_sum"] += float(node.performance)
        depth_summary = {}
        for d, s in depth_stats.items():
            depth_summary[str(d)] = {
                "count": s["count"],
                "avg_performance": float(s["perf_sum"] / max(s["count"], 1)),
            }

        return {
            "summary": self.get_statistics(),
            "top_tasks": top_details,
            "bottom_tasks": bottom_details,
            "valid_count": len(valid_items),
            "total_count": len(self.nodes),
            "module_usage": {m: int(c) for m, c in module_usage.items()},
            "module_best_performance": {
                m: float(v) if v > float("-inf") else None for m, v in module_best_perf.items()
            },
            "depth_breakdown": depth_summary,
        }

    @staticmethod
    def format_report_table(
        report: Dict[str, object],
        title: str,
        items: List[Dict[str, object]],
    ) -> str:
        if not items:
            return f"  [{title}] No valid tasks\n"
        lines = [f"  [{title}] Top {len(items)} by performance:"]
        lines.append(
            f"  {'Rank':<4} {'Perf':>8} {'Depth':>5} {'Visits':>6} {'Mods':>4} "
            f"{'Parent':<10} {'HP-LR':>7} {'HP-Drop':>7}"
        )
        lines.append("  " + "-" * 70)
        for rank, item in enumerate(items, 1):
            perf = f"{item['performance']:.3f}" if item["performance"] is not None else "N/A"
            mods = len(item["active_modules"])
            parent = (item["parent_id"] or "-")[:8]
            lr = f"{item['hyper_params']['learning_rate']:.4f}"
            drop = f"{item['hyper_params']['dropout_rate']:.3f}"
            lines.append(
                f"  {rank:<4} {perf:>8} {item['depth']:>5} "
                f"{item['visit_count']:>6} {mods:>4} {parent:<10} {lr:>7} {drop:>7}"
            )
        return "\n".join(lines) + "\n"

    def print_report(self, top_k: int = 10):
        report = self.generate_report(top_k=top_k)
        print("=" * 70)
        print("LINEAGE TASK RANKING")
        print("=" * 70)
        summary = report["summary"]
        print(
            f"  Valid: {report['valid_count']}/{report['total_count']} nodes | "
            f"BestPerf: {summary['best_perf']:.4f} | "
            f"AvgPerf: {summary['avg_perf']:.4f} | "
            f"AvgDepth: {summary['avg_depth']:.2f}"
        )
        print()
        print(self.format_report_table(report, "TOP PERFORMERS", report["top_tasks"]))
        print(self.format_report_table(report, "BOTTOM PERFORMERS", report["bottom_tasks"]))

        module_items = sorted(
            report["module_usage"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if module_items:
            print("  [MODULE USAGE] (usage_count, best_perf)")
            line_parts = []
            for m, c in module_items:
                bp = report["module_best_performance"].get(m)
                bp_str = f"{bp:.3f}" if bp is not None else "N/A"
                line_parts.append(f"{m}({c},{bp_str})")
            print("  " + "  ".join(line_parts))
            print()
        print("=" * 70)
        return report

    def print_tree(self, max_depth: int = 5):
        def _print_subtree(nid: str, indent: int):
            node = self.nodes.get(nid)
            if node is None:
                return
            prefix = "  " * indent
            marker = "├─ " if indent > 0 else "◆ "
            perf_str = f"{node.performance:.3f}" if node.performance > float("-inf") else "N/A"
            print(
                f"{prefix}{marker}[{nid}] "
                f"depth={node.depth} perf={perf_str} "
                f"visits={node.visit_count} mods={len(node.pattern.active_modules)}"
            )
            if indent >= max_depth:
                return
            for cid in node.children_ids:
                _print_subtree(cid, indent + 1)

        for rid in self.root_ids:
            _print_subtree(rid, 0)
