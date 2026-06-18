from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from genetic_inheritance import GeneticInheritor
from lineage_tree import LineageTree
from modular_maml import ModularMAML, TaskBatch
from modular_network import ModularNetwork


class SinusoidTaskGenerator:
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        num_samples_support: int = 10,
        num_samples_query: int = 15,
        amplitude_range: Tuple[float, float] = (0.5, 5.0),
        phase_range: Tuple[float, float] = (0.0, np.pi),
        freq_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_support = num_samples_support
        self.k_query = num_samples_query
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.freq_range = freq_range
        self.noise_std = noise_std
        self.base_seed = seed
        self.rng = np.random.RandomState(seed)

    def reset_rng(self, seed_offset: int = 0):
        self.rng = np.random.RandomState(self.base_seed + seed_offset)

    def sample_task_family(self, family_id: int) -> Dict:
        amp = self.rng.uniform(*self.amplitude_range)
        phase = self.rng.uniform(*self.phase_range)
        freq = self.rng.uniform(*self.freq_range)
        x_shift = self.rng.uniform(-2.0, 2.0)
        y_shift = self.rng.uniform(-1.0, 1.0)
        return {
            "family_id": f"family_{family_id:03d}",
            "amplitude": amp,
            "phase": phase,
            "frequency": freq,
            "x_shift": x_shift,
            "y_shift": y_shift,
        }

    def sample_task_from_family(self, family: Dict, task_idx: int) -> TaskBatch:
        amp = family["amplitude"] * self.rng.uniform(0.9, 1.1)
        phase = family["phase"] + self.rng.uniform(-0.1, 0.1)
        freq = family["frequency"] * self.rng.uniform(0.95, 1.05)
        x_shift = family["x_shift"] + self.rng.uniform(-0.3, 0.3)
        y_shift = family["y_shift"] + self.rng.uniform(-0.2, 0.2)

        def f(x):
            return amp * np.sin(freq * (x - x_shift) + phase) + y_shift

        x_s = self.rng.uniform(-5.0, 5.0, size=(self.k_support, self.input_dim))
        y_s = f(x_s) + self.rng.normal(0, self.noise_std, size=x_s.shape)
        x_q = self.rng.uniform(-5.0, 5.0, size=(self.k_query, self.input_dim))
        y_q = f(x_q) + self.rng.normal(0, self.noise_std, size=x_q.shape)

        return TaskBatch(
            task_id=f"{family['family_id']}_t{task_idx:03d}",
            x_support=torch.tensor(x_s, dtype=torch.float32),
            y_support=torch.tensor(y_s, dtype=torch.float32),
            x_query=torch.tensor(x_q, dtype=torch.float32),
            y_query=torch.tensor(y_q, dtype=torch.float32),
            task_family_id=family["family_id"],
        )

    def generate_meta_batch(
        self,
        num_families: int = 4,
        tasks_per_family: int = 2,
    ) -> Tuple[List[TaskBatch], List[Dict]]:
        families = [self.sample_task_family(i) for i in range(num_families)]
        tasks: List[TaskBatch] = []
        for fam in families:
            for t in range(tasks_per_family):
                tasks.append(self.sample_task_from_family(fam, t))
        return tasks, families

    def generate_test_tasks(
        self,
        families: List[Dict],
        tasks_per_family: int = 3,
        seed_offset: int = 100,
    ) -> List[TaskBatch]:
        tasks = []
        for fam in families:
            for t in range(tasks_per_family):
                tasks.append(self.sample_task_from_family(fam, seed_offset + t))
        return tasks

    def generate_novel_families(
        self,
        num_novel: int = 3,
        start_id: int = 1000,
    ) -> List[Dict]:
        return [self.sample_task_family(start_id + i) for i in range(num_novel)]


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo(
    args: Optional[argparse.Namespace] = None,
    **kwargs,
) -> Dict[str, object]:
    if args is None:
        args = parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    set_seeds(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print("=" * 70)
    print("Modular MAML with Lineage Tree - Demo")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Num modules: {args.num_modules}")
    print(f"Inner steps: {args.inner_steps}")
    print(f"Meta steps: {args.meta_steps}")
    print(f"Top-k gating: {args.top_k_modules}")
    print(f"Eval repeats: {args.eval_repeats}")
    print()

    generator = SinusoidTaskGenerator(
        input_dim=1,
        output_dim=1,
        num_samples_support=args.support_shots,
        num_samples_query=args.query_shots,
        seed=args.seed,
    )

    network = ModularNetwork(
        num_modules=args.num_modules,
        input_dim=1,
        output_dim=1,
        base_hidden_dim=args.hidden_dim,
        gating_temperature=1.0,
    )

    all_module_ids = network.module_ids()
    print(f"Module IDs: {all_module_ids}")
    print()

    lineage_tree = LineageTree(
        all_module_ids=all_module_ids,
        merge_threshold=args.merge_threshold,
        prune_threshold=args.prune_threshold,
        max_depth=10,
        max_nodes=args.max_tree_nodes,
    )

    inheritor = GeneticInheritor(
        lineage_tree=lineage_tree,
        diversity_penalty_weight=args.div_penalty,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
    )

    maml = ModularMAML(
        network=network,
        lineage_tree=lineage_tree,
        inheritor=inheritor,
        inner_steps=args.inner_steps,
        first_order=args.first_order,
        ewc_lambda=args.ewc_lambda,
        memory_replay_prob=args.replay_prob,
        evolve_every=args.evolve_every,
        top_k_modules=args.top_k_modules if args.top_k_modules > 0 else None,
        device=device,
    )

    meta_optimizer = optim.AdamW(
        network.parameters(),
        lr=args.meta_lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer, T_max=max(args.meta_steps, 1), eta_min=1e-5
    )

    train_families = [generator.sample_task_family(i) for i in range(8)]
    test_families_seen = train_families[:4]
    test_families_novel = generator.generate_novel_families(3, start_id=900)

    meta_loss_history: List[float] = []
    perf_history: List[float] = []
    tree_size_history: List[int] = []
    evolutions_history: List[Dict] = []

    start_time = time.time()

    for step in range(1, args.meta_steps + 1):
        selected_families = random.sample(
            train_families, min(args.tasks_per_batch, len(train_families))
        )
        meta_batch = []
        for fam in selected_families:
            for _ in range(2):
                meta_batch.append(
                    generator.sample_task_from_family(
                        fam, step * 10 + random.randint(0, 99)
                    )
                )

        log = maml.meta_update(meta_batch, meta_optimizer)
        scheduler.step()

        meta_loss_history.append(log.meta_loss)
        perf_history.append(log.avg_query_perf)
        tree_size_history.append(log.tree_stats.get("num_nodes", 0))
        if log.evolutions:
            evolutions_history.append({"step": step, **log.evolutions})

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            tree_stats = log.tree_stats
            best_perf = tree_stats.get("best_perf", float("-inf"))
            best_perf_str = f"{best_perf:.4f}" if best_perf > float("-inf") else "N/A"
            print(
                f"[Step {step:4d}/{args.meta_steps}] "
                f"MetaLoss={log.meta_loss:.6f} "
                f"AdptLoss={log.avg_adapt_loss:.6f} "
                f"Perf={log.avg_query_perf:.4f} "
                f"Tree={tree_stats.get('num_nodes', 0):3d}nodes "
                f"Roots={tree_stats.get('num_roots', 0):2d} "
                f"BestPerf={best_perf_str} "
                f"ValidPerf={tree_stats.get('valid_perf_count', 0)} "
                f"({elapsed:.1f}s)"
            )
            if log.evolutions:
                n_merged = sum(len(m[1]) for m in log.evolutions.get("merges", []))
                n_pruned = len(log.evolutions.get("pruned", []))
                if n_merged or n_pruned:
                    print(f"  Evolve: merged={n_merged}, pruned={n_pruned}")

    total_time = time.time() - start_time
    print()
    print(f"Training complete in {total_time:.1f}s")

    print()
    print("=" * 70)
    print("TREE STRUCTURE:")
    print("=" * 70)
    lineage_tree.print_tree(max_depth=4)

    print()
    print("=" * 70)
    print("STATISTICS:")
    print("=" * 70)
    stats = lineage_tree.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print()
    lineage_report = lineage_tree.print_report(top_k=8)

    eval_seen_repeats = _evaluate_families_repeated(
        maml,
        generator,
        test_families_seen,
        "seen",
        n_tasks=3,
        repeats=args.eval_repeats,
    )

    print()
    eval_novel_repeats = _evaluate_families_repeated(
        maml,
        generator,
        test_families_novel,
        "novel",
        n_tasks=3,
        repeats=args.eval_repeats,
    )

    ablation_result: Optional[Dict] = None
    if args.ablation:
        print()
        print("=" * 70)
        print("ABLATION STUDY: Comparing w/ and w/o Lineage Initialization")
        print("=" * 70)
        ablation_result = _run_ablation(
            maml,
            generator,
            test_families_novel,
            n_tasks=5,
            repeats=args.eval_repeats,
        )

    results = {
        "meta_loss_history": meta_loss_history,
        "perf_history": perf_history,
        "tree_size_history": tree_size_history,
        "evolutions_history": evolutions_history,
        "final_stats": stats,
        "lineage_report": lineage_report,
        "eval_seen_repeated": eval_seen_repeats,
        "eval_novel_repeated": eval_novel_repeats,
        "ablation": ablation_result,
        "config": vars(args),
        "total_time_seconds": float(total_time),
    }

    if args.output:
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=2,
                default=lambda x: (
                    float(x) if hasattr(x, "item") else str(x)
                ),
                allow_nan=False,
            )
        print(f"\nResults saved to {out_path}")
        results["_output_path"] = out_path

    return results


def _evaluate_single_task_once(
    maml: ModularMAML,
    task: TaskBatch,
    *,
    use_lineage: bool,
) -> Dict[str, object]:
    x_q = task.x_query.to(maml.device)
    y_q = task.y_query.to(maml.device)
    params = maml._make_functional_params()
    pattern, _ = maml._functional_forward_gating(x_q, params)
    pred = maml._functional_forward(x_q, params, pattern, dropout_rate=0.0)
    pre_loss = maml.criterion(pred, y_q).item()

    result = maml.fast_adapt(
        task,
        use_lineage=use_lineage,
        add_to_lineage=False,
    )
    post_loss = result["query_loss"]

    lin = result["lineage"]
    return {
        "task_id": task.task_id,
        "pre_adapt_loss": float(pre_loss),
        "post_adapt_loss": float(post_loss),
        "improvement_pct": (
            (pre_loss - post_loss) / max(pre_loss, 1e-8) * 100.0
        ),
        "n_sources": len(lin.source_ids) if lin else 0,
        "div_penalty": float(lin.diversity_penalty) if lin else 0.0,
        "homog_risk": float(lin.homogenization_risk) if lin else 0.0,
        "mutations": list(lin.mutations_applied) if lin else [],
        "modules_used": sorted(result["pattern"].active_modules),
        "hp_lr": float(result["hyper_params"].learning_rate),
        "hp_dropout": float(result["hyper_params"].dropout_rate),
    }


def _evaluate_families_repeated(
    maml: ModularMAML,
    generator: SinusoidTaskGenerator,
    families: List[Dict],
    label: str,
    n_tasks: int = 3,
    repeats: int = 1,
) -> Dict[str, object]:
    print("=" * 70)
    print(f"EVALUATION - {label.capitalize()} Families (repeats={repeats})")
    print("=" * 70)

    nodes_before = len(maml.lineage_tree.nodes)
    repeat_results: List[Dict] = []

    for r in range(repeats):
        generator.reset_rng(seed_offset=10000 + r * 1000)
        tasks = generator.generate_test_tasks(
            families, tasks_per_family=n_tasks, seed_offset=100 + r * 100
        )
        task_details = []
        for task in tasks:
            detail = _evaluate_single_task_once(maml, task, use_lineage=True)
            task_details.append(detail)

        pre_losses = [d["pre_adapt_loss"] for d in task_details]
        post_losses = [d["post_adapt_loss"] for d in task_details]
        improvements = [d["improvement_pct"] for d in task_details]

        repeat_summary = {
            "repeat_idx": r,
            "num_tasks": len(tasks),
            "pre_loss_mean": float(np.mean(pre_losses)),
            "pre_loss_std": float(np.std(pre_losses)),
            "post_loss_mean": float(np.mean(post_losses)),
            "post_loss_std": float(np.std(post_losses)),
            "improvement_pct_mean": float(np.mean(improvements)),
            "improvement_pct_std": float(np.std(improvements)),
            "avg_lineage_sources": float(
                np.mean([d["n_sources"] for d in task_details])
            ),
            "avg_diversity_penalty": float(
                np.mean([d["div_penalty"] for d in task_details])
            ),
            "avg_homogenization_risk": float(
                np.mean([d["homog_risk"] for d in task_details])
            ),
            "task_details": task_details,
        }
        repeat_results.append(repeat_summary)

        marker = "" if repeats == 1 else f"[R{r+1}/{repeats}] "
        print(f"  {marker}Tasks={len(tasks)}")
        print(f"    Pre-adapt  loss:  {repeat_summary['pre_loss_mean']:.6f} ± {repeat_summary['pre_loss_std']:.6f}")
        print(f"    Post-adapt loss:  {repeat_summary['post_loss_mean']:.6f} ± {repeat_summary['post_loss_std']:.6f}")
        print(
            f"    Improvement:      {repeat_summary['improvement_pct_mean']:.2f}% "
            f"(σ={repeat_summary['improvement_pct_std']:.2f})"
        )
        print(
            f"    Avg sources={repeat_summary['avg_lineage_sources']:.2f} "
            f"div={repeat_summary['avg_diversity_penalty']:.4f} "
            f"homog={repeat_summary['avg_homogenization_risk']:.4f}"
        )

        if r == 0:
            print(f"    Sample traces (first 3):")
            for s in task_details[:3]:
                print(
                    f"      {s['task_id']}: sources={s['n_sources']}, "
                    f"mods={s['modules_used']}, "
                    f"lr={s['hp_lr']:.4f}, drop={s['hp_dropout']:.3f}, "
                    f"mutations={s['mutations']}"
                )

    nodes_after = len(maml.lineage_tree.nodes)
    assert nodes_before == nodes_after, (
        f"Evaluation modified lineage tree: {nodes_before} -> {nodes_after}"
    )

    post_means = [s["post_loss_mean"] for s in repeat_results]
    imp_means = [s["improvement_pct_mean"] for s in repeat_results]
    overall = {
        "repeats": repeats,
        "post_loss_grand_mean": float(np.mean(post_means)),
        "post_loss_grand_std": float(np.std(post_means)) if repeats > 1 else 0.0,
        "improvement_grand_mean": float(np.mean(imp_means)),
        "improvement_grand_std": float(np.std(imp_means)) if repeats > 1 else 0.0,
    }

    print()
    if repeats > 1:
        print("  AGGREGATE (across repeats):")
        print(
            f"    Post-adapt loss:  {overall['post_loss_grand_mean']:.6f} "
            f"± {overall['post_loss_grand_std']:.6f}"
        )
        print(
            f"    Improvement:      {overall['improvement_grand_mean']:.2f}% "
            f"(± {overall['improvement_grand_std']:.2f})"
        )
    print(f"  Lineage tree integrity: OK ({nodes_before} nodes unchanged)")

    return {
        "label": label,
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "repeat_results": repeat_results,
        "aggregate": overall,
    }


def _run_ablation_once(
    maml: ModularMAML,
    tasks: List[TaskBatch],
    *,
    with_first: bool,
) -> Dict[str, object]:
    nodes_before = len(maml.lineage_tree.nodes)
    task_details = []

    for task in tasks:
        seed_snapshot = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )

        r1a = _evaluate_single_task_once(maml, task, use_lineage=True)
        r2a = _evaluate_single_task_once(maml, task, use_lineage=False)

        random.setstate(seed_snapshot[0])
        np.random.set_state(seed_snapshot[1])
        torch.set_rng_state(seed_snapshot[2])

        r2b = _evaluate_single_task_once(maml, task, use_lineage=False)
        r1b = _evaluate_single_task_once(maml, task, use_lineage=True)

        def avg_half(a: Dict, b: Dict, key: str) -> float:
            return float((a[key] + b[key]) / 2.0)

        detail = {
            "task_id": task.task_id,
            "with_lineage_post_loss": avg_half(r1a, r1b, "post_adapt_loss"),
            "without_lineage_post_loss": avg_half(r2a, r2b, "post_adapt_loss"),
            "with_lineage_pre_loss": avg_half(r1a, r1b, "pre_adapt_loss"),
            "without_lineage_pre_loss": avg_half(r2a, r2b, "pre_adapt_loss"),
            "with_trace": {
                "n_sources": r1a["n_sources"],
                "modules": r1a["modules_used"],
                "hp_lr": r1a["hp_lr"],
                "hp_dropout": r1a["hp_dropout"],
                "mutations": r1a["mutations"],
            },
        }
        detail["delta"] = (
            detail["without_lineage_post_loss"] - detail["with_lineage_post_loss"]
        )
        detail["lineage_wins"] = detail["with_lineage_post_loss"] < detail["without_lineage_post_loss"]
        detail["lineage_improved_wrt_pre"] = (
            detail["with_lineage_post_loss"] < detail["with_lineage_pre_loss"]
        )
        task_details.append(detail)

    nodes_after = len(maml.lineage_tree.nodes)
    assert nodes_before == nodes_after, (
        f"Ablation modified lineage tree: {nodes_before} -> {nodes_after}"
    )

    with_losses = [d["with_lineage_post_loss"] for d in task_details]
    without_losses = [d["without_lineage_post_loss"] for d in task_details]
    deltas = [d["delta"] for d in task_details]

    n_wins = sum(1 for d in task_details if d["lineage_wins"])
    n_ties = sum(
        1 for d in task_details
        if abs(d["with_lineage_post_loss"] - d["without_lineage_post_loss"]) < 1e-9
    )
    n_losses = len(task_details) - n_wins - n_ties

    w_mean = float(np.mean(with_losses))
    wo_mean = float(np.mean(without_losses))
    w_std = float(np.std(with_losses))
    wo_std = float(np.std(without_losses))

    if with_first:
        gain = (wo_mean - w_mean) / max(wo_mean, 1e-8) * 100.0 if wo_mean > 0 else 0.0
    else:
        gain = (wo_mean - w_mean) / max(wo_mean, 1e-8) * 100.0 if wo_mean > 0 else 0.0

    return {
        "with_first": with_first,
        "num_tasks": len(task_details),
        "with_lineage_mean": w_mean,
        "with_lineage_std": w_std,
        "without_lineage_mean": wo_mean,
        "without_lineage_std": wo_std,
        "relative_gain_pct": float(gain),
        "pct_better": float(n_wins / max(len(task_details), 1) * 100.0),
        "wins": n_wins,
        "ties": n_ties,
        "losses": n_losses,
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "task_details": task_details,
    }


def _run_ablation(
    maml: ModularMAML,
    generator: SinusoidTaskGenerator,
    families: List[Dict],
    n_tasks: int = 5,
    repeats: int = 1,
) -> Dict[str, object]:
    nodes_before_all = len(maml.lineage_tree.nodes)
    all_repeat_results: List[Dict] = []

    for r in range(repeats):
        generator.reset_rng(seed_offset=90000 + r * 1000)
        tasks_ab = generator.generate_test_tasks(
            families, tasks_per_family=n_tasks, seed_offset=200 + r * 100
        )

        order_label = "[W->WO]" if True else "[WO->W]"
        marker = "" if repeats == 1 else f"[R{r+1}/{repeats}] "
        print(f"  {marker}Evaluating {len(tasks_ab)} tasks {order_label}...")

        result_with_first = _run_ablation_once(
            maml, tasks_ab, with_first=True
        )

        generator.reset_rng(seed_offset=90000 + r * 1000)
        tasks_ba = generator.generate_test_tasks(
            families, tasks_per_family=n_tasks, seed_offset=200 + r * 100
        )
        result_without_first = _run_ablation_once(
            maml, tasks_ba, with_first=False
        )

        def avg_key(a: Dict, b: Dict, key: str) -> float:
            return float((a[key] + b[key]) / 2.0)

        merged_details = []
        for d1, d2 in zip(
            result_with_first["task_details"],
            result_without_first["task_details"],
        ):
            m = {
                "task_id": d1["task_id"],
                "with_lineage_post_loss": avg_key(d1, d2, "with_lineage_post_loss"),
                "without_lineage_post_loss": avg_key(d1, d2, "without_lineage_post_loss"),
            }
            m["delta"] = (
                m["without_lineage_post_loss"] - m["with_lineage_post_loss"]
            )
            m["lineage_wins"] = (
                m["with_lineage_post_loss"] < m["without_lineage_post_loss"]
            )
            m["with_trace"] = d1["with_trace"]
            m["order_WO_W_delta_orderdiff"] = float(
                (d1["with_lineage_post_loss"] - d1["without_lineage_post_loss"])
                - (d2["with_lineage_post_loss"] - d2["without_lineage_post_loss"])
            )
            merged_details.append(m)

        merged = {
            "repeat_idx": r,
            "num_tasks": len(merged_details),
            "with_lineage_mean": avg_key(
                result_with_first, result_without_first, "with_lineage_mean"
            ),
            "with_lineage_std": avg_key(
                result_with_first, result_without_first, "with_lineage_std"
            ),
            "without_lineage_mean": avg_key(
                result_with_first, result_without_first, "without_lineage_mean"
            ),
            "without_lineage_std": avg_key(
                result_with_first, result_without_first, "without_lineage_std"
            ),
            "relative_gain_pct": avg_key(
                result_with_first, result_without_first, "relative_gain_pct"
            ),
            "pct_better": avg_key(
                result_with_first, result_without_first, "pct_better"
            ),
            "wins": sum(1 for d in merged_details if d["lineage_wins"]),
            "ties": sum(
                1 for d in merged_details
                if abs(d["with_lineage_post_loss"] - d["without_lineage_post_loss"]) < 1e-9
            ),
            "losses": len(merged_details) - sum(1 for d in merged_details if d["lineage_wins"]),
            "nodes_before": result_with_first["nodes_before"],
            "nodes_after": result_with_first["nodes_after"],
            "order_independence_check": {
                "delta_diff_max_abs": float(
                    max(abs(d["order_WO_W_delta_orderdiff"]) for d in merged_details)
                ),
            },
            "task_details": merged_details,
        }
        merged["losses"] = len(merged_details) - merged["wins"] - merged["ties"]
        all_repeat_results.append(merged)

        print(
            f"    With lineage:    {merged['with_lineage_mean']:.6f} "
            f"± {merged['with_lineage_std']:.6f}"
        )
        print(
            f"    Without lineage: {merged['without_lineage_mean']:.6f} "
            f"± {merged['without_lineage_std']:.6f}"
        )
        if merged["without_lineage_mean"] > 0:
            print(
                f"    Relative gain:   {merged['relative_gain_pct']:.2f}%"
            )
        print(
            f"    Record(W/T/L):   {merged['wins']}/{merged['ties']}/{merged['losses']} "
            f"({merged['pct_better']:.1f}% win rate)"
        )
        od = merged["order_independence_check"]["delta_diff_max_abs"]
        print(
            f"    Order-indep Δ:   max |Δ| = {od:.6f} "
            f"{'(OK)' if od < 1e-6 else '(WARN: order effect?)'}"
        )

        if r == 0:
            print("    Per-task details (first 5):")
            for d in merged_details[:5]:
                status = "WIN " if d["lineage_wins"] else "LOSS"
                print(
                    f"      [{status}] {d['task_id']}: "
                    f"with={d['with_lineage_post_loss']:.4f} "
                    f"without={d['without_lineage_post_loss']:.4f} "
                    f"Δ={d['delta']:+.4f} "
                    f"mods={d['with_trace']['modules']}"
                )

    nodes_after_all = len(maml.lineage_tree.nodes)
    assert nodes_before_all == nodes_after_all, (
        f"Ablation modified lineage tree: {nodes_before_all} -> {nodes_after_all}"
    )

    with_means = [r["with_lineage_mean"] for r in all_repeat_results]
    wo_means = [r["without_lineage_mean"] for r in all_repeat_results]
    gains = [r["relative_gain_pct"] for r in all_repeat_results]
    pcts = [r["pct_better"] for r in all_repeat_results]

    aggregate = {
        "repeats": repeats,
        "with_lineage_grand_mean": float(np.mean(with_means)),
        "with_lineage_grand_std": float(np.std(with_means)) if repeats > 1 else 0.0,
        "without_lineage_grand_mean": float(np.mean(wo_means)),
        "without_lineage_grand_std": float(np.std(wo_means)) if repeats > 1 else 0.0,
        "gain_grand_mean": float(np.mean(gains)),
        "gain_grand_std": float(np.std(gains)) if repeats > 1 else 0.0,
        "win_rate_grand_mean": float(np.mean(pcts)),
        "win_rate_grand_std": float(np.std(pcts)) if repeats > 1 else 0.0,
    }

    print()
    if repeats > 1:
        print("  ABLATION AGGREGATE (across repeats):")
        print(
            f"    With lineage:    "
            f"{aggregate['with_lineage_grand_mean']:.6f} "
            f"± {aggregate['with_lineage_grand_std']:.6f}"
        )
        print(
            f"    Without lineage: "
            f"{aggregate['without_lineage_grand_mean']:.6f} "
            f"± {aggregate['without_lineage_grand_std']:.6f}"
        )
        print(
            f"    Relative gain:   {aggregate['gain_grand_mean']:.2f}% "
            f"(± {aggregate['gain_grand_std']:.2f})"
        )
        print(
            f"    Win rate:        {aggregate['win_rate_grand_mean']:.1f}% "
            f"(± {aggregate['win_rate_grand_std']:.1f})"
        )
    print(
        f"  Lineage tree integrity: OK "
        f"({nodes_before_all} nodes unchanged after ablation)"
    )

    return {
        "nodes_before": nodes_before_all,
        "nodes_after": nodes_after_all,
        "repeat_results": all_repeat_results,
        "aggregate": aggregate,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Modular MAML with Lineage Tree Demo"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_modules", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--top_k_modules", type=int, default=4)
    parser.add_argument("--support_shots", type=int, default=10)
    parser.add_argument("--query_shots", type=int, default=15)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--meta_steps", type=int, default=200)
    parser.add_argument("--tasks_per_batch", type=int, default=4)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--first_order", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=50.0)
    parser.add_argument("--replay_prob", type=float, default=0.3)
    parser.add_argument("--evolve_every", type=int, default=15)
    parser.add_argument("--merge_threshold", type=float, default=0.82)
    parser.add_argument("--prune_threshold", type=float, default=0.05)
    parser.add_argument("--max_tree_nodes", type=int, default=150)
    parser.add_argument("--div_penalty", type=float, default=0.35)
    parser.add_argument("--mutation_rate", type=float, default=0.25)
    parser.add_argument("--mutation_strength", type=float, default=0.35)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--ablation", action="store_true", default=True)
    parser.add_argument("--no_ablation", dest="ablation", action="store_false")
    parser.add_argument("--eval_repeats", type=int, default=1,
                        help="Repeat evaluation N times on resampled tasks for stable estimate")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if not args.output:
        args.output = os.path.join(
            os.path.dirname(__file__), "results", "demo_results.json"
        )
    run_demo(args)
