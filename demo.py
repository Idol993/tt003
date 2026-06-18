from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple

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
        self.rng = np.random.RandomState(seed)

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
    ) -> List[TaskBatch]:
        tasks = []
        for fam in families:
            for t in range(tasks_per_family):
                tasks.append(self.sample_task_from_family(fam, 100 + t))
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


def run_demo(args: argparse.Namespace):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"=" * 70)
    print(f"Modular MAML with Lineage Tree - Demo")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"Num modules: {args.num_modules}")
    print(f"Inner steps: {args.inner_steps}")
    print(f"Meta steps: {args.meta_steps}")
    print(f"Top-k gating: {args.top_k_modules}")
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
        meta_optimizer, T_max=args.meta_steps, eta_min=1e-5
    )

    train_families = [generator.sample_task_family(i) for i in range(8)]
    test_families_seen = train_families[:4]
    test_families_novel = generator.generate_novel_families(3, start_id=900)

    meta_loss_history: List[float] = []
    perf_history: List[float] = []
    tree_size_history: List[int] = []

    start_time = time.time()

    for step in range(1, args.meta_steps + 1):
        selected_families = random.sample(train_families, min(args.tasks_per_batch, len(train_families)))
        meta_batch = []
        for fam in selected_families:
            for _ in range(2):
                meta_batch.append(generator.sample_task_from_family(fam, step * 10 + random.randint(0, 99)))

        log = maml.meta_update(meta_batch, meta_optimizer)
        scheduler.step()

        meta_loss_history.append(log.meta_loss)
        perf_history.append(log.avg_query_perf)
        tree_size_history.append(log.tree_stats.get("num_nodes", 0))

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[Step {step:4d}/{args.meta_steps}] "
                f"MetaLoss={log.meta_loss:.6f} "
                f"AdptLoss={log.avg_adapt_loss:.6f} "
                f"Perf={log.avg_query_perf:.4f} "
                f"Tree={log.tree_stats.get('num_nodes', 0):3d}nodes "
                f"Roots={log.tree_stats.get('num_roots', 0):2d} "
                f"BestPerf={log.tree_stats.get('best_perf', 0):.4f} "
                f"({elapsed:.1f}s)"
            )
            if log.evolutions:
                n_merged = sum(len(m[1]) for m in log.evolutions.get("merges", []))
                n_pruned = len(log.evolutions.get("pruned", []))
                if n_merged or n_pruned:
                    print(f"  └─ Evolution: merged={n_merged}, pruned={n_pruned}")

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
    print("=" * 70)
    print("EVALUATION - Seen Families:")
    print("=" * 70)
    _evaluate_families(maml, generator, test_families_seen, "seen", n_tasks=3)

    print()
    print("=" * 70)
    print("EVALUATION - Novel (Unseen) Families:")
    print("=" * 70)
    _evaluate_families(maml, generator, test_families_novel, "novel", n_tasks=3)

    if args.ablation:
        print()
        print("=" * 70)
        print("ABLATION STUDY: Comparing w/ and w/o Lineage Initialization")
        print("=" * 70)
        _run_ablation(maml, generator, test_families_novel, n_tasks=5)

    results = {
        "meta_loss_history": meta_loss_history,
        "perf_history": perf_history,
        "tree_size_history": tree_size_history,
        "final_stats": stats,
        "config": vars(args),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
        print(f"\nResults saved to {args.output}")

    return results


def _evaluate_families(
    maml: ModularMAML,
    generator: SinusoidTaskGenerator,
    families: List[Dict],
    label: str,
    n_tasks: int = 3,
):
    tasks = generator.generate_test_tasks(families, tasks_per_family=n_tasks)

    pre_adapt_losses = []
    post_adapt_losses = []
    lineage_info_summary = []

    for task in tasks:
        x_q = task.x_query.to(maml.device)
        y_q = task.y_query.to(maml.device)
        params = maml._make_functional_params()
        pattern, _ = maml._functional_forward_gating(x_q, params)
        pred = maml._functional_forward(x_q, params, pattern, dropout_rate=0.0)
        pre_loss = maml.criterion(pred, y_q).item()
        pre_adapt_losses.append(pre_loss)

        result = maml.fast_adapt(task, use_lineage=True)
        post_adapt_losses.append(result["query_loss"])

        lin = result["lineage"]
        lineage_info_summary.append({
            "task_id": task.task_id,
            "n_sources": len(lin.source_ids) if lin else 0,
            "div_penalty": lin.diversity_penalty if lin else 0.0,
            "homog_risk": lin.homogenization_risk if lin else 0.0,
            "mutations": lin.mutations_applied if lin else [],
            "modules_used": sorted(result["pattern"].active_modules),
            "hp_lr": result["hyper_params"].learning_rate,
            "hp_dropout": result["hyper_params"].dropout_rate,
        })

    avg_pre = float(np.mean(pre_adapt_losses))
    avg_post = float(np.mean(post_adapt_losses))
    impr = (avg_pre - avg_post) / max(avg_pre, 1e-8) * 100

    print(f"  [{label.upper()}] Tasks={len(tasks)}")
    print(f"    Pre-adapt  loss:  {avg_pre:.6f} ± {float(np.std(pre_adapt_losses)):.6f}")
    print(f"    Post-adapt loss:  {avg_post:.6f} ± {float(np.std(post_adapt_losses)):.6f}")
    print(f"    Improvement:      {impr:.2f}%")

    avg_sources = float(np.mean([s["n_sources"] for s in lineage_info_summary]))
    avg_div = float(np.mean([s["div_penalty"] for s in lineage_info_summary]))
    avg_homog = float(np.mean([s["homog_risk"] for s in lineage_info_summary]))
    print(f"    Avg lineage sources: {avg_sources:.2f}")
    print(f"    Avg diversity penalty: {avg_div:.4f}")
    print(f"    Avg homogenization risk: {avg_homog:.4f}")

    print(f"    Sample lineage traces (first 3):")
    for s in lineage_info_summary[:3]:
        print(f"      {s['task_id']}: sources={s['n_sources']}, "
              f"mods={s['modules_used']}, "
              f"lr={s['hp_lr']:.4f}, drop={s['hp_dropout']:.3f}, "
              f"mutations={s['mutations']}")


def _run_ablation(
    maml: ModularMAML,
    generator: SinusoidTaskGenerator,
    families: List[Dict],
    n_tasks: int = 5,
):
    tasks = generator.generate_test_tasks(families, tasks_per_family=n_tasks)

    results_with = []
    results_without = []
    for task in tasks:
        result_with = maml.fast_adapt(task, use_lineage=True)
        result_without = maml.fast_adapt(task, use_lineage=False)
        results_with.append(result_with["query_loss"])
        results_without.append(result_without["query_loss"])

    w_mean = float(np.mean(results_with))
    wo_mean = float(np.mean(results_without))
    w_std = float(np.std(results_with))
    wo_std = float(np.std(results_without))

    print(f"  With lineage inheritance:    {w_mean:.6f} ± {w_std:.6f}")
    print(f"  Without lineage (default):   {wo_mean:.6f} ± {wo_std:.6f}")

    if wo_mean > 0:
        gain = (wo_mean - w_mean) / wo_mean * 100
        print(f"  Relative gain from lineage:  {gain:.2f}%")

    pct_better = sum(1 for w, wo in zip(results_with, results_without) if w < wo) / len(results_with) * 100
    print(f"  % tasks where lineage wins:  {pct_better:.1f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Modular MAML with Lineage Tree Demo")
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
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.output:
        args.output = os.path.join(os.path.dirname(__file__), "results", "demo_results.json")
    run_demo(args)
