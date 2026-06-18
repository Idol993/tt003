from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from lineage_tree import LineageTree, TaskNode
from modular_network import ActivationPattern, HyperParams


@dataclass
class InheritanceResult:
    hyper_params: HyperParams
    pattern: ActivationPattern
    source_ids: List[str]
    source_weights: Dict[str, float]
    diversity_penalty: float
    homogenization_risk: float
    mutations_applied: List[str] = field(default_factory=list)


class GeneticInheritor:
    def __init__(
        self,
        lineage_tree: LineageTree,
        diversity_penalty_weight: float = 0.3,
        crossover_alpha_range: Tuple[float, float] = (0.3, 0.7),
        mutation_rate: float = 0.25,
        mutation_strength: float = 0.35,
        homogenization_threshold: float = 0.92,
    ):
        self.lineage_tree = lineage_tree
        self.diversity_penalty_weight = diversity_penalty_weight
        self.crossover_alpha_range = crossover_alpha_range
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.homogenization_threshold = homogenization_threshold

    def _collect_inheritance_sources(
        self,
        nearest_ids: List[Tuple[str, float]],
        pattern: ActivationPattern,
    ) -> Tuple[List[str], Dict[str, float]]:
        if not nearest_ids:
            return [], {}

        sources = []
        raw_weights = {}

        for nid, affinity in nearest_ids:
            if nid not in self.lineage_tree.nodes:
                continue
            node = self.lineage_tree.nodes[nid]
            sources.append(nid)
            freq_factor = 1.0 + 0.1 * np.log1p(node.visit_count)
            depth_decay = 1.0 / (1.0 + node.depth * 0.08)
            perf_factor = 0.5 + 0.5 * min(node.performance, 1.0)
            raw_weights[nid] = affinity * freq_factor * depth_decay * perf_factor

        if not raw_weights:
            return [], {}

        total = sum(raw_weights.values())
        normalized = {k: v / total for k, v in raw_weights.items()}

        selected = []
        cum_weight = 0.0
        for nid in sorted(normalized, key=normalized.get, reverse=True):
            if cum_weight >= 0.9 and len(selected) >= 2:
                break
            selected.append(nid)
            cum_weight += normalized[nid]

        final_weights = {nid: normalized[nid] for nid in selected}
        total_final = sum(final_weights.values())
        final_weights = {k: v / total_final for k, v in final_weights.items()}
        return selected, final_weights

    def _compute_homogenization_risk(
        self,
        sources: List[str],
        new_hp: HyperParams,
        new_pattern: ActivationPattern,
    ) -> float:
        if not sources:
            return 0.0

        all_module_ids = self.lineage_tree.all_module_ids
        risks = []

        sibling_pools: Dict[str, List[str]] = {}
        for sid in sources:
            siblings = self.lineage_tree.get_siblings(sid)
            sibling_pools[sid] = siblings

            path = self.lineage_tree.find_path_to_root(sid)
            for ancestor_id in path[:3]:
                if ancestor_id not in self.lineage_tree.nodes:
                    continue
                ancestor = self.lineage_tree.nodes[ancestor_id]
                sim_hp = self._hp_similarity(new_hp, ancestor.hyper_params)
                sim_pattern = new_pattern.jaccard_similarity(ancestor.pattern)
                combined = 0.5 * sim_hp + 0.5 * sim_pattern
                risks.append(combined)

            for sib_id in siblings:
                if sib_id not in self.lineage_tree.nodes:
                    continue
                sib = self.lineage_tree.nodes[sib_id]
                sim_hp = self._hp_similarity(new_hp, sib.hyper_params)
                sim_pattern = new_pattern.jaccard_similarity(sib.pattern)
                combined = 0.6 * sim_hp + 0.4 * sim_pattern
                risks.append(combined)

        if not risks:
            return 0.0
        avg_risk = float(np.mean(risks))
        return np.clip(avg_risk, 0.0, 1.0)

    def _hp_similarity(self, hp_a: HyperParams, hp_b: HyperParams) -> float:
        vec_a = hp_a.to_tensor().numpy()
        vec_b = hp_b.to_tensor().numpy()
        dist = np.linalg.norm(vec_a - vec_b)
        return float(np.exp(-dist * 4.0))

    def _diversity_penalty(
        self,
        homogenization_risk: float,
        new_pattern: ActivationPattern,
        sources: List[str],
    ) -> float:
        base_penalty = 0.0

        if homogenization_risk >= self.homogenization_threshold:
            excess = (homogenization_risk - self.homogenization_threshold) / (
                1.0 - self.homogenization_threshold + 1e-8
            )
            base_penalty += self.diversity_penalty_weight * excess

        active_modules = new_pattern.active_modules
        module_counts: Dict[str, int] = {}
        for sid in sources:
            if sid not in self.lineage_tree.nodes:
                continue
            for m in self.lineage_tree.nodes[sid].pattern.active_modules:
                module_counts[m] = module_counts.get(m, 0) + 1

        overused_penalty = 0.0
        for m in active_modules:
            count = module_counts.get(m, 0)
            if count >= len(sources) * 0.8 and len(sources) >= 2:
                overused_penalty += 0.05
        base_penalty += min(overused_penalty, 0.2)

        return base_penalty

    def _apply_diversity_correction(
        self,
        hp: HyperParams,
        pattern: ActivationPattern,
        penalty: float,
        sources: List[str],
    ) -> Tuple[HyperParams, ActivationPattern, List[str]]:
        mutations = []
        new_hp = hp
        new_pattern = pattern

        if penalty <= 0.0:
            return new_hp, new_pattern, mutations

        correction_strength = min(penalty * 2.0, 1.0)

        boosted_mutation_rate = self.mutation_rate + correction_strength * 0.4
        boosted_mutation_strength = self.mutation_strength + correction_strength * 0.3
        new_hp = hp.mutate(
            mutation_rate=boosted_mutation_rate,
            mutation_strength=boosted_mutation_strength,
        )
        mutations.append("hyperparameter_diversity_boost")

        source_modules: Dict[str, int] = {}
        for sid in sources:
            if sid not in self.lineage_tree.nodes:
                continue
            for m in self.lineage_tree.nodes[sid].pattern.active_modules:
                source_modules[m] = source_modules.get(m, 0) + 1

        all_modules = self.lineage_tree.all_module_ids
        underused = [
            m
            for m in all_modules
            if source_modules.get(m, 0) < max(1, len(sources) // 3)
        ]

        if underused and penalty > 0.15:
            current_modules = set(new_pattern.active_modules)
            num_swap = min(1, len(underused), max(1, len(current_modules) // 3))

            overused = [
                m
                for m in current_modules
                if source_modules.get(m, 0) >= len(sources) * 0.7
            ]
            to_remove = np.random.choice(
                overused or list(current_modules),
                size=num_swap,
                replace=False,
            )
            to_add = np.random.choice(underused, size=num_swap, replace=False)

            new_modules = (current_modules - set(to_remove)) | set(to_add)
            if new_modules:
                new_weights = {}
                for m in new_modules:
                    if m in new_pattern.weights:
                        new_weights[m] = new_pattern.weights[m] * 0.8
                    else:
                        new_weights[m] = 0.5 / len(to_add)
                total_w = sum(new_weights.values())
                new_weights = {k: v / total_w for k, v in new_weights.items()}
                new_pattern = ActivationPattern(new_modules, new_weights)
                mutations.append("module_swap_diversity")

        return new_hp, new_pattern, mutations

    def _recombine_patterns(
        self,
        sources: List[str],
        weights: Dict[str, float],
        base_pattern: Optional[ActivationPattern] = None,
    ) -> ActivationPattern:
        module_scores: Dict[str, float] = {}

        for sid in sources:
            if sid not in self.lineage_tree.nodes:
                continue
            node = self.lineage_tree.nodes[sid]
            w = weights.get(sid, 0.0)
            for m in node.pattern.active_modules:
                contrib = w * node.pattern.weights.get(m, 1.0)
                causal_bonus = 0.1 * w * node.causal_effects.get(m, 0.0)
                module_scores[m] = module_scores.get(m, 0.0) + contrib + causal_bonus

        if base_pattern is not None:
            for m in base_pattern.active_modules:
                module_scores[m] = module_scores.get(m, 0.0) + 0.3 * base_pattern.weights.get(m, 1.0)

        if not module_scores:
            if base_pattern is not None:
                return base_pattern
            default_modules = set(self.lineage_tree.all_module_ids[:2])
            return ActivationPattern(default_modules)

        threshold = 0.15 * max(module_scores.values())
        selected = {m: s for m, s in module_scores.items() if s >= threshold}

        if not selected:
            top = max(module_scores, key=module_scores.get)
            selected = {top: module_scores[top]}

        total = sum(selected.values())
        normalized = {k: v / total for k, v in selected.items()}
        return ActivationPattern(set(selected.keys()), normalized)

    def _recombine_hyperparams(
        self,
        sources: List[str],
        weights: Dict[str, float],
    ) -> HyperParams:
        if not sources:
            return HyperParams()

        lr_list, mom_list, drop_list, wd_list, hd_list = [], [], [], [], []
        w_list = []

        for sid in sources:
            if sid not in self.lineage_tree.nodes:
                continue
            node = self.lineage_tree.nodes[sid]
            hp = node.hyper_params
            w = weights.get(sid, 0.0)
            lr_list.append(hp.learning_rate)
            mom_list.append(hp.momentum)
            drop_list.append(hp.dropout_rate)
            wd_list.append(hp.weight_decay)
            hd_list.append(hp.hidden_dim)
            w_list.append(w)

        if not w_list:
            return HyperParams()

        total_w = sum(w_list)
        if total_w <= 0:
            return HyperParams()
        w_norm = [w / total_w for w in w_list]

        lr = float(np.exp(sum(w * np.log(lr) for w, lr in zip(w_norm, lr_list))))
        mom = float(sum(w * m for w, m in zip(w_norm, mom_list)))
        drop = float(sum(w * d for w, d in zip(w_norm, drop_list)))
        wd = float(np.exp(sum(w * np.log(wd) for w, wd in zip(w_norm, wd_list))))
        hd = int(round(sum(w * h for w, h in zip(w_norm, hd_list))))

        if len(sources) >= 2:
            alpha = np.random.uniform(*self.crossover_alpha_range)
            idx_a, idx_b = np.argsort(w_list)[-2:]
            hp_a = self.lineage_tree.nodes[sources[idx_a]].hyper_params
            hp_b = self.lineage_tree.nodes[sources[idx_b]].hyper_params
            cross = HyperParams.crossover(hp_a, hp_b, alpha=alpha)

            beta = 0.6
            lr = beta * cross.learning_rate + (1 - beta) * lr
            mom = beta * cross.momentum + (1 - beta) * mom
            drop = beta * cross.dropout_rate + (1 - beta) * drop
            wd = beta * cross.weight_decay + (1 - beta) * wd
            hd = int(round(beta * cross.hidden_dim + (1 - beta) * hd))

        return HyperParams(
            learning_rate=float(np.clip(lr, 1e-5, 1.0)),
            momentum=float(np.clip(mom, 0.0, 0.999)),
            dropout_rate=float(np.clip(drop, 0.0, 0.8)),
            weight_decay=float(np.clip(wd, 1e-6, 1e-2)),
            hidden_dim=int(np.clip(hd, 16, 512)),
        )

    def inherit(
        self,
        task_pattern: ActivationPattern,
        task_causal_effects: Optional[Dict[str, float]] = None,
        candidate_hp: Optional[HyperParams] = None,
        num_sources: int = 3,
    ) -> InheritanceResult:
        nearest = self.lineage_tree.find_nearest_ancestor(
            pattern=task_pattern,
            causal_effects=task_causal_effects,
            candidate_hp=candidate_hp,
            top_k=num_sources + 2,
        )

        source_ids, source_weights = self._collect_inheritance_sources(
            nearest, task_pattern
        )

        if not source_ids:
            default_hp = HyperParams().mutate(mutation_rate=0.5, mutation_strength=0.5)
            return InheritanceResult(
                hyper_params=default_hp,
                pattern=task_pattern,
                source_ids=[],
                source_weights={},
                diversity_penalty=0.0,
                homogenization_risk=0.0,
                mutations_applied=["default_random_init"],
            )

        recombined_pattern = self._recombine_patterns(
            source_ids, source_weights, base_pattern=task_pattern
        )

        final_pattern = ActivationPattern(
            recombined_pattern.active_modules | task_pattern.active_modules,
            recombined_pattern.weights,
        )
        for m, w in task_pattern.weights.items():
            final_pattern.weights[m] = 0.5 * final_pattern.weights.get(m, 0.0) + 0.5 * w
        total = sum(final_pattern.weights.values())
        if total > 0:
            final_pattern.weights = {k: v / total for k, v in final_pattern.weights.items()}

        base_hp = self._recombine_hyperparams(source_ids, source_weights)

        homog_risk = self._compute_homogenization_risk(
            source_ids, base_hp, final_pattern
        )

        div_penalty = self._diversity_penalty(homog_risk, final_pattern, source_ids)

        corrected_hp, corrected_pattern, mutations = self._apply_diversity_correction(
            base_hp, final_pattern, div_penalty, source_ids
        )

        if np.random.random() < self.mutation_rate:
            corrected_hp = corrected_hp.mutate(
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength,
            )
            mutations.append("base_mutation")

        return InheritanceResult(
            hyper_params=corrected_hp,
            pattern=corrected_pattern,
            source_ids=source_ids,
            source_weights=source_weights,
            diversity_penalty=div_penalty,
            homogenization_risk=homog_risk,
            mutations_applied=mutations,
        )
