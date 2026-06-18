from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HyperParams:
    learning_rate: float = 0.01
    momentum: float = 0.9
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    hidden_dim: int = 64

    def mutate(self, mutation_rate: float = 0.2, mutation_strength: float = 0.3) -> "HyperParams":
        new_hp = HyperParams(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            dropout_rate=self.dropout_rate,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
        )
        if np.random.random() < mutation_rate:
            factor = np.exp(np.random.normal(0, mutation_strength))
            new_hp.learning_rate = np.clip(self.learning_rate * factor, 1e-5, 1.0)
        if np.random.random() < mutation_rate:
            delta = np.random.normal(0, mutation_strength * 0.1)
            new_hp.momentum = np.clip(self.momentum + delta, 0.0, 0.999)
        if np.random.random() < mutation_rate:
            delta = np.random.normal(0, mutation_strength * 0.05)
            new_hp.dropout_rate = np.clip(self.dropout_rate + delta, 0.0, 0.8)
        if np.random.random() < mutation_rate:
            factor = np.exp(np.random.normal(0, mutation_strength))
            new_hp.weight_decay = np.clip(self.weight_decay * factor, 1e-6, 1e-2)
        return new_hp

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.learning_rate,
            self.momentum,
            self.dropout_rate,
            self.weight_decay,
            float(self.hidden_dim),
        ])

    @staticmethod
    def crossover(hp_a: "HyperParams", hp_b: "HyperParams", alpha: float = 0.5) -> "HyperParams":
        return HyperParams(
            learning_rate=hp_a.learning_rate * alpha + hp_b.learning_rate * (1 - alpha),
            momentum=hp_a.momentum * alpha + hp_b.momentum * (1 - alpha),
            dropout_rate=hp_a.dropout_rate * alpha + hp_b.dropout_rate * (1 - alpha),
            weight_decay=hp_a.weight_decay * alpha + hp_b.weight_decay * (1 - alpha),
            hidden_dim=int(round(hp_a.hidden_dim * alpha + hp_b.hidden_dim * (1 - alpha))),
        )


class ExpertModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, module_id: Optional[str] = None):
        super().__init__()
        self.module_id = module_id or f"mod_{uuid.uuid4().hex[:8]}"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.head(features)


class ActivationPattern:
    def __init__(self, active_modules: Set[str], weights: Optional[Dict[str, float]] = None):
        self.active_modules: Set[str] = set(active_modules)
        self.weights: Dict[str, float] = weights or {m: 1.0 / len(active_modules) for m in active_modules}

    def jaccard_similarity(self, other: "ActivationPattern") -> float:
        if not self.active_modules and not other.active_modules:
            return 1.0
        intersection = len(self.active_modules & other.active_modules)
        union = len(self.active_modules | other.active_modules)
        return intersection / union if union > 0 else 0.0

    def weighted_overlap(self, other: "ActivationPattern") -> float:
        common = self.active_modules & other.active_modules
        return sum(min(self.weights.get(m, 0), other.weights.get(m, 0)) for m in common)

    def to_vector(self, all_module_ids: List[str]) -> np.ndarray:
        vec = np.zeros(len(all_module_ids), dtype=np.float32)
        for i, mid in enumerate(all_module_ids):
            if mid in self.active_modules:
                vec[i] = self.weights.get(mid, 1.0)
        return vec

    def __repr__(self) -> str:
        return f"ActivationPattern(modules={sorted(self.active_modules)})"


class ModularNetwork(nn.Module):
    def __init__(
        self,
        num_modules: int,
        input_dim: int,
        output_dim: int,
        base_hidden_dim: int = 64,
        gating_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_hidden_dim = base_hidden_dim
        self.gating_temperature = gating_temperature

        self.experts: nn.ModuleDict = nn.ModuleDict()
        for i in range(num_modules):
            mid = f"mod_{i:03d}"
            self.experts[mid] = ExpertModule(input_dim, base_hidden_dim, output_dim, module_id=mid)

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_modules),
        )

    def module_ids(self) -> List[str]:
        return list(self.experts.keys())

    def compute_gating(self, x: torch.Tensor, top_k: Optional[int] = None) -> Tuple[ActivationPattern, torch.Tensor]:
        logits = self.gating_network(x.mean(dim=0, keepdim=True)) / self.gating_temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        if top_k is not None and top_k < self.num_modules:
            topk_vals, topk_idx = torch.topk(probs, top_k)
            selected_ids = [self.module_ids()[i.item()] for i in topk_idx]
            normalized = topk_vals / topk_vals.sum()
            weights = {mid: w.item() for mid, w in zip(selected_ids, normalized)}
            return ActivationPattern(set(selected_ids), weights), probs
        else:
            threshold = 1.0 / self.num_modules
            selected = [self.module_ids()[i] for i, p in enumerate(probs) if p.item() >= threshold]
            if not selected:
                selected = [self.module_ids()[torch.argmax(probs).item()]]
            weights = {mid: probs[self.module_ids().index(mid)].item() for mid in selected}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            return ActivationPattern(set(selected), weights), probs

    def forward_with_pattern(
        self,
        x: torch.Tensor,
        pattern: ActivationPattern,
        dropout_rate: float = 0.0,
    ) -> torch.Tensor:
        outputs = []
        total_weight = 0.0
        for mid in pattern.active_modules:
            weight = pattern.weights.get(mid, 1.0)
            module = self.experts[mid]
            out = module(x)
            if dropout_rate > 0.0 and self.training:
                mask = (torch.rand_like(out) > dropout_rate).float() / (1.0 - dropout_rate)
                out = out * mask
            outputs.append(out * weight)
            total_weight += weight
        if not outputs:
            raise ValueError("No active modules in pattern")
        return sum(outputs) / total_weight if total_weight > 0 else sum(outputs)

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, ActivationPattern, torch.Tensor]:
        pattern, gating_probs = self.compute_gating(x, top_k=top_k)
        output = self.forward_with_pattern(x, pattern)
        return output, pattern, gating_probs

    def causal_contribution(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pattern: ActivationPattern,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        contributions = {}
        baseline_output = self.forward_with_pattern(x, pattern)
        baseline_loss = criterion(baseline_output, y).item()

        for mid in pattern.active_modules:
            ablated_modules = pattern.active_modules - {mid}
            if not ablated_modules:
                continue
            ablated_pattern = ActivationPattern(
                ablated_modules,
                {m: pattern.weights[m] for m in ablated_modules},
            )
            total_w = sum(ablated_pattern.weights.values())
            ablated_pattern.weights = {k: v / total_w for k, v in ablated_pattern.weights.items()}

            ablated_output = self.forward_with_pattern(x, ablated_pattern)
            ablated_loss = criterion(ablated_output, y).item()
            contributions[mid] = max(0.0, ablated_loss - baseline_loss)

        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        return contributions
