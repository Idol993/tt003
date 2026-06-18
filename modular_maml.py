from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genetic_inheritance import GeneticInheritor, InheritanceResult
from lineage_tree import LineageTree
from modular_network import ActivationPattern, HyperParams, ModularNetwork


@dataclass
class TaskBatch:
    task_id: str
    x_support: torch.Tensor
    y_support: torch.Tensor
    x_query: torch.Tensor
    y_query: torch.Tensor
    task_family_id: Optional[str] = None


@dataclass
class AdaptationResult:
    adapted_params: Dict[str, torch.Tensor]
    train_losses: List[float]
    final_pattern: ActivationPattern
    final_hp: HyperParams
    causal_effects: Dict[str, float]
    node_id: Optional[str] = None
    lineage_info: Optional[InheritanceResult] = None


@dataclass
class MetaStepLog:
    step: int
    meta_loss: float
    avg_adapt_loss: float
    avg_query_perf: float
    num_tasks: int
    tree_stats: Dict[str, object] = field(default_factory=dict)
    evolutions: Dict[str, object] = field(default_factory=dict)


class ModularMAML:
    def __init__(
        self,
        network: ModularNetwork,
        lineage_tree: LineageTree,
        inheritor: GeneticInheritor,
        inner_steps: int = 5,
        first_order: bool = False,
        ewc_lambda: float = 100.0,
        memory_replay_prob: float = 0.3,
        evolve_every: int = 10,
        top_k_modules: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.lineage_tree = lineage_tree
        self.inheritor = inheritor
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.ewc_lambda = ewc_lambda
        self.memory_replay_prob = memory_replay_prob
        self.evolve_every = evolve_every
        self.top_k_modules = top_k_modules
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network.to(self.device)

        self.meta_step_counter = 0
        self.ewc_fisher: Dict[str, torch.Tensor] = {}
        self.ewc_params: Dict[str, torch.Tensor] = {}
        self.memory_buffer: List[TaskBatch] = []
        self.memory_max_size = 50
        self.criterion = nn.MSELoss()

        self._init_ewc()

    def _init_ewc(self):
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                self.ewc_fisher[name] = torch.zeros_like(param.data, device=self.device)
                self.ewc_params[name] = param.data.clone().to(self.device)

    def _update_ewc(
        self,
        tasks: List[TaskBatch],
        pattern: Optional[ActivationPattern] = None,
    ):
        temp_fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.data, device=self.device)
            for name, p in self.network.named_parameters()
            if p.requires_grad
        }
        num_samples = 0

        self.network.eval()
        for task in tasks:
            x = task.x_support.to(self.device)
            y = task.y_support.to(self.device)
            self.network.zero_grad()
            if pattern is not None:
                pred = self.network.forward_with_pattern(x, pattern)
            else:
                pred, _, _ = self.network(x, top_k=self.top_k_modules)
            loss = self.criterion(pred, y)
            loss.backward()

            for name, param in self.network.named_parameters():
                if param.requires_grad and param.grad is not None:
                    temp_fisher[name] += param.grad.data ** 2
            num_samples += 1

        if num_samples > 0:
            for name in temp_fisher:
                self.ewc_fisher[name] = (
                    0.8 * self.ewc_fisher[name]
                    + 0.2 * (temp_fisher[name] / max(num_samples, 1))
                )

        for name, param in self.network.named_parameters():
            if param.requires_grad:
                self.ewc_params[name] = param.data.clone().to(self.device)

    def _ewc_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        if self.ewc_lambda <= 0:
            return loss
        for name, param in self.network.named_parameters():
            if name not in params or name not in self.ewc_fisher:
                continue
            fisher = self.ewc_fisher[name]
            old_p = self.ewc_params[name]
            new_p = params[name]
            if fisher.abs().sum() > 0:
                loss += (fisher * (new_p - old_p) ** 2).sum()
        return self.ewc_lambda * 0.5 * loss

    def _make_functional_params(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.clone().to(self.device).requires_grad_(True)
            for name, param in self.network.named_parameters()
            if param.requires_grad
        }

    def _functional_forward(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
        pattern: ActivationPattern,
        dropout_rate: float = 0.0,
    ) -> torch.Tensor:
        module_outputs = []
        total_weight = 0.0

        for mid in pattern.active_modules:
            if mid not in self.network.experts:
                continue
            weight = pattern.weights.get(mid, 1.0)
            expert = self.network.experts[mid]

            h = x
            layer_idx = 0
            for layer in expert.net:
                if isinstance(layer, nn.Linear):
                    w_name = f"experts.{mid}.net.{layer_idx}.weight"
                    b_name = f"experts.{mid}.net.{layer_idx}.bias"
                    w = params.get(w_name, layer.weight)
                    b = params.get(b_name, layer.bias)
                    h = F.linear(h, w, b)
                elif isinstance(layer, nn.LayerNorm):
                    w_name = f"experts.{mid}.net.{layer_idx}.weight"
                    b_name = f"experts.{mid}.net.{layer_idx}.bias"
                    w = params.get(w_name, layer.weight)
                    b = params.get(b_name, layer.bias)
                    h = F.layer_norm(h, layer.normalized_shape, w, b, layer.eps)
                elif isinstance(layer, nn.GELU):
                    h = F.gelu(h)
                layer_idx += 1

            head_w_name = f"experts.{mid}.head.weight"
            head_b_name = f"experts.{mid}.head.bias"
            head_w = params.get(head_w_name, expert.head.weight)
            head_b = params.get(head_b_name, expert.head.bias)
            out = F.linear(h, head_w, head_b)

            if dropout_rate > 0.0 and self.network.training:
                mask = (torch.rand_like(out) > dropout_rate).float() / (1.0 - dropout_rate)
                out = out * mask
            module_outputs.append(out * weight)
            total_weight += weight

        if not module_outputs:
            raise ValueError(f"No modules found for pattern: {pattern}")
        return sum(module_outputs) / max(total_weight, 1e-8)

    def _functional_forward_gating(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[ActivationPattern, torch.Tensor]:
        h = x.mean(dim=0, keepdim=True)
        layer_idx = 0
        for layer in self.network.gating_network:
            if isinstance(layer, nn.Linear):
                w_name = f"gating_network.{layer_idx}.weight"
                b_name = f"gating_network.{layer_idx}.bias"
                w = params.get(w_name, layer.weight)
                b = params.get(b_name, layer.bias)
                h = F.linear(h, w, b)
            elif isinstance(layer, nn.GELU):
                h = F.gelu(h)
            layer_idx += 1

        logits = h / self.network.gating_temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        module_ids = self.network.module_ids()

        if self.top_k_modules is not None and self.top_k_modules < self.network.num_modules:
            topk_vals, topk_idx = torch.topk(probs, self.top_k_modules)
            selected_ids = [module_ids[i.item()] for i in topk_idx]
            normalized = topk_vals / topk_vals.sum()
            weights = {mid: w.item() for mid, w in zip(selected_ids, normalized)}
            return ActivationPattern(set(selected_ids), weights), probs

        threshold = 1.0 / self.network.num_modules
        selected = [module_ids[i] for i, p in enumerate(probs) if p.item() >= threshold]
        if not selected:
            selected = [module_ids[torch.argmax(probs).item()]]
        weights = {mid: probs[module_ids.index(mid)].item() for mid in selected}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return ActivationPattern(set(selected), weights), probs

    def _estimate_causal_effects(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pattern: ActivationPattern,
        params: Dict[str, torch.Tensor],
        hp: HyperParams,
    ) -> Dict[str, float]:
        effects = {}
        try:
            baseline_out = self._functional_forward(
                x, params, pattern, dropout_rate=0.0
            )
            baseline_loss = self.criterion(baseline_out, y).item()
        except Exception:
            return {m: 1.0 / max(len(pattern.active_modules), 1) for m in pattern.active_modules}

        for mid in pattern.active_modules:
            ablated = pattern.active_modules - {mid}
            if not ablated:
                effects[mid] = 0.5
                continue
            ablated_pattern = ActivationPattern(
                ablated,
                {m: pattern.weights[m] for m in ablated},
            )
            total_w = sum(ablated_pattern.weights.values())
            if total_w > 0:
                ablated_pattern.weights = {
                    k: v / total_w for k, v in ablated_pattern.weights.items()
                }
            try:
                ablated_out = self._functional_forward(
                    x, params, ablated_pattern, dropout_rate=0.0
                )
                ablated_loss = self.criterion(ablated_out, y).item()
                effects[mid] = max(0.0, ablated_loss - baseline_loss)
            except Exception:
                effects[mid] = 0.1

        total = sum(effects.values())
        if total > 0:
            effects = {k: v / total for k, v in effects.items()}
        return effects

    def adapt_to_task(
        self,
        task: TaskBatch,
        use_inheritance: bool = True,
    ) -> AdaptationResult:
        x_s = task.x_support.to(self.device)
        y_s = task.y_support.to(self.device)
        x_q = task.x_query.to(self.device)
        y_q = task.y_query.to(self.device)

        params = self._make_functional_params()

        init_pattern, _ = self._functional_forward_gating(x_s, params)
        init_hp = HyperParams()

        lineage_info: Optional[InheritanceResult] = None
        adapted_pattern = init_pattern
        adapted_hp = init_hp
        parent_node_id: Optional[str] = None

        if use_inheritance and self.lineage_tree.nodes:
            lineage_info = self.inheritor.inherit(
                task_pattern=init_pattern,
                candidate_hp=init_hp,
                num_sources=3,
            )
            adapted_pattern = lineage_info.pattern
            adapted_hp = lineage_info.hyper_params
            if lineage_info.source_ids:
                parent_node_id = lineage_info.source_ids[0]

        lr = adapted_hp.learning_rate
        drop_rate = adapted_hp.dropout_rate
        wd = adapted_hp.weight_decay
        momentum = adapted_hp.momentum

        velocity: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in params.items()
        }

        train_losses: List[float] = []
        for step in range(self.inner_steps):
            pred = self._functional_forward(
                x_s, params, adapted_pattern, dropout_rate=drop_rate
            )
            task_loss = self.criterion(pred, y_s)

            wd_loss = torch.tensor(0.0, device=self.device)
            if wd > 0:
                for name, p in params.items():
                    wd_loss += (p ** 2).sum()
                wd_loss = wd * wd_loss

            ewc = self._ewc_loss(params)
            total_loss = task_loss + wd_loss + ewc

            grads = torch.autograd.grad(
                total_loss,
                list(params.values()),
                create_graph=(not self.first_order and step == self.inner_steps - 1),
                allow_unused=True,
            )

            for (name, p), g in zip(params.items(), grads):
                if g is None:
                    continue
                if momentum > 0:
                    velocity[name] = momentum * velocity[name] + g
                    update = velocity[name]
                else:
                    update = g
                params[name] = p - lr * update

            train_losses.append(float(task_loss.item()))

        causal = self._estimate_causal_effects(
            x_s, y_s, adapted_pattern, params, adapted_hp
        )

        node_id = self.lineage_tree.add_node(
            pattern=adapted_pattern,
            hyper_params=adapted_hp,
            performance=0.0,
            causal_effects=causal,
            parent_id=parent_node_id,
        )

        return AdaptationResult(
            adapted_params=params,
            train_losses=train_losses,
            final_pattern=adapted_pattern,
            final_hp=adapted_hp,
            causal_effects=causal,
            node_id=node_id,
            lineage_info=lineage_info,
        )

    def _evaluate_on_query(
        self,
        x_q: torch.Tensor,
        y_q: torch.Tensor,
        params: Dict[str, torch.Tensor],
        pattern: ActivationPattern,
    ) -> Tuple[torch.Tensor, float]:
        x = x_q.to(self.device)
        y = y_q.to(self.device)
        pred = self._functional_forward(x, params, pattern, dropout_rate=0.0)
        loss = self.criterion(pred, y)
        perf = float(-loss.item())
        return loss, perf

    def meta_update(
        self,
        tasks: List[TaskBatch],
        meta_optimizer: torch.optim.Optimizer,
    ) -> MetaStepLog:
        meta_optimizer.zero_grad()
        total_meta_loss = torch.tensor(0.0, device=self.device)
        total_query_perf = 0.0
        total_adapt_loss = 0.0
        node_ids_to_update: List[Tuple[str, float]] = []

        replay_tasks: List[TaskBatch] = []
        if self.memory_buffer and np.random.random() < self.memory_replay_prob:
            replay_count = min(len(tasks), len(self.memory_buffer))
            replay_tasks = list(
                np.random.choice(
                    self.memory_buffer, size=replay_count, replace=False
                )
            )

        all_tasks = tasks + replay_tasks

        for i, task in enumerate(all_tasks):
            adaptation = self.adapt_to_task(task, use_inheritance=True)
            x_q = task.x_query.to(self.device)
            y_q = task.y_query.to(self.device)

            query_loss, perf = self._evaluate_on_query(
                x_q, y_q, adaptation.adapted_params, adaptation.final_pattern
            )
            total_meta_loss = total_meta_loss + query_loss
            total_query_perf += perf
            total_adapt_loss += adaptation.train_losses[-1] if adaptation.train_losses else 0.0

            if adaptation.node_id:
                node_ids_to_update.append((adaptation.node_id, perf))

            self.memory_buffer.append(task)
            if len(self.memory_buffer) > self.memory_max_size:
                self.memory_buffer = self.memory_buffer[-self.memory_max_size:]

        meta_loss = total_meta_loss / max(len(all_tasks), 1)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        meta_optimizer.step()

        for nid, perf in node_ids_to_update:
            self.lineage_tree.update_node(nid, performance=perf)

        evolutions: Dict[str, object] = {}
        if self.meta_step_counter > 0 and self.meta_step_counter % self.evolve_every == 0:
            evolutions = self.lineage_tree.evolve()
            self._update_ewc(tasks)

        self.meta_step_counter += 1

        avg_perf = total_query_perf / max(len(all_tasks), 1)
        avg_adapt = total_adapt_loss / max(len(all_tasks), 1)

        for nid, perf in node_ids_to_update:
            self.lineage_tree.update_node(nid, performance=max(avg_perf, perf))

        return MetaStepLog(
            step=self.meta_step_counter,
            meta_loss=float(meta_loss.item()),
            avg_adapt_loss=float(avg_adapt),
            avg_query_perf=float(avg_perf),
            num_tasks=len(all_tasks),
            tree_stats=self.lineage_tree.get_statistics(),
            evolutions=evolutions,
        )

    def fast_adapt(
        self,
        task: TaskBatch,
        use_lineage: bool = True,
    ) -> Dict[str, object]:
        adaptation = self.adapt_to_task(task, use_inheritance=use_lineage)
        x_q = task.x_query.to(self.device)
        y_q = task.y_query.to(self.device)
        query_loss, perf = self._evaluate_on_query(
            x_q, y_q, adaptation.adapted_params, adaptation.final_pattern
        )

        return {
            "node_id": adaptation.node_id,
            "pattern": adaptation.final_pattern,
            "hyper_params": adaptation.final_hp,
            "causal_effects": adaptation.causal_effects,
            "query_loss": float(query_loss.item()),
            "query_performance": perf,
            "adapt_losses": adaptation.train_losses,
            "lineage": adaptation.lineage_info,
        }

    def get_state(self) -> Dict[str, object]:
        return {
            "network_state": self.network.state_dict(),
            "ewc_fisher": self.ewc_fisher,
            "ewc_params": self.ewc_params,
            "memory_buffer": self.memory_buffer,
            "meta_step": self.meta_step_counter,
        }
