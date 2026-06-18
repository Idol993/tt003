from __future__ import annotations

import unittest

import numpy as np
import torch

from genetic_inheritance import GeneticInheritor
from lineage_tree import LineageTree, TaskNode
from modular_network import ActivationPattern, ExpertModule, HyperParams, ModularNetwork


class TestHyperParams(unittest.TestCase):
    def test_default_values(self):
        hp = HyperParams()
        self.assertAlmostEqual(hp.learning_rate, 0.01)
        self.assertAlmostEqual(hp.momentum, 0.9)
        self.assertAlmostEqual(hp.dropout_rate, 0.1)

    def test_mutate_changes_values(self):
        torch.manual_seed(0)
        np.random.seed(0)
        hp = HyperParams()
        mutated = hp.mutate(mutation_rate=1.0, mutation_strength=1.0)
        self.assertNotEqual(mutated.learning_rate, hp.learning_rate)

    def test_crossover(self):
        hp_a = HyperParams(learning_rate=0.1, momentum=0.5, dropout_rate=0.0)
        hp_b = HyperParams(learning_rate=0.001, momentum=0.99, dropout_rate=0.5)
        child = HyperParams.crossover(hp_a, hp_b, alpha=0.5)
        self.assertAlmostEqual(child.learning_rate, 0.0505, places=4)
        self.assertAlmostEqual(child.momentum, 0.745, places=3)
        self.assertAlmostEqual(child.dropout_rate, 0.25)

    def test_clipping(self):
        hp = HyperParams(learning_rate=10.0, dropout_rate=2.0, weight_decay=0.0)
        mutated = hp.mutate(mutation_rate=1.0, mutation_strength=0.0)
        self.assertLessEqual(mutated.learning_rate, 1.0)
        self.assertLessEqual(mutated.dropout_rate, 0.8)
        self.assertGreaterEqual(mutated.learning_rate, 1e-5)


class TestActivationPattern(unittest.TestCase):
    def test_jaccard_identical(self):
        p1 = ActivationPattern({"a", "b", "c"})
        p2 = ActivationPattern({"a", "b", "c"})
        self.assertAlmostEqual(p1.jaccard_similarity(p2), 1.0)

    def test_jaccard_disjoint(self):
        p1 = ActivationPattern({"a", "b"})
        p2 = ActivationPattern({"c", "d"})
        self.assertAlmostEqual(p1.jaccard_similarity(p2), 0.0)

    def test_jaccard_partial(self):
        p1 = ActivationPattern({"a", "b", "c"})
        p2 = ActivationPattern({"b", "c", "d"})
        self.assertAlmostEqual(p1.jaccard_similarity(p2), 0.5)

    def test_jaccard_empty(self):
        p1 = ActivationPattern(set())
        p2 = ActivationPattern(set())
        self.assertAlmostEqual(p1.jaccard_similarity(p2), 1.0)

    def test_to_vector(self):
        p = ActivationPattern({"b", "d"}, {"b": 0.7, "d": 0.3})
        all_ids = ["a", "b", "c", "d"]
        vec = p.to_vector(all_ids)
        self.assertEqual(len(vec), 4)
        self.assertAlmostEqual(vec[0], 0.0)
        self.assertAlmostEqual(vec[1], 0.7)
        self.assertAlmostEqual(vec[2], 0.0)
        self.assertAlmostEqual(vec[3], 0.3)

    def test_weighted_overlap(self):
        p1 = ActivationPattern({"a", "b"}, {"a": 0.6, "b": 0.4})
        p2 = ActivationPattern({"b", "c"}, {"b": 0.8, "c": 0.2})
        overlap = p1.weighted_overlap(p2)
        self.assertAlmostEqual(overlap, 0.4)


class TestModularNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.net = ModularNetwork(num_modules=4, input_dim=2, output_dim=1, base_hidden_dim=16)

    def test_module_count(self):
        self.assertEqual(len(self.net.experts), 4)

    def test_module_ids(self):
        ids = self.net.module_ids()
        self.assertEqual(len(ids), 4)
        for i in range(4):
            self.assertIn(f"mod_{i:03d}", ids)

    def test_forward_shape(self):
        x = torch.randn(8, 2)
        out, pattern, gating = self.net(x, top_k=2)
        self.assertEqual(out.shape, (8, 1))
        self.assertEqual(gating.shape, (4,))
        self.assertGreaterEqual(len(pattern.active_modules), 1)
        self.assertLessEqual(len(pattern.active_modules), 2)

    def test_forward_with_pattern(self):
        x = torch.randn(5, 2)
        pattern = ActivationPattern({"mod_000", "mod_001"})
        out = self.net.forward_with_pattern(x, pattern)
        self.assertEqual(out.shape, (5, 1))

    def test_causal_contribution(self):
        x = torch.randn(6, 2)
        y = torch.randn(6, 1)
        pattern = ActivationPattern({"mod_000", "mod_001", "mod_002"})
        criterion = torch.nn.MSELoss()
        contributions = self.net.causal_contribution(x, y, pattern, criterion)
        self.assertGreaterEqual(len(contributions), 2)
        total = sum(contributions.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestLineageTree(unittest.TestCase):
    def setUp(self):
        self.all_modules = [f"mod_{i:03d}" for i in range(6)]
        self.tree = LineageTree(
            all_module_ids=self.all_modules,
            merge_threshold=0.9,
            prune_threshold=0.01,
            max_nodes=20,
        )

    def test_add_node(self):
        pattern = ActivationPattern({"mod_000", "mod_001"})
        hp = HyperParams()
        nid = self.tree.add_node(pattern, hp, performance=0.5)
        self.assertIn(nid, self.tree.nodes)
        self.assertEqual(self.tree.nodes[nid].performance, 0.5)
        self.assertEqual(len(self.tree.root_ids), 1)

    def test_add_node_with_parent(self):
        pattern = ActivationPattern({"mod_000"})
        hp = HyperParams()
        parent_id = self.tree.add_node(pattern, hp, performance=0.8)
        child_id = self.tree.add_node(
            ActivationPattern({"mod_000", "mod_001"}),
            HyperParams(),
            performance=0.6,
            parent_id=parent_id,
        )
        self.assertEqual(self.tree.nodes[child_id].depth, 1)
        self.assertIn(child_id, self.tree.nodes[parent_id].children_ids)

    def test_find_nearest_ancestor(self):
        for i in range(5):
            mods = {f"mod_{i:03d}", f"mod_{(i + 1) % 6:03d}"}
            self.tree.add_node(
                ActivationPattern(mods),
                HyperParams(learning_rate=0.01 * (i + 1)),
                performance=0.3 + i * 0.1,
            )
        query_pattern = ActivationPattern({"mod_000", "mod_001"})
        nearest = self.tree.find_nearest_ancestor(query_pattern, top_k=3)
        self.assertEqual(len(nearest), 3)
        self.assertGreaterEqual(nearest[0][1], nearest[-1][1])

    def test_node_composite_affinity(self):
        n1 = TaskNode(
            task_id="a",
            pattern=ActivationPattern({"mod_000", "mod_001"}, {"mod_000": 0.5, "mod_001": 0.5}),
            hyper_params=HyperParams(learning_rate=0.01),
            causal_effects={"mod_000": 0.7, "mod_001": 0.3},
        )
        n2 = TaskNode(
            task_id="b",
            pattern=ActivationPattern({"mod_000", "mod_001"}, {"mod_000": 0.6, "mod_001": 0.4}),
            hyper_params=HyperParams(learning_rate=0.011),
            causal_effects={"mod_000": 0.65, "mod_001": 0.35},
        )
        n3 = TaskNode(
            task_id="c",
            pattern=ActivationPattern({"mod_004", "mod_005"}),
            hyper_params=HyperParams(learning_rate=0.5),
            causal_effects={"mod_004": 0.9},
        )
        aff_ab = n1.composite_affinity(n2, self.all_modules)
        aff_ac = n1.composite_affinity(n3, self.all_modules)
        self.assertGreater(aff_ab, aff_ac)

    def test_path_to_root(self):
        r = self.tree.add_node(ActivationPattern({"mod_000"}), HyperParams())
        c = self.tree.add_node(ActivationPattern({"mod_001"}), HyperParams(), parent_id=r)
        gc = self.tree.add_node(ActivationPattern({"mod_002"}), HyperParams(), parent_id=c)
        path = self.tree.find_path_to_root(gc)
        self.assertEqual(path, [gc, c, r])

    def test_cluster_nodes(self):
        for family in range(3):
            for t in range(4):
                mods = {f"mod_{family * 2:03d}", f"mod_{family * 2 + 1:03d}"}
                self.tree.add_node(
                    ActivationPattern(mods),
                    HyperParams(learning_rate=0.001 * (family + 1)),
                    performance=0.5,
                )
        clusters = self.tree.cluster_nodes(num_clusters=3)
        self.assertEqual(len(clusters), 3)

    def test_merge_and_prune(self):
        for i in range(25):
            mods = {f"mod_{i % 6:03d}", f"mod_{(i + 1) % 6:03d}"}
            perf = 0.01 if i < 15 else 0.8
            self.tree.add_node(
                ActivationPattern(mods),
                HyperParams(learning_rate=0.01),
                performance=perf,
            )
        init_count = len(self.tree.nodes)
        merges = self.tree.merge_similar_nodes(force=False)
        pruned = self.tree.prune()
        self.assertLessEqual(len(self.tree.nodes), init_count)


class TestGeneticInheritor(unittest.TestCase):
    def setUp(self):
        self.all_modules = [f"mod_{i:03d}" for i in range(5)]
        self.tree = LineageTree(
            all_module_ids=self.all_modules,
            merge_threshold=0.85,
            max_nodes=50,
        )
        for i in range(10):
            mods = set(np.random.choice(self.all_modules, size=2, replace=False))
            self.tree.add_node(
                ActivationPattern(mods),
                HyperParams(
                    learning_rate=0.005 + 0.02 * np.random.random(),
                    dropout_rate=0.05 + 0.2 * np.random.random(),
                ),
                performance=0.3 + 0.6 * np.random.random(),
                causal_effects={m: np.random.random() for m in mods},
            )
        self.inheritor = GeneticInheritor(
            lineage_tree=self.tree,
            diversity_penalty_weight=0.3,
        )

    def test_inherit_returns_valid_result(self):
        pattern = ActivationPattern({"mod_000", "mod_001"})
        result = self.inheritor.inherit(pattern, num_sources=3)
        self.assertIsInstance(result.hyper_params, HyperParams)
        self.assertIsInstance(result.pattern, ActivationPattern)
        self.assertGreater(len(result.pattern.active_modules), 0)
        self.assertGreaterEqual(result.diversity_penalty, 0.0)
        self.assertGreaterEqual(result.homogenization_risk, 0.0)
        self.assertLessEqual(result.homogenization_risk, 1.0)

    def test_inherit_without_nodes(self):
        empty_tree = LineageTree(all_module_ids=self.all_modules)
        empty_inheritor = GeneticInheritor(lineage_tree=empty_tree)
        pattern = ActivationPattern({"mod_000"})
        result = empty_inheritor.inherit(pattern)
        self.assertEqual(len(result.source_ids), 0)
        self.assertIn("default_random_init", result.mutations_applied)

    def test_hp_similarity(self):
        hp_a = HyperParams(learning_rate=0.01, momentum=0.9, dropout_rate=0.1, weight_decay=1e-4)
        hp_b = HyperParams(learning_rate=0.01, momentum=0.9, dropout_rate=0.1, weight_decay=1e-4)
        hp_c = HyperParams(learning_rate=0.5, momentum=0.0, dropout_rate=0.8, weight_decay=1e-2)
        sim_ab = self.inheritor._hp_similarity(hp_a, hp_b)
        sim_ac = self.inheritor._hp_similarity(hp_a, hp_c)
        self.assertGreater(sim_ab, 0.99)
        self.assertLess(sim_ac, sim_ab)

    def test_diversity_correction(self):
        hp = HyperParams(learning_rate=0.01)
        pattern = ActivationPattern({"mod_000", "mod_001"})
        sources = list(self.tree.nodes.keys())[:3]
        corrected_hp, corrected_pattern, mutations = self.inheritor._apply_diversity_correction(
            hp, pattern, penalty=0.8, sources=sources
        )
        self.assertGreaterEqual(len(mutations), 1)


class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        import torch.nn as nn
        from modular_maml import ModularMAML, TaskBatch

        torch.manual_seed(99)
        np.random.seed(99)

        all_modules = [f"mod_{i:03d}" for i in range(4)]
        network = ModularNetwork(
            num_modules=4, input_dim=1, output_dim=1, base_hidden_dim=16
        )
        tree = LineageTree(all_module_ids=all_modules, max_nodes=20)
        inheritor = GeneticInheritor(lineage_tree=tree)
        maml = ModularMAML(
            network=network,
            lineage_tree=tree,
            inheritor=inheritor,
            inner_steps=3,
            first_order=True,
            ewc_lambda=10.0,
            evolve_every=5,
            top_k_modules=2,
        )
        opt = torch.optim.Adam(network.parameters(), lr=1e-3)

        def make_task():
            amp = 0.5 + 4.5 * np.random.random()
            phase = np.pi * np.random.random()
            freq = 0.8 + 0.4 * np.random.random()
            x_s = np.random.uniform(-5, 5, (8, 1)).astype(np.float32)
            y_s = (amp * np.sin(freq * x_s + phase)).astype(np.float32)
            x_q = np.random.uniform(-5, 5, (10, 1)).astype(np.float32)
            y_q = (amp * np.sin(freq * x_q + phase)).astype(np.float32)
            return TaskBatch(
                task_id=f"t_{np.random.randint(1000)}",
                x_support=torch.from_numpy(x_s),
                y_support=torch.from_numpy(y_s),
                x_query=torch.from_numpy(x_q),
                y_query=torch.from_numpy(y_q),
            )

        for _ in range(10):
            tasks = [make_task() for _ in range(3)]
            log = maml.meta_update(tasks, opt)
            self.assertIsInstance(log.meta_loss, float)

        self.assertGreater(len(tree.nodes), 0)

        test_task = make_task()
        result = maml.fast_adapt(test_task, use_lineage=True)
        self.assertIn("query_performance", result)
        self.assertIsInstance(result["pattern"], ActivationPattern)

    def test_nodes_have_valid_performance_after_training(self):
        from modular_maml import ModularMAML, TaskBatch

        torch.manual_seed(42)
        np.random.seed(42)

        all_modules = [f"mod_{i:03d}" for i in range(4)]
        network = ModularNetwork(
            num_modules=4, input_dim=1, output_dim=1, base_hidden_dim=16
        )
        tree = LineageTree(all_module_ids=all_modules, max_nodes=30)
        inheritor = GeneticInheritor(lineage_tree=tree)
        maml = ModularMAML(
            network=network,
            lineage_tree=tree,
            inheritor=inheritor,
            inner_steps=2,
            first_order=True,
            ewc_lambda=1.0,
            evolve_every=20,
            top_k_modules=2,
        )
        opt = torch.optim.Adam(network.parameters(), lr=1e-3)

        def make_task():
            amp = 0.5 + 4.5 * np.random.random()
            phase = np.pi * np.random.random()
            freq = 0.8 + 0.4 * np.random.random()
            x_s = np.random.uniform(-5, 5, (6, 1)).astype(np.float32)
            y_s = (amp * np.sin(freq * x_s + phase)).astype(np.float32)
            x_q = np.random.uniform(-5, 5, (8, 1)).astype(np.float32)
            y_q = (amp * np.sin(freq * x_q + phase)).astype(np.float32)
            return TaskBatch(
                task_id=f"t_{np.random.randint(10000)}",
                x_support=torch.from_numpy(x_s),
                y_support=torch.from_numpy(y_s),
                x_query=torch.from_numpy(x_q),
                y_query=torch.from_numpy(y_q),
            )

        for _ in range(6):
            tasks = [make_task() for _ in range(2)]
            maml.meta_update(tasks, opt)

        stats = tree.get_statistics()
        self.assertGreater(stats["valid_perf_count"], 0,
                           "After training, some nodes should have valid performance scores")
        self.assertGreater(stats["best_perf"], float("-inf"),
                           "best_perf should not be -inf after training")
        self.assertIsInstance(stats["avg_perf"], float)

        for nid, node in tree.nodes.items():
            if node.performance > float("-inf"):
                self.assertIsInstance(node.performance, float)
                self.assertLess(node.performance, 0,
                                "For regression with MSE, performance (-loss) should be negative")
                break

    def test_fast_adapt_without_adding_to_lineage(self):
        from modular_maml import ModularMAML, TaskBatch

        torch.manual_seed(7)
        np.random.seed(7)

        all_modules = [f"mod_{i:03d}" for i in range(3)]
        network = ModularNetwork(
            num_modules=3, input_dim=1, output_dim=1, base_hidden_dim=16
        )
        tree = LineageTree(all_module_ids=all_modules, max_nodes=30)
        inheritor = GeneticInheritor(lineage_tree=tree)
        maml = ModularMAML(
            network=network,
            lineage_tree=tree,
            inheritor=inheritor,
            inner_steps=2,
            first_order=True,
            ewc_lambda=0.0,
            evolve_every=50,
            top_k_modules=2,
        )
        opt = torch.optim.Adam(network.parameters(), lr=1e-3)

        def make_task():
            amp = 1.0 + 3.0 * np.random.random()
            x_s = np.random.uniform(-5, 5, (6, 1)).astype(np.float32)
            y_s = (amp * np.sin(x_s)).astype(np.float32)
            x_q = np.random.uniform(-5, 5, (8, 1)).astype(np.float32)
            y_q = (amp * np.sin(x_q)).astype(np.float32)
            return TaskBatch(
                task_id=f"eval_{np.random.randint(1000)}",
                x_support=torch.from_numpy(x_s),
                y_support=torch.from_numpy(y_s),
                x_query=torch.from_numpy(x_q),
                y_query=torch.from_numpy(y_q),
            )

        for _ in range(4):
            tasks = [make_task() for _ in range(2)]
            maml.meta_update(tasks, opt)

        nodes_before = len(tree.nodes)
        self.assertGreater(nodes_before, 0)

        for _ in range(5):
            test_task = make_task()
            maml.fast_adapt(test_task, use_lineage=True, add_to_lineage=False)
            maml.fast_adapt(test_task, use_lineage=False, add_to_lineage=False)

        nodes_after = len(tree.nodes)
        self.assertEqual(nodes_before, nodes_after,
                         "fast_adapt with add_to_lineage=False should not change tree size")

    def test_output_json_current_directory(self):
        import json
        import os
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            data = {"test": 123, "nested": {"a": [1, 2, 3]}}
            out_path = "test_output.json"
            with open(out_path, "w") as f:
                json.dump(data, f)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["test"], 123)
            os.remove(out_path)

            nested_path = os.path.join("subdir", "nested.json")
            nested_dir = os.path.dirname(nested_path)
            os.makedirs(nested_dir, exist_ok=True)
            with open(nested_path, "w") as f:
                json.dump(data, f)
            self.assertTrue(os.path.exists(nested_path))
            import shutil
            shutil.rmtree("subdir")
        finally:
            os.chdir(cwd)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_demo_saves_current_directory_output(self):
        import json
        import os
        import sys
        import tempfile

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        tmp_dir = tempfile.mkdtemp()
        cwd = os.getcwd()
        created_files = []
        try:
            os.chdir(tmp_dir)
            import demo
            out_name = "demo_shortrun.json"
            result = demo.run_demo(
                None,
                seed=1,
                num_modules=3,
                hidden_dim=16,
                top_k_modules=2,
                inner_steps=2,
                meta_steps=3,
                tasks_per_batch=2,
                log_every=3,
                evolve_every=50,
                eval_repeats=1,
                merge_threshold=0.95,
                max_tree_nodes=80,
                ablation=False,
                output=out_name,
            )
            created_files.append(os.path.abspath(out_name))
            self.assertIn("_output_path", result)
            self.assertTrue(os.path.exists(result["_output_path"]))
            with open(result["_output_path"], "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("final_stats", loaded)
            self.assertIn("lineage_report", loaded)
            self.assertIn("eval_seen_repeated", loaded)
            self.assertIn("eval_novel_repeated", loaded)
            self.assertIn("meta_loss_history", loaded)
            self.assertEqual(len(loaded["meta_loss_history"]), 3)
            report = loaded["lineage_report"]
            self.assertIn("top_tasks", report)
            self.assertIn("bottom_tasks", report)
            self.assertIn("summary", report)
            self.assertIn("module_usage", report)
            self.assertGreater(report["valid_count"], 0)
            self.assertGreater(report["summary"]["best_perf"], float("-inf"))
        finally:
            os.chdir(cwd)
            for fp in created_files:
                try:
                    os.remove(fp)
                except Exception:
                    pass
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_demo_saves_nested_output_directory(self):
        import json
        import os
        import sys
        import tempfile

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        tmp_dir = tempfile.mkdtemp()
        cwd = os.getcwd()
        created_dirs = []
        try:
            os.chdir(tmp_dir)
            import demo
            out_path = os.path.join("nested", "a", "b", "out.json")
            result = demo.run_demo(
                None,
                seed=2,
                num_modules=3,
                hidden_dim=16,
                top_k_modules=2,
                inner_steps=2,
                meta_steps=3,
                tasks_per_batch=2,
                log_every=3,
                evolve_every=50,
                eval_repeats=1,
                merge_threshold=0.95,
                max_tree_nodes=80,
                ablation=False,
                output=out_path,
            )
            created_dirs.append(os.path.abspath("nested"))
            self.assertIn("_output_path", result)
            self.assertTrue(os.path.exists(result["_output_path"]))
            self.assertTrue(
                result["_output_path"].endswith(os.path.join("nested", "a", "b", "out.json"))
            )
            with open(result["_output_path"], "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("config", loaded)
            self.assertEqual(loaded["config"]["meta_steps"], 3)
        finally:
            os.chdir(cwd)
            for dp in created_dirs:
                import shutil
                shutil.rmtree(dp, ignore_errors=True)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_demo_eval_repeats_produces_aggregate(self):
        import os
        import sys
        import tempfile

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        tmp_dir = tempfile.mkdtemp()
        cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            import demo
            result = demo.run_demo(
                None,
                seed=3,
                num_modules=3,
                hidden_dim=16,
                top_k_modules=2,
                inner_steps=2,
                meta_steps=2,
                tasks_per_batch=2,
                log_every=2,
                evolve_every=50,
                eval_repeats=2,
                merge_threshold=0.95,
                max_tree_nodes=80,
                ablation=True,
                output="",
            )
            seen_eval = result["eval_seen_repeated"]
            self.assertEqual(seen_eval["aggregate"]["repeats"], 2)
            self.assertEqual(len(seen_eval["repeat_results"]), 2)
            self.assertGreaterEqual(
                seen_eval["repeat_results"][0]["num_tasks"],
                2,
            )
            self.assertIsInstance(seen_eval["aggregate"]["post_loss_grand_mean"], float)
            self.assertIsInstance(seen_eval["aggregate"]["post_loss_grand_std"], float)

            novel_eval = result["eval_novel_repeated"]
            self.assertEqual(novel_eval["aggregate"]["repeats"], 2)

            ablation = result["ablation"]
            self.assertIsNotNone(ablation)
            self.assertEqual(ablation["aggregate"]["repeats"], 2)
            self.assertEqual(len(ablation["repeat_results"]), 2)
            first_rep = ablation["repeat_results"][0]
            self.assertIn("order_independence_check", first_rep)
            self.assertIsInstance(
                first_rep["order_independence_check"]["delta_diff_max_abs"],
                float,
            )
            self.assertIn("wins", first_rep)
            self.assertIn("ties", first_rep)
            self.assertIn("losses", first_rep)
            self.assertEqual(
                first_rep["wins"] + first_rep["ties"] + first_rep["losses"],
                first_rep["num_tasks"],
            )
        finally:
            os.chdir(cwd)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_lineage_tree_report_contains_top_bottom_structure(self):
        tree = LineageTree(
            all_module_ids=[f"mod_{i:03d}" for i in range(4)],
            max_nodes=20,
        )
        perfs = [0.1, 0.5, -0.2, -1.0, 0.3]
        for i, p in enumerate(perfs):
            mods = {f"mod_{i % 4:03d}", f"mod_{(i + 1) % 4:03d}"}
            tree.add_node(
                ActivationPattern(mods),
                HyperParams(learning_rate=0.001 * (i + 1)),
                performance=p,
                causal_effects={m: 0.5 for m in mods},
            )

        report = tree.generate_report(top_k=3, recent_history=3)
        self.assertEqual(len(report["top_tasks"]), 3)
        self.assertEqual(len(report["bottom_tasks"]), 3)
        self.assertEqual(report["valid_count"], 5)
        top_perfs = [t["performance"] for t in report["top_tasks"]]
        bot_perfs = [t["performance"] for t in report["bottom_tasks"]]
        self.assertGreaterEqual(top_perfs[0], top_perfs[-1])
        self.assertLessEqual(bot_perfs[0], bot_perfs[-1])
        for t in report["top_tasks"]:
            self.assertIn("task_id", t)
            self.assertIn("active_modules", t)
            self.assertIn("hyper_params", t)
            self.assertIn("parent_id", t)
            self.assertIn("visit_count", t)
            self.assertIn("recent_performances", t)
            self.assertIn("module_weights", t)
            self.assertIn("causal_effects", t)
            self.assertIn("num_children", t)
            self.assertIn("parent_performance", t)
        self.assertIn("summary", report)
        self.assertIn("module_usage", report)
        self.assertIn("depth_breakdown", report)
        self.assertIn("module_best_performance", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
