"""Tests for embedded PocketFlow core."""

import pytest


class TestNode:
    """Tests for Node base class."""

    def test_node_creation(self):
        """Test creating a node with prep/exec/post methods."""
        from dspy_examples.pocketflow import Node

        class MyNode(Node):
            def prep(self, shared):
                return {"input": shared["value"]}

            def exec(self, prep_res):
                return prep_res["input"] * 2

            def post(self, shared, prep_res, exec_res):
                shared["result"] = exec_res
                return "default"

        node = MyNode()
        shared = {"value": 5}
        result = node.run(shared)

        assert shared["result"] == 10
        assert result == "default"

    def test_node_with_successors(self):
        """Test node with successor nodes."""
        from dspy_examples.pocketflow import Node

        class StartNode(Node):
            def exec(self, prep_res):
                return "next"

        class EndNode(Node):
            def exec(self, prep_res):
                return "done"

        start = StartNode()
        end = EndNode()
        start >> end  # Connect nodes

        assert end in start.successors.values()
        assert len(end.successors) == 0

    def test_node_orchestrator_pattern(self):
        """Test orchestrator returns action name for routing."""
        from dspy_examples.pocketflow import Node

        class RouterNode(Node):
            def prep(self, shared):
                shared["path"] = "path_a"

            def exec(self, prep_res):
                return "path_a"

            def post(self, shared, prep_res, exec_res):
                # Return the action for routing
                return exec_res

        node = RouterNode()
        result = node.run({})

        assert result == "path_a"


class TestFlow:
    """Tests for Flow orchestration."""

    def test_flow_sequential_execution(self):
        """Test running nodes in sequence."""
        from dspy_examples.pocketflow import Node, Flow

        class AddNode(Node):
            def __init__(self, amount):
                super().__init__()
                self.amount = amount

            def prep(self, shared):
                return shared.get("value", 0)

            def exec(self, prep_res):
                return prep_res + self.amount

            def post(self, shared, prep_res, exec_res):
                shared["value"] = exec_res

        # Create nodes
        add5 = AddNode(5)
        add10 = AddNode(10)

        # Connect
        add5 >> add10

        # Run flow
        flow = Flow(add5)
        shared = {}
        flow.run(shared)

        assert shared["value"] == 15

    def test_flow_with_routing(self):
        """Test flow with conditional routing."""
        from dspy_examples.pocketflow import Node, Flow

        class RouterNode(Node):
            def prep(self, shared):
                return shared.get("type", "a")

            def exec(self, prep_res):
                return f"path_{prep_res}"

        class PathANode(Node):
            def exec(self, prep_res):
                return "a_result"

        class PathBNode(Node):
            def exec(self, prep_res):
                return "b_result"

        router = RouterNode()
        path_a = PathANode()
        path_b = PathBNode()

        # Connect with actions
        router.successors["path_a"] = path_a
        router.successors["path_b"] = path_b

        # Test path A
        flow = Flow(router)
        shared = {"type": "a"}
        flow.run(shared)


class TestBatchNode:
    """Tests for BatchNode."""

    def test_batch_node_processes_items(self):
        """Test processing items in batch."""
        from dspy_examples.pocketflow import BatchNode

        class SumBatchNode(BatchNode):
            def prep(self, shared):
                return shared.get("items", [])

            def exec(self, item):
                return item * 2

            def post(self, shared, prep_res, exec_res):
                shared["results"] = exec_res

        node = SumBatchNode()
        shared = {"items": [1, 2, 3, 4]}
        node.run(shared)

        assert shared["results"] == [2, 4, 6, 8]

    def test_batch_node_empty_list(self):
        """Test batch node with empty list."""
        from dspy_examples.pocketflow import BatchNode

        class EmptyBatchNode(BatchNode):
            def prep(self, shared):
                return []

            def exec(self, item):
                return item

        node = EmptyBatchNode()
        shared = {}
        node.run(shared)

        assert shared.get("batch_results") == []


class TestBatchFlow:
    """Tests for BatchFlow."""

    def test_batch_flow_multiple_configs(self):
        """Test running flow with multiple configurations."""
        from dspy_examples.pocketflow import Node, BatchFlow

        class ConfigNode(Node):
            def prep(self, shared):
                return shared.get("config_value", 0)

            def exec(self, prep_res):
                return prep_res * 2

            def post(self, shared, prep_res, exec_res):
                shared["result"] = exec_res

        node = ConfigNode()
        flow = BatchFlow(node)

        configs = [
            {"config_value": 5},
            {"config_value": 10},
            {"config_value": 15},
        ]

        flow.set_batch_configs(configs)
        # Call run - it updates shared state internally
        shared = {}
        flow.run(shared)

        # BatchFlow.exec returns results directly, but run() calls post()
        # which returns None. To get results, we call exec() directly.
        results = flow.exec({})

        assert len(results) == 3
        assert results[0]["result"] == 10
        assert results[1]["result"] == 20
        assert results[2]["result"] == 30