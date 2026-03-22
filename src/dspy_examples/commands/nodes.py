"""PocketFlow nodes for optimization workflow.

Nodes implement the prep/exec/post pattern:
- prep: Extract data from shared state
- exec: Execute the node's logic
- post: Update shared state with results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dspy_examples.pocketflow import Node
from dspy_examples.template import PromptTemplate


class LoadPromptNode(Node):
    """Node that loads and optionally processes a prompt file.

    Shared state inputs:
        None required

    Shared state outputs:
        prompt_content: The loaded prompt content
        prompt_template: PromptTemplate object (if frontmatter exists)
        prompt_path: Path to the prompt file
    """

    def __init__(
        self,
        prompt_path: Path,
        variables: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_path = Path(prompt_path)
        self.variables = variables or {}

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare prompt loading parameters."""
        return {
            "prompt_path": self.prompt_path,
            "variables": self.variables,
        }

    def exec(self, prep_res: dict[str, Any]) -> str:
        """Load and process the prompt."""
        prompt_path = prep_res["prompt_path"]
        variables = prep_res["variables"]

        # Try to load as template with frontmatter
        try:
            template = PromptTemplate.from_file(prompt_path)
            # Validate and substitute
            errors = template.validate(variables)
            if errors:
                # Filter warnings
                actual_errors = [e for e in errors if not e.startswith("Warning:")]
                if actual_errors:
                    raise ValueError(f"Variable validation failed: {'; '.join(actual_errors)}")

            content = template.substitute(variables)
            return content
        except ValueError as e:
            if "Invalid YAML frontmatter" in str(e):
                # Load as plain prompt
                return prompt_path.read_text()
            raise

    def post(
        self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: str
    ) -> str | None:
        """Update shared state with loaded prompt."""
        shared["prompt_content"] = exec_res
        shared["prompt_path"] = prep_res["prompt_path"]
        return "default"


class OptimizeNode(Node):
    """Node that runs prompt optimization.

    Uses the existing OptimizationPipeline to optimize prompts.

    Shared state inputs:
        prompt_content: The prompt to optimize
        provider_name: LLM provider to use
        optimizer_name: Optimizer to use

    Shared state outputs:
        result: OptimizationResult object
        optimized_prompt: The optimized prompt string
    """

    def __init__(
        self,
        prompt_path: Path,
        output_path: Path,
        provider_name: str,
        optimizer_name: str,
        model_name: str | None = None,
        variables: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_path = Path(prompt_path)
        self.output_path = Path(output_path)
        self.provider_name = provider_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.variables = variables or {}

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare optimization parameters."""
        return {
            "prompt_path": self.prompt_path,
            "output_path": self.output_path,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "optimizer_name": self.optimizer_name,
            "variables": self.variables,
            "batch_id": shared.get("batch_id"),
        }

    def exec(self, prep_res: dict[str, Any]) -> dict[str, Any]:
        """Run the optimization using the pipeline.

        Note: This is a placeholder that returns mock results.
        Actual optimization requires DSPy setup.
        """
        from dspy_examples.pipeline import OptimizationPipeline, PipelineConfig

        config = PipelineConfig(
            provider_name=prep_res["provider_name"],
            optimizer_name=prep_res["optimizer_name"],
            input_path=prep_res["prompt_path"],
            output_path=prep_res["output_path"],
            variables=prep_res["variables"],
            use_cache=False,  # Don't use cache in batch
        )

        # Run optimization
        pipeline = OptimizationPipeline(config)
        result = pipeline.run(prep_res["variables"])

        return {
            "status": "success",
            "result": result,
            "output_path": prep_res["output_path"],
        }

    def post(
        self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: dict[str, Any]
    ) -> str | None:
        """Update shared state with optimization result."""
        shared["result"] = exec_res.get("result")
        shared["status"] = exec_res.get("status", "unknown")
        shared["output_path"] = exec_res.get("output_path")

        if exec_res.get("status") == "failed":
            shared["error"] = exec_res.get("error")
            return "failed"

        return "default"


class SaveResultNode(Node):
    """Node that saves optimization results.

    Shared state inputs:
        result: OptimizationResult object
        output_path: Path to save to

    Shared state outputs:
        saved_path: Actual path where result was saved
    """

    def __init__(self, output_path: Path | None = None) -> None:
        super().__init__()
        self.output_path = output_path

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepare save parameters."""
        return {
            "result": shared.get("result"),
            "output_path": self.output_path or shared.get("output_path"),
        }

    def exec(self, prep_res: dict[str, Any]) -> Path:
        """Save the result to file."""
        from dspy_examples.prompts import save_prompt

        result = prep_res["result"]
        output_path = prep_res["output_path"]

        if result is None:
            raise ValueError("No result to save")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the optimized prompt
        save_prompt(result.optimized_prompt, output_path)

        return output_path

    def post(
        self, shared: dict[str, Any], prep_res: dict[str, Any], exec_res: Path
    ) -> str | None:
        """Update shared state with saved path."""
        shared["saved_path"] = exec_res
        return "default"