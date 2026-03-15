# src/dspy_examples/bootstrap_fewshot.py
"""BootstrapFewShot optimization script."""

import dspy
from dspy.teleprompt import BootstrapFewShot


class PromptOptimization(dspy.Signature):
    """Optimize a prompt by adding few-shot examples."""

    unoptimized_prompt: str = dspy.InputField(desc="The original unoptimized prompt")
    optimized_prompt: str = dspy.OutputField(desc="The optimized prompt with examples")


class PromptOptimizer(dspy.Module):
    """Module for optimizing prompts using DSPy."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(PromptOptimization)

    def forward(self, unoptimized_prompt: str) -> dspy.Prediction:
        """Process an unoptimized prompt and return optimized version."""
        return self.prog(unoptimized_prompt=unoptimized_prompt)


def optimize_prompt(
    prompt: str,
) -> str:
    """Optimize a prompt using BootstrapFewShot.

    Args:
        prompt: The unoptimized prompt text.

    Returns:
        The optimized prompt with few-shot examples.
    """
    from dspy_examples.config import configure_dspy

    configure_dspy()

    # Note: The metric function currently accepts all examples (lambda x, y: True).
    # In production, you would provide a proper metric function that evaluates
    # whether the optimization improved the prompt quality.
    optimizer = BootstrapFewShot(
        metric=lambda x, y: True,  # Accept all examples for demo purposes
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    # Create a simple module to optimize
    module = PromptOptimizer()

    # Compile with example prompt
    # In real usage, you'd provide training examples
    optimized_module = optimizer.compile(
        module,
        trainset=[
            dspy.Example(
                unoptimized_prompt="Write a greeting.",
                optimized_prompt="Write a greeting.\n\nExample:\nInput: Write a greeting for a friend.\nOutput: Hey there! Great to see you again!",
            )
        ],
    )

    # Run the optimized module
    result = optimized_module(unoptimized_prompt=prompt)

    return result.optimized_prompt


def main() -> None:
    """Run BootstrapFewShot optimization on unoptimized_prompt.md."""
    from pathlib import Path

    from dspy_examples.prompts import load_prompt, save_prompt

    # Load the unoptimized prompt
    input_path = Path("prompts/unoptimized_prompt.md")
    prompt = load_prompt(input_path)

    print(f"Loaded prompt from {input_path}")
    print(f"Original prompt length: {len(prompt)} characters")

    # Optimize
    optimized = optimize_prompt(prompt)

    print(f"Optimized prompt length: {len(optimized)} characters")

    # Save with version number if file exists
    output_path = Path("prompts/optimized_prompt.md")
    if output_path.exists():
        # Find next available version number
        version = 1
        while output_path.with_name(f"optimized_prompt_{version:02d}.md").exists():
            version += 1
        save_prompt(optimized, output_path, version=version)
        print(f"Saved optimized prompt to {output_path.with_name(f'optimized_prompt_{version:02d}.md')}")
    else:
        save_prompt(optimized, output_path)
        print(f"Saved optimized prompt to {output_path}")


if __name__ == "__main__":
    main()