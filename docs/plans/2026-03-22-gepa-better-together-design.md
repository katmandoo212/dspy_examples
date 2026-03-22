# GEPA and BetterTogether Optimizers Design

**Date**: 2026-03-22
**Status**: Implemented

## Summary

Added two new DSPy optimizers:
- **GEPA**: Reflective prompt evolution with Pareto-based candidate selection
- **BetterTogether**: Meta-optimizer combining prompt and weight optimization

## GEPA Optimizer

### Design Decisions

1. **Reflection Model**: Uses configured LM by default, with optional override via `reflection_model` parameter
2. **Metric**: Adapted to return `ScoreWithFeedback` dict for GEPA's reflective learning
3. **Auto Mode**: Defaults to `"medium"` (balanced), supports `"light"` and `"heavy"`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto` | `"medium"` | Budget preset |
| `reflection_model` | `None` | Optional override model for reflection |

### Implementation

```python
from dspy_examples.optimizers.gepa import GEPAOptimizer

optimizer = GEPAOptimizer(
    auto="medium",
    reflection_model="gpt-4"  # Optional override
)
```

## BetterTogether Optimizer

### Design Decisions

1. **Default Strategy**: `"p -> w"` (prompt → weight) - simpler and empirically effective
2. **Prompt Optimizer**: Defaults to `bootstrap_random`, supports `gepa`
3. **Fine-tuning**: Requires provider with fine-tuning support (OpenAI, Databricks, Local)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt_optimizer` | `"bootstrap_random"` | Which prompt optimizer to use |
| `strategy` | `"p -> w"` | Optimization sequence |
| `auto` | `"medium"` | Budget preset for GEPA if used |

### Valid Strategies

- `"p -> w"`: Prompt optimization, then fine-tuning
- `"w -> p"`: Fine-tuning first, then prompt optimization
- `"p -> w -> p"`: Full cycle (most thorough)

### Implementation

```python
from dspy_examples.optimizers.better_together import BetterTogetherOptimizer

optimizer = BetterTogetherOptimizer(
    prompt_optimizer="gepa",
    strategy="p -> w"
)
```

## Files Created

- `src/dspy_examples/optimizers/gepa.py` - GEPA implementation
- `src/dspy_examples/optimizers/better_together.py` - BetterTogether implementation
- `tests/test_optimizer_gepa.py` - GEPA tests (8 tests)
- `tests/test_optimizer_better_together.py` - BetterTogether tests (13 tests)

## Files Modified

- `src/dspy_examples/optimizers/__init__.py` - Added exports
- `src/dspy_examples/factory/optimizer_factory.py` - Registered new optimizers
- `tests/test_optimizer_factory.py` - Added factory tests for new optimizers
- `README.md` - Updated optimizer documentation

## Test Results

- 100 tests passed
- 2 tests skipped (integration tests)
- All new optimizer tests pass

## References

- [GEPA Overview - DSPy](https://dspy.ai/api/optimizers/GEPA)
- [BetterTogether - DSPy](https://dspy.ai/api/optimizers/BetterTogether/)
- GEPA Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (arXiv:2507.19457)
- BetterTogether Paper: "Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together"