# Issue #126 Fix - Refined Implementation

## Key Discovery

**JAX treats all keys the same**: Both BaseModel fields and dict keys appear as `DictKey` objects after `tree_flatten_with_path()`. This is because BaseModels are converted to dicts (via `model_dump()`) before JAX sees them.

Therefore, **we cannot distinguish dict keys from model fields by inspecting the JAX path alone**. We must use schema metadata.

## Implementation Strategy

### 1. Helper Function: Identify Dict Field Patterns

```python
def _build_dict_field_map(schema_paths: dict[str, Any]) -> dict[str, re.Pattern]:
    """Build a mapping of path prefixes to their dict key patterns.

    Args:
        schema_paths: Dictionary from OpenAPI schema differentiable_arrays

    Returns:
        Dict mapping path prefixes to compiled regex patterns for dict keys

    Example:
        Input: {"parameters.{}": {...}, "scalars.a": {...}}
        Output: {"parameters": re.compile(r"parameters\.(.+)")}
    """
    dict_field_map = {}

    for schema_path in schema_paths.keys():
        if "{}" in schema_path:
            # Extract prefix before the wildcard
            # e.g., "parameters.{}" → "parameters"
            # e.g., "nested.data.{}" → "nested.data"
            prefix = schema_path.rsplit(".{}", 1)[0]

            # Create a regex pattern to match paths under this prefix
            # The pattern captures the dict key
            pattern = re.escape(prefix) + r"\.(.+)"
            dict_field_map[prefix] = re.compile(pattern)

    return dict_field_map
```

### 2. Modified `_pytree_to_tesseract_flat()`

```python
import re
from typing import Any

def _pytree_to_tesseract_flat(
    pytree: PyTree,
    schema_paths: dict[str, Any] | None = None
) -> list[tuple]:
    """Flatten a pytree to tesseract path format.

    Dict keys are wrapped in curly braces {key}, while BaseModel fields
    are left as plain identifiers, based on the schema_paths metadata.

    Args:
        pytree: The pytree to flatten
        schema_paths: Optional dict of schema paths (from differentiable_arrays)
                     If provided, dict fields will be formatted as {key}

    Returns:
        List of (path_string, value) tuples
    """
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    # Build dict field mapping if schema provided
    dict_field_map = {}
    if schema_paths:
        dict_field_map = _build_dict_field_map(schema_paths)

    flat_list = []
    for jax_path, val in leaves:
        # Build path incrementally, checking at each step
        path_parts = []

        for elem in jax_path:
            if hasattr(elem, "key"):
                path_parts.append(elem.key)
            elif hasattr(elem, "idx"):
                # Sequence index - format as [idx]
                path_parts.append(f"[{elem.idx}]")

        # Construct the final path string with proper dict key formatting
        tesseract_path = _format_path_with_dict_keys(path_parts, dict_field_map)

        flat_list.append((tesseract_path, val))

    return flat_list


def _format_path_with_dict_keys(
    path_parts: list[str],
    dict_field_map: dict[str, re.Pattern]
) -> str:
    """Format a path with proper dict key notation.

    Args:
        path_parts: List of path components (field names or indices)
        dict_field_map: Mapping of prefixes to patterns for dict fields

    Returns:
        Formatted path string with dict keys wrapped in curly braces

    Example:
        path_parts = ["parameters", "x"]
        dict_field_map = {"parameters": ...}
        Returns: "parameters.{x}"
    """
    if not dict_field_map:
        # No schema info - use plain dot notation
        return ".".join(path_parts)

    # Check each possible prefix to see if it's a dict field
    formatted_parts = []

    for i, part in enumerate(path_parts):
        # Build the prefix up to (but not including) this part
        prefix = ".".join(path_parts[:i])

        # Check if this prefix indicates a dict field
        is_dict_key = prefix in dict_field_map

        if is_dict_key and not part.startswith("["):
            # This is a dict key - wrap in curly braces
            formatted_parts.append(f"{{{part}}}")
        else:
            # Regular field or sequence index
            formatted_parts.append(part)

    return ".".join(formatted_parts)
```

### 3. Update Jaxeract Methods

#### In `jacobian_vector_product()` (line 213)

```python
def jacobian_vector_product(self, ...) -> PyTree:
    # ... existing code ...

    # OLD:
    # flat_tangents = dict(_pytree_to_tesseract_flat(tangent_inputs))

    # NEW:
    flat_tangents = dict(_pytree_to_tesseract_flat(
        tangent_inputs,
        schema_paths=self.differentiable_input_paths
    ))

    jvp_inputs = list(flat_tangents.keys())
    # ... rest of method ...
```

#### In `vector_jacobian_product()` (line 259)

```python
def vector_jacobian_product(self, ...) -> PyTree:
    # ... existing code ...

    # OLD:
    # in_keys = [k for k, _ in _pytree_to_tesseract_flat(primal_inputs)]

    # NEW:
    in_keys = [k for k, _ in _pytree_to_tesseract_flat(
        primal_inputs,
        schema_paths=self.differentiable_input_paths
    )]

    vjp_inputs = [o for o, m in zip(in_keys, is_static_mask, strict=True) if not m]
    # ... rest of method ...
```

#### Output Path Construction (lines 225-230, 263-268)

```python
# Both JVP and VJP need to construct output paths
# OLD:
# paths = [
#     p for p, _ in _pytree_to_tesseract_flat(
#         jax.tree.unflatten(output_pytreedef, range(len(output_avals)))
#     )
# ]

# NEW:
paths = [
    p for p, _ in _pytree_to_tesseract_flat(
        jax.tree.unflatten(output_pytreedef, range(len(output_avals))),
        schema_paths=self.differentiable_output_paths
    )
]
```

### 4. Testing Strategy

#### Unit Tests for Helper Functions

```python
def test_build_dict_field_map():
    schema_paths = {
        "parameters.{}": {"dtype": "float32", "shape": [None]},
        "scalars.a": {"dtype": "float32", "shape": []},
        "nested.data.{}": {"dtype": "float32", "shape": [3]},
    }

    result = _build_dict_field_map(schema_paths)

    assert "parameters" in result
    assert "nested.data" in result
    assert "scalars" not in result  # Not a dict field


def test_format_path_with_dict_keys():
    dict_field_map = {
        "parameters": re.compile(r"parameters\.(.+)")
    }

    # Dict key should be wrapped
    path = ["parameters", "x"]
    assert _format_path_with_dict_keys(path, dict_field_map) == "parameters.{x}"

    # Nested dict keys
    path = ["parameters", "x", "y"]
    result = _format_path_with_dict_keys(path, dict_field_map)
    # After "parameters", both "x" and "y" are dict keys
    assert result == "parameters.{x}.{y}"

    # Non-dict field
    path = ["scalars", "a"]
    assert _format_path_with_dict_keys(path, dict_field_map) == "scalars.a"


def test_pytree_to_tesseract_flat_with_dicts():
    pytree = {"parameters": {"x": jnp.array([1.0]), "y": jnp.array([2.0])}}
    schema_paths = {"parameters.{}": {...}}

    result = _pytree_to_tesseract_flat(pytree, schema_paths)

    paths = [p for p, _ in result]
    assert "parameters.{x}" in paths
    assert "parameters.{y}" in paths
```

#### Integration Tests

Run all tests from PR #127:
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_endtoend.py::test_dict_tesseract_apply -v
```

All 8 tests should pass.

### 5. Edge Cases to Handle

1. **Nested dicts**: `dict[str, dict[str, Array]]`
   - Schema: `"outer.{}.{}"`
   - After first dict key, all subsequent keys are also dict keys
   - Solution: Check at each level if we're inside a dict field

2. **Dict of models**: `dict[str, BaseModel]`
   - Schema: `"items.{}.field"`
   - After the dict key, revert to regular field names
   - Solution: Check each prefix independently

3. **Sequence indices**: `list[dict[str, Array]]`
   - Schema: `"items.[].{}"`
   - Path: `"items.[0].{x}"`
   - Solution: Handle `[idx]` separately, don't wrap in dict format

### 6. Alternative: Simpler Pattern Matching

Instead of building a map, directly check if a schema pattern matches:

```python
def _is_dict_key_at_position(
    path_parts: list[str],
    position: int,
    schema_paths: dict[str, Any]
) -> bool:
    """Check if the key at position is a dict key based on schema."""
    # Build the path up to this position
    prefix = ".".join(path_parts[:position])

    # Check if any schema pattern suggests this is a dict key
    for schema_path in schema_paths.keys():
        if "{}" in schema_path:
            # Extract the pattern
            pattern_parts = schema_path.split(".")
            prefix_parts = path_parts[:position]

            # Match pattern: if pattern has {} at this position, it's a dict key
            if len(pattern_parts) > position:
                if pattern_parts[position] == "{}" and ".".join(pattern_parts[:position]) == prefix:
                    return True

    return False
```

This approach checks each position against the schema pattern structure, which might be more robust for complex nesting.

## Recommended Next Steps

1. Implement the core `_pytree_to_tesseract_flat()` modification
2. Add unit tests for the helper functions
3. Update the Jaxeract call sites
4. Run PR #127 tests to verify
5. Test edge cases (nested dicts, dict of models, etc.)
6. Consider upstreaming dict detection logic to tesseract-core
