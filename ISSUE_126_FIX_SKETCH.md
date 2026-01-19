# Issue #126 Fix Implementation Sketch

## Key Insights from tesseract-core

### Path Conventions (from `tree_transforms.py` and `schema_generation.py`)

1. **Dictionary keys**: Use `{key}` format (e.g., `parameters.{x}`)
2. **Sequence indices**: Use `[idx]` format (e.g., `list.[0]`)
3. **BaseModel fields**: Use plain identifier (e.g., `scalars.a`)

### Relevant Constants and Functions

From `tesseract_core.runtime.schema_generation`:
```python
DICT_INDEX_SENTINEL = object()  # Marks dict indexing in schema tree
SEQ_INDEX_SENTINEL = object()   # Marks sequence indexing in schema tree

def _path_to_pattern(path: Sequence[str | object]) -> str:
    """Converts path with sentinels to regex pattern."""
    # DICT_INDEX_SENTINEL → r"\{[\w \-]+\}"
    # SEQ_INDEX_SENTINEL → r"\[-?\d+\]"
```

## Proposed Implementation Approach

### Step 1: Modify `_pytree_to_tesseract_flat()`

**Current implementation** (lines 80-97):
```python
def _pytree_to_tesseract_flat(pytree: PyTree) -> list[tuple]:
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    flat_list = []
    for jax_path, val in leaves:
        tesseract_path = ""
        first_elem = True
        for elem in jax_path:
            if hasattr(elem, "key"):
                if not first_elem:
                    tesseract_path += "."
                tesseract_path += elem.key              # ← PROBLEM
            elif hasattr(elem, "idx"):
                tesseract_path += f"[{elem.idx}]"
            first_elem = False
        flat_list.append((tesseract_path, val))

    return flat_list
```

**Problem**: Cannot distinguish dict keys from BaseModel field names - both use `elem.key`.

### Step 2: Use Schema Metadata to Detect Dict Fields

The `Jaxeract` class already has access to differentiable paths from the OpenAPI schema:
```python
self.differentiable_input_paths = self.client.openapi_schema["components"]["schemas"]["ApplyInputSchema"]["differentiable_arrays"]
```

For dict fields, these paths contain `{}` wildcards:
- BaseModel: `"scalars.a"` (literal)
- Dict: `"parameters.{}"` (wildcard)

### Step 3: Enhanced Implementation

```python
import re
from typing import Optional

def _pytree_to_tesseract_flat(
    pytree: PyTree,
    schema_paths: Optional[dict[str, Any]] = None
) -> list[tuple]:
    """Flatten a pytree to tesseract path format.

    Args:
        pytree: The pytree to flatten
        schema_paths: Optional dict of schema paths (from differentiable_arrays)
                     Used to detect dict fields that need {key} formatting
    """
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    # Build a set of dict field prefixes from schema
    dict_field_prefixes = set()
    if schema_paths:
        for schema_path in schema_paths.keys():
            # Find patterns with {} wildcards
            if "{}" in schema_path:
                # Extract the prefix before {}
                # e.g., "parameters.{}" → "parameters"
                prefix = schema_path.rsplit(".{}", 1)[0]
                dict_field_prefixes.add(prefix)

    flat_list = []
    for jax_path, val in leaves:
        tesseract_path = ""
        first_elem = True
        path_parts = []

        for elem in jax_path:
            if hasattr(elem, "key"):
                path_parts.append(elem.key)
            elif hasattr(elem, "idx"):
                # For now, just add the bracketed index
                # We'll handle proper path construction below
                pass

        # Now construct the path, checking if we're in a dict field
        tesseract_path = ""
        for i, part in enumerate(path_parts):
            if i > 0:
                tesseract_path += "."

            # Check if the current prefix indicates we're about to enter a dict
            current_prefix = ".".join(path_parts[:i])
            if current_prefix in dict_field_prefixes:
                # This part is a dict key - wrap in curly braces
                tesseract_path += f"{{{part}}}"
            else:
                # This is a regular field
                tesseract_path += part

        # Handle sequence indices (separate loop for clarity)
        # ... (add [idx] handling)

        flat_list.append((tesseract_path, val))

    return flat_list
```

### Step 4: Update Call Sites

**In `Jaxeract.jacobian_vector_product()`** (line 213):
```python
# OLD:
flat_tangents = dict(_pytree_to_tesseract_flat(tangent_inputs))

# NEW:
flat_tangents = dict(_pytree_to_tesseract_flat(
    tangent_inputs,
    schema_paths=self.differentiable_input_paths
))
```

**In `Jaxeract.vector_jacobian_product()`** (line 259):
```python
# OLD:
in_keys = [k for k, _ in _pytree_to_tesseract_flat(primal_inputs)]

# NEW:
in_keys = [k for k, _ in _pytree_to_tesseract_flat(
    primal_inputs,
    schema_paths=self.differentiable_input_paths
)]
```

**Similar updates needed in**:
- Line 227-229 (output path construction)
- Line 263-267 (output path construction)

## Alternative Approach: Use JAX Tree Context

JAX might provide information about whether we're accessing a dict vs an object attribute. Need to investigate:
- `jax.tree_util.tree_flatten_with_path()` path element types
- Can we detect dict access from the path element itself?

## Testing Strategy

1. **Unit test for `_pytree_to_tesseract_flat()`**:
   ```python
   # Test dict input formatting
   pytree = {"parameters": {"x": array1, "y": array2}}
   schema_paths = {"parameters.{}": {...}}
   result = _pytree_to_tesseract_flat(pytree, schema_paths)
   assert result == [("parameters.{x}", array1), ("parameters.{y}", array2)]

   # Test nested model formatting (unchanged)
   pytree = {"scalars": {"a": 1.0}, "vectors": {"v": array1}}
   schema_paths = {"scalars.a": {...}, "vectors.v": {...}}
   result = _pytree_to_tesseract_flat(pytree, schema_paths)
   assert result == [("scalars.a", 1.0), ("vectors.v", array1)]
   ```

2. **Integration test**: Run existing PR #127 tests - all should pass

## Potential Issues

1. **Mixed dict/model nesting**: What if we have `dict[str, BaseModel]`?
   - Path would be like `params.{key}.field`
   - Need to handle partial paths correctly

2. **Multiple dict levels**: `dict[str, dict[str, Array]]`
   - Path would be like `outer.{key1}.{key2}`
   - Schema would show `outer.{}.{}`

3. **Backward compatibility**: Existing code may rely on current behavior
   - Make schema_paths optional (default None)
   - Only apply curly brace formatting when schema_paths provided

## Questions

1. Should we use regex matching against schema patterns, or just check for `{}` suffix?
2. How to handle edge cases with list[dict], dict[str, list], etc.?
3. Should this logic live in tesseract-core instead for consistency?
