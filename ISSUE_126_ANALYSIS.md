# Issue #126 Analysis: Dict Inputs Validation Failure

## Summary
Bug #126 describes validation errors when using tesseracts with dictionary input signatures like `dict[str, Differentiable[Array]]`. PR #127 adds reproducer tests but **does not implement a fix** - all VJP/JVP/Jacobian tests are currently failing.

## Root Cause

### The Problem
When tesseract-jax flattens dictionary inputs, it creates paths using dot notation:
- Input: `{"parameters": {"x": array1, "y": array2}}`
- Flattened paths: `["parameters.x", "parameters.y"]`

However, tesseract-core's validation schemas expect wildcard dictionary keys to use **curly brace notation**:
- Expected format: `["parameters.{x}", "parameters.{y}"]`

### Evidence

**1. OpenAPI Schema Registration**

Nested BaseModel (works):
```json
{
  "scalars.a": {"dtype": "float32", "shape": []},
  "vectors.v": {"dtype": "float32", "shape": [null]}
}
```

Dict type (fails):
```json
{
  "parameters.{}": {"dtype": "float32", "shape": [null]}
}
```

The `{}` wildcard indicates dictionary fields require curly brace syntax.

**2. Validation Error**
```
ValidationError: 2 validation errors for VJPInputSchema
vjp_inputs.0
  String should match pattern 'parameters\.\{[\w \-]+\}'
  [input_value='parameters.x']
```

The regex pattern `parameters\.\{[\w \-]+\}` explicitly requires:
- `parameters.` prefix
- Literal `{` opening brace
- One or more word characters, spaces, or hyphens
- Literal `}` closing brace

### Where the Bug Occurs

**File**: `tesseract_jax/tesseract_compat.py`
**Function**: `_pytree_to_tesseract_flat()` (lines 80-97)

```python
def _pytree_to_tesseract_flat(pytree: PyTree) -> list[tuple]:
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]

    flat_list = []
    for jax_path, val in leaves:
        tesseract_path = ""
        first_elem = True
        for elem in jax_path:
            if hasattr(elem, "key"):                    # Dict key
                if not first_elem:
                    tesseract_path += "."               # Add dot separator
                tesseract_path += elem.key              # ← PROBLEM: should add {key}
            elif hasattr(elem, "idx"):                  # List/tuple index
                tesseract_path += f"[{elem.idx}]"
            first_elem = False
        flat_list.append((tesseract_path, val))

    return flat_list
```

**Current behavior**: Creates `"parameters.x"`
**Required behavior**: Should create `"parameters.{x}"` for dict keys

## PR #127 Status

### What PR #127 Includes
✅ New dict_tesseract fixture with `dict[str, Differentiable[Array]]` input
✅ Comprehensive test coverage (primal, VJP, JVP, Jacobian tests)
✅ Tests correctly reproduce the bug

### What PR #127 is Missing
❌ **No fix implemented** - all gradient tests are failing
❌ The dict_tesseract implementation itself has bugs (see test failures)
❌ No changes to `_pytree_to_tesseract_flat()` function

### Test Results
```
tests/test_api.py::test_dict_tesseract_primal[True]       PASSED (12%)
tests/test_api.py::test_dict_tesseract_primal[False]      PASSED (25%)
tests/test_api.py::test_dict_tesseract_vjp[True]          FAILED (37%)
tests/test_api.py::test_dict_tesseract_vjp[False]         FAILED (50%)
tests/test_api.py::test_dict_tesseract_jacobian[fwd-True] FAILED (62%)
tests/test_api.py::test_dict_tesseract_jacobian[fwd-False] FAILED (75%)
tests/test_api.py::test_dict_tesseract_jacobian[rev-True] FAILED (87%)
tests/test_api.py::test_dict_tesseract_jacobian[rev-False] FAILED (100%)
```

## Required Fix

### Approach 1: Fix Path Generation in tesseract-jax (Recommended)
Modify `_pytree_to_tesseract_flat()` to detect when a key comes from a dict type and wrap it in curly braces.

**Challenge**: How to distinguish dict keys from BaseModel field names at the JAX tree level?

### Approach 2: Fix Validation in tesseract-core
Relax the validation pattern to accept both `parameters.x` and `parameters.{x}` formats.

**Challenge**: Requires changes to tesseract-core (different repository), and may break existing behavior.

### Approach 3: Hybrid Solution
- Use OpenAPI schema metadata to identify dict fields (those with `{}` pattern)
- In `_pytree_to_tesseract_flat()`, check if parent path matches a dict pattern
- Apply curly brace formatting only for dict keys

## Additional Issues in PR #127

The dict_tesseract fixture implementation has bugs:
1. `jacobian()` function initializes output dict incorrectly
2. VJP/JVP functions reference wrong keys in some paths

These need to be fixed alongside the path formatting issue.

## Recommendation

**PR #127 does NOT sufficiently address the bug** - it only adds reproducers. A complete fix requires:

1. ✅ Reproducer tests (already in PR #127)
2. ❌ Fix `_pytree_to_tesseract_flat()` to handle dict keys with curly braces
3. ❌ Fix bugs in dict_tesseract fixture implementation
4. ❌ All tests passing

The PR is correctly marked as "Draft" and should not be merged without implementing the actual fix.
