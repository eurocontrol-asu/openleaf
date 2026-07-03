# Multi-backend dispatch

`openleaf-performance` supports four array backends through a unified dispatch
layer in the `numpy/` module. Code written against this layer works unchanged
with any supported backend.

## Supported backends

| Backend | Import | Use case |
|---|---|---|
| NumPy | `numpy` | General analysis, data processing |
| CasADi | `casadi` | Trajectory optimization (symbolic AD) |
| JAX | `jax.numpy` | GPU acceleration, JIT compilation |
| PyTorch | `torch` | GPU acceleration, ML integration |

## How dispatch works

Every math function in `openleaf.performance.numpy` (e.g., `sqrt`, `where`,
`sin`, `clip`) follows the same two-step dispatch:

1. **CasADi check first.** If any input is a CasADi type (`MX`, `DM`, or `SX`),
   the function delegates to the corresponding `casadi` method. This check
   happens first because CasADi types do not implement the Python Array API.

2. **Array API fallback.** For all other types, `get_array_namespace(*arrays)`
   (from `array-api-compat`) returns the correct namespace (`numpy`, `jax.numpy`,
   `torch`), and the function calls the standard method on that namespace.

```
Input arrays
    |
    +-- is_casadi_type? --> casadi.sqrt, casadi.if_else, ...
    |
    +-- else --> get_array_namespace() --> xp.sqrt, xp.where, ...
                    |
                    +-- numpy.ndarray   --> numpy
                    +-- jax.Array       --> jax.numpy
                    +-- torch.Tensor    --> torch
                    +-- int/float       --> numpy (default)
```

## CasADi specifics

CasADi is the primary optimization backend. Its symbolic types (`MX` for
optimization variables, `DM` for dense matrices, `SX` for scalar expressions)
require special handling:

- **No boolean indexing.** Use `np.where(condition, a, b)` instead of `a[mask]`.
- **No shape inspection at build time.** Symbolic dimensions are resolved at
  solve time.
- **Interpolation.** `interpn` with CasADi inputs uses CasADi's built-in
  B-spline or linear interpolation instead of SciPy.

Because all physics and model code uses `openleaf.performance.numpy` as a
drop-in replacement for NumPy, switching to CasADi requires no code changes
in the model layer -- only the input types change.

## When to use which backend

| Scenario | Backend | Why |
|---|---|---|
| Batch analysis of flight data | NumPy | Fast, no dependencies beyond NumPy |
| Trajectory optimization (NLP) | CasADi | Symbolic automatic differentiation for NLP solvers |
| Large-scale GPU computation | JAX or PyTorch | Hardware acceleration, JIT compilation |
| ML pipeline integration | PyTorch | Native tensor interop with PyTorch models |

## Interpolation dispatch

The `interpn` function has its own dispatch logic because interpolation methods
differ significantly between backends:

| Input type | Method | Backend used |
|---|---|---|
| NumPy | `linear`, `nearest` | SciPy (direct delegation) |
| NumPy | `bspline` | CasADi (SciPy has no B-spline interpn) |
| CasADi | `linear`, `bspline` | CasADi native interpolation |

See [Interpolation reference](../reference/interpolation.md) for the full API.
