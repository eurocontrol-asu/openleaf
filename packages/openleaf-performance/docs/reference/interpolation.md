# Interpolation Reference

## `interpn`

```python
from openleaf.performance.numpy.interpolate import interpn
```

Performs multidimensional interpolation on regular grids, analogous to
[`scipy.interpolate.interpn`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html).

Automatically dispatches between a NumPy/SciPy fast path and a CasADi
backend depending on input types and method.

### Signature

```python
def interpn(
    points: tuple[ndarray],
    values: ndarray,
    xi: ndarray,
    method: Literal["linear", "bspline", "nearest"] = "linear",
    bounds_error: bool = True,
    fill_value: float = nan,
) -> ndarray
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `points` | `tuple[ndarray]` | Coordinates of each grid axis. Shapes `(m1,), ..., (mn,)` |
| `values` | `ndarray` | Data on the regular grid. Shape `(m1, ..., mn)` |
| `xi` | `ndarray` | Coordinates to sample at. Shape `(..., ndim)` |
| `method` | `str` | `"linear"`, `"bspline"`, or `"nearest"` |
| `bounds_error` | `bool` | Raise `ValueError` for out-of-bounds xi (default `True`) |
| `fill_value` | `float \| None` | Value for out-of-bounds points. `None` clamps to grid edge |

### Methods

- **`"linear"`** — Differentiable but C1-discontinuous. Uses SciPy for pure NumPy inputs, CasADi otherwise.
- **`"bspline"`** — Piecewise-cubic, differentiable, suitable for optimization. CasADi only. May overshoot (not monotonicity-preserving).
- **`"nearest"`** — Not differentiable. Uses SciPy fast path only.

### Backend dispatch

| xi type | method | Backend |
|---|---|---|
| NumPy | `linear`, `nearest` | SciPy (direct delegation) |
| NumPy | `bspline` | CasADi |
| CasADi MX/DM | `linear`, `bspline` | CasADi |
