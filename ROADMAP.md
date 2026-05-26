# OpenLeaf — Audit Follow-Up Roadmap

This document tracks the work identified by the May 2026 audit of `main` (commit `620331b`) that has **not** yet landed on `main`. The work that *has* landed is on `claude/audit-main-branch-VngCU` (Phases 0–3 in full, Phase 4 as scaffold only) — see that branch's commit history for the diff.

Each follow-up below is a self-contained workstream. Items can land in the order shown, or be parallelised where they don't share files.

---

## 1. Phase 4 — BADA4 engine migration (~1–2 weeks)

**What:** derive `BADA4JetEngineModel`, `BADA4TurbopropEngineModel`, and `BADA4PistonEngineModel` from `BaseEngineModel[BADA4EngineData, BADA4Coeffs]`. Move the shared polynomial-evaluation, idle/max threshold, and fuel-flow derivation logic into the base; each engine type becomes a thin coefficient binding plus type-specific overrides.

**Why this work:** `packages/openleaf-performance/src/openleaf/performance/models/bada4/models.py` is **2 077 lines**. The three engine subclasses re-implement the same algorithmic skeleton with different coefficient interpretations. The audit's success bar is ≥40% line reduction — roughly 800 lines of duplication gone. More importantly, every BADA4 maintenance change today happens in three places; after migration it happens in one.

**Why BADA4 first:** the most surface to leverage AND the cleanest polynomial structure to extract. Subsequent migrations will benefit from helpers this PR introduces in `BaseEngineModel` (polynomial eval, idle/max selectors).

**Gate:** characterization fixtures in `tests/characterization/fixtures/` must be regenerated against the live BADA4 models (maintainer-only — needs `OPENLEAF_BADA4_PATH`) and then held bit-equal under `rtol=1e-12` through the migration. The infrastructure for this is already in place on the audit branch.

---

## 2. Drag-model scaffolding (~1 hour)

**What:** ship `BaseDragModel` + a `DragCoefficientSource` Protocol in `openleaf/performance/core/`, with the same shape as `BaseEngineModel`. No provider migrations.

**Why:** each provider's `drag_coef(data)` mixes four concerns — flight-state extraction, clean/non-clean configuration selection, the coefficient math, and unit conversions. They're structurally identical across providers but live in four separate files. Without scaffolding planted now, the BADA4 migration PR ends up doing engines AND inventing the drag pattern from scratch. Cheap to plant; expensive to retrofit later.

**Gate:** mypy clean, ruff clean, a single sanity unit test confirming a trivial subclass instantiates and is rejected without the abstract `drag_coef` implementation. Pairs naturally with the audit branch's `engine_base.py`.

---

## 3. Architecture doc update (~30 min)

**What:** rewrite `packages/openleaf-performance/docs/explanation/architecture.md` (and add a section to `packages/openleaf-core/README.md` if the public surface grows) to reflect the two-tier layering: `openleaf-core` carries the contract (types, protocols, base exceptions); `openleaf-performance` carries the implementations, the physics, and the soon-to-be-merged `BaseEngineModel` / `BaseDragModel` bases.

**Why:** the current doc predates `openleaf-core`. A new contributor today reads "everything lives in `openleaf-performance`" and guesses where new types go — probably wrong. Documentation rot is silent but it's what makes the next contributor's first PR take three days instead of three hours.

**Gate:** `make docs-build --strict` passes; the diagram (if any) is regenerated.

---

## 4. Phase 4 — BADA3 / OpenAP / PS engine migrations (~1 week each)

**What:** derive the remaining three providers' engine and drag models from the two base classes.

**Why one PR each, in this order:**

- **BADA3** uses table-based coefficient interpolation, not polynomial — exposes whether `BaseEngineModel`'s "coefficient set" abstraction generalises beyond polynomials, or needs a strategy-pattern helper. The first migration that's not BADA4 is where the real design pressure shows up.
- **OpenAP** wraps an external library and does CasADi-symbolic unwrapping. Tests whether the base's `__init__` is flexible enough for a wrapper provider, or whether `BaseEngineModel` needs to expose a hook for external-library coupling.
- **PS** (Poll-Schumann) uses parametric formulas, not tabulated data. Smallest provider; goes last because by then the abstraction is mature enough to absorb a fourth shape without churning.

**Why not one mega-PR:** each migration must hold `rtol=1e-12` against its golden fixtures. Combining means a regression in any one provider rejects the whole batch — and `rtol=1e-12` regressions in BADA models are typically subtle (wrong sign on a temperature gradient, wrong unit conversion). Fast feedback per provider is worth the extra review cycles.

**Gate per migration:** characterization fixtures bit-equal at `rtol=1e-12`; mypy clean; line-count reduction matches the audit's per-provider target (≥30% for BADA3, comparable for the others).

---

## 5. Tighten `models.*` mypy override (incremental — lands with each migration)

**What:** shrink (eventually delete) the wholesale `disable_error_code = ["arg-type", "return-value", "call-overload", "operator", "attr-defined", "assignment", "override"]` block on `models.*` in `packages/openleaf-performance/pyproject.toml`.

**Why:** that block silences every type error in 4 000+ lines of model code. A refactor that breaks a method signature, a wrong dataclass field name, an enum that drifted — none of those fail mypy today. The Phase-3 exercise on `physics.*` surfaced exactly one real fix (`_thrust_computed` annotation in `tem.py`); the models layer is roughly 10× the surface, so expect 5–15 real fixes plus a handful of targeted `# type: ignore[code]` annotations at known cross-backend operator-overload edges.

**Why not a separate PR:** the structural cleanup and the typing cleanup share context. Lifting the override one provider at a time, alongside each Phase-4 migration, halves the diff a reviewer has to load and means the migration cannot accidentally rely on a type error that the override was hiding.

**End state:** the override either disappears entirely or shrinks to targeted line-level `# type: ignore` comments at documented edges, matching the pattern `physics.*` ended up at in Phase 3.

---

## Net effect when everything above has landed

- `models/bada4/models.py`: **2 077 → ~1 100 lines** (audit's ≥40% bar)
- `models/bada3/models.py`: similar ratio, **~30% reduction**
- Whole-module mypy suppressions: **gone** from both `models.*` and `physics.*`; only line-level ignores remain at documented cross-backend edges.
- Adding a fifth aircraft model provider goes from "write 800 lines of duplicated thrust/fuel-flow scaffolding" to "**write a coefficient class + a 50-line subclass + golden fixtures**".

That last point is the actual payoff. The audit isn't about cleanup for its own sake — it's about making the next aircraft model a weekend's work instead of a fortnight's.
