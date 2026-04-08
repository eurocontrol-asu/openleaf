# Getting Started

This tutorial walks you through installing `openleaf-performance` and verifying your setup.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
uv add openleaf-performance
```

Or with pip:

```bash
pip install openleaf-performance
```

## Step 1: Verify Installation

```python
from openleaf_performance import __version__

print(f"openleaf-performance v{__version__}")
```

## Step 2: Run the Tests

```bash
cd packages/openleaf-performance
make check
```

This runs lint + type check + security audit + tests.

## Next Steps

- [CLI Reference](../reference/cli.md) — Full command documentation
- [Architecture](../explanation/architecture.md) — How the project is structured
