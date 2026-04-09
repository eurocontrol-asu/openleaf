# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in OpenLeaf, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **ramon.dalmau@eurocontrol.int**

Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You should receive an acknowledgement within **72 hours**. We will work with you to
understand the issue and coordinate a fix before any public disclosure.

## Scope

OpenLeaf is a scientific computing library for aircraft performance modeling. Security
considerations include:

- **File path handling**: BADA data paths are configured via environment variables and
  settings files. OpenLeaf validates paths but does not sandbox file access.
- **Dependency supply chain**: We pin minimum versions for all dependencies and run
  `pip-audit` in CI to detect known vulnerabilities.
- **Serialization**: OpenLeaf does not use `pickle` or other unsafe deserialization.
  Model data is loaded from structured text/XML files with dedicated parsers.
- **No network access**: OpenLeaf does not make network requests at runtime. All data
  is loaded from local files.

## Best Practices for Users

- Keep dependencies up to date (`uv sync --upgrade`)
- Do not expose BADA data files in public repositories (they are proprietary)
- Use environment variables or `.secrets.toml` (excluded from version control) for
  sensitive configuration
