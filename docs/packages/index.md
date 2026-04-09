---
hide:
  - toc
---

# Packages

<div class="grid cards" markdown>

-   :material-airplane:{ .lg .middle } **openleaf-performance**

    ---

    A modular, differentiable aircraft performance modeling framework supporting BADA, Poll-Schumann, OpenAP and more.

    [:octicons-arrow-right-24: Getting Started](../performance/)

</div>

## Architecture

```mermaid
graph TD
    classDef perf fill:#42A5F5,stroke:#1E88E5
    classDef core fill:#78909C,stroke:#546E7A

    PERF["openleaf-performance\nAircraft performance"]:::perf --> CORE
    CORE["openleaf-core\nShared types & interfaces"]:::core
```
