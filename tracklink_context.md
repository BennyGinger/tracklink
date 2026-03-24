# **Project overview**

This project is an in-house Python tracking module for segmented microscopy data.

The goal is to build a flexible tracking wrapper that links segmented objects across time, primarily from Cellpose masks. The module should work as a standalone component that can later be integrated into different analysis pipelines, but it should remain independent from higher-level application logic.

This project is not about inventing a new tracking algorithm from scratch. The intention is to use existing tracking approaches where possible, and wrap them in a design that fits our microscopy workflows and data structures.

# **Tracking problem definition**

The input data are microscopy images of living cells and tissues, where segmentation is already available frame by frame, usually from Cellpose.

The purpose of tracking is to associate segmented masks across time.

There are several tracking scenarios to support:

1. **Static or near-static cell cultures**
   Cells move little or not at all between frames. Simple IoU-based tracking works well here.

2. **Highly motile cells in living tissue**
   Examples include neutrophils and macrophages. These cells can move substantially and unpredictably, cross paths, leave the field of view, and re-enter later.

3. **Tissue drift / global motion**
   Individual cells may stay relatively fixed with respect to the tissue, but the whole tissue moves between frames. Pure IoU tracking is not sufficient in this case.

The overall objective is to design one tracking wrapper that can cover most of these use cases and achieve good practical performance across them, ideally tracking around 90% of cells in typical experiments.

# **Scope**

This project is focused on the design of the wrapper module and its internal structure.

It is not necessary to overengineer for enterprise-level robustness. This is in-house software, so the design should favor:

* simplicity
* clarity
* modularity
* practical usability

Safety, abstraction depth, and framework complexity should be kept moderate.

# **Design goals**

The module should:

* expose one clear, user-friendly public entry point, ideally a class
* internally separate concerns cleanly into focused submodules
* remain independent from higher-level pipelines and orchestration code
* support multiple tracking backends or strategies later if needed
* be easy to integrate into other projects
* accept either a sequence of masks or a stack
* eventually support possible 3D / z-aware tracking
* handle explicit axis-order conventions

The output should likely mirror the input style, meaning sequence in / sequence out, stack in / stack out, although this is not fully fixed yet and may be refined later.

# **Input assumptions**

The primary input is segmented mask data over time.

Possible accepted forms:

* sequence of single-frame masks
* stacked masks

Possible future extension:

* 3D structures over z
* time + z data, where axis order must be explicit and configurable

The tracker should operate on masks rather than raw image segmentation.

# **Architecture preferences**

I prefer a simple and modular architecture.

Important preferences:

* one module, one responsibility
* clear separation of concerns
* avoid hidden coupling between components
* public API should be convenient, but internals should stay well decomposed
* keep higher-level orchestration outside this module

A good pattern is:

* one clean public-facing class as the main client entry point
* separate internal modules for distinct responsibilities such as input normalization, feature extraction, candidate generation, cost computation, matching, track management, and output formatting

# **Scalability philosophy**

The design should remain scalable over time.

This does not mean overengineering the initial implementation. The goal is to keep the first version simple, but structure the code so that it can grow cleanly later.

Scalability should include:

* adding new tracking strategies or backends without rewriting the public API
* supporting additional data layouts or axis conventions later
* extending from 2D time series to possible 3D / z-aware tracking
* improving performance later without major structural redesign
* keeping modules decoupled enough that individual parts can be replaced or upgraded independently

The code should therefore favor simple interfaces, clearly separated responsibilities, and internal components that are easy to extend without breaking the overall structure.

# **Coding style preferences**

* use type hints extensively
* prefer simple, readable code over clever abstractions
* function parameter lists should stay on one line where possible
* function calls inside bodies can be split over multiple lines if needed
* imports should stay one per line
* avoid deeply nested logic
* avoid multiple layers of nested `if` blocks inside `for` loops where possible
* prefer flatter control flow
* use logging, not `print`
* logging is managed by the higher-level caller
* when raising errors, also log them

# **Engineering philosophy**

This is in-house software. I do not want heavy frameworks or unnecessary abstraction layers.

Preferred approach:

* practical over theoretical
* modular over monolithic
* simple over overengineered
* explicit over magical

# **Guidelines for ChatGPT**

* Prefer simple, modular solutions over complex abstractions
* When suggesting code, make it directly compatible with my structure
* Avoid suggesting heavy frameworks
* Ask clarifying questions if architecture is unclear

# **Working style with ChatGPT and Copilot**

ChatGPT is used mainly for:

* structural thinking
* architecture design
* API design
* keeping track of design decisions
* preparing implementation plans

GitHub Copilot in VS Code is used mainly for:

* code execution
* repository inspection
* implementation details
* checking local files and existing code structure

When conceptual discussion is needed, normal discussion with ChatGPT is appropriate.

When implementation work is needed, ChatGPT should provide a **clear Copilot handoff**: concise, actionable instructions that can be pasted into Copilot so it can inspect files and implement changes in the repository.

If repository-specific clarification is needed, prefer asking Copilot to inspect the codebase rather than making ChatGPT guess. Specific files can be provided to ChatGPT when needed, but direct repository inspection via Copilot should be favored.

# **Current project state**

The project is still in design phase. Some details, especially exact I/O representation and supported tracking backends, will be finalized progressively during development.

The immediate goal is to design a clean standalone tracking module structure that can later support different tracking strategies under one consistent interface.