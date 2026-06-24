# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.13] - 2026-06-24

### Added

- Added Memory Steelprint: content-hashed provenance sources, exact evidence spans, memory-to-evidence seals, automatic claim conflict detection, answer-level citations, strict evidence refusal, post-answer verification, and persisted hallucination evaluation metrics.
- Added dual MCP deployment: client-owned stdio and a shared single-instance Streamable HTTP service with `/mcp` and `/health` endpoints.
- Added profile-aware memory growth with built-in generalist, software engineer, research scientist, and clinical reasoner profiles.
- Added memory profile tools, project profile binding, layered memory routing, store/retrieve scoring metadata, and outcome feedback records.
- Added optional first-run profile setup with `memery configure`, `memery profiles`, `memery setup-status`, `get_setup_status`, and `configure_memory_defaults`.
- New projects and contexts can now inherit the configured default personality/profession profile.
- Expanded setup into a two-step personality-then-profession composition flow with 36 personality presets and 216 profession presets.
- Added preset-id based `create_memory_profile` creation, compact/pageable preset listing, and `delete_memory_profile`.
- Added an OS-level single-instance service lock so only one MCP service runs on a machine at a time.
- Added repeatable contract coverage for all 62 MCP tools and malformed structured inputs.

### Fixed

- Hardened all MCP contracts around native JSON arrays/objects, malformed numeric values, bounded pagination, project scoping, vector-index partial failures, evidence locators, and malformed answer claims.
- Prevented missing named projects, wings, and rooms from accidentally widening list/search operations to global data.
- Added a SQLite temporal-graph fallback when the optional MemPalace package is unavailable.
- Profile JSON errors now identify the exact field, such as `calibration`.
- Environment variables now correctly override `~/.memery/config.json`.
- Batched retrieval hit counters into one SQLite transaction per result set.

## [1.12] - 2026-06-18

### Fixed

- Hardened data writes so old SQLite databases are migrated before insert.
- Made `write_memory` and `write_memories_batch` tolerate plain strings, bad JSON fields, and unknown `hall_id` values.
- Normalized blank `memory_type`, `title`, tags, source files, and numeric scores before SQLite writes.
- Returned clean empty-content errors instead of leaking SQLite `NOT NULL` failures.
- Prevented vector index failures from blocking durable SQLite writes.

## [1.11] - 2026-06-17

### Added

- Added `memery doctor` for user-friendly installation diagnostics.
- Added robust MCP stdio handling that ignores blank terminal input.

### Fixed

- Fixed LanceDB startup when an existing `memories.lance` table is present.
- Fixed direct source-tree startup so it does not depend on the clone directory name.

### Removed

- Removed obsolete duplicate vector backend files and local runtime artifacts from the clean release.

### Added

- Standard Python packaging with an installable `memery` command.
- Apache License 2.0, security policy, contribution guide, disclaimer, and GitHub templates.
- User-profile runtime storage under `~/.memery` by default.

## [1.10] - 2026-06-16

### Added

- Pinned project core, latest-conversation summary, and current project summary.
- Generalized software, research, business, learning, and general contexts.
- Revision-aware context caching and bounded summary rebuilding.
- Thread-local SQLite connections, WAL tuning, composite indexes, and batch writes.
- Project-prefiltered LanceDB retrieval and repeatable stress benchmarks.
- Temporal knowledge graph, code graph analysis, task snapshots, and decisions.
