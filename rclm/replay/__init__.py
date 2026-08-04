"""Read-only replay of the shipped compression mechanisms over captured sessions.

See docs/work_context/PRD_Replay_MCP.md. Every module here is offline and
deterministic: no network, no disk, no model calls, no wall-clock dependence
in the accounting itself.
"""
