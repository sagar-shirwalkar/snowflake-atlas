"""Snowflake Atlas — local-first AI knowledge layer for Snowflake documentation.

Two Model Context Protocol (MCP) servers expose the entire Snowflake
documentation corpus to any AI agent:

* snowflake-fs — deterministic filesystem server backed by ripgrep
* snowflake-rag — semantic search over precomputed embeddings

The RAG bundle is built once by the maintainer, then distributed as a
single download. End users never embed, never chunk, never run a vector
database, never pull a model.
"""

__version__ = "0.1.0"
