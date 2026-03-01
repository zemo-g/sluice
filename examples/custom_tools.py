"""Example: extending sluice with cloud tool use.

Shows how to define tools for Claude and execute them in a multi-turn loop.
This pattern lets Claude query databases, APIs, or filesystems mid-conversation.

Requires: pip install sluice-llm[cloud]
"""

import json
import sqlite3

# Define tools Claude can call (Anthropic tool_use format)
TOOLS = [
    {
        "name": "query_database",
        "description": "Run a read-only SQL query against the application database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SELECT query to execute (read-only).",
                },
            },
            "required": ["sql"],
        },
    },
]


def execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "query_database":
        return _query_db(inputs["sql"])
    return json.dumps({"error": f"Unknown tool: {name}"})


def _query_db(sql: str) -> str:
    """Execute a read-only SQL query. Replace DB_PATH with your database."""
    DB_PATH = "app.db"  # Change this to your database path
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries allowed"})
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        conn.close()
        return json.dumps([dict(r) for r in rows[:100]], default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
