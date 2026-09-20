"""MCP 数据库查询 server - 演示如何用 MCP 封装数据库操作。

基于 SQLite（Python 内置，零依赖），包含 3 个工具：
  - list_tables:    列出所有表
  - describe_table: 查看表结构
  - query_db:       执行只读 SQL 查询（禁止写操作）

启动方式：
    venv/bin/python mcp_servers/db_server.py

CLI 接入（mcp.json）：
    "db": {
      "command": "venv/bin/python",
      "args": ["mcp_servers/db_server.py"],
      "transport": "stdio"
    }

环境变量：
    DB_PATH  数据库文件路径（默认: mcp_servers/demo.db）
"""
import os
import sqlite3

from mcp.server.mcpserver import MCPServer

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "demo.db"))

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create",
    "replace", "attach", "detach", "pragma", "vacuum",
}


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


mcp = MCPServer("db-server", version="0.1.0")


@mcp.tool()
def list_tables() -> str:
    """列出数据库中所有的表名。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        if not rows:
            return "数据库中没有表"
        names = [r["name"] for r in rows]
        return f"共 {len(names)} 个表:\n" + "\n".join(f"  - {n}" for n in names)
    finally:
        conn.close()


@mcp.tool()
def describe_table(table_name: str) -> str:
    """查看指定表的结构（列名、类型、是否可空、主键等）。

    Args:
        table_name: 表名
    """
    conn = _get_conn()
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        rows = cur.fetchall()
        if not rows:
            return f"表 '{table_name}' 不存在"
        lines = [f"表 '{table_name}' 结构:"]
        for r in rows:
            pk = " [PK]" if r["pk"] else ""
            nullable = "" if r["notnull"] else " (可空)"
            lines.append(
                f"  - {r['name']}  类型={r['type'] or '无'}{pk}{nullable}"
            )
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
def query_db(sql: str, limit: int = 20) -> str:
    """执行只读 SQL 查询（SELECT），返回结果。禁止任何写操作。

    Args:
        sql: SELECT 查询语句，如 "SELECT * FROM users WHERE age > 18"
        limit: 最多返回的行数，默认 20，最大 100
    """
    normalized = sql.strip().lower()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in normalized:
            return f"[拒绝] 禁止执行写操作，检测到关键词: {kw}"

    if not normalized.startswith("select"):
        return "[拒绝] 只允许 SELECT 查询"

    limit = max(1, min(limit, 100))

    conn = _get_conn()
    try:
        cur = conn.execute(sql)
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(limit)
        total = len(rows)

        if total == 0:
            return "查询结果为空"

        lines = [" | ".join(col_names), "-" * (len(" | ".join(col_names)))]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))

        lines.append(f"\n共 {total} 行（limit={limit}）")
        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"[SQL 错误] {e}"
    finally:
        conn.close()


def _init_demo_db():
    """创建示例数据库（仅首次运行时创建）。"""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        return

    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                city TEXT,
                email TEXT
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                product TEXT NOT NULL,
                amount REAL,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            INSERT INTO users (name, age, city, email) VALUES
                ('张三', 28, '北京', 'zhangsan@example.com'),
                ('李四', 35, '上海', 'lisi@example.com'),
                ('王五', 22, '杭州', 'wangwu@example.com'),
                ('赵六', 31, '深圳', 'zhaoliu@example.com');
            INSERT INTO orders (user_id, product, amount, status) VALUES
                (1, '笔记本电脑', 6999.00, 'completed'),
                (1, '鼠标', 89.00, 'completed'),
                (2, '手机', 3999.00, 'pending'),
                (3, '键盘', 299.00, 'completed'),
                (4, '显示器', 1599.00, 'pending');
        """)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    _init_demo_db()
    mcp.run("stdio")
