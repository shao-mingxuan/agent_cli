"""共享 pytest fixtures。"""
import pytest
from unittest.mock import MagicMock

from agent_cli.tools.builtin.safety.backend import (
    LocalSafetyBackend,
    set_backend,
    get_backend,
)


@pytest.fixture(autouse=True)
def fresh_backend():
    """每个测试前重置 safety backend singleton，防止跨测试污染。"""
    original = get_backend()
    set_backend(LocalSafetyBackend())
    yield
    set_backend(original)


@pytest.fixture
def safety_backend_instance():
    """返回一个独立的 LocalSafetyBackend 实例（不触及全局 singleton）。"""
    return LocalSafetyBackend()


@pytest.fixture
def fake_provider():
    """返回一个模拟的 BaseProvider 实现。"""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="mock summary")
    return SimpleNamespace(
        get_model=lambda: mock_model,
        get_embeddings=lambda: None,
        model_name="test-model",
        base_url="http://test",
    )


class FakeState:
    """模拟 langgraph StateSnapshot。"""

    def __init__(self, values):
        self.values = values


class FakeAgent:
    """模拟 create_agent() 返回值，由 script 驱动。

    script 是一个 (mode, data) 元组列表，stream() 会依次 yield。
    """

    def __init__(self, script=None, state_values=None):
        self._script = list(script or [])
        self._messages = list(
            (state_values or {}).get("messages", [])
        )
        self.stream_calls = []
        self.update_state_calls = []

    def stream(self, input, config=None, stream_mode=None):
        self.stream_calls.append((input, config, stream_mode))
        for chunk in self._script:
            mode, data = chunk
            if mode == "updates" and "__interrupt__" not in data:
                for _node, node_output in data.items():
                    for msg in node_output.get("messages", []):
                        self._messages.append(msg)
            yield chunk
        self._script = []

    def get_state(self, config=None):
        return FakeState({"messages": list(self._messages)})

    def update_state(self, config, updates, as_node=None):
        self.update_state_calls.append((config, updates, as_node))
        new_msgs = updates.get("messages", [])
        self._messages.extend(new_msgs)


# ── 流式 chunk 构建辅助函数 ──


def make_msg_chunk(msg, node="model"):
    """构建 ('messages', (msg, metadata)) chunk。"""
    return ("messages", (msg, {"langgraph_node": node}))


def make_update_chunk(data):
    """构建 ('updates', data) chunk。"""
    return ("updates", data)


def make_interrupt_chunk():
    """构建中断标记 chunk。"""
    return ("updates", {"__interrupt__": True})


@pytest.fixture
def mock_create_agent(monkeypatch):
    """Patch create_agent 在 orchestrator 模块中的引用。

    返回 mock 对象，测试可设置 return_value 为 FakeAgent。
    """
    mock = MagicMock()
    monkeypatch.setattr("agent_cli.agent.orchestrator.create_agent", mock)
    return mock


@pytest.fixture
def make_orchestrator(mock_create_agent, fake_provider):
    """工厂 fixture：创建 Orchestrator + FakeAgent。"""

    def _make(script=None, state_values=None, **kwargs):
        fake_agent = FakeAgent(script=script, state_values=state_values)
        mock_create_agent.return_value = fake_agent
        from agent_cli.agent.orchestrator import Orchestrator

        if kwargs.get("enable_long_term_memory") and "db_path" not in kwargs:
            kwargs["db_path"] = ":memory:"

        orch = Orchestrator(provider=fake_provider, **kwargs)
        return orch, fake_agent

    return _make
