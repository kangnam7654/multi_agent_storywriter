"""app.generate_story 완료 메시지 테스트 (Ollama 호출 없음)

실제 LangGraph 그래프를 실행하되, 에이전트만 결정적 스크립트로 대체합니다.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app as app_module
from tests.test_graph import (
    ScriptedDirector,
    ScriptedParser,
    ScriptedWriter,
    install_scripted_agents,
)


@pytest.fixture
def offline_app(monkeypatch):
    """ChatOllama와 프롬프트 파일 로딩을 대체"""
    monkeypatch.setattr(app_module, "ChatOllama", MagicMock())
    monkeypatch.setattr(
        app_module,
        "load_system_prompts",
        lambda: SimpleNamespace(story_writer="", director=""),
    )
    return app_module


def _final_output(app, max_retries: int) -> str:
    outputs = list(app.generate_story("아이디어", "test-model", max_retries))
    assert outputs, "generate_story가 아무것도 출력하지 않았습니다"
    return outputs[-1]


def test_approved_run_shows_approved_completion(offline_app, monkeypatch):
    writer = ScriptedWriter(["draft 1", "draft 2"])
    director = ScriptedDirector([(False, "fb 1"), (True, "good")])
    install_scripted_agents(monkeypatch, ScriptedParser(), writer, director)

    final = _final_output(offline_app, max_retries=3)

    assert offline_app.APPROVED_MESSAGE in final
    assert "미승인" not in final
    assert final.count("## ✍️") == 2
    assert "## ✍️ 스토리 초안" in final
    assert "## ✍️ 스토리 수정 (시도 2)" in final
    assert len(writer.calls) == 2
    assert len(director.calls) == 2


def test_max_review_exit_shows_unapproved_completion(offline_app, monkeypatch):
    writer = ScriptedWriter(["draft 1", "draft 2", "draft 3"])
    director = ScriptedDirector([(False, "fb 1"), (False, "fb 2"), (True, "ok")])
    install_scripted_agents(monkeypatch, ScriptedParser(), writer, director)

    final = _final_output(offline_app, max_retries=2)

    expected = offline_app.UNAPPROVED_MESSAGE_TEMPLATE.format(max_reviews=2)
    assert expected in final
    assert offline_app.APPROVED_MESSAGE not in final
    assert "🎉" not in final
    assert "## ✅ 검수 통과" not in final
    assert final.count("## 🔄 검수 피드백") == 2
    assert "draft 3" not in final
    assert len(writer.calls) == 2
    assert len(director.calls) == 2


def test_empty_input_short_circuits(offline_app):
    outputs = list(offline_app.generate_story("   ", "test-model", 3))
    assert outputs == ["스토리 아이디어를 입력해주세요."]
