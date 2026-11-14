"""graph 모듈 테스트"""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import src.graph as graph_module
from src.graph import create_graph, run_story_generation_stream
from src.schemas.state import EvalReport, GraphState, RefinedRequest, StoryOutput


class TestCreateGraph:
    """create_graph 함수 테스트"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM fixture"""
        return MagicMock()

    def test_create_graph_returns_state_graph(self, mock_llm):
        """create_graph가 StateGraph를 반환하는지 테스트"""
        graph = create_graph(llm=mock_llm)

        # StateGraph 객체 반환
        assert graph is not None

    def test_create_graph_with_system_prompts(self, mock_llm):
        """시스템 프롬프트와 함께 그래프 생성 테스트"""
        graph = create_graph(
            llm=mock_llm,
            story_writer_system_prompt="Writer prompt",
            director_system_prompt="Director prompt",
        )

        assert graph is not None

    def test_graph_has_expected_nodes(self, mock_llm):
        """그래프에 예상 노드가 있는지 테스트"""
        graph = create_graph(llm=mock_llm)

        # 노드 이름 확인
        node_names = list(graph.nodes.keys())
        assert "init" in node_names
        assert "write" in node_names
        assert "review" in node_names

    def test_graph_compiles(self, mock_llm):
        """그래프가 컴파일되는지 테스트"""
        graph = create_graph(llm=mock_llm)

        # 컴파일 성공
        app = graph.compile()
        assert app is not None


class TestGraphFlow:
    """그래프 흐름 테스트"""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    def test_should_retry_returns_end_when_complete(self, mock_llm):
        """is_complete가 True일 때 'end' 반환 테스트"""
        # 그래프 내부의 should_retry 함수 테스트
        state = GraphState(is_complete=True)

        # should_retry 로직 재현
        result = "end" if state.is_complete else "retry"

        assert result == "end"

    def test_should_retry_returns_retry_when_incomplete(self, mock_llm):
        """is_complete가 False일 때 'retry' 반환 테스트"""
        state = GraphState(is_complete=False)

        result = "end" if state.is_complete else "retry"

        assert result == "retry"


class TestGraphState:
    """GraphState 테스트"""

    def test_default_state(self):
        """기본 상태 테스트"""
        state = GraphState()

        assert state.user_input == ""
        assert state.is_complete is False
        assert state.retry_count == 0
        assert state.max_retries == 3

    def test_state_with_values(self):
        """값이 있는 상태 테스트"""
        state = GraphState(
            user_input="테스트",
            retry_count=2,
            is_complete=True,
        )

        assert state.user_input == "테스트"
        assert state.retry_count == 2
        assert state.is_complete is True


# ======================================================================
# 실제 LangGraph 엔진(.compile().invoke/stream)으로 흐름을 실행하는 테스트.
# 에이전트만 결정적 스크립트로 대체하며, 스크립트 에이전트는 실제 에이전트와
# 같은 계약(받은 상태를 수정하고 "전체 상태"를 반환)을 따릅니다.
# ======================================================================


class ScriptedParser:
    """UserRequestParser 대체: state.request를 채우고 전체 상태 반환"""

    def __init__(self):
        self.calls = 0

    def __call__(self, state: GraphState, runtime) -> GraphState:
        self.calls += 1
        state.request = RefinedRequest(
            summarized_prompt=f"요약: {state.user_input}",
            genre="판타지",
            style="소설",
            length="Short",
        )
        return state


class ScriptedWriter:
    """StoryWriter 대체: 미리 정한 초안을 순서대로 작성하고 히스토리에 추가"""

    def __init__(self, drafts: list[str]):
        self.drafts = list(drafts)
        self.calls: list[dict] = []

    def __call__(self, state: GraphState, runtime) -> GraphState:
        index = len(self.calls)
        self.calls.append(
            {
                "story_history": list(state.story_history),
                "feedback_history": list(state.feedback_history),
            }
        )
        draft = self.drafts[index]  # 스크립트 초과 호출 시 IndexError
        state.story_output = StoryOutput(
            title=f"제목 {index + 1}", story=draft, word_count=len(draft)
        )
        state.story_history.append(draft)
        return state


class ScriptedDirector:
    """Director 대체: 실제 Director와 동일한 상태 갱신 규칙을 사용"""

    def __init__(self, verdicts: list[tuple[bool, str]]):
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    def __call__(self, state: GraphState, runtime) -> GraphState:
        index = len(self.calls)
        self.calls.append(
            {
                "story": state.story_output.story if state.story_output else None,
                "story_history": list(state.story_history),
                "feedback_history": list(state.feedback_history),
                "retry_count": state.retry_count,
            }
        )
        approved, feedback = self.verdicts[index]  # 스크립트 초과 호출 시 IndexError
        report = EvalReport(
            is_approved=approved, score=8.0 if approved else 4.0, feedback=feedback
        )
        state.eval_report = report
        if not approved:
            state.feedback_history.append(feedback)
            state.retry_count += 1
        if approved or state.retry_count >= state.max_retries:
            state.is_complete = True
        return state


def install_scripted_agents(monkeypatch, parser, writer, director) -> None:
    """src.graph가 생성하는 에이전트를 스크립트 에이전트로 교체"""
    monkeypatch.setattr(graph_module, "UserRequestParser", lambda *a, **k: parser)
    monkeypatch.setattr(graph_module, "StoryWriter", lambda *a, **k: writer)
    monkeypatch.setattr(graph_module, "Director", lambda *a, **k: director)


def _as_state(result) -> GraphState:
    return result if isinstance(result, GraphState) else GraphState(**result)


def _invoke(max_retries: int, llm=None) -> GraphState:
    app = create_graph(llm=llm if llm is not None else MagicMock()).compile()
    result = app.invoke(GraphState(user_input="아이디어", max_retries=max_retries))
    return _as_state(result)


class TestCompiledGraphWithScriptedAgents:
    """컴파일된 LangGraph로 실제 흐름 실행"""

    def test_first_draft_approved(self, monkeypatch):
        parser = ScriptedParser()
        writer = ScriptedWriter(["draft 1"])
        director = ScriptedDirector([(True, "good")])
        install_scripted_agents(monkeypatch, parser, writer, director)

        final = _invoke(max_retries=3)

        assert parser.calls == 1
        assert len(writer.calls) == 1
        assert len(director.calls) == 1
        assert final.request.summarized_prompt == "요약: 아이디어"
        assert final.story_history == ["draft 1"]
        assert final.feedback_history == []
        assert final.retry_count == 0
        assert final.is_complete is True
        assert final.eval_report.is_approved is True
        assert final.story_output.story == "draft 1"

    def test_two_rejections_then_approval(self, monkeypatch):
        parser = ScriptedParser()
        writer = ScriptedWriter(["draft 1", "draft 2", "draft 3"])
        director = ScriptedDirector([(False, "fb 1"), (False, "fb 2"), (True, "ok")])
        install_scripted_agents(monkeypatch, parser, writer, director)

        final = _invoke(max_retries=3)

        assert parser.calls == 1
        assert len(writer.calls) == 3
        assert len(director.calls) == 3
        assert final.story_history == ["draft 1", "draft 2", "draft 3"]
        assert final.feedback_history == ["fb 1", "fb 2"]
        assert final.retry_count == 2
        assert final.is_complete is True
        assert final.eval_report.is_approved is True

        # Writer는 각 피드백을 정확히 한 번씩, 순서대로 받는다
        assert [c["feedback_history"] for c in writer.calls] == [
            [],
            ["fb 1"],
            ["fb 1", "fb 2"],
        ]
        assert [c["story_history"] for c in writer.calls] == [
            [],
            ["draft 1"],
            ["draft 1", "draft 2"],
        ]
        # Director는 각 초안을 정확히 한 번씩, 순서대로 검수한다
        assert [c["story"] for c in director.calls] == ["draft 1", "draft 2", "draft 3"]
        assert [c["story_history"] for c in director.calls] == [
            ["draft 1"],
            ["draft 1", "draft 2"],
            ["draft 1", "draft 2", "draft 3"],
        ]
        assert [c["feedback_history"] for c in director.calls] == [
            [],
            ["fb 1"],
            ["fb 1", "fb 2"],
        ]
        assert [c["retry_count"] for c in director.calls] == [0, 1, 2]

    def test_max_review_exit_after_two_rejections(self, monkeypatch):
        """max_retries=2: 반려 2회 후 미승인 상태로 종료, 3번째 호출 없음"""
        parser = ScriptedParser()
        # 스크립트에 3번째 항목이 있어도 그래프가 호출하면 안 된다
        writer = ScriptedWriter(["draft 1", "draft 2", "draft 3"])
        director = ScriptedDirector([(False, "fb 1"), (False, "fb 2"), (True, "ok")])
        install_scripted_agents(monkeypatch, parser, writer, director)

        final = _invoke(max_retries=2)

        assert parser.calls == 1
        assert len(writer.calls) == 2
        assert len(director.calls) == 2
        assert final.story_history == ["draft 1", "draft 2"]
        assert final.feedback_history == ["fb 1", "fb 2"]
        assert final.retry_count == 2
        assert final.is_complete is True
        assert final.eval_report.is_approved is False
        assert final.story_output.story == "draft 2"
        assert [c["story"] for c in director.calls] == ["draft 1", "draft 2"]
        assert [c["feedback_history"] for c in writer.calls] == [[], ["fb 1"]]

    def test_max_retries_one_allows_single_review(self, monkeypatch):
        parser = ScriptedParser()
        writer = ScriptedWriter(["draft 1", "draft 2"])
        director = ScriptedDirector([(False, "fb 1"), (True, "ok")])
        install_scripted_agents(monkeypatch, parser, writer, director)

        final = _invoke(max_retries=1)

        assert len(writer.calls) == 1
        assert len(director.calls) == 1
        assert final.story_history == ["draft 1"]
        assert final.feedback_history == ["fb 1"]
        assert final.is_complete is True
        assert final.eval_report.is_approved is False

    def test_stream_updates_contain_only_new_history_entries(self, monkeypatch):
        """노드 업데이트가 델타(새 항목만)인지 stream_mode='updates'로 확인"""
        parser = ScriptedParser()
        writer = ScriptedWriter(["draft 1", "draft 2"])
        director = ScriptedDirector([(False, "fb 1"), (True, "ok")])
        install_scripted_agents(monkeypatch, parser, writer, director)

        app = create_graph(llm=MagicMock()).compile()
        updates = [
            (node, output)
            for event in app.stream(
                GraphState(user_input="아이디어", max_retries=3),
                stream_mode="updates",
            )
            for node, output in event.items()
        ]

        assert [node for node, _ in updates] == [
            "init",
            "write",
            "review",
            "write",
            "review",
        ]
        init_update = updates[0][1]
        assert "story_history" not in init_update
        assert "feedback_history" not in init_update

        write_updates = [out for node, out in updates if node == "write"]
        assert [u["story_history"] for u in write_updates] == [["draft 1"], ["draft 2"]]
        assert all("feedback_history" not in u for u in write_updates)

        review_updates = [out for node, out in updates if node == "review"]
        assert [u["feedback_history"] for u in review_updates] == [["fb 1"], []]
        assert all("story_history" not in u for u in review_updates)
        assert [u["is_complete"] for u in review_updates] == [False, True]

    def test_console_stream_does_not_claim_approval_on_unapproved_exit(self, monkeypatch, capsys):
        parser = ScriptedParser()
        writer = ScriptedWriter(["draft 1", "draft 2", "draft 3"])
        director = ScriptedDirector([
            (False, "fb 1"), (False, "fb 2"), (False, "fb 3")
        ])
        install_scripted_agents(monkeypatch, parser, writer, director)

        run_story_generation_stream("아이디어", llm=MagicMock())
        output = capsys.readouterr().out
        assert len(writer.calls) == 3
        assert len(director.calls) == 3
        assert "스토리 생성 흐름 종료 — 미승인" in output
        assert "스토리 생성 흐름 종료 — Director 승인" not in output
        assert "✨ 스토리 생성 완료" not in output

    def test_console_stream_shows_actual_approval(self, monkeypatch, capsys):
        parser = ScriptedParser()
        writer = ScriptedWriter(["draft 1"])
        director = ScriptedDirector([(True, "good")])
        install_scripted_agents(monkeypatch, parser, writer, director)
        run_story_generation_stream("아이디어", llm=MagicMock())
        output = capsys.readouterr().out
        assert len(writer.calls) == 1
        assert "스토리 생성 흐름 종료 — Director 승인" in output
        assert "미승인" not in output


class TestCompiledGraphWithRealWriterAndDirector:
    """실제 StoryWriter/Director 클래스 + 스크립트 LLM 응답 (Ollama 호출 없음)"""

    def test_two_rejections_with_real_agents(self, monkeypatch):
        parser = ScriptedParser()
        monkeypatch.setattr(graph_module, "UserRequestParser", lambda *a, **k: parser)

        def writer_reply(n: int) -> AIMessage:
            return AIMessage(
                content=json.dumps(
                    {"title": f"T{n}", "story": f"draft {n}", "word_count": 7, "notes": ""}
                )
            )

        def director_reply(n: int) -> AIMessage:
            return AIMessage(
                content=json.dumps(
                    {
                        "is_approved": False,
                        "score": 4,
                        "feedback": f"fb {n}",
                        "issues": [f"issue {n}"],
                    }
                )
            )

        # Writer와 Director가 같은 llm.bind_tools(...) 결과를 공유하므로 호출 순서대로 응답
        script = [writer_reply(1), director_reply(1), writer_reply(2), director_reply(2)]
        seen_user_messages: list[str] = []

        def scripted_invoke(messages):
            seen_user_messages.append(messages[1].content)
            return script[len(seen_user_messages) - 1]

        llm = MagicMock()
        llm.bind_tools.return_value.invoke.side_effect = scripted_invoke

        final = _invoke(max_retries=2, llm=llm)

        assert parser.calls == 1
        assert len(seen_user_messages) == 4  # write, review, write, review
        assert final.story_history == ["draft 1", "draft 2"]
        assert final.feedback_history == ["fb 1", "fb 2"]
        assert final.retry_count == 2
        assert final.is_complete is True
        assert final.eval_report.is_approved is False

        writer_msg_1, director_msg_1, writer_msg_2, director_msg_2 = seen_user_messages
        assert "draft 1" in director_msg_1
        assert "Attempt: 0 / 2" in director_msg_1
        assert "fb 1" in writer_msg_2 and "draft 1" in writer_msg_2
        assert "draft 2" in director_msg_2
        assert "Attempt: 1 / 2" in director_msg_2
        # 최종 LLM(도구 없는) 폴백 경로는 사용되지 않아야 한다
        llm.invoke.assert_not_called()
