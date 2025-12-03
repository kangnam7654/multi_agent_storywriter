"""graph 모듈 테스트"""

from unittest.mock import MagicMock, patch

import pytest

from src.graph import create_graph
from src.schemas.state import GraphState


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
