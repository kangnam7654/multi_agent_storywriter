"""BaseAgent 테스트"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.schemas.state import GraphState


class DummyOutput(BaseModel):
    """테스트용 출력 스키마"""

    result: str = ""


class ConcreteAgent(BaseAgent):
    """테스트용 구체 에이전트"""

    def __call__(self, state: GraphState, runtime) -> GraphState:
        user_message = self._build_user_message(state)
        messages = self._create_messages(user_message)
        response_text = self._handle_tool_calls(messages)
        output = self._parse_response(response_text)
        state.is_complete = True
        return state

    def _build_user_message(self, state: GraphState) -> str:
        return f"User input: {state.user_input}"

    def _parse_response(self, response: str) -> DummyOutput:
        return DummyOutput(result=response)


class TestBaseAgent:
    """BaseAgent 추상 클래스 테스트"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM fixture"""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_llm):
        """ConcreteAgent fixture"""
        return ConcreteAgent(llm=mock_llm, system_prompt="Test system prompt")

    @pytest.fixture
    def sample_state(self):
        """샘플 GraphState fixture"""
        return GraphState(user_input="테스트 입력")

    def test_init(self, mock_llm):
        """초기화 테스트"""
        agent = ConcreteAgent(llm=mock_llm, system_prompt="Test prompt")
        assert agent.llm == mock_llm
        assert agent.system_prompt == "Test prompt"

    def test_init_with_empty_system_prompt(self, mock_llm):
        """빈 시스템 프롬프트로 초기화 테스트"""
        agent = ConcreteAgent(llm=mock_llm, system_prompt="")
        assert agent.system_prompt == ""

    def test_create_messages(self, agent):
        """메시지 생성 테스트"""
        messages = agent._create_messages("Hello")

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Test system prompt"
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hello"

    def test_set_system_prompt(self, agent):
        """시스템 프롬프트 설정 테스트"""
        agent.set_system_prompt("New prompt")
        assert agent.get_system_prompt() == "New prompt"

    def test_get_system_prompt(self, agent):
        """시스템 프롬프트 반환 테스트"""
        assert agent.get_system_prompt() == "Test system prompt"


class TestBaseAgentJsonExtraction:
    """BaseAgent JSON 추출 테스트"""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_llm):
        return ConcreteAgent(llm=mock_llm, system_prompt="Test")

    def test_extract_json_direct(self, agent):
        """직접 JSON 추출 테스트"""
        content = '{"key": "value"}'
        result = agent._extract_json(content)

        assert result == {"key": "value"}

    def test_extract_json_with_json_code_block(self, agent):
        """```json 코드 블록 내 JSON 추출 테스트"""
        content = """Here is the result:
```json
{"key": "value"}
```
Done."""
        result = agent._extract_json(content)

        assert result == {"key": "value"}

    def test_extract_json_with_plain_code_block(self, agent):
        """``` 코드 블록 내 JSON 추출 테스트"""
        content = """Result:
```
{"key": "value"}
```"""
        result = agent._extract_json(content)

        assert result == {"key": "value"}

    def test_extract_json_nested(self, agent):
        """중첩 JSON 추출 테스트"""
        content = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = agent._extract_json(content)

        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_extract_json_invalid_raises_error(self, agent):
        """잘못된 JSON 추출 시 에러 테스트"""
        with pytest.raises(Exception):  # json.JSONDecodeError
            agent._extract_json("not json at all")


class TestBaseAgentToolHandling:
    """BaseAgent 도구 호출 처리 테스트"""

    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def agent(self, mock_llm):
        return ConcreteAgent(llm=mock_llm, system_prompt="Test")

    def test_handle_tool_calls_no_tools(self, agent):
        """도구 호출이 없을 때 테스트"""
        # tool_calls가 없는 응답
        mock_response = AIMessage(content="Final response")
        mock_response.tool_calls = []
        agent.llm_with_tools.invoke.return_value = mock_response

        messages = agent._create_messages("Hello")
        result = agent._handle_tool_calls(messages)

        assert result == "Final response"

    def test_handle_tool_calls_with_search(self, agent):
        """search_lorebook 도구 호출 테스트"""
        # 첫 번째 호출: tool call 포함
        tool_call = {
            "id": "call_123",
            "name": "search_lorebook",
            "args": {"query": "스카이림"},
        }
        first_response = AIMessage(content="", tool_calls=[tool_call])

        # 두 번째 호출: 최종 응답
        final_response = AIMessage(content="스카이림에 대한 이야기입니다.")
        final_response.tool_calls = []

        agent.llm_with_tools.invoke.side_effect = [first_response, final_response]

        with patch("src.agents.base.search_lorebook") as mock_search:
            mock_search.invoke.return_value = "스카이림은 노르드의 땅입니다."

            messages = agent._create_messages("스카이림 이야기")
            result = agent._handle_tool_calls(messages)

        assert result == "스카이림에 대한 이야기입니다."
        mock_search.invoke.assert_called_once()


class TestBaseAgentCallable:
    """BaseAgent __call__ 메서드 테스트"""

    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def sample_state(self):
        return GraphState(user_input="테스트")

    def test_call_returns_updated_state(self, mock_llm, sample_state):
        """__call__이 업데이트된 state를 반환하는지 테스트"""
        agent = ConcreteAgent(llm=mock_llm, system_prompt="Test")

        mock_response = AIMessage(content="Response")
        mock_response.tool_calls = []
        agent.llm_with_tools.invoke.return_value = mock_response

        result = agent(sample_state, runtime=None)

        assert result.is_complete is True

    def test_call_invokes_llm(self, mock_llm, sample_state):
        """__call__이 LLM을 호출하는지 테스트"""
        agent = ConcreteAgent(llm=mock_llm, system_prompt="Test")

        mock_response = AIMessage(content="Response")
        mock_response.tool_calls = []
        agent.llm_with_tools.invoke.return_value = mock_response

        agent(sample_state, runtime=None)

        agent.llm_with_tools.invoke.assert_called()
