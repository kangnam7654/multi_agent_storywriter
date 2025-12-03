"""StoryWriter 테스트"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.story_writer import StoryWriter
from src.schemas.state import GraphState, RefinedRequest, StoryOutput


class TestStoryWriter:
    """StoryWriter 테스트"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM fixture"""
        mock = MagicMock()
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def writer(self, mock_llm):
        """StoryWriter fixture"""
        return StoryWriter(llm=mock_llm, system_prompt="테스트 시스템 프롬프트")

    @pytest.fixture
    def sample_state(self):
        """샘플 GraphState fixture"""
        return GraphState(
            user_input="용사 이야기",
            request=RefinedRequest(
                summarized_prompt="용사가 드래곤을 물리치는 이야기",
                genre="판타지",
                style="소설",
                length="Medium",
            ),
        )

    def test_init(self, mock_llm):
        """초기화 테스트"""
        writer = StoryWriter(llm=mock_llm, system_prompt="Test")
        assert writer.llm == mock_llm
        assert writer.system_prompt == "Test"

    def test_build_user_message_includes_request_info(self, writer, sample_state):
        """유저 메시지에 요청 정보가 포함되는지 테스트"""
        message = writer._build_user_message(sample_state)

        assert "용사가 드래곤을 물리치는 이야기" in message
        assert "판타지" in message
        assert "소설" in message
        assert "Medium" in message

    def test_build_user_message_includes_feedback(self, writer, sample_state):
        """피드백이 있을 때 메시지에 포함되는지 테스트"""
        sample_state.feedback_history = ["더 긴장감 있게 써주세요"]
        message = writer._build_user_message(sample_state)

        assert "더 긴장감 있게 써주세요" in message

    def test_build_user_message_includes_previous_story(self, writer, sample_state):
        """이전 스토리가 있을 때 메시지에 포함되는지 테스트"""
        # story_history에 이전 스토리 추가
        sample_state.story_history = ["이전 스토리 내용"]
        sample_state.feedback_history = ["수정 필요"]
        message = writer._build_user_message(sample_state)

        assert "이전 스토리 내용" in message

    def test_parse_response_creates_story_output(self, writer):
        """응답 파싱이 StoryOutput을 생성하는지 테스트"""
        response = """{
            "title": "용사의 여정",
            "story": "옛날 옛적에 용사가 있었습니다.",
            "word_count": 20,
            "notes": "판타지 스타일로 작성"
        }"""

        result = writer._parse_response(response)

        assert isinstance(result, StoryOutput)
        assert result.title == "용사의 여정"
        assert result.story == "옛날 옛적에 용사가 있었습니다."
        assert result.word_count == 20
        assert result.notes == "판타지 스타일로 작성"

    def test_parse_response_handles_json_in_code_block(self, writer):
        """코드 블록 내 JSON 파싱 테스트"""
        response = """Here is the story:
```json
{
    "title": "테스트 제목",
    "story": "테스트 스토리",
    "word_count": 10,
    "notes": ""
}
```"""

        result = writer._parse_response(response)

        assert result.title == "테스트 제목"
        assert result.story == "테스트 스토리"

    def test_parse_response_returns_default_on_invalid_json(self, writer):
        """잘못된 JSON일 때 기본값 반환 테스트"""
        response = "이것은 JSON이 아닙니다"

        result = writer._parse_response(response)

        assert isinstance(result, StoryOutput)
        # 원본 텍스트가 story에 들어감
        assert "이것은 JSON이 아닙니다" in result.story

    def test_call_updates_state(self, writer, mock_llm, sample_state):
        """__call__이 state를 업데이트하는지 테스트"""
        mock_response = AIMessage(
            content="""{
            "title": "테스트",
            "story": "테스트 스토리",
            "word_count": 10,
            "notes": ""
        }"""
        )
        mock_response.tool_calls = []
        writer.llm_with_tools.invoke.return_value = mock_response

        result = writer(sample_state, runtime=None)

        assert result.story_output is not None
        assert result.story_output.title == "테스트"
        assert len(result.story_history) > 0


class TestStoryWriterEdgeCases:
    """StoryWriter 엣지 케이스 테스트"""

    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def writer(self, mock_llm):
        return StoryWriter(llm=mock_llm, system_prompt="Test")

    def test_empty_request(self, writer):
        """빈 요청 처리 테스트"""
        state = GraphState(
            user_input="",
            request=RefinedRequest(),
        )

        message = writer._build_user_message(state)

        # 빈 요청이어도 메시지 생성됨
        assert message is not None

    def test_story_history_appended(self, writer, mock_llm):
        """스토리 히스토리 추가 테스트"""
        state = GraphState(
            user_input="테스트",
            request=RefinedRequest(summarized_prompt="테스트"),
            story_history=[],
        )

        mock_response = AIMessage(
            content='{"title":"t","story":"s","word_count":1,"notes":""}'
        )
        mock_response.tool_calls = []
        writer.llm_with_tools.invoke.return_value = mock_response

        result = writer(state, runtime=None)

        # story_history에 새 스토리 추가됨
        assert len(result.story_history) == 1
        assert result.story_history[0] == "s"
