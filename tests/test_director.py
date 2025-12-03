"""Director 테스트"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.director import Director
from src.schemas.state import EvalReport, GraphState, RefinedRequest, StoryOutput


class TestDirector:
    """Director 테스트"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM fixture"""
        mock = MagicMock()
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def director(self, mock_llm):
        """Director fixture"""
        return Director(llm=mock_llm, system_prompt="테스트 시스템 프롬프트")

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
            story_output=StoryOutput(
                title="용사의 여정",
                story="용사가 드래곤을 물리쳤습니다.",
                word_count=15,
                notes="",
            ),
        )

    def test_init(self, mock_llm):
        """초기화 테스트"""
        director = Director(llm=mock_llm, system_prompt="Test")
        assert director.llm == mock_llm
        assert director.system_prompt == "Test"

    def test_build_user_message_includes_story(self, director, sample_state):
        """유저 메시지에 스토리가 포함되는지 테스트"""
        message = director._build_user_message(sample_state)

        assert "용사가 드래곤을 물리쳤습니다" in message

    def test_build_user_message_includes_request_info(self, director, sample_state):
        """유저 메시지에 요청 정보가 포함되는지 테스트"""
        message = director._build_user_message(sample_state)

        assert "판타지" in message
        assert "소설" in message

    def test_parse_response_creates_eval_report(self, director):
        """응답 파싱이 EvalReport을 생성하는지 테스트"""
        response = """{
            "is_approved": true,
            "score": 8.5,
            "feedback": "잘 작성되었습니다.",
            "issues": []
        }"""

        result = director._parse_response(response)

        assert isinstance(result, EvalReport)
        assert result.is_approved is True
        assert result.score == 8.5
        assert result.feedback == "잘 작성되었습니다."
        assert result.issues == []

    def test_parse_response_with_issues(self, director):
        """이슈가 있는 응답 파싱 테스트"""
        response = """{
            "is_approved": false,
            "score": 5.0,
            "feedback": "수정이 필요합니다.",
            "issues": ["문장이 어색합니다", "더 긴장감이 필요합니다"]
        }"""

        result = director._parse_response(response)

        assert result.is_approved is False
        assert result.score == 5.0
        assert len(result.issues) == 2

    def test_parse_response_handles_json_in_code_block(self, director):
        """코드 블록 내 JSON 파싱 테스트"""
        response = """검토 결과:
```json
{
    "is_approved": true,
    "score": 9.0,
    "feedback": "훌륭합니다!",
    "issues": []
}
```"""

        result = director._parse_response(response)

        assert result.is_approved is True
        assert result.score == 9.0

    def test_parse_response_returns_default_on_invalid_json(self, director):
        """잘못된 JSON일 때 기본값 반환 테스트"""
        response = "이것은 JSON이 아닙니다"

        result = director._parse_response(response)

        assert isinstance(result, EvalReport)
        # 파싱 실패 시 불합격 처리
        assert result.is_approved is False
        assert "Failed to parse" in result.feedback

    def test_call_approved_sets_complete(self, director, mock_llm, sample_state):
        """승인 시 is_complete가 True가 되는지 테스트"""
        mock_response = MagicMock()
        mock_response.content = """{
            "is_approved": true,
            "score": 8.0,
            "feedback": "좋습니다",
            "issues": []
        }"""
        mock_response.tool_calls = []
        director.llm_with_tools.invoke.return_value = mock_response

        result = director(sample_state, runtime=None)

        assert result.is_complete is True
        assert result.eval_report.is_approved is True

    def test_call_rejected_sets_incomplete(self, director, mock_llm, sample_state):
        """거부 시 is_complete가 False인지 테스트"""
        mock_response = MagicMock()
        mock_response.content = """{
            "is_approved": false,
            "score": 4.0,
            "feedback": "수정 필요",
            "issues": ["문제점"]
        }"""
        mock_response.tool_calls = []
        director.llm_with_tools.invoke.return_value = mock_response

        result = director(sample_state, runtime=None)

        assert result.is_complete is False
        assert result.eval_report.is_approved is False

    def test_call_max_retries_forces_complete(self, director, mock_llm, sample_state):
        """최대 재시도 도달 시 강제 완료 테스트"""
        sample_state.retry_count = 3
        sample_state.max_retries = 3

        mock_response = MagicMock()
        mock_response.content = """{
            "is_approved": false,
            "score": 4.0,
            "feedback": "수정 필요",
            "issues": ["문제점"]
        }"""
        mock_response.tool_calls = []
        director.llm_with_tools.invoke.return_value = mock_response

        result = director(sample_state, runtime=None)

        # 거부되었어도 max_retries 도달로 완료
        assert result.is_complete is True


class TestDirectorEdgeCases:
    """Director 엣지 케이스 테스트"""

    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def director(self, mock_llm):
        return Director(llm=mock_llm, system_prompt="Test")

    def test_no_story_output(self, director):
        """스토리 출력이 없을 때 테스트"""
        state = GraphState(
            user_input="테스트",
            request=RefinedRequest(summarized_prompt="테스트"),
            story_output=None,
        )

        message = director._build_user_message(state)

        # 빈 스토리여도 메시지 생성됨
        assert message is not None

    def test_feedback_history_updated(self, director, mock_llm):
        """피드백 히스토리가 업데이트되는지 테스트"""
        state = GraphState(
            user_input="테스트",
            request=RefinedRequest(summarized_prompt="테스트"),
            story_output=StoryOutput(title="t", story="s"),
            feedback_history=[],
        )

        mock_response = MagicMock()
        mock_response.content = """{
            "is_approved": false,
            "score": 5.0,
            "feedback": "더 나은 설명이 필요합니다",
            "issues": []
        }"""
        mock_response.tool_calls = []
        director.llm_with_tools.invoke.return_value = mock_response

        result = director(state, runtime=None)

        assert len(result.feedback_history) > 0
        assert "더 나은 설명이 필요합니다" in result.feedback_history[0]
