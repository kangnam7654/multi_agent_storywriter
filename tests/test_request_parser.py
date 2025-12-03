import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.request_parser import UserRequestParser
from src.schemas.state import GraphState


class TestUserRequestParser:
    """UserRequestParser 테스트"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM fixture"""
        return MagicMock()

    @pytest.fixture
    def parser(self, mock_llm):
        """UserRequestParser fixture"""
        return UserRequestParser(llm=mock_llm)

    @pytest.fixture
    def sample_state(self):
        """샘플 GraphState fixture"""
        return GraphState(user_input="용사가 마왕을 물리치는 이야기를 써줘")

    def test_init(self, mock_llm):
        """초기화 테스트"""
        parser = UserRequestParser(llm=mock_llm)
        assert parser.llm == mock_llm
        assert parser.system_prompt is not None
        assert "JSON" in parser.system_prompt

    def test_init_sets_system_prompt(self, parser):
        """시스템 프롬프트 설정 테스트"""
        assert "summarized_prompt" in parser.system_prompt
        assert "genre" in parser.system_prompt
        assert "style" in parser.system_prompt
        assert "length" in parser.system_prompt

    def test_call_raises_error_when_llm_is_none(self, sample_state):
        """LLM이 None일 때 에러 발생 테스트"""
        parser = UserRequestParser(llm=None)
        parser.llm = None  # 명시적으로 None 설정

        with pytest.raises(ValueError, match="LLM not set"):
            parser(sample_state, runtime=None)

    def test_call_invokes_chain_with_user_input(self, parser, mock_llm, sample_state):
        """chain이 user_input으로 호출되는지 테스트"""
        # Mock chain response
        mock_response = AIMessage(
            content=json.dumps(
                {
                    "summarized_prompt": "용사가 마왕을 물리치는 판타지 이야기",
                    "genre": "판타지",
                    "style": "소설",
                    "length": "Medium",
                }
            )
        )

        # Mock the chain
        with patch.object(parser, "llm") as patched_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            patched_llm.__or__ = MagicMock(return_value=mock_chain)

            # __call__에서 prompt | llm 체인을 mock
            with patch(
                "src.agents.request_parser.ChatPromptTemplate"
            ) as mock_prompt_template:
                mock_prompt = MagicMock()
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt_template.from_messages.return_value = mock_prompt

                result = parser(sample_state, runtime=None)

                # chain.invoke가 올바른 input으로 호출되었는지 확인
                mock_chain.invoke.assert_called_once_with(
                    {"input": sample_state.user_input}
                )

    def test_system_prompt_contains_json_format_instructions(self, parser):
        """시스템 프롬프트에 JSON 형식 지침이 포함되어 있는지 테스트"""
        prompt = parser.system_prompt

        # 핵심 지침 포함 여부
        assert "JSON" in prompt
        assert "summarized_prompt" in prompt
        assert "genre" in prompt
        assert "style" in prompt
        assert "length" in prompt

    def test_system_prompt_contains_default_values(self, parser):
        """시스템 프롬프트에 기본값 안내가 포함되어 있는지 테스트"""
        prompt = parser.system_prompt

        # 기본값 안내
        assert "판타지" in prompt  # genre 기본값
        assert "소설" in prompt  # style 기본값
        assert "Medium" in prompt  # length 기본값

    def test_system_prompt_escaped_json_braces(self, parser):
        """시스템 프롬프트의 JSON 예시가 이스케이프되어 있는지 테스트"""
        prompt = parser.system_prompt

        # LangChain 템플릿 변수로 해석되지 않도록 {{ }} 사용
        assert "{{" in prompt
        assert "}}" in prompt


class TestUserRequestParserIntegration:
    """UserRequestParser 통합 테스트 (실제 LLM 호출 없이)"""

    @pytest.fixture
    def mock_llm_with_response(self):
        """응답을 반환하는 Mock LLM"""
        mock_llm = MagicMock()
        return mock_llm

    def test_parse_fantasy_request(self, mock_llm_with_response):
        """판타지 요청 파싱 테스트"""
        parser = UserRequestParser(llm=mock_llm_with_response)
        state = GraphState(user_input="드래곤과 마법사가 싸우는 이야기")

        # Mock response
        expected_response = {
            "summarized_prompt": "드래곤과 마법사의 대결을 그린 판타지 이야기",
            "genre": "판타지",
            "style": "소설",
            "length": "Medium",
        }

        with patch(
            "src.agents.request_parser.ChatPromptTemplate"
        ) as mock_prompt_template:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(
                content=json.dumps(expected_response)
            )

            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_template.from_messages.return_value = mock_prompt

            result = parser(state, runtime=None)

            assert result.content == json.dumps(expected_response)

    def test_parse_scifi_request(self, mock_llm_with_response):
        """SF 요청 파싱 테스트"""
        parser = UserRequestParser(llm=mock_llm_with_response)
        state = GraphState(user_input="우주에서 외계인과 만나는 SF 소설을 써줘")

        expected_response = {
            "summarized_prompt": "우주에서 외계인과의 첫 만남을 그린 SF 이야기",
            "genre": "SF",
            "style": "소설",
            "length": "Medium",
        }

        with patch(
            "src.agents.request_parser.ChatPromptTemplate"
        ) as mock_prompt_template:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(
                content=json.dumps(expected_response)
            )

            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_template.from_messages.return_value = mock_prompt

            result = parser(state, runtime=None)

            assert result.content == json.dumps(expected_response)

    def test_parse_request_with_specific_length(self, mock_llm_with_response):
        """길이가 지정된 요청 파싱 테스트"""
        parser = UserRequestParser(llm=mock_llm_with_response)
        state = GraphState(user_input="짧은 공포 이야기를 써줘")

        expected_response = {
            "summarized_prompt": "짧은 공포 이야기",
            "genre": "공포",
            "style": "소설",
            "length": "Short",
        }

        with patch(
            "src.agents.request_parser.ChatPromptTemplate"
        ) as mock_prompt_template:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(
                content=json.dumps(expected_response)
            )

            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_template.from_messages.return_value = mock_prompt

            result = parser(state, runtime=None)

            parsed = json.loads(result.content)
            assert parsed["length"] == "Short"
            assert parsed["genre"] == "공포"
