"""UserRequestParser 테스트"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.request_parser import RequestParserError, UserRequestParser
from src.schemas.state import GraphState, RefinedRequest


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


class TestUserRequestParserExtractJson:
    """UserRequestParser JSON 추출 테스트"""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def parser(self, mock_llm):
        return UserRequestParser(llm=mock_llm)

    def test_extract_json_direct(self, parser):
        """직접 JSON 추출 테스트"""
        content = '{"summarized_prompt": "테스트", "genre": "판타지"}'
        result = parser._extract_json(content)

        assert result["summarized_prompt"] == "테스트"
        assert result["genre"] == "판타지"

    def test_extract_json_with_code_block(self, parser):
        """코드 블록 내 JSON 추출 테스트"""
        content = """결과입니다:
```json
{"summarized_prompt": "테스트", "genre": "SF"}
```"""
        result = parser._extract_json(content)

        assert result["summarized_prompt"] == "테스트"
        assert result["genre"] == "SF"

    def test_extract_json_with_surrounding_text(self, parser):
        """텍스트로 둘러싸인 JSON 추출 테스트"""
        content = '요청을 분석한 결과: {"summarized_prompt": "테스트"} 입니다.'
        result = parser._extract_json(content)

        assert result["summarized_prompt"] == "테스트"

    def test_extract_json_empty_returns_none(self, parser):
        """빈 내용에서 None 반환 테스트"""
        assert parser._extract_json("") is None
        assert parser._extract_json("   ") is None

    def test_extract_json_invalid_returns_none(self, parser):
        """잘못된 JSON에서 None 반환 테스트"""
        assert parser._extract_json("not json") is None


class TestUserRequestParserFallback:
    """UserRequestParser 폴백 테스트"""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def parser(self, mock_llm):
        return UserRequestParser(llm=mock_llm)

    def test_create_fallback_request(self, parser):
        """폴백 요청 생성 테스트"""
        user_input = "테스트 입력"
        result = parser._create_fallback_request(user_input)

        assert isinstance(result, RefinedRequest)
        assert result.summarized_prompt == user_input
        assert result.genre == "판타지"
        assert result.style == "소설"
        assert result.length == "Medium"

    def test_create_refined_request(self, parser):
        """RefinedRequest 생성 테스트"""
        data = {
            "summarized_prompt": "용사 이야기",
            "genre": "판타지",
            "style": "소설",
            "length": "Short",
        }
        result = parser._create_refined_request(data)

        assert result.summarized_prompt == "용사 이야기"
        assert result.genre == "판타지"
        assert result.length == "Short"

    def test_create_refined_request_missing_key_raises(self, parser):
        """필수 키가 없을 때 에러 테스트"""
        data = {"genre": "판타지"}  # summarized_prompt 누락

        with pytest.raises(KeyError):
            parser._create_refined_request(data)

    def test_create_refined_request_with_defaults(self, parser):
        """기본값으로 RefinedRequest 생성 테스트"""
        data = {"summarized_prompt": "테스트"}  # 다른 필드 누락
        result = parser._create_refined_request(data)

        assert result.summarized_prompt == "테스트"
        assert result.genre == "판타지"  # 기본값
        assert result.style == "소설"  # 기본값
        assert result.length == "Medium"  # 기본값


class TestUserRequestParserIntegration:
    """UserRequestParser 통합 테스트"""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    def test_call_success(self, mock_llm):
        """성공적인 호출 테스트"""
        parser = UserRequestParser(llm=mock_llm)
        state = GraphState(user_input="드래곤과 마법사 이야기")

        expected_response = {
            "summarized_prompt": "드래곤과 마법사의 대결 이야기",
            "genre": "판타지",
            "style": "소설",
            "length": "Medium",
        }

        with patch("src.agents.request_parser.ChatPromptTemplate") as mock_template:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = AIMessage(
                content=json.dumps(expected_response)
            )

            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_template.from_messages.return_value = mock_prompt

            result = parser(state, runtime=None)

            assert result.request is not None
            assert result.request.summarized_prompt == "드래곤과 마법사의 대결 이야기"
            assert result.request.genre == "판타지"

    def test_call_fallback_on_invalid_json(self, mock_llm):
        """잘못된 JSON 시 폴백 테스트"""
        parser = UserRequestParser(llm=mock_llm)
        state = GraphState(user_input="테스트 입력")

        with patch("src.agents.request_parser.ChatPromptTemplate") as mock_template:
            mock_chain = MagicMock()
            # 항상 잘못된 응답 반환
            mock_chain.invoke.return_value = AIMessage(content="잘못된 응답")

            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_template.from_messages.return_value = mock_prompt

            result = parser(state, runtime=None)

            # 폴백으로 원본 입력 사용
            assert result.request is not None
            assert result.request.summarized_prompt == "테스트 입력"
            assert result.request.genre == "판타지"  # 기본값
