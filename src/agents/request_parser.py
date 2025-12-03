import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.schemas.state import GraphState, RefinedRequest

logger = logging.getLogger(__name__)


class RequestParserError(Exception):
    """Request parsing 중 발생하는 에러"""

    pass


class UserRequestParser:
    """사용자 요청을 구조화된 RefinedRequest로 변환하는 파서"""

    MAX_RETRIES = 2  # JSON 파싱 실패 시 재시도 횟수

    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.system_prompt = self._system_prompt()

    def __call__(self, state: GraphState, runtime) -> GraphState:
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{input}")]
        )
        chain = prompt | self.llm

        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                result = chain.invoke({"input": state.user_input})
                result_json = self._extract_json(result.content)

                if result_json is None:
                    raise RequestParserError("JSON 추출 실패")

                request = self._create_refined_request(result_json)
                state.request = request
                logger.info(
                    f"Request parsing 성공: {request.summarized_prompt[:50]}..."
                )
                return state

            except (json.JSONDecodeError, RequestParserError, KeyError) as e:
                last_error = e
                logger.warning(
                    f"Parsing 시도 {attempt + 1}/{self.MAX_RETRIES + 1} 실패: {e}"
                )
                continue

        # 모든 재시도 실패 시 기본값으로 폴백
        logger.error(f"모든 parsing 시도 실패, 기본값 사용: {last_error}")
        state.request = self._create_fallback_request(state.user_input)
        return state

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        """LLM 응답에서 JSON을 추출"""
        if not content or not content.strip():
            return None

        # 1차 시도: 직접 파싱
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # 2차 시도: 코드 블록 내 JSON 추출
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, content)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3차 시도: { } 사이의 내용 추출
        brace_pattern = r"\{[\s\S]*\}"
        match = re.search(brace_pattern, content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _create_refined_request(self, data: dict[str, Any]) -> RefinedRequest:
        """딕셔너리에서 RefinedRequest 생성"""
        if "summarized_prompt" not in data:
            raise KeyError("필수 키 'summarized_prompt'가 없습니다")

        return RefinedRequest(
            summarized_prompt=data["summarized_prompt"],
            genre=data.get("genre", "판타지"),
            style=data.get("style", "소설"),
            length=data.get("length", "Medium"),
        )

    def _create_fallback_request(self, user_input: str) -> RefinedRequest:
        """파싱 실패 시 기본 RefinedRequest 생성"""
        return RefinedRequest(
            summarized_prompt=user_input,  # 원본 입력을 그대로 사용
            genre="판타지",
            style="소설",
            length="Medium",
        )

    def _system_prompt(self) -> str:
        return """
당신은 사용자의 모호한 스토리 아이디어를 분석하여 구조화된 데이터로 변환하는 '전문 스토리 기획 AI'입니다.
당신의 목표는 사용자 입력에서 핵심 요소를 추출하여 반드시 '순수한 JSON' 형식으로만 응답하는 것입니다.

[지침]
1. summarized_prompt: 사용자의 요청을 1-2문장의 명확한 지시문으로 요약/재작성하십시오.
2. genre: 명시되지 않았다면 문맥을 파악하여 결정하되, 불확실하면 "판타지"로 설정하십시오.
3. style: 글의 어조나 서술 방식(예: 소설, 대본, 시, 뉴스 기사 등). 기본값은 "소설"입니다.
4. length: 스토리의 분량. (Short: 500자 이내, Medium: 1000자 내외, Long: 2000자 이상). 기본값은 "Medium"입니다.

[제약 사항]
- **반드시 JSON 형식의 텍스트만 출력하십시오.**
- 설명, 인사말, 마크다운 코드 블록(```json) 등을 절대 포함하지 마십시오.
- JSON의 키(Key)는 반드시 아래 예시와 동일해야 합니다.

[출력 예시]
{{
    "summarized_prompt": "전설의 검을 찾아 떠나는 용사가 숲에서 길을 잃고 고대 유적을 발견하는 이야기",
    "genre": "판타지",
    "style": "소설",
    "length": "Medium"
}}
"""
