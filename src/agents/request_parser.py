import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.schemas.state import GraphState, RefinedRequest


class UserRequestParser:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.system_prompt = self._system_prompt()

    def __call__(self, state: GraphState, runtime):
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{input}")]
        )
        chain = prompt | self.llm
        result = chain.invoke({"input": state.user_input})

        result_json = json.loads(result.content)
        print(result_json)
        request = RefinedRequest(
            summarized_prompt=result_json["summarized_prompt"],
            genre=result_json.get("genre"),
            style=result_json.get("style"),
            length=result_json.get("length"),
        )
        state.request = request
        return state

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
