import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from src.agents.tools.search_lorebook import search_lorebook
from src.schemas.state import GraphState, StoryOutput


class StoryWriter:
    """스토리 작성 에이전트"""

    def __init__(self, llm: ChatOllama, system_prompt: str = ""):
        self.llm_with_tools = llm.bind_tools([search_lorebook])
        self.system_prompt = system_prompt

    def __call__(self, state: GraphState, runtime) -> GraphState:
        """스토리 작성 실행"""
        user_message = self._build_user_message(state)

        # 에이전트별 독립적인 메시지 리스트 사용 (context 크기 제한)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]

        # Tool call 처리를 위한 반복 (최대 3회)
        max_tool_iterations = 3
        response_text = ""
        ai_message = None

        for iteration in range(max_tool_iterations):
            ai_message = self.llm_with_tools.invoke(messages)
            messages.append(ai_message)

            # 로그 출력
            content_preview = ai_message.content[:100] if ai_message.content else ""
            print(
                f"🔍 Story Writer {iteration+1}차 응답 content: '{content_preview}'..."
            )

            if ai_message.tool_calls:
                print(
                    f"🔍 Story Writer {iteration+1}차 응답 tool_calls: {ai_message.tool_calls}"
                )
                for tool_call in ai_message.tool_calls:
                    # 함수 실행 (Lorebook 검색)
                    if tool_call["name"] == "search_lorebook":
                        tool_result = search_lorebook.invoke(tool_call)
                        # 검색 결과 크기 제한 (최대 1000자)
                        tool_result_str = str(tool_result)[:1000]
                        print(f"📚 Lorebook 검색 결과: {tool_result_str[:200]}...")

                        # 결과를 메시지에 추가 (ToolMessage)
                        messages.append(
                            ToolMessage(
                                content=tool_result_str, tool_call_id=tool_call["id"]
                            )
                        )
                # tool call이 있으면 계속 반복
                continue
            else:
                # tool call이 없으면 응답 텍스트 사용
                response_text = ai_message.content
                break

        # JSON 파싱 (response_text가 문자열인지 확인)
        response_preview = response_text[:200] if response_text else ""
        print(f"🔍 Story Writer 최종 응답: '{response_preview}'...")

        if isinstance(response_text, str) and response_text.strip():
            print("📝 Story Writer 응답 파싱 중...")
            story_output = self._parse_response(response_text)
        else:
            print(
                f"⚠️ Story Writer 응답이 비어있거나 문자열이 아닙니다: {type(response_text)}"
            )
            story_output = StoryOutput(
                title="제목 없음",
                story=str(response_text) if response_text else "스토리 생성 실패",
                word_count=len(str(response_text)) if response_text else 0,
                notes="응답 형식 오류",
            )
        state.story_output = story_output
        state.story_history.append(story_output.story)
        return state

    def _build_user_message(self, state: GraphState) -> str:
        """유저 메시지 구성"""
        parts = []

        # 이전 피드백이 있으면 추가
        if len(state.feedback_history) > 0:
            latest_feedback = state.feedback_history[-1]
            parts.append("## 마지막 리뷰 피드백")
            parts.append(latest_feedback)
            parts.append("피드백을 반영하여 스토리를 수정해 주세요.")

        if len(state.story_history) > 0:
            latest_story = state.story_history[-1]
            parts.append("## 이전 스토리 버전")
            parts.append(latest_story)
            parts.append("이전 버전을 참고하여 스토리를 수정해 주세요.")

        # 스토리 요청
        if state.request:
            parts.append("## Story Request")
            if state.request.summarized_prompt:
                parts.append(f"\nPrompt: {state.request.summarized_prompt}")
            if state.request.genre:
                parts.append(f"\nGenre: {state.request.genre}")
            if state.request.style:
                parts.append(f"\nStyle: {state.request.style}")
            if state.request.length:
                parts.append(f"\nLength: {state.request.length}")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> StoryOutput:
        """LLM 응답을 StoryOutput으로 파싱"""
        try:
            # JSON 블록 추출 시도
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            data = json.loads(json_str)
            return StoryOutput(
                title=data.get("title", ""),
                story=data.get("story", ""),
                word_count=int(data.get("word_count", 0)),
                notes=data.get("notes", ""),
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 파싱 실패 시 원본 텍스트를 스토리로 사용
            return StoryOutput(
                title="제목 없음",
                story=response,
                word_count=len(response),
                notes=f"JSON 파싱 실패: {str(e)}",
            )

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt
