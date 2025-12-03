import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from src.agents.tools.search_lorebook import search_lorebook
from src.schemas.state import EvalReport, GraphState


class Director:
    """스토리 검수 에이전트 (Director)"""

    def __init__(self, llm: ChatOllama, system_prompt: str = ""):
        self.llm_with_tools = llm.bind_tools([search_lorebook])
        self.system_prompt = system_prompt

    def __call__(self, state: GraphState, runtime) -> GraphState:
        """스토리 검수 실행"""
        user_message = self._build_user_message(state)

        # 에이전트별 독립적인 메시지 리스트 사용 (context 크기 제한)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]

        ai_message = self.llm_with_tools.invoke(messages)
        messages.append(ai_message)

        print(f"🔍 Director 1차 응답 content: '{ai_message.content}'")
        print(f"🔍 Director 1차 응답 tool_calls: {ai_message.tool_calls}")

        if ai_message.tool_calls:
            print(f"🕵️ Director가 설정을 검색합니다: {ai_message.tool_calls}")
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

            final_response = self.llm_with_tools.invoke(messages)
            response_text = final_response.content
        else:
            # 도구를 안 썼으면 바로 결과 사용
            response_text = ai_message.content

        # 5. JSON 파싱 (response_text 타입 확인)
        print(f"🔍 Director 원본 응답: '{response_text}'")
        print(
            f"🔍 Director 응답 타입: {type(response_text)}, 길이: {len(response_text) if response_text else 0}"
        )

        if not response_text or (
            isinstance(response_text, str) and not response_text.strip()
        ):
            print("⚠️ Director 응답이 비어있습니다.")
            eval_report = EvalReport(
                is_approved=False,
                score=0.0,
                feedback="Director 응답 오류: 응답이 비어있습니다.",
                issues=["응답 형식 오류"],
            )
        elif not isinstance(response_text, str):
            print(f"⚠️ Director 응답이 문자열이 아닙니다: {type(response_text)}")
            eval_report = EvalReport(
                is_approved=False,
                score=0.0,
                feedback=f"Director 응답 오류: 응답 타입이 {type(response_text)}입니다.",
                issues=["응답 형식 오류"],
            )
        else:
            print(f"🔍 Director 응답 파싱 중: {response_text[:200]}...")
            eval_report = self._parse_response(response_text)

        # 결과 반환 (기존 로직 동일)
        state.eval_report = eval_report
        if not eval_report.is_approved:
            state.feedback_history.append(eval_report.feedback)
            state.retry_count += 1
        if eval_report.is_approved:
            state.is_complete = True
            print("✅ 스토리가 승인되었습니다.")
        elif state.retry_count >= state.max_retries:
            state.is_complete = True
            print("⚠️ 최대 재시도 횟수에 도달하여 종료합니다.")
        return state

    def _build_user_message(self, state: GraphState) -> str:
        """유저 메시지 구성"""
        parts = []

        # 원본 요청 정보
        if state.request:
            parts.append("## Request")
            if state.request.summarized_prompt:
                parts.append(f"Prompt: {state.request.summarized_prompt}")
            if state.request.genre:
                parts.append(f"Genre: {state.request.genre}")
            if state.request.style:
                parts.append(f"Style: {state.request.style}")
            if state.request.length:
                parts.append(f"Length: {state.request.length}")
            parts.append("")

        # 검수할 스토리
        parts.append("## Story to Review")
        if state.story_output and state.story_output.story:
            parts.append(state.story_output.story)
        else:
            parts.append("(스토리 없음)")
        parts.append("")

        # 재시도 정보
        parts.append("## Review Info")
        parts.append(f"Attempt: {state.retry_count} / {state.max_retries}")
        return "\n".join(parts)

    def _parse_response(self, response: str) -> EvalReport:
        """LLM 응답을 EvalReport로 파싱"""
        try:
            # JSON 블록 추출 시도
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            data = json.loads(json_str)
            return EvalReport(
                is_approved=data.get("is_approved", False),
                score=float(data.get("score", 0.0)),
                feedback=data.get("feedback", ""),
                issues=data.get("issues", []),
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 파싱 실패 시 기본값 반환 (불합격 처리)
            return EvalReport(
                is_approved=False,
                score=0.0,
                feedback=f"Failed to parse evaluation response: {str(e)}. Raw response: {response[:500]}",
                issues=["Evaluation parsing failed"],
            )

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt
