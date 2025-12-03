"""스토리 검수 에이전트"""

from src.agents.base import BaseAgent
from src.schemas.state import EvalReport, GraphState


class Director(BaseAgent):
    """스토리 검수 에이전트 (Director)"""

    def __call__(self, state: GraphState, runtime) -> GraphState:
        """스토리 검수 실행"""
        user_message = self._build_user_message(state)
        messages = self._create_messages(user_message)

        # Tool call 처리 (Director는 tool 검색 후 최종 응답까지 받아야 함)
        response_text = self._handle_tool_calls(messages, max_iterations=4)

        # 응답 파싱
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

        # 결과 반환
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
            data = self._extract_json(response)
            return EvalReport(
                is_approved=data.get("is_approved", False),
                score=float(data.get("score", 0.0)),
                feedback=data.get("feedback", ""),
                issues=data.get("issues", []),
            )
        except Exception as e:
            # 파싱 실패 시 기본값 반환 (불합격 처리)
            return EvalReport(
                is_approved=False,
                score=0.0,
                feedback=f"Failed to parse evaluation response: {str(e)}. Raw: {response[:500]}",
                issues=["Evaluation parsing failed"],
            )
