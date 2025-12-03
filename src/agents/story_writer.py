"""스토리 작성 에이전트"""

from src.agents.base import BaseAgent
from src.schemas.state import GraphState, StoryOutput


class StoryWriter(BaseAgent):
    """스토리 작성 에이전트"""

    def __call__(self, state: GraphState, runtime) -> GraphState:
        """스토리 작성 실행"""
        user_message = self._build_user_message(state)
        messages = self._create_messages(user_message)

        # Tool call 처리
        response_text = self._handle_tool_calls(messages, max_iterations=3)

        # 응답 파싱
        if isinstance(response_text, str) and response_text.strip():
            print("📝 Story Writer 응답 파싱 중...")
            story_output = self._parse_response(response_text)
        else:
            print(f"⚠️ Story Writer 응답이 비어있습니다: {type(response_text)}")
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
            data = self._extract_json(response)
            return StoryOutput(
                title=data.get("title", ""),
                story=data.get("story", ""),
                word_count=int(data.get("word_count", 0)),
                notes=data.get("notes", ""),
            )
        except Exception as e:
            # 파싱 실패 시 원본 텍스트를 스토리로 사용
            return StoryOutput(
                title="제목 없음",
                story=response,
                word_count=len(response),
                notes=f"JSON 파싱 실패: {str(e)}",
            )
