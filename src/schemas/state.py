from operator import add
from typing import Annotated

from pydantic import BaseModel, Field


class RefinedRequest(BaseModel):
    """사용자 요청 스키마"""

    summarized_prompt: str = Field(
        default="", description="사용자가 입력한 스토리 요청 문장"
    )
    genre: str | None = Field(default=None, description="스토리 장르")
    style: str | None = Field(default=None, description="스토리 스타일")
    length: str | None = Field(default=None, description="스토리 길이")


class StoryOutput(BaseModel):
    """Story Writer의 스토리 출력 스키마"""

    title: str = Field(default="", description="스토리 제목")
    story: str = Field(default="", description="스토리 본문")
    word_count: int = Field(default=0, description="스토리 글자 수")
    notes: str = Field(default="", description="작성 시 참고 사항")


class EvalReport(BaseModel):
    """Director의 스토리 검수 결과"""

    is_approved: bool = Field(..., description="스토리 승인 여부")
    score: float = Field(default=0.0, ge=0.0, le=10.0, description="스토리 점수 (0-10)")
    feedback: str = Field(default="", description="스토리에 대한 피드백")
    issues: list[str] = Field(default_factory=list, description="발견된 문제점 목록")


class GraphState(BaseModel):
    """LangGraph 상태 스키마"""

    user_input: str = Field(default="", description="사용자 입력 텍스트")
    request: RefinedRequest = Field(
        default_factory=RefinedRequest, description="파싱된 사용자 요청"
    )
    story_output: StoryOutput | None = Field(
        default=None, description="Story Writer의 구조화된 출력"
    )
    story_history: Annotated[list[str], add] = Field(
        default_factory=list, description="이전 스토리 버전 히스토리"
    )

    chat_history: list = Field(
        default_factory=list, description="대화형 에이전트의 채팅 히스토리"
    )
    # Director 출력
    eval_report: EvalReport | None = Field(default=None, description="검수 결과 보고서")
    feedback_history: Annotated[list[str], add] = Field(
        default_factory=list, description="Director 피드백 히스토리"
    )

    # 흐름 제어
    retry_count: int = Field(default=0, description="현재 재시도 횟수")
    max_retries: int = Field(default=3, description="최대 재시도 횟수")
    is_complete: bool = Field(default=False, description="작업 완료 여부")
