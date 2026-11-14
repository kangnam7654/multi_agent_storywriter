"""
스토리 생성 워크플로우 그래프 모듈

LangGraph를 사용하여 멀티 에이전트 스토리 생성 파이프라인을 구성합니다.

워크플로우:
    1. init: 사용자 요청을 파싱하여 구조화된 RefinedRequest 생성
    2. write: StoryWriter가 스토리 작성
    3. review: Director가 스토리 검수 및 피드백 제공
    4. 조건부 분기: 승인되면 종료, 아니면 write로 재시도

노드 업데이트 계약:
    `story_history`, `feedback_history`는 `operator.add` 리듀서를 사용하므로,
    각 노드는 전체 상태가 아니라 "변경된 필드"와 "새로 추가된 히스토리 항목"만
    반환합니다. 에이전트는 기존처럼 상태를 수정해 반환하지만, 그래프 노드가
    에이전트에 사본을 넘기고 그 결과에서 델타만 추출합니다.

Example:
    >>> from src.graph import run_story_generation
    >>> result = run_story_generation("용사가 드래곤을 물리치는 이야기")
    >>> print(result.story_output.story)
"""

from typing import Any

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.agents.director import Director
from src.agents.request_parser import UserRequestParser
from src.agents.story_writer import StoryWriter
from src.schemas.state import GraphState


def _appended(before: list[str], after: list[str]) -> list[str]:
    """에이전트가 히스토리 끝에 새로 추가한 항목만 반환합니다.

    `operator.add` 리듀서 필드에 전체 리스트를 다시 반환하면 기존 항목이
    중복 누적되므로, 노드 입력 이후에 추가된 꼬리 부분만 델타로 사용합니다.
    """
    return list(after[len(before) :])


def create_graph(
    llm: ChatOllama,
    story_writer_system_prompt: str = "",
    director_system_prompt: str = "",
) -> StateGraph:
    """
    스토리 작성 워크플로우 그래프를 생성합니다.

    Args:
        llm: 사용할 LLM 인스턴스 (ChatOllama)
        story_writer_system_prompt: StoryWriter의 시스템 프롬프트 (빈 문자열이면 기본값 사용)
        director_system_prompt: Director의 시스템 프롬프트 (빈 문자열이면 기본값 사용)

    Returns:
        StateGraph: 컴파일 가능한 LangGraph StateGraph 객체

    Note:
        반환된 그래프는 .compile() 호출 후 사용해야 합니다.
    """
    # 에이전트 초기화
    request_parser = UserRequestParser(llm=llm)
    story_writer = StoryWriter(llm=llm, system_prompt=story_writer_system_prompt)
    director = Director(llm=llm, system_prompt=director_system_prompt)

    # 그래프 정의
    graph = StateGraph(GraphState)

    # ========== 노드 정의 ==========
    # 에이전트는 받은 상태를 수정하여 전체 상태를 반환합니다. 리듀서 필드가
    # 중복 누적되지 않도록 사본을 넘기고, 노드별로 필요한 델타만 반환합니다.
    def init_node(state: GraphState, runtime) -> dict[str, Any]:
        """사용자 요청을 파싱하여 RefinedRequest로 변환"""
        result = request_parser(state.model_copy(deep=True), runtime)
        return {"request": result.request}

    def write_node(state: GraphState, runtime) -> dict[str, Any]:
        """StoryWriter가 스토리를 작성하고 StoryOutput 생성"""
        result = story_writer(state.model_copy(deep=True), runtime)
        return {
            "story_output": result.story_output,
            "story_history": _appended(state.story_history, result.story_history),
        }

    def review_node(state: GraphState, runtime) -> dict[str, Any]:
        """Director가 스토리를 검수하고 EvalReport 생성"""
        result = director(state.model_copy(deep=True), runtime)
        return {
            "eval_report": result.eval_report,
            "feedback_history": _appended(
                state.feedback_history, result.feedback_history
            ),
            "retry_count": result.retry_count,
            "is_complete": result.is_complete,
        }

    # 노드 추가
    graph.add_node("init", init_node)
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    # ========== 엣지 정의 ==========
    # 기본 흐름: START → init → write → review
    graph.add_edge(START, "init")
    graph.add_edge("init", "write")
    graph.add_edge("write", "review")

    def should_retry(state: GraphState) -> str:
        """
        검수 결과에 따라 분기를 결정합니다.

        Returns:
            "end": 스토리가 승인되었거나 최대 재시도 횟수 도달
            "retry": 스토리 수정이 필요함
        """
        if state.is_complete:
            return "end"
        return "retry"

    # 조건부 엣지: review 후 승인 여부에 따라 분기
    graph.add_conditional_edges(
        "review",
        should_retry,
        {
            "end": END,  # 승인 → 종료
            "retry": "write",  # 미승인 → 재작성
        },
    )
    return graph


def run_story_generation(
    user_input: str = "", llm: ChatOllama | None = None
) -> GraphState:
    """
    동기 방식으로 스토리를 생성합니다.

    Args:
        user_input: 사용자의 스토리 요청 텍스트
        llm: 사용할 LLM 인스턴스 (None이면 기본 모델 사용)

    Returns:
        GraphState: 최종 상태 (story_output에 생성된 스토리 포함)

    Example:
        >>> result = run_story_generation("마법사가 되고 싶은 소년의 이야기")
        >>> print(result.story_output.title)
        >>> print(result.story_output.story)
    """
    # 기본 LLM 설정
    if llm is None:
        llm = ChatOllama(model="gpt-oss:20b")

    # 그래프 생성 및 컴파일
    graph = create_graph(llm=llm)
    app = graph.compile()

    # 그래프 실행
    initial_state = GraphState(user_input=user_input)
    final_state = app.invoke(initial_state)

    return GraphState(**final_state)


def run_story_generation_stream(
    user_input: str = "",
    llm: ChatOllama | None = None,
    story_writer_system_prompt: str = "",
    director_system_prompt: str = "",
) -> None:
    """
    노드 단위 스트리밍으로 스토리를 생성합니다.

    각 노드의 실행 상태를 실시간으로 출력합니다.
    주로 디버깅이나 진행 상황 모니터링에 사용됩니다.

    Args:
        user_input: 사용자의 스토리 요청 텍스트
        llm: 사용할 LLM 인스턴스 (None이면 기본 모델 사용)
        story_writer_system_prompt: StoryWriter의 시스템 프롬프트
        director_system_prompt: Director의 시스템 프롬프트

    Note:
        결과를 반환하지 않고 콘솔에 직접 출력합니다.
    """
    # 기본 LLM 설정
    if llm is None:
        llm = ChatOllama(model="gpt-oss:20b", reasoning=True)

    # 그래프 생성 및 컴파일
    graph = create_graph(
        llm=llm,
        story_writer_system_prompt=story_writer_system_prompt,
        director_system_prompt=director_system_prompt,
    )
    app = graph.compile()

    # 그래프 스트리밍 실행
    initial_state = GraphState(user_input=user_input)

    print("=" * 50)
    print("🚀 스토리 생성 시작")
    print("=" * 50)

    review_result = None
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n📍 노드: {node_name}")
            print("-" * 30)
            if node_name == "review":
                review_result = node_output.get("eval_report")

    print("\n" + "=" * 50)
    if review_result is not None:
        if review_result.is_approved:
            print("스토리 생성 흐름 종료 — Director 승인")
        else:
            print("스토리 생성 흐름 종료 — 미승인")
    else:
        print("스토리 생성 흐름 종료 — 검수 결과 없음")
    print("=" * 50)


async def run_story_generation_stream_tokens(
    user_input: str = "",
    llm: ChatOllama | None = None,
    story_writer_system_prompt: str = "",
    director_system_prompt: str = "",
) -> None:
    """
    토큰 단위 스트리밍으로 스토리를 생성합니다.

    LLM의 출력을 토큰 단위로 실시간 스트리밍하여 타이핑 효과를 제공합니다.
    Gradio나 웹 인터페이스에서 실시간 출력이 필요할 때 사용합니다.

    Args:
        user_input: 사용자의 스토리 요청 텍스트
        llm: 사용할 LLM 인스턴스 (None이면 기본 모델 사용)
        story_writer_system_prompt: StoryWriter의 시스템 프롬프트
        director_system_prompt: Director의 시스템 프롬프트

    Note:
        비동기 함수이므로 await와 함께 호출해야 합니다.

    Example:
        >>> import asyncio
        >>> asyncio.run(run_story_generation_stream_tokens("용사 이야기"))
    """
    # 기본 LLM 설정
    if llm is None:
        llm = ChatOllama(model="gpt-oss:20b")

    # 그래프 생성 및 컴파일
    graph = create_graph(
        llm=llm,
        story_writer_system_prompt=story_writer_system_prompt,
        director_system_prompt=director_system_prompt,
    )
    app = graph.compile()

    # 그래프 스트리밍 실행
    initial_state = GraphState(user_input=user_input)

    print("=" * 50)
    print("🚀 스토리 생성 시작 (토큰 스트리밍)")
    print("=" * 50)

    async for event in app.astream_events(initial_state, version="v2"):
        kind = event["event"]

        # LLM 스트리밍 토큰 출력
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

        # 노드 시작/종료 이벤트
        elif kind == "on_chain_start" and event.get("name"):
            if event["name"] in ["init", "write", "review"]:
                print(f"\n\n📍 노드 시작: {event['name']}")
                print("-" * 30)
        elif kind == "on_chain_end" and event.get("name"):
            if event["name"] in ["init", "write", "review"]:
                print(f"\n📍 노드 종료: {event['name']}")

    print("\n" + "=" * 50)
    print("스토리 생성 흐름 종료")
    print("=" * 50)
