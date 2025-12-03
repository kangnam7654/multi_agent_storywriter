from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.agents.director import Director
from src.agents.request_parser import UserRequestParser
from src.agents.story_writer import StoryWriter
from src.schemas.state import GraphState


def create_graph(
    llm: ChatOllama, story_writer_system_prompt="", director_system_prompt=""
) -> StateGraph:
    """스토리 작성 그래프 생성"""

    request_parser = UserRequestParser(llm=llm)
    story_writer = StoryWriter(llm=llm, system_prompt=story_writer_system_prompt)
    director = Director(llm=llm, system_prompt=director_system_prompt)

    # 그래프 정의
    graph = StateGraph(GraphState)

    # 노드 정의
    def init_node(state: GraphState, runtime) -> GraphState:
        """초기화 노드: Lorebook 컨텍스트 주입"""
        return request_parser(state, runtime)

    def write_node(state: GraphState, runtime) -> GraphState:
        """스토리 작성 노드"""
        return story_writer(state, runtime)

    def review_node(state: GraphState, runtime) -> GraphState:
        """스토리 검수 노드"""
        return director(state, runtime)

    # 노드 추가
    graph.add_node("init", init_node)
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    # 엣지 정의
    graph.add_edge(START, "init")
    graph.add_edge("init", "write")
    graph.add_edge("write", "review")

    # 조건부 엣지: 검수 결과에 따라 분기
    def should_retry(state: GraphState) -> str:
        """재시도 여부 결정"""
        if state.is_complete:
            return "end"
        return "retry"

    graph.add_conditional_edges(
        "review",
        should_retry,
        {
            "end": END,
            "retry": "write",
        },
    )
    return graph


def run_story_generation(user_input="", llm=None) -> GraphState:
    """스토리 생성 실행"""

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
    user_input="",
    llm=None,
    story_writer_system_prompt="",
    director_system_prompt="",
):
    """스토리 생성 실행 (스트리밍 모드)"""

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

    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n📍 노드: {node_name}")
            print("-" * 30)

            # if isinstance(node_output, dict):
            #     # 주요 필드만 출력
            #     if "request" in node_output and node_output["request"]:
            #         print(f"📝 파싱된 요청: {node_output['request']}")
            #     if "story_output" in node_output and node_output["story_output"]:
            #         story_out = node_output["story_output"]
            #         print(
            #             f"📖 제목: {story_out.title if hasattr(story_out, 'title') else 'N/A'}"
            #         )
            #         story_content = (
            #             story_out.story if hasattr(story_out, "story") else ""
            #         )

            #         print(f"📖 스토리:\n{story_content}")
            #     elif "story" in node_output and node_output["story"]:
            #         print(f"📖 스토리:\n{node_output['story']}")
            #     if "feedback_history" in node_output and node_output["feedback_history"]:
            #         print(f"💬 피드백: {node_output['feedback_history'][-1]}")
            #     if "is_complete" in node_output:
            #         print(f"✅ 완료 여부: {node_output['is_complete']}")
            #     if "iteration" in node_output:
            #         print(f"🔄 반복 횟수: {node_output['iteration']}")
            # else:
            #     print(node_output)

    print("\n" + "=" * 50)
    print("✨ 스토리 생성 완료")
    print("=" * 50)


async def run_story_generation_stream_tokens(
    user_input="",
    llm=None,
    story_writer_system_prompt="",
    director_system_prompt="",
):
    """스토리 생성 실행 (토큰 단위 스트리밍 모드)"""

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
    print("✨ 스토리 생성 완료")
    print("=" * 50)
