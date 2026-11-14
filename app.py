import gradio as gr
from langchain_ollama import ChatOllama

from src.graph import create_graph
from src.schemas.state import GraphState
from src.utils.prompt_loader import load_system_prompts

# 최종 결과 메시지: 승인 종료와 최대 검수 횟수 도달(미승인) 종료를 구분합니다.
APPROVED_MESSAGE = "## 🎉 스토리 생성 완료 — Director 검수 승인\n"
UNAPPROVED_MESSAGE_TEMPLATE = (
    "## ⚠️ 최대 검수 횟수({max_reviews}회) 도달 — 미승인 상태로 종료\n\n"
    "마지막 초안은 Director 승인을 받지 못했습니다. 위 피드백을 참고하세요.\n"
)


def generate_story(
    user_input: str,
    model_name: str,
    max_retries: int,
    progress=gr.Progress(),
):
    """스토리 생성 (스트리밍)"""

    if not user_input.strip():
        yield "스토리 아이디어를 입력해주세요."
        return

    # 시스템 프롬프트 로드 (캐싱됨)
    prompts = load_system_prompts()

    # LLM 설정
    llm = ChatOllama(model=model_name)

    # 그래프 생성
    graph = create_graph(
        llm=llm,
        story_writer_system_prompt=prompts.story_writer,
        director_system_prompt=prompts.director,
    )
    app = graph.compile()

    # 초기 상태
    initial_state = GraphState(user_input=user_input, max_retries=max_retries)

    # 출력 버퍼
    output_parts = []
    # 노드는 변경분(델타)만 반환하므로 작성 시도 횟수는 여기서 직접 셉니다.
    write_attempts = 0

    # 스트리밍 실행
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():

            if node_name == "init":
                # 요청 파싱 결과
                if "request" in node_output and node_output["request"]:
                    req = node_output["request"]
                    output_parts.append("## 📝 요청 분석\n")
                    output_parts.append(f"- **프롬프트**: {req.summarized_prompt}\n")
                    output_parts.append(f"- **장르**: {req.genre}\n")
                    output_parts.append(f"- **스타일**: {req.style}\n")
                    output_parts.append(f"- **분량**: {req.length}\n\n")
                    output_parts.append("---\n\n")
                    yield "".join(output_parts)

            elif node_name == "write":
                # 스토리 작성 결과
                if "story_output" in node_output and node_output["story_output"]:
                    story_out = node_output["story_output"]
                    write_attempts += 1

                    if write_attempts > 1:
                        output_parts.append(
                            f"## ✍️ 스토리 수정 (시도 {write_attempts})\n\n"
                        )
                    else:
                        output_parts.append("## ✍️ 스토리 초안\n\n")

                    if hasattr(story_out, "title") and story_out.title:
                        output_parts.append(f"### {story_out.title}\n\n")

                    if hasattr(story_out, "story") and story_out.story:
                        output_parts.append(f"{story_out.story}\n\n")

                    if hasattr(story_out, "notes") and story_out.notes:
                        output_parts.append(f"*📌 참고: {story_out.notes}*\n\n")

                    output_parts.append("---\n\n")
                    yield "".join(output_parts)

            elif node_name == "review":
                # 검수 결과
                if "eval_report" in node_output and node_output["eval_report"]:
                    report = node_output["eval_report"]

                    if report.is_approved:
                        output_parts.append("## ✅ 검수 통과\n\n")
                        output_parts.append(f"**점수**: {report.score}/10\n\n")
                        output_parts.append(f"**피드백**: {report.feedback}\n\n")
                    else:
                        output_parts.append("## 🔄 검수 피드백\n\n")
                        output_parts.append(f"**점수**: {report.score}/10\n\n")
                        output_parts.append(f"**피드백**: {report.feedback}\n\n")
                        if report.issues:
                            output_parts.append("**개선 필요 사항**:\n")
                            for issue in report.issues:
                                output_parts.append(f"- {issue}\n")
                            output_parts.append("\n")

                    output_parts.append("---\n\n")
                    yield "".join(output_parts)

                # 완료 여부 확인: 승인 종료와 미승인(최대 검수 횟수) 종료 구분
                if node_output.get("is_complete"):
                    report = node_output.get("eval_report")
                    if report is not None and report.is_approved:
                        output_parts.append(APPROVED_MESSAGE)
                    else:
                        output_parts.append(
                            UNAPPROVED_MESSAGE_TEMPLATE.format(
                                max_reviews=int(max_retries)
                            )
                        )
                    yield "".join(output_parts)


def create_demo():
    """Gradio 데모 생성"""

    with gr.Blocks(
        title="Multi-Agent Story Writer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📖 Multi-Agent Story Writer
            
            LangGraph 기반 멀티 에이전트 스토리 생성기입니다.  
            Lorebook(세계관 설정집)을 참고하여 스토리를 작성하고, 자동으로 검수합니다.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 입력 영역
                user_input = gr.Textbox(
                    label="스토리 아이디어",
                    placeholder="예: 스카이림 지방을 배경으로 한 판타지 모험 이야기를 작성해줘.",
                    lines=3,
                )

                with gr.Accordion("⚙️ 설정", open=False):
                    model_name = gr.Textbox(
                        label="Ollama 모델",
                        value="gpt-oss:20b",
                        info="사용할 Ollama 모델 이름",
                    )
                    max_retries = gr.Slider(
                        label="최대 검수 횟수",
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        info=(
                            "작성+검수 라운드 최대 횟수 (첫 초안 포함). "
                            "이 횟수 안에 승인되지 않으면 미승인 상태로 종료됩니다."
                        ),
                    )

                generate_btn = gr.Button("✨ 스토리 생성", variant="primary")

            with gr.Column(scale=2):
                # 출력 영역
                output = gr.Markdown(
                    label="생성 결과",
                    value="스토리 아이디어를 입력하고 생성 버튼을 눌러주세요.",
                )

        # 예시
        gr.Examples(
            examples=[
                ["스카이림 지방을 배경으로 한 드래곤본의 모험 이야기"],
                ["화이트런에서 벌어지는 도둑과 경비병의 추격전"],
                ["윈드헬름의 어둠의 형제단 암살자 이야기를 짧게 써줘"],
            ],
            inputs=user_input,
        )

        # 이벤트 연결
        generate_btn.click(
            fn=generate_story,
            inputs=[user_input, model_name, max_retries],
            outputs=output,
        )

        user_input.submit(
            fn=generate_story,
            inputs=[user_input, model_name, max_retries],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
