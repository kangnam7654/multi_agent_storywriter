import asyncio

from src.graph import (
    run_story_generation,
    run_story_generation_stream,
    run_story_generation_stream_tokens,
)


def main():
    # 예시 실행
    user_input = "스카이림 지방의 소설 형태의 판타지 이야기를 작성해줘."
    run_story_generation(user_input=user_input)


def main_stream():
    """노드별 스트리밍 모드"""
    user_input = "스카이림 지방의 소설 형태의 판타지 이야기를 작성해줘. 짧게."
    with open("system_prompts/story_writer.md", "r") as f:
        story_writer_system_prompt = f.read()

    with open("system_prompts/director.md", "r") as f:
        director_system_prompt = f.read()

    run_story_generation_stream(
        user_input=user_input,
        story_writer_system_prompt=story_writer_system_prompt,
        director_system_prompt=director_system_prompt,
    )


async def main_stream_tokens():
    """토큰 단위 스트리밍 모드"""
    user_input = "스카이림 지방의 소설 형태의 판타지 이야기를 작성해줘."
    with open("system_prompts/story_writer.md", "r") as f:
        story_writer_system_prompt = f.read()

    with open("system_prompts/director.md", "r") as f:
        director_system_prompt = f.read()
    await run_story_generation_stream_tokens(
        user_input=user_input,
        story_writer_system_prompt=story_writer_system_prompt,
        director_system_prompt=director_system_prompt,
    )


if __name__ == "__main__":
    # 일반 실행
    # main()

    # 노드별 스트리밍 (각 노드 완료 시 출력)
    main_stream()

    # 토큰 단위 스트리밍 (LLM 출력을 실시간으로 확인)
    # asyncio.run(main_stream_tokens())
