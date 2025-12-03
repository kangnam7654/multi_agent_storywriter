import asyncio

from src.graph import (
    run_story_generation,
    run_story_generation_stream,
    run_story_generation_stream_tokens,
)
from src.utils.prompt_loader import load_system_prompts


def main():
    """동기 방식 스토리 생성"""
    user_input = "스카이림 지방의 소설 형태의 판타지 이야기를 작성해줘."
    run_story_generation(user_input=user_input)


def main_stream():
    """노드별 스트리밍 모드"""
    user_input = "스카이림 지방의 소설 형태의 판타지 이야기를 작성해줘. 짧게."
    prompts = load_system_prompts()

    run_story_generation_stream(
        user_input=user_input,
        story_writer_system_prompt=prompts.story_writer,
        director_system_prompt=prompts.director,
    )


async def main_stream_tokens():
    """토큰 단위 스트리밍 모드"""
    user_input = "스카이림 지방의 소설 형태의 판타지 이야기를 작성해줘."
    prompts = load_system_prompts()

    await run_story_generation_stream_tokens(
        user_input=user_input,
        story_writer_system_prompt=prompts.story_writer,
        director_system_prompt=prompts.director,
    )


if __name__ == "__main__":
    # 일반 실행
    # main()

    # 노드별 스트리밍 (각 노드 완료 시 출력)
    main_stream()

    # 토큰 단위 스트리밍 (LLM 출력을 실시간으로 확인)
    # asyncio.run(main_stream_tokens())
