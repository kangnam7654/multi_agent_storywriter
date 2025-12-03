"""
시스템 프롬프트 로딩 유틸리티 모듈

시스템 프롬프트 파일을 로드하고 관리하는 기능을 제공합니다.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# 프로젝트 루트 디렉토리 (src/utils/prompt_loader.py 기준)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "system_prompts"


@dataclass
class SystemPrompts:
    """시스템 프롬프트 컨테이너"""

    story_writer: str
    director: str


@lru_cache(maxsize=1)
def load_system_prompts() -> SystemPrompts:
    """
    시스템 프롬프트를 로드합니다.

    캐싱되어 있어 여러 번 호출해도 파일을 한 번만 읽습니다.

    Returns:
        SystemPrompts: story_writer와 director 프롬프트를 담은 객체

    Raises:
        FileNotFoundError: 프롬프트 파일이 없을 경우

    Example:
        >>> prompts = load_system_prompts()
        >>> print(prompts.story_writer[:50])
        >>> print(prompts.director[:50])
    """
    story_writer_path = PROMPTS_DIR / "story_writer.md"
    director_path = PROMPTS_DIR / "director.md"

    story_writer_prompt = story_writer_path.read_text(encoding="utf-8")
    director_prompt = director_path.read_text(encoding="utf-8")

    return SystemPrompts(
        story_writer=story_writer_prompt,
        director=director_prompt,
    )


def load_prompt(name: str) -> str:
    """
    특정 시스템 프롬프트를 로드합니다.

    Args:
        name: 프롬프트 이름 ("story_writer" 또는 "director")

    Returns:
        str: 프롬프트 내용

    Raises:
        ValueError: 알 수 없는 프롬프트 이름
        FileNotFoundError: 프롬프트 파일이 없을 경우

    Example:
        >>> writer_prompt = load_prompt("story_writer")
        >>> director_prompt = load_prompt("director")
    """
    prompts = load_system_prompts()

    if name == "story_writer":
        return prompts.story_writer
    elif name == "director":
        return prompts.director
    else:
        raise ValueError(
            f"알 수 없는 프롬프트 이름: {name}. 'story_writer' 또는 'director'를 사용하세요."
        )


def clear_prompt_cache() -> None:
    """
    프롬프트 캐시를 클리어합니다.

    프롬프트 파일이 변경되었을 때 호출하세요.
    """
    load_system_prompts.cache_clear()
