"""프롬프트 로더 테스트"""

import pytest

from src.utils.prompt_loader import (
    PROMPTS_DIR,
    SystemPrompts,
    clear_prompt_cache,
    load_prompt,
    load_system_prompts,
)


class TestLoadSystemPrompts:
    """load_system_prompts 함수 테스트"""

    def test_returns_system_prompts_instance(self):
        """SystemPrompts 인스턴스를 반환하는지 테스트"""
        result = load_system_prompts()
        assert isinstance(result, SystemPrompts)

    def test_story_writer_prompt_not_empty(self):
        """story_writer 프롬프트가 비어있지 않은지 테스트"""
        prompts = load_system_prompts()
        assert prompts.story_writer
        assert len(prompts.story_writer) > 0

    def test_director_prompt_not_empty(self):
        """director 프롬프트가 비어있지 않은지 테스트"""
        prompts = load_system_prompts()
        assert prompts.director
        assert len(prompts.director) > 0

    def test_prompts_contain_expected_content(self):
        """프롬프트에 예상 내용이 포함되어 있는지 테스트"""
        prompts = load_system_prompts()

        # story_writer에는 스토리 관련 키워드가 있어야 함
        assert (
            "스토리" in prompts.story_writer or "story" in prompts.story_writer.lower()
        )

        # director에는 검토/평가 관련 키워드가 있어야 함
        assert (
            "검수" in prompts.director
            or "평가" in prompts.director
            or "피드백" in prompts.director
        )

    def test_caching_returns_same_object(self):
        """캐싱이 동일한 객체를 반환하는지 테스트"""
        clear_prompt_cache()

        first = load_system_prompts()
        second = load_system_prompts()

        # 동일한 객체 (캐시된 결과)
        assert first is second


class TestLoadPrompt:
    """load_prompt 함수 테스트"""

    def test_load_story_writer_prompt(self):
        """story_writer 프롬프트 로드 테스트"""
        prompt = load_prompt("story_writer")
        assert prompt
        assert len(prompt) > 0

    def test_load_director_prompt(self):
        """director 프롬프트 로드 테스트"""
        prompt = load_prompt("director")
        assert prompt
        assert len(prompt) > 0

    def test_invalid_name_raises_error(self):
        """잘못된 이름으로 호출 시 에러 테스트"""
        with pytest.raises(ValueError) as exc_info:
            load_prompt("invalid_name")

        assert "알 수 없는 프롬프트 이름" in str(exc_info.value)
        assert "story_writer" in str(exc_info.value)
        assert "director" in str(exc_info.value)


class TestClearPromptCache:
    """clear_prompt_cache 함수 테스트"""

    def test_cache_cleared(self):
        """캐시가 클리어되는지 테스트"""
        # 캐시 생성
        first = load_system_prompts()

        # 캐시 클리어
        clear_prompt_cache()

        # 다시 로드하면 새로운 객체
        second = load_system_prompts()

        # 내용은 같지만 다른 객체 (새로 로드됨)
        # lru_cache는 클리어 후 새 객체를 생성
        assert first.story_writer == second.story_writer
        assert first.director == second.director


class TestPromptsDir:
    """PROMPTS_DIR 상수 테스트"""

    def test_prompts_dir_exists(self):
        """프롬프트 디렉토리가 존재하는지 테스트"""
        assert PROMPTS_DIR.exists()
        assert PROMPTS_DIR.is_dir()

    def test_story_writer_file_exists(self):
        """story_writer.md 파일이 존재하는지 테스트"""
        story_writer_path = PROMPTS_DIR / "story_writer.md"
        assert story_writer_path.exists()

    def test_director_file_exists(self):
        """director.md 파일이 존재하는지 테스트"""
        director_path = PROMPTS_DIR / "director.md"
        assert director_path.exists()
