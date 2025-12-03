"""에이전트 베이스 클래스"""

import json
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.agents.tools.search_lorebook import search_lorebook
from src.schemas.state import GraphState


class BaseAgent(ABC):
    """모든 에이전트의 베이스 클래스"""

    def __init__(self, llm: ChatOllama, system_prompt: str = ""):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools([search_lorebook])
        self.system_prompt = system_prompt

    @abstractmethod
    def __call__(self, state: GraphState, runtime: Any) -> GraphState:
        """에이전트 실행 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    def _build_user_message(self, state: GraphState) -> str:
        """유저 메시지 구성 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    def _parse_response(self, response: str) -> BaseModel:
        """LLM 응답 파싱 (하위 클래스에서 구현)"""
        pass

    def _create_messages(self, user_message: str) -> list:
        """초기 메시지 리스트 생성"""
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]

    def _handle_tool_calls(
        self,
        messages: list,
        max_iterations: int = 3,
    ) -> str:
        """
        Tool call을 처리하고 최종 응답 텍스트를 반환

        Args:
            messages: 현재 메시지 리스트
            max_iterations: 최대 tool call 반복 횟수

        Returns:
            최종 응답 텍스트
        """
        response_text: str = ""

        for iteration in range(max_iterations):
            ai_message = self.llm_with_tools.invoke(messages)
            messages.append(ai_message)

            # 로그 출력
            content = ai_message.content
            if isinstance(content, str):
                content_preview = content[:100]
            else:
                content_preview = str(content)[:100] if content else ""
            print(
                f"🔍 {self.__class__.__name__} {iteration+1}차 응답: '{content_preview}'..."
            )

            if ai_message.tool_calls:
                print(f"🔧 Tool calls: {ai_message.tool_calls}")

                for tool_call in ai_message.tool_calls:
                    if tool_call["name"] == "search_lorebook":
                        tool_result = search_lorebook.invoke(tool_call)
                        # 검색 결과 크기 제한 (최대 1000자)
                        tool_result_str = str(tool_result)[:1000]
                        print(f"📚 Lorebook 검색 결과: {tool_result_str[:200]}...")

                        messages.append(
                            ToolMessage(
                                content=tool_result_str,
                                tool_call_id=tool_call["id"],
                            )
                        )
                # tool call이 있으면 계속 반복
                continue
            else:
                # tool call이 없으면 응답 텍스트 사용
                if isinstance(content, str):
                    response_text = content
                else:
                    response_text = str(content) if content else ""
                break

        return response_text

    def _extract_json(self, response: str) -> dict:
        """
        LLM 응답에서 JSON 추출

        Args:
            response: LLM 응답 텍스트

        Returns:
            파싱된 JSON 딕셔너리

        Raises:
            json.JSONDecodeError: JSON 파싱 실패 시
        """
        # JSON 블록 추출 시도
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        return json.loads(json_str)

    def set_system_prompt(self, prompt: str) -> None:
        """시스템 프롬프트 설정"""
        self.system_prompt = prompt

    def get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return self.system_prompt
