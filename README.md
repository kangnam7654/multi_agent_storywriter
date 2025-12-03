# Multi-Agent Story Writer

LangGraph 기반의 멀티 에이전트 스토리 작성 시스템입니다.  
Lorebook(세계관 설정집)을 참고하여 스토리를 생성하고, 자동으로 검수 및 피드백 반영까지 수행합니다.

## 구조

```
├── src/
│   ├── agents/
│   │   ├── request_parser.py   # 사용자 요청 파싱
│   │   ├── story_writer.py     # 스토리 작성
│   │   ├── director.py         # 스토리 검수 및 피드백
│   │   └── tools/
│   │       └── search_lorebook.py  # Lorebook 검색 도구
│   ├── schemas/
│   │   └── state.py            # 상태 스키마 정의
│   └── graph.py                # LangGraph 워크플로우
├── system_prompts/             # 에이전트 시스템 프롬프트
├── lorebooks/                  # 세계관 설정 문서
├── chroma_db/                  # 벡터 DB 저장소
└── parent_docs_store/          # 원본 문서 저장소
```

## 워크플로우

```
사용자 입력 → Request Parser → Story Writer ←→ Director → 최종 스토리
                                    ↑              ↓
                                    └── 피드백 반영 ──┘
```

1. **Request Parser**: 사용자 입력을 분석하여 장르, 스타일, 분량 등을 추출
2. **Story Writer**: Lorebook을 검색하여 세계관에 맞는 스토리 작성
3. **Director**: 작성된 스토리를 검수하고 설정 오류나 개선점 피드백
4. 피드백이 있으면 Story Writer가 수정 후 재검수 (최대 3회 반복)

## 기술 스택

- Python 3.12+
- LangGraph (워크플로우 오케스트레이션)
- LangChain (LLM 통합)
- ChromaDB (벡터 검색)
- Ollama (로컬 LLM)
- HuggingFace Embeddings

## 설치

```bash
# uv 사용
uv sync

# 또는 pip 사용
pip install -e .
```

## 실행

### CLI 모드
```bash
python main.py
```

### 웹 데모 (Gradio)
```bash
python app.py
```
브라우저에서 `http://localhost:7860` 접속

Ollama가 실행 중이어야 합니다.

## 라이선스

MIT License