# Multi-Agent Story Writer

스토리 아이디어를 세계관 설정집(Lorebook)에 맞춰 작성하고, 검수 결과를 반영해 수정하는 개인 프로젝트입니다. LangGraph가 요청 분석 → 초안 작성 → Director 검수의 흐름을 관리하고, 작성·검수 에이전트는 필요할 때 ChromaDB에서 설정을 검색합니다. 로컬 Ollama 모델과 검색 데이터가 있어야 실제 스토리를 생성할 수 있습니다.

## 검수 결과와 수정 이력

검수가 초안을 반려하면 Writer에 직전 초안과 해당 피드백을 전달합니다. 초안·피드백은 각각 한 번씩만 기록되도록 LangGraph 노드가 새 항목만 상태에 반환합니다. 이전에는 전체 이력을 다시 반환해 리듀서가 중복 누적했습니다. Gradio 화면은 Director가 승인한 종료와 최대 검수 횟수에 도달해 **미승인**으로 끝난 경우를 구분합니다. 검수 횟수에는 첫 초안도 포함됩니다(기본 최대 3회).

컴파일된 LangGraph 흐름에 결정적 에이전트/LLM 응답을 넣은 검증에서는 두 번 반려될 때 초안과 피드백이 순서대로 두 개씩 기록되고, 세 번째 검수를 호출하지 않고 미승인으로 종료했습니다. 이 응답은 테스트용이며 실제 모델 출력이나 스토리 품질의 증거가 아닙니다. 실제 Ollama 생성·ChromaDB 검색·설정 일치도는 별도로 확인해야 합니다.

## 테스트

```bash
uv sync --python 3.12 --locked --extra dev
uv run --no-sync pytest
```

테스트는 그래프의 승인·반려 흐름과 Gradio 출력 텍스트를 확인하며, Ollama나 외부 서비스는 호출하지 않습니다.

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
4. 피드백이 있으면 Story Writer가 수정 후 재검수 (기본 `max_retries=3`: 첫 초안 포함 최대 3회 검수, 그 안에 승인되지 않으면 미승인 상태로 종료)

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