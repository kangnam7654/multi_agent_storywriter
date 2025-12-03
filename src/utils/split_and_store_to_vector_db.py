# from langchain.retrievers import ParentDocumentRetriever
# from langchain.storage import InMemoryStore
import pickle

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    with open("lorebooks/ElderScrolls_Skyrim.md", "r", encoding="utf-8") as f:
        long_text = f.read()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    embedding_function = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

    # # Vectorstore: 자식 청크를 검색하기 위한 벡터 DB (여기선 Chroma 사용)
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=embedding_function,  # API Key 필요
        persist_directory="./chroma_db",
    )
    fs = LocalFileStore("./parent_docs_store")

    def pickle_dumps(obj: object) -> bytes:
        return pickle.dumps(obj)

    def pickle_loads(data: bytes) -> object:
        return pickle.loads(data)

    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle_dumps,
        value_deserializer=pickle_loads,
    )

    # # 3. ParentDocumentRetriever 초기화
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # # --- 테스트 데이터 추가 ---
    docs = [Document(page_content=long_text)]

    # 4. 문서 주입 (add_documents)
    # 이 과정에서 자동으로 Parent로 자르고 -> 다시 Child로 잘라서 -> 각각 저장소에 넣습니다.
    retriever.add_documents(docs, ids=None)

    # # 5. 검색 실행
    # # 검색은 '드래곤' 키워드로 자식 청크를 찾지만, 결과는 그 자식이 포함된 '부모 청크' 전체가 나옵니다.
    # result_docs = retriever.invoke("탐리엘")

    # print(f"검색된 문서 개수: {len(result_docs)}")
    # print(
    #     f"내용 길이: {len(result_docs[0].page_content)}"
    # )  # 400자가 아니라 2000자 단위의 긴 글이 나옴
    # print(result_docs[0].page_content)


if __name__ == "__main__":
    main()
