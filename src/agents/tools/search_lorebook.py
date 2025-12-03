# tools.py
import pickle

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pickle_dumps(obj: object) -> bytes:
    return pickle.dumps(obj)


def pickle_loads(data: bytes) -> object:
    return pickle.loads(data)


def get_retriever(
    model="Qwen/Qwen3-Embedding-0.6B",
    collection_name="split_parents",
    persist_directory="./chroma_db",
) -> ParentDocumentRetriever:
    embedding_function = HuggingFaceEmbeddings(model_name=model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
    fs = LocalFileStore("./parent_docs_store")
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle_dumps,
        value_deserializer=pickle_loads,
    )
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever


@tool
def search_lorebook(query: str) -> str:
    """
    게임 설정집(Lorebook)에서 정보를 검색합니다.
    1. 스토리의 내용이 설정과 맞는지 확인할 때 하거나
    2. 스토리 아이디어를 얻기 위해 사용할 수 있습니다.

    Args:
        query: 검색할 키워드나 질문 (예: "화이트 런", "블러드 드래곤")
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "관련된 설정을 찾을 수 없습니다."

    # 검색된 문서들의 내용을 합쳐서 반환
    return "\n\n".join([f"[설정 자료]: {doc.page_content}" for doc in docs])
