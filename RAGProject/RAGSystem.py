import hashlib
import operator
from typing import TypedDict,Annotated

import chromadb
from dotenv import load_dotenv
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path


from langgraph.graph import StateGraph, START, END


load_dotenv(override=True)
key = os.getenv("DEEPSEEK_FLASH_KEY")
url = os.getenv("DEEPSEEK_FLASH_URL")
embed_models_url = os.getenv("EMBED_MODEL")
embed_model_ch = os.getenv("EMBED_MODEL_ch")
dir_root = os.getenv("DIR_ROOT")
if not dir_root:
    raise ValueError("DEEPSEEK_FLASH_DIR environment variable is not set")
path_dir_root = Path(dir_root)

dir_chroma = path_dir_root / "chroma_store"
dir_chroma.mkdir(exist_ok=True, parents=True)



class IngestionState(TypedDict):
    file_path: str
    raw_document: list[Document]
    text_chunks: Annotated[list[Document],operator.add]
    table_chunks: Annotated[list[Document],operator.add]
    chunk_count: int
    store_vs: Annotated[list[Chroma],operator.add]
    status: str
    errors: Annotated[list[str],operator.add]



def build_RAG():
    # build pdf file
    pdf_path = create_sample_documents()
    pdf_path1 = r"C:\Users\qxx\Desktop\Langchain1.0-Study-main\MyProject\RAGProject\test.pdf"
    if not os.path.exists(pdf_path1):
        print("样本 PDF 不存在，跳过此示例")
        return None

    # step 1: load pdf
    print("Step 1: 解析 PDF=======================》")
    pages = parse_pdf(pdf_path1)

    # step 2: 文本切片
    print("文本切片===================》")
    chunks = chunk_documents(pages)

    # step 3:embedding + chromaDB 入库
    print("Step 3: Embedding + 入库")
    vectorstore = embed_and_store(chunks)

    return vectorstore

def build_RAG_graph():
    file_path = create_sample_documents()
    if not os.path.exists(file_path):
        print("样本 PDF 不存在，跳过此示例")
        return None


    builder = StateGraph(IngestionState)

    builder.add_node("parse", _parse_pdf)
    builder.add_node("chunk", _chunk_documents)
    builder.add_node("store",_embed_and_store)

    builder.add_edge(START,"parse")
    builder.add_edge("parse", "chunk")
    builder.add_edge("chunk", "store")
    builder.add_edge("store", END)

    build = builder.compile()

    results = build.invoke({
        "file_path": file_path,
        "raw_document": [],
        "text_chunks":[],
        "table_chunks":[],
        "store_vs":[],
        "chunk_count": 0,
        "status": "初始化",
        "errors": [],
    })
    print(f"状态：{results['status']}")
    print(f"切片数：{results['chunk_count']}")
    if results.get('errors'):
        for err in results["errors"]:
            print(err)


    return results["store_vs"][0]



def _parse_pdf(state: IngestionState)-> dict:
    file_path = state["file_path"]
    try:
        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load()
        return {
            "raw_document": pages,
            "status": f"已解析{len(pages)}页"
        }
    except Exception as e:
        return {
            "raw_document": [],
            "errors":[f"解析文档失败：{e}"],
            "status": "解析失败",
        }

chunk_size=500
chunk_overlap=50

def _chunk_documents(state: IngestionState) -> dict:
    load = state["raw_document"]
    if not load:
        return {
            "text_chunks":[],
            "status":"无文档可切分"
        }
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],

    )
    chunks = splitter.split_documents(load)
    return {
        "text_chunks":chunks,
        "chunk_count": len(chunks),
        "status":f"已切分成功：{len(chunks)}快"
    }

collection_name = "rag_collection"
def _embed_and_store(state: IngestionState) -> dict:
    chunks = state["text_chunks"]
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_ch)

        # client = chromadb.PersistentClient(path=str(dir_chroma))
        reset_collection(collection_name)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(dir_chroma),
        )
        ids = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            content_hash = hashlib.md5(
                chunk.page_content.encode("utf-8")
            ).hexdigest()[:8]
            ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")

        vectorstore.add_documents(documents=chunks, ids=ids)
        return {
            "store_vs":[vectorstore]
        }
    except Exception as e:
        return{
            "store_vs":[],
            "errors":[f"存入错误：{e}"]
        }


def retrieval_docs(vectorstore,content: str):
    #docs = vectorstore.similarity_search("2024 AI Report", k=3)

    # retrieval = vectorstore.as_retriever(
    #     search_type = "mmr", # search_type="similarity"
    #     search_kwargs = {
    #         "k":3, "fetch_k":8,"lambda_mult":0.5
    #     }
    # )
    retrieval = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k":1
        }
    )
    docs= retrieval.invoke(content)
    return docs


def create_sample_documents():
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 未安装，跳过样本 PDF 生成")
        print("安装方式：pip install fpdf2")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2024 AI Technology Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0,6,
        "This report provides an overview of major AI developments in 2024. "
        "Large Language Models (LLMs) have seen significant improvements in "
        "reasoning capabilities, multilingual understanding, and tool use."
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Metrics Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 表格
    pdf.set_font("Helvetica", size=10)
    headers = ["Model", "Parameters", "Context Length", "MMLU Score"]
    col_widths = [40, 35, 40, 35]

    # 表头
    pdf.set_fill_color(200, 220, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, fill=True)
    pdf.ln()

    rows = [
        ["GPT-4o", "1.8T (est.)", "128K", "88.7%"],
        ["Claude 3.5", "Unknown", "200K", "88.3%"],
        ["Gemini Ultra", "Unknown", "1M", "90.0%"],
        ["Llama 3.1", "405B", "128K", "85.2%"],
    ]
    for row in rows:
        for i,cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
                   "The table above shows that Gemini Ultra achieves the highest MMLU score "
                   "at 90.0%, while Claude 3.5 offers the longest context window at 200K tokens. "
                   "Open-source models like Llama 3.1 are closing the gap with proprietary models."
                   )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Retrieval-Augmented Generation", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
                   "RAG (Retrieval-Augmented Generation) combines information retrieval with "
                   "text generation. The system first retrieves relevant documents from a knowledge "
                   "base, then uses an LLM to generate answers grounded in those documents.\n\n"
                   "Key RAG improvements in 2024:\n"
                   "- Hybrid search combining BM25 and vector retrieval\n"
                   "- Cross-encoder reranking for higher precision\n"
                   "- Multi-modal RAG handling tables, images, and structured data\n"
                   "- Query rewriting and expansion techniques"
                   )
    pdf_path = "./sample_report.pdf"

    pdf.output(pdf_path)
    return pdf_path

def parse_pdf(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()
    #print(type(pages))
    #print(pages)
    return pages

def chunk_documents(documents: list[Document],
                    chunk_size: int = 500,
                    chunk_overlap: int = 50,) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],

    )
    chunks = splitter.split_documents(documents)
    # print(type(chunks))
    # print(type(chunks[0]))
    # print(chunks)
    return chunks

def embed_and_store(chunks: list[Document],
                    collection_name: str = "rag_collection",) -> Chroma:
    """
    将文档切片向量化并存入 ChromaDB

    流程：chunks → Embedding → ChromaDB.upsert
    ChromaDB 自动持久化到 config.CHROMA_DIR。

    参数:
        chunks: 文档切片列表
        collection_name: ChromaDB 集合名

    返回:
        ChromaDB 实例
    """

    embeddings = HuggingFaceEmbeddings(model_name = embed_model_ch)


    #client = chromadb.PersistentClient(path=str(dir_chroma))
    reset_collection(collection_name)
    vectorstore = Chroma(
        collection_name = collection_name,
        embedding_function=embeddings,
        persist_directory=str(dir_chroma),
    )
    # vectorstore = Chroma(
    #     collection_name = collection_name,
    #     embedding_function=embeddings,
    #     client=client,
    # )

   # vectorstore.delete_collection()
    # 为每个 chunk 生成唯一 ID（基于内容哈希 + 来源），避免重复入库
    ids = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source","unknown")
        page = chunk.metadata.get("page",0)
        content_hash = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:8]
        ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")

    vectorstore.add_documents(documents = chunks,ids = ids)
    return vectorstore

def reset_collection(collection_name: str = "rag_collection"):
    """
    清空指定集合（用于重复运行示例时避免重复数据）

    参数:
        collection_name: 要清空的集合名
    """
    import chromadb



    client = chromadb.PersistentClient(path=str(dir_chroma))
    try:
        client.delete_collection(collection_name)
        print(f"  [OK] 已清空集合: {collection_name}")
    except Exception:
        pass  # 集合不存在时忽略
def main():
    # vectorstore = build_RAG()
    vectorstore = build_RAG_graph()
    if vectorstore is None:
        print("RAG构建失败")
        return
    # out = retrieval_docs(vectorstore,"RAG")
    # for i in out:
    #     print(i)
    # print("hello world")





if __name__ == "__main__":
    main()