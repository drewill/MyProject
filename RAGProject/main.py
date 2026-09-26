import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "data" / "upload_temp"

logger = logging.getLogger(__name__)

# 本地学习版：串行执行入库和问答，避免同时操作
operation_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    from PDFDatabase import init_database

    init_database()

    # 启动时导入，加载一次模型和向量库
    from Agent import ask_agent

    app.state.ask_agent = ask_agent

    yield


app = FastAPI(
    title="PDF 知识库",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.get("/")
def homepage():
    """返回前端页面。"""
    return FileResponse(
        BASE_DIR / "static" / "index.html",
        media_type="text/html",
    )


@app.get("/api/documents")
def list_documents():
    """读取 PDF 数据库，返回文档列表。"""
    from PDFDatabase import get_all_pdfs

    pdfs = get_all_pdfs()

    return [
        {
            "id": pdf.id,
            "name": pdf.original_name,
            "status": pdf.ingest_status,
            "size_kb": round(pdf.file_size / 1024, 1),
        }
        for pdf in pdfs
    ]


@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)):
    """接收 PDF，保存后调用现有入库函数。"""
    from PDFDatabase import ingest_pdf

    try:
        # 去除上传文件名中可能存在的目录部分
        filename = (file.filename or "").replace("\\", "/")
        filename = filename.rsplit("/", 1)[-1]

        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="请选择 PDF 文件",
            )

        max_size = 20 * 1024 * 1024

        # 每次请求使用独立目录，保留原始文件名
        with TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir) / filename
            total_size = 0

            with temp_path.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    total_size += len(chunk)

                    if total_size > max_size:
                        raise HTTPException(
                            status_code=413,
                            detail="文件不能超过 20 MB",
                        )

                    output.write(chunk)

            with temp_path.open("rb") as uploaded:
                header = uploaded.read(1024)

            if b"%PDF-" not in header:
                raise HTTPException(
                    status_code=400,
                    detail="文件内容不是有效的 PDF 格式",
                )

            with operation_lock:
                # 你的 ingest_pdf 会把文件复制到 data/pdfs
                # 完成后才能清理这里的临时文件
                store = ingest_pdf(str(temp_path))

                if store is None:
                    raise RuntimeError("入库未返回向量库")

        return {
            "message": "上传并入库成功",
            "filename": filename,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("PDF 入库失败")
        raise HTTPException(
            status_code=500,
            detail="PDF 入库失败，请查看 PyCharm 终端中的错误",
        )
    finally:
        file.file.close()


@app.post("/api/chat")
def chat(body: ChatRequest):
    """将网页问题交给 Agent。"""
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空",
        )

    try:
        with operation_lock:
            answer = app.state.ask_agent(question)

        return {"answer": answer}

    except Exception:
        logger.exception("问答失败")
        raise HTTPException(
            status_code=500,
            detail="问答失败，请查看 PyCharm 终端中的错误",
        )