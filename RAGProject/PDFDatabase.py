from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, sessionmaker
import hashlib
import shutil
import uuid
from RAGSystem import build_RAG

BASE_DIR = Path(__file__).resolve().parent
print(type(BASE_DIR))
DB_PATH = BASE_DIR / "pdf_store.db"

PDF_DIR = BASE_DIR / "data" / "pdfs"

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ============================================================
# 2. 创建数据库引擎
# ============================================================
engine = create_engine(
    DATABASE_URL,
    echo=False
)

# ============================================================
# 3. Session
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False
)


# ============================================================
# 4. SQLAlchemy 基类
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# 5. PDF 表
# ============================================================

class PDFDocument(Base):

    __tablename__ = "pdf_documents"

    # 主键
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # 用户原始文件名
    original_name: Mapped[str] = mapped_column(
        String(255)
    )

    # 实际保存文件名
    stored_name: Mapped[str] = mapped_column(
        String(255)
    )

    # PDF保存位置
    file_path: Mapped[str] = mapped_column(
        String(500)
    )

    # 文件hash
    file_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True
    )

    # 文件大小
    file_size: Mapped[int] = mapped_column(
        Integer
    )

    # RAG入库状态
    ingest_status: Mapped[str] = mapped_column(
        String(50),
        default="未入库"
    )

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now
    )


# ============================================================
# 6. 创建数据库表
# ============================================================

def init_database():
    Base.metadata.create_all(engine)
    print("[OK] PDF数据库初始化完成")


def calculate_file_hash(
    file_path: str
) -> str:

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()

def save_pdf(file_path: str):

    source_path = Path(file_path)

    # --------------------------
    # 检查文件
    # --------------------------

    if not source_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{file_path}"
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError(
            "只能存储 PDF 文件"
        )

    # --------------------------
    # 计算hash
    # --------------------------

    file_hash = calculate_file_hash(
        str(source_path)
    )

    # --------------------------
    # 查询数据库是否已经存在
    # --------------------------

    with SessionLocal() as session:

        old_pdf = (
            session.query(PDFDocument)
            .filter(
                PDFDocument.file_hash == file_hash
            )
            .first()
        )

        if old_pdf:

            print(

                f"[INFO] PDF 已存在："
                f"{old_pdf.original_name}"
            )

            return old_pdf

        # --------------------------
        # 生成新的保存文件名
        # --------------------------

        stored_name = (
            f"{uuid.uuid4().hex}_"
            f"{source_path.name}"
        )

        target_path = (
            PDF_DIR /
            stored_name
        )

        # --------------------------
        # 复制PDF
        # --------------------------

        shutil.copy2(
            source_path,
            target_path
        )

        # --------------------------
        # 获取文件大小
        # --------------------------

        file_size = target_path.stat().st_size

        # --------------------------
        # 写入数据库
        # --------------------------

        pdf = PDFDocument(
            original_name=source_path.name,
            stored_name=stored_name,
            file_path=str(target_path),
            file_hash=file_hash,
            file_size=file_size,
            ingest_status="未入库"
        )

        session.add(pdf)

        session.commit()

        session.refresh(pdf)

        print(
            f"[OK] PDF 已保存，ID={pdf.id}"
        )

        return pdf

def get_all_pdfs():

    with SessionLocal() as session:

        pdfs = (
            session.query(PDFDocument)
            .order_by(PDFDocument.id)
            .all()
        )

        return pdfs

def get_pdf_by_id(pdf_id: int):

    with SessionLocal() as session:

        pdf = session.get(
            PDFDocument,
            pdf_id
        )

        return pdf

def update_pdf_status(
    pdf_id: int,
    status: str
):

    with SessionLocal() as session:

        pdf = session.get(
            PDFDocument,
            pdf_id
        )

        if not pdf:
            return False

        pdf.ingest_status = status

        session.commit()

        return True

def ingest_pdf(file_path: str):

    # ==========================================
    # 1. 保存PDF + 数据库记录
    # ==========================================

    pdf = save_pdf(file_path)

    pdf_id = pdf.id

    stored_path = pdf.file_path

    print(
        f"PDF ID：{pdf_id}"
    )

    # ==========================================
    # 2. 修改状态
    # ==========================================

    update_pdf_status(
        pdf_id,
        "处理中"
    )

    try:
        vectorstore = build_RAG(stored_path)

        update_pdf_status(
            pdf_id,
            "已入库"
        )
        print(
            f"[OK] PDF入库完成："
            f"{pdf.original_name}"
        )
        return vectorstore


    except Exception as e:

        update_pdf_status(
            pdf_id,
            "入库失败"
        )

        raise e

if __name__ == "__main__":
    init_database()



    ingest_pdf(r"C:\Users\qxx\Desktop\sample_report.pdf")
    ingest_pdf(r"C:\Users\qxx\Desktop\MyProject\RAGProject\咖啡烘焙与风味基础.pdf")
    ingest_pdf(r"C:\Users\qxx\Desktop\MyProject\RAGProject\城市公园生态观察记录.pdf")
    print(get_pdf_by_id(1).ingest_status)