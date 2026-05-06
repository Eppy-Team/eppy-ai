import os, uuid, asyncio
import fitz
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from dotenv import load_dotenv
load_dotenv()

_executor = ThreadPoolExecutor(max_workers=4)

connection = os.getenv("CONNECTION_STRING")
HF_API_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class StorePipeline:
    def __init__(self, db_url: str = connection, hf_token: str = HF_API_TOKEN, groq_api_key: str = GROQ_API_KEY):
        self.db_url = db_url
        self.hf_token = hf_token
        self.groq_api_key = groq_api_key

        ## HuggingFace Inference API Embeddings
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            task="feature-extraction",
            huggingfacehub_api_token=self.hf_token,
        )

    def _get_store(self, col_name: str) -> PGVector:
        return PGVector(
            embeddings=self.embeddings,
            collection_name=col_name,
            connection=self.db_url,
            use_jsonb=True,
        )

    def _parse_document(self, s3_url: str) -> str:
        """Parse PDF menggunakan pymupdf — ringan, tanpa torch."""
        doc = fitz.open(s3_url)
        markdown_parts = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        line_text = " ".join(
                            span["text"] for span in line["spans"]
                        ).strip()
                        if not line_text:
                            continue
                        # Deteksi heading berdasarkan font size
                        max_size = max(span["size"] for span in line["spans"])
                        if max_size >= 18:
                            markdown_parts.append(f"# {line_text}")
                        elif max_size >= 14:
                            markdown_parts.append(f"## {line_text}")
                        elif max_size >= 12:
                            markdown_parts.append(f"### {line_text}")
                        else:
                            markdown_parts.append(line_text)

        doc.close()
        return "\n".join(markdown_parts)

    def _chunk_document(self, markdown_content: str, file_name: str, article_id: str) -> List[Document]:
        """Chunk document based on headers using LangChain."""
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        chunks = markdown_splitter.split_text(markdown_content)

        # Merge small chunks
        proc_chunks = []
        for doc in chunks:
            is_header_1_only = (
                doc.metadata.get("Header_1") is not None and
                doc.metadata.get("Header_2") is None and
                doc.metadata.get("Header_3") is None
            )
            is_small = len(doc.page_content.strip()) < 200
            if proc_chunks and is_small and not is_header_1_only:
                prev_doc = proc_chunks[-1]
                if prev_doc.metadata.get("Header_1") == doc.metadata.get("Header_1"):
                    section_title = (
                        doc.metadata.get("Header_3") or
                        doc.metadata.get("Header_2") or
                        ""
                    )
                    prefix = f"\n{section_title}\n" if section_title else "\n"
                    prev_doc.page_content += prefix + doc.page_content
                else:
                    proc_chunks.append(doc)
            else:
                proc_chunks.append(doc)

        # Add metadata
        documents = []
        for i, chunk in enumerate(proc_chunks):
            header_parts = [h for h in [
                chunk.metadata.get("Header_1"),
                chunk.metadata.get("Header_2"),
                chunk.metadata.get("Header_3"),
            ] if h]
            documents.append(Document(
                page_content=chunk.page_content,
                metadata={
                    "article_id": article_id,
                    "file_name": file_name,
                    "chunk_id": i,
                    "header": "/".join(header_parts)
                }
            ))
        return documents

    def _add_new_document(self, article_id: str, file_name: str, s3_url: str, collection_name: str):
        store = self._get_store(collection_name)
        markdown_content = self._parse_document(s3_url)
        chunks = self._chunk_document(markdown_content, file_name, article_id)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_chunk_{i}")) for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)

    def _delete_document_by_filename(self, collection_name: str, article_id: str):
        store = self._get_store(collection_name)
        search_results = store.similarity_search("", k=10000, filter={"article_id": article_id})
        file_name = search_results[0].metadata["file_name"] if search_results else ""
        ids_to_delete = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_chunk_{i}"))
            for i in range(len(search_results))
        ]
        if ids_to_delete:
            store.delete(ids=ids_to_delete)

    async def add_new_document(self, article_id: str, file_name: str, s3_url: str, collection_name: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._add_new_document, article_id, file_name, s3_url, collection_name
        )

    async def delete_document_by_filename(self, article_id: str, collection_name: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._delete_document_by_filename, collection_name, article_id
        )