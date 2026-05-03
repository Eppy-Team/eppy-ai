import os, uuid, asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions, PictureDescriptionApiOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownPictureSerializer, MarkdownDocSerializer
from docling_core.types.doc.document import DoclingDocument, ImageRefMode, PictureDescriptionData, PictureItem

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from typing_extensions import override
from dotenv import load_dotenv
load_dotenv()

_executor = ThreadPoolExecutor(max_workers=4)

# Database connection string for PostgreSQL with pgvector
connection = os.getenv("CONNECTION_STRING")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AnnotationPictureSerializer(MarkdownPictureSerializer):
    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: Optional[str] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        text_parts: list[str] = []

        # reusing the existing result:
        parent_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **kwargs,
        )
        text_parts.append(parent_res.text)

        # appending annotations:
        if item.meta is not None and item.meta.description is not None:
            text_parts.append(
                f"<!-- Picture description: {item.meta.description.text} -->"
            )

        text_res = (separator or "\n").join(text_parts)
        return create_ser_result(text=text_res, span_source=item)

class StorePipeline:
    def __init__(self, db_url: str = connection, openai_api_key: str = OPENAI_API_KEY, hf_token: str = HF_API_TOKEN, groq_api_key: str = GROQ_API_KEY):
        self.db_url = db_url
        self.openai_api_key = openai_api_key
        self.hf_token = hf_token
        self.groq_api_key = groq_api_key

        ## HuggingFace Embeddings (local inference)
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}) # dimension 384
        # self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}) # dimension 768
        
        ## HuggingFace Inference API Embeddings (requires API calls)
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            task="feature-extraction",
            huggingfacehub_api_token=self.hf_token,
        )

        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True

        ## Example picture description options with a custom API (no money wkwk)
        # self.pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        #     url="https://api.openai.com/v1/chat/completions",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     model="gpt-4o-mini",
        #     scale=2.0,
        # )
        
        self.pipeline_options.do_picture_description = True
        # Alternative, a lightweight HuggingFace model for picture description
        self.pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
            repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
            # repo_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            prompt="In this manual document for epson products, Analysis and Describe image in detail",
        )
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.images_scale = 2.0

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF:  PdfFormatOption(pipeline_options=self.pipeline_options),
            }
        )

    def _get_store(self, col_name: str) -> PGVector:
        return PGVector(
            embeddings=self.embeddings,
            collection_name=col_name,
            connection=self.db_url,
            use_jsonb=True,
        )
    
    def _parse_document(self, file_path: str) -> str:
        """Parse document using Docling with picture annotations."""
        doc = self.converter.convert(file_path).document
        serializer = MarkdownDocSerializer(
            doc=doc,
            picture_serializer=AnnotationPictureSerializer(),
            params=MarkdownParams(
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder="",
            ),
        )
        return serializer.serialize().text

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
                prev_header_1 = prev_doc.metadata.get("Header_1")
                curr_header_1 = doc.metadata.get("Header_1")

                if prev_header_1 == curr_header_1:
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
            header_1 = chunk.metadata.get("Header_1")
            header_2 = chunk.metadata.get("Header_2")
            header_3 = chunk.metadata.get("Header_3")
            header_parts = [h for h in [header_1, header_2, header_3] if h]
            header = "/".join(header_parts)
            
            metadata = {
                "article_id": article_id,
                "file_name": file_name,
                "chunk_id": i,
                "header": header
            }
            documents.append(Document(page_content=chunk.page_content, metadata=metadata))
        return documents

    def _add_new_document(self, article_id: str, file_name: str, file_path: str, collection_name: str):
        """Process a new document and add to vector store."""
        store = self._get_store(collection_name)
        # print(f"Parsing document: {file_name}")
        markdown_content = self._parse_document(file_path)
        # print("Chunking document...")
        chunks = self._chunk_document(markdown_content, file_name, article_id)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_chunk_{i}")) for i in range(len(chunks))]
    
        # print(f"✓ Successfully Chunk: {file_name} with {len(chunks)} chunks.")
        # print(f"Adding {len(chunks)} chunks to vector store...")
        store.add_documents(chunks, ids=ids)
        # print(f"✓ Successfully added document: {file_name} with {len(chunks)} chunks.")
        # try:
        #     store.add_documents(chunks, ids=ids)
        #     print(f"✓ Successfully added document: {file_name} with {len(ch
        # except Exception as e:
        #     print(f"Error adding document: {file_name}. Error: {str(e

    def _delete_document_by_filename(self, collection_name: str, article_id: str):
        """Delete all chunks by article_id using ID pattern matching."""
        store = self._get_store(collection_name)
        search_results = store.similarity_search("", k=10000, filter={"article_id": article_id})
        file_name = search_results[0].metadata["file_name"] if search_results else ""
        ids_to_delete = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_chunk_{i}"))
            for i in range(len(search_results))
        ]
        if ids_to_delete:
            store.delete(ids=ids_to_delete)
            # print(f"✓ Deleted {len(ids_to_delete)} chunks from document: {file_name}")
    
    #-- Async methods 
    async def add_new_document(self, article_id: str, file_name: str, file_path: str, collection_name: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._add_new_document, article_id, file_name, file_path, collection_name)

    async def delete_document_by_filename(self, article_id:str, collection_name: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._delete_document_by_filename, collection_name, article_id)



# Example usage
# if __name__ == "__main__":
#     pipeline = StorePipeline()

    # parse = pipeline._parse_document("data/sample2.pdf")
    # chunk = pipeline.chunk_document(parse, "sample2.pdf")
    # print(parse)
    # print("=====================================================")
    # print(chunk)
    # Add a new document
    # async def main():
    #     await pipeline.add_new_document(
    #         "id_article_1",
    #         "DS-1730-12-13.pdf",
    #         "example_documents/DS-1730-12-13.pdf",
    #         "epson_collection"
    #     )
    #     # await pipeline.delete_document_by_filename("id_article_1", "epson_collection")  
    # asyncio.run(main())