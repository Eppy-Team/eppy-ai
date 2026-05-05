import os, time, asyncio, base64
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ai_services.store_pipeline import StorePipeline
from contextvars import ContextVar

from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain.tools import tool

app = FastAPI()
sPipeline = StorePipeline()
col_name = "epson_collection"
_retrieved_docs_store = {"docs": []}
_retrieved_image_analyze = {"docs": []}
class HistoryMessage(BaseModel):
    role: str
    content: str    

class ChatRequest(BaseModel):
    conversation_id: str
    query: str
    image_url: Optional[str] = None
    history: List[HistoryMessage] = []

class EmbedRequest(BaseModel):
    article_id: str
    title: str
    s3_url: str

class SourceItem(BaseModel):
    article_id: str         # id dari artikel
    chunks_id: str          # id chunk
    title: str              # nama file
    header: str             # header dari chunk
    snippet: str            # isi dokumen

class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    sources: List[SourceItem]
    image_analyses: List[str]

#-- Helper
def _build_lc_history(history: List[HistoryMessage]) -> List[HumanMessage | AIMessage]:
    message = []
    for msg in history:
        if msg.role == "user":
            message.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            message.append(AIMessage(content=msg.content))
    return message

def _extract_sources_and_score(retrieved_docs: list) -> tuple[List[SourceItem], float]:
    if not retrieved_docs:
        return [], 0.0  
    sources = []
    scores = []
    for doc, score in retrieved_docs:
        scores.append(score)
        metadata = doc.metadata
        source_item = SourceItem(
            article_id=metadata.get("article_id", "unknown"),
            chunks_id=str(metadata.get("chunk_id", "unknown")),
            title=metadata.get("file_name", "unknown"),
            header=metadata.get("header", "unknown"),
            snippet=doc.page_content[:200]  # Ambil potongan awal sebagai snippet
        )
        sources.append(source_item)
    confidence_score = sum(scores) / len(scores) if scores else 0.0
    return sources, confidence_score

#-- Tools
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = sPipeline._get_store(col_name).similarity_search_with_relevance_scores(query=query, k=5)
    _retrieved_docs_store["docs"] = retrieved_docs
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc, _ in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def analyze_image(image_url: str, query: str) -> str:
    """Analyze an image from URL to help answer user query about Epson products."""
    model = ChatGroq(
        groq_api_key=sPipeline.groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # Cek apakah path lokal atau URL
    if image_url.startswith("http://") or image_url.startswith("https://"):
        # URL normal
        image_content = {"type": "image_url", "image_url": {"url": image_url}}
    else:
        # Path lokal → convert ke base64
        with open(image_url, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Deteksi format gambar dari ekstensi
        ext = image_url.split(".")[-1].lower()
        mime_types = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        mime_type = mime_types.get(ext, "image/jpeg")
        
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
        }

    response = model.invoke([
        HumanMessage(content=[
            image_content,
            {
                "type": "text",
                "text": (
                    f"Kamu adalah teknisi Epson. Analisis gambar ini dalam konteks pertanyaan berikut: '{query}'. "
                    "Deskripsikan kondisi produk, error yang terlihat, atau informasi relevan lainnya secara detail."
                )
            }
        ])
    ])
    _retrieved_image_analyze["docs"].append(response.content)
    return response.content

tools = [retrieve_context, analyze_image]

#-- API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def message(req: ChatRequest):
    _retrieved_docs_store["docs"] = []
    _retrieved_image_analyze["docs"] = []

    try:
        user_query = req.query
        system_prompt = (
            "Kamu adalah asisten helpdesk Epson bernama Eppy."
            "Wajib menggunakan tool retrieve_context untuk mencari informasi dari dokumen."
            "Jika user mengirim gambar (image_url tersedia), gunakan tool analyze_image"
            "Jawab HANYA berdasarkan hasil retrieve dan analisis gambar."
            "Jika tidak ditemukan, jawab: 'Maaf, saya tidak menemukan jawabannya di dokumen.'"
        )
        message_history = _build_lc_history(req.history)
        if req.image_url:
            user_query += f"\n[image_url: {req.image_url}]"
        model = ChatGroq(
            groq_api_key=sPipeline.groq_api_key,
            model_name="qwen/qwen3-32b"
        )
        agent = create_agent(model, tools, system_prompt=system_prompt)

        # Generate answer
        response = await agent.ainvoke(
            {"messages": message_history + [{"role": "user", "content": user_query}]}
        )
        answer = response["messages"][-1].content
        retrieved_docs = _retrieved_docs_store["docs"]
        sources, confidence_score = _extract_sources_and_score(retrieved_docs)
        image_analyses = _retrieved_image_analyze["docs"]

        return {"answer": answer, "confidence_score": confidence_score, "sources": sources, "image_analyses": image_analyses}

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "LLM_ERROR", "message": str(e)})

@app.post("/embed")
async def embed_document(req: EmbedRequest):
    try:
        await sPipeline.add_new_document(req.article_id, req.title, req.s3_url, col_name)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "EMBED_FAILED", "message": str(e)})

@app.delete("/embed/{article_id}")
async def delete_embed(article_id: str):
    try:
        await sPipeline.delete_document_by_filename(article_id=article_id, collection_name=col_name)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "DELETE_FAILED", "message": str(e)})

@app.get("/health")
async def health_check():
    try:
        engine = create_engine(sPipeline.db_url)
        
        with engine.connect() as conn:
            # Hitung total chunks di collection
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :col_name
            """), {"col_name": col_name})
            knowledge_count = result.scalar()
    except Exception:
        knowledge_count = 0

    return {
        "status": "ok",
        "model": "qwen/qwen3-32b",
        "knowledge_count": knowledge_count
    }



# Example usage
# if __name__ == "__main__":
#     async def main():
        # add & embedd doc test
        # await embed_document(EmbedRequest(
        #     article_id="id_article_3",
        #     title="DS-1730-15-16.pdf",
        #     s3_url="example_documents/DS-1730-15-16.pdf"
        # ))

        ## delete doc test
        # await delete_embed("id_article_3")

        ## chat test
        # response = await message(ChatRequest(
        #     conversation_id="conv1",
        #     query="Why the scanner epson ds-1730 error jamp paper?",
        #     image_url="example_documents/error_paper_jamp.jpg",
        #     history=[]
        # ))
        # print("\n" + "="*50)
        # print(f"ANSWER:\n{response['answer']}")
        # print(f"\nCONFIDENCE SCORE: {response['confidence_score']:.2f}")
        # print("\nSOURCES:")
        # for src in response['sources']:
        #     print(f"  - [{src.header}] {src.title} (chunk {src.chunks_id})")
        #     print(f"    {src.snippet[:100]}...")
        # print("="*50)
        # print("\nIMAGE ANALYSES:")
        # for analysis in response['image_analyses']:
        #     print(f"  - {analysis}")

        ## health check test
        # health = await health_check()
        # print(f"Health Check: {health}")
    # asyncio.run(main())