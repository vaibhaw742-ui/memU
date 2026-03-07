import os
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from memu.app.service import MemoryService
from memu.config.settings import (
    DatabaseConfig,
    MetadataStoreConfig,
    VectorIndexConfig,
    LLMProfilesConfig,
    MemorizeConfig,
    RetrieveConfig,
    CategoryConfig,
)

app = FastAPI(title="Lumen MCP Bridge", version="0.1.0")

# ── Initialize MemoryService ──────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def build_service() -> MemoryService:
    llm_profiles = LLMProfilesConfig.model_validate({
        "default": {
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
            "chat_model": "gpt-4o-mini",
            "client_backend": "sdk",
            "embed_model": "text-embedding-3-small",
        },
        "embedding": {
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
            "chat_model": "gpt-4o-mini",
            "client_backend": "sdk",
            "embed_model": "text-embedding-3-small",
        }
    })

    database_config = DatabaseConfig(
        metadata_store=MetadataStoreConfig(
            provider="postgres",
            dsn=DATABASE_URL,
            ddl_mode="create",
        ),
        vector_index=VectorIndexConfig(
            provider="pgvector",
            dsn=DATABASE_URL,
        )
    )

    memorize_config = MemorizeConfig(
        memory_categories=[
            CategoryConfig(name="agents", description="AI agents, autonomous systems, orchestration"),
            CategoryConfig(name="rag", description="Retrieval augmented generation, vector search, embeddings"),
            CategoryConfig(name="llm_training", description="LLM training, fine-tuning, RLHF, datasets"),
            CategoryConfig(name="engineering", description="Software engineering, system design, architecture"),
            CategoryConfig(name="general", description="General knowledge and everything else"),
        ]
    )

    retrieve_config = RetrieveConfig(
        method="rag",
        route_intention=True,
        sufficiency_check=True,
    )

    return MemoryService(
        llm_profiles=llm_profiles,
        database_config=database_config,
        memorize_config=memorize_config,
        retrieve_config=retrieve_config,
        category_md_output_dir="./categories",
    )

service = build_service()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Tool 1: Memorize URL ──────────────────────────────────────────────────────

class MemorizeRequest(BaseModel):
    url: str
    user_id: str
    workspace_id: Optional[str] = None

@app.post("/tools/memorize")
async def memorize(req: MemorizeRequest):
    try:
        result = await service.memorize(
            resource_url=req.url,
            user={"user_id": req.user_id, "workspace_id": req.workspace_id},
        )
        return {
            "status": "ok",
            "user_id": req.user_id,
            "url": req.url,
            "items_extracted": len(result.get("items", [])),
            "categories": [c.get("name") for c in result.get("categories", [])],
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tool 2: Retrieve / Search KB ─────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str
    user_id: str
    workspace_id: Optional[str] = None

@app.post("/tools/retrieve")
async def retrieve(req: RetrieveRequest):
    try:
        result = await service.retrieve(
            queries=[{"role": "user", "content": {"text": req.query}}],
            where={"user_id": req.user_id, "workspace_id": req.workspace_id},
        )

        # Build context string for OpenClaw to synthesize
        context_parts = []
        for cat in result.get("categories", [])[:3]:
            summary = cat.get("summary") or cat.get("name", "")
            if summary:
                context_parts.append(f"[Category: {cat.get('name')}]\n{summary}")
        for item in result.get("items", [])[:5]:
            desc = item.get("description") or item.get("summary", "")
            if desc:
                context_parts.append(f"- {desc}")

        return {
            "status": "ok",
            "query": req.query,
            "context": "\n\n".join(context_parts),
            "items": result.get("items", []),
            "categories": result.get("categories", []),
            "resources": result.get("resources", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tool 3: List memory items ─────────────────────────────────────────────────

class ListItemsRequest(BaseModel):
    user_id: str
    workspace_id: Optional[str] = None
    limit: int = 10

@app.post("/tools/list_items")
async def list_items(req: ListItemsRequest):
    try:
        result = await service.list_memory_items(
            where={"user_id": req.user_id, "workspace_id": req.workspace_id},
        )
        items = result.get("items", [])
        return {
            "status": "ok",
            "total": len(items),
            "items": items[:req.limit],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tool 4: List categories ───────────────────────────────────────────────────

class ListCategoriesRequest(BaseModel):
    user_id: str
    workspace_id: Optional[str] = None

@app.post("/tools/list_categories")
async def list_categories(req: ListCategoriesRequest):
    try:
        result = await service.list_memory_categories(
            where={"user_id": req.user_id, "workspace_id": req.workspace_id},
        )
        return {
            "status": "ok",
            "categories": result.get("categories", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tool 5: Clear memory ──────────────────────────────────────────────────────

class ClearMemoryRequest(BaseModel):
    user_id: str
    workspace_id: Optional[str] = None

@app.post("/tools/clear_memory")
async def clear_memory(req: ClearMemoryRequest):
    try:
        await service.clear_memory(
            where={"user_id": req.user_id, "workspace_id": req.workspace_id},
        )
        return {"status": "ok", "message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))