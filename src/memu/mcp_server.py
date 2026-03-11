import os
import re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg

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

app = FastAPI(title="Lumen MCP Bridge", version="0.2.0")

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CATEGORIES_MD_DIR = Path(os.getenv("CATEGORIES_MD_DIR", "./categories"))
DEFAULT_USER_ID = "default"
DEFAULT_WORKSPACE_ID = "default"

# ── DB helpers ────────────────────────────────────────────────────────────────

async def get_categories_from_db(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID) -> list[dict]:
    """Read categories from memory_categories table."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT id, name, description, summary
            FROM learning.memory_categories
            WHERE user_id = $1 AND workspace_id = $2
            ORDER BY name
            """,
            user_id, workspace_id
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def upsert_category_in_db(name: str, description: str,
                                 user_id: str = DEFAULT_USER_ID,
                                 workspace_id: str = DEFAULT_WORKSPACE_ID) -> str:
    """Insert or update a category, return its id."""
    import uuid
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        existing = await conn.fetchrow(
            """
            SELECT id FROM learning.memory_categories
            WHERE name = $1 AND user_id = $2 AND workspace_id = $3
            """,
            name, user_id, workspace_id
        )
        if existing:
            await conn.execute(
                """
                UPDATE learning.memory_categories
                SET description = $1, updated_at = now()
                WHERE id = $2
                """,
                description, existing["id"]
            )
            return existing["id"]
        else:
            new_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO learning.memory_categories
                  (id, name, description, user_id, workspace_id, updated_at)
                VALUES ($1, $2, $3, $4, $5, now())
                """,
                new_id, name, description, user_id, workspace_id
            )
            return new_id
    finally:
        await conn.close()


async def delete_category_from_db(slug: str, user_id: str = DEFAULT_USER_ID,
                                   workspace_id: str = DEFAULT_WORKSPACE_ID) -> bool:
    """Delete category by slug (name). Returns True if deleted."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        result = await conn.execute(
            """
            DELETE FROM learning.memory_categories
            WHERE name = $1 AND user_id = $2 AND workspace_id = $3
            """,
            slug, user_id, workspace_id
        )
        return result == "DELETE 1"
    finally:
        await conn.close()


# ── Markdown helpers ──────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_category_md(name: str, description: str, summary: str = ""):
    CATEGORIES_MD_DIR.mkdir(parents=True, exist_ok=True)
    path = CATEGORIES_MD_DIR / f"{slug(name)}.md"
    existing_items = ""
    if path.exists():
        content = path.read_text()
        if "## Items" in content:
            existing_items = content.split("## Items", 1)[1]
    content = f"""# {name}

## Description
{description}

## Summary
{summary or "(auto-populated as content is ingested)"}

## Items
{existing_items.strip() if existing_items else "(auto-populated by Lumen when content is ingested)"}
"""
    path.write_text(content)


def delete_category_md(name: str):
    path = CATEGORIES_MD_DIR / f"{slug(name)}.md"
    if path.exists():
        path.unlink()


def append_item_to_category_md(category_name: str, summary_line: str):
    from datetime import date
    path = CATEGORIES_MD_DIR / f"{slug(category_name)}.md"
    if path.exists():
        with open(path, "a") as f:
            f.write(f"\n- [{date.today()}] {summary_line}")


# ── Service builder ───────────────────────────────────────────────────────────

def build_llm_profiles() -> LLMProfilesConfig:
    return LLMProfilesConfig.model_validate({
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


def build_database_config() -> DatabaseConfig:
    return DatabaseConfig(
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


async def build_service_from_db() -> MemoryService:
    """Build MemoryService with categories loaded from DB."""
    rows = await get_categories_from_db()
    if rows:
        categories = [CategoryConfig(name=r["name"], description=r["description"]) for r in rows]
    else:
        # fallback until onboarding is done
        categories = [CategoryConfig(name="general", description="General knowledge and everything else")]

    return MemoryService(
        llm_profiles=build_llm_profiles(),
        database_config=build_database_config(),
        memorize_config=MemorizeConfig(memory_categories=categories),
        retrieve_config=RetrieveConfig(
            method="rag",
            route_intention=True,
            sufficiency_check=True,
        ),
        category_md_output_dir=str(CATEGORIES_MD_DIR),
    )


# ── Startup ───────────────────────────────────────────────────────────────────

service: MemoryService = None  # type: ignore

@app.on_event("startup")
async def startup():
    global service
    service = await build_service_from_db()


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
            user={"user_id": req.user_id, "workspace_id": req.workspace_id or DEFAULT_WORKSPACE_ID},
        )
        updated_categories = [c.get("name") for c in result.get("categories", [])]

        # Append to category .md files
        for cat in result.get("categories", []):
            summary = cat.get("summary") or cat.get("description", "")
            if summary:
                append_item_to_category_md(cat.get("name", ""), summary[:120])

        return {
            "status": "ok",
            "user_id": req.user_id,
            "url": req.url,
            "items_extracted": len(result.get("items", [])),
            "categories": updated_categories,
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
            where={"user_id": req.user_id, "workspace_id": req.workspace_id or DEFAULT_WORKSPACE_ID},
        )
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
            where={"user_id": req.user_id, "workspace_id": req.workspace_id or DEFAULT_WORKSPACE_ID},
        )
        items = result.get("items", [])
        return {"status": "ok", "total": len(items), "items": items[:req.limit]}
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
            where={"user_id": req.user_id, "workspace_id": req.workspace_id or DEFAULT_WORKSPACE_ID},
        )
        return {"status": "ok", "categories": result.get("categories", [])}
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
            where={"user_id": req.user_id, "workspace_id": req.workspace_id or DEFAULT_WORKSPACE_ID},
        )
        return {"status": "ok", "message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Admin: Onboarding ─────────────────────────────────────────────────────────

class CategoryPayload(BaseModel):
    name: str
    description: str

class OnboardRequest(BaseModel):
    categories: list[CategoryPayload]
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID

@app.post("/admin/onboard")
async def onboard(req: OnboardRequest):
    """
    Called at end of onboarding flow.
    Saves all categories to DB, rebuilds service, writes .md files.
    """
    global service
    saved = []
    for cat in req.categories:
        cat_id = await upsert_category_in_db(cat.name, cat.description, req.user_id, req.workspace_id)
        write_category_md(cat.name, cat.description)
        saved.append({"id": cat_id, "name": cat.name, "description": cat.description})

    # Rebuild service with new categories
    service = await build_service_from_db()

    return {
        "status": "ok",
        "message": f"Onboarded with {len(saved)} categories",
        "categories": saved,
    }


# ── Admin: Get categories ─────────────────────────────────────────────────────

@app.get("/admin/categories")
async def get_categories(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID):
    rows = await get_categories_from_db(user_id, workspace_id)
    return {"status": "ok", "categories": rows}


# ── Admin: Add category ───────────────────────────────────────────────────────

class AddCategoryRequest(BaseModel):
    name: str
    description: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID

@app.post("/admin/categories/add")
async def add_category(req: AddCategoryRequest):
    global service
    cat_id = await upsert_category_in_db(req.name, req.description, req.user_id, req.workspace_id)
    write_category_md(req.name, req.description)
    service = await build_service_from_db()
    return {"status": "ok", "id": cat_id, "name": req.name, "slug": slug(req.name)}


# ── Admin: Update category ────────────────────────────────────────────────────

class UpdateCategoryRequest(BaseModel):
    description: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID

@app.put("/admin/categories/{category_name}")
async def update_category(category_name: str, req: UpdateCategoryRequest):
    global service
    cat_id = await upsert_category_in_db(category_name, req.description, req.user_id, req.workspace_id)
    write_category_md(category_name, req.description)
    service = await build_service_from_db()
    return {"status": "ok", "id": cat_id, "name": category_name, "description": req.description}


# ── Admin: Delete category ────────────────────────────────────────────────────

class DeleteCategoryRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID

@app.delete("/admin/categories/{category_name}")
async def delete_category(category_name: str, req: DeleteCategoryRequest):
    global service
    deleted = await delete_category_from_db(category_name, req.user_id, req.workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
    delete_category_md(category_name)
    service = await build_service_from_db()
    return {"status": "ok", "deleted": category_name}


# ── Admin: Reload service ─────────────────────────────────────────────────────

@app.post("/admin/reload")
async def reload_service():
    global service
    service = await build_service_from_db()
    rows = await get_categories_from_db()
    return {
        "status": "reloaded",
        "categories": [r["name"] for r in rows],
    }