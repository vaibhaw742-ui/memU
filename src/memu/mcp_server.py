import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncpg
import httpx
from bs4 import BeautifulSoup

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
    rows = await get_categories_from_db()
    if rows:
        categories = [CategoryConfig(name=r["name"], description=r["description"]) for r in rows]
    else:
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


# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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


# ── Tool 1b: Memorize URL with SSE streaming ──────────────────────────────────

@app.post("/tools/memorize/stream")
async def memorize_stream(req: MemorizeRequest):
    async def event_generator():
        try:
            yield sse("progress", {
                "step": 1,
                "status": "fetching",
                "message": f"Got it! Fetching content from {req.url}..."
            })

            queue: asyncio.Queue = asyncio.Queue()

            async def run_memorize():
                try:
                    await queue.put(("progress", {
                        "step": 2,
                        "status": "extracting",
                        "message": "Extracting and processing content..."
                    }))

                    async def heartbeat():
                        elapsed = 0
                        while True:
                            await asyncio.sleep(30)
                            elapsed += 30
                            await queue.put(("progress", {
                                "step": 2,
                                "status": "extracting",
                                "message": f"Still processing... ({elapsed}s elapsed)"
                            }))

                    hb_task = asyncio.create_task(heartbeat())

                    try:
                        result = await service.memorize(
                            resource_url=req.url,
                            user={
                                "user_id": req.user_id,
                                "workspace_id": req.workspace_id or DEFAULT_WORKSPACE_ID
                            },
                        )
                    finally:
                        hb_task.cancel()
                        try:
                            await hb_task
                        except asyncio.CancelledError:
                            pass

                    await queue.put(("progress", {
                        "step": 3,
                        "status": "categorizing",
                        "message": "Categorizing and saving to knowledge base..."
                    }))

                    updated_categories = [c.get("name") for c in result.get("categories", [])]

                    for cat in result.get("categories", []):
                        summary = cat.get("summary") or cat.get("description", "")
                        if summary:
                            append_item_to_category_md(cat.get("name", ""), summary[:120])

                    await queue.put(("done", {
                        "status": "ok",
                        "user_id": req.user_id,
                        "url": req.url,
                        "items_extracted": len(result.get("items", [])),
                        "categories": updated_categories,
                        "message": f"✅ Saved to: {', '.join(updated_categories)}" if updated_categories else "✅ Saved to knowledge base"
                    }))

                except Exception as e:
                    await queue.put(("error", {
                        "status": "error",
                        "message": f"❌ Failed: {str(e)}"
                    }))

            task = asyncio.create_task(run_memorize())

            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=900)
                    yield sse(event, data)
                    if event in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield sse("error", {
                        "status": "error",
                        "message": "❌ Timeout: memorize did not complete within 15 minutes"
                    })
                    break

            await task

        except Exception as e:
            yield sse("error", {"status": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


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

        # Extract source URLs from resources
        source_urls = []
        for resource in result.get("resources", []):
            url = resource.get("url") or resource.get("resource_url", "")
            if url and url.startswith("http"):
                source_urls.append(url)

        return {
            "status": "ok",
            "query": req.query,
            "context": "\n\n".join(context_parts),
            "items": result.get("items", []),
            "categories": result.get("categories", []),
            "resources": result.get("resources", []),
            "source_urls": list(dict.fromkeys(source_urls)),  # deduplicated
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


# ── Tool 6: Deep Process URL (SSE streaming) ──────────────────────────────────

class DeepProcessRequest(BaseModel):
    url: str
    user_id: str
    workspace_id: Optional[str] = None


@app.post("/tools/deep_process/stream")
async def deep_process_stream(req: DeepProcessRequest):
    async def event_generator():
        try:
            yield sse("progress", {
                "status": "fetching",
                "message": f"Fetching {req.url}..."
            })

            async with httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Lumen/1.0)"}
            ) as client:
                response = await client.get(req.url)
                response.raise_for_status()
                html = response.text
                base_url = f"{response.url.scheme}://{response.url.host}"

            yield sse("progress", {
                "status": "parsing",
                "message": "Parsing page content..."
            })

            soup = BeautifulSoup(html, "html.parser")

            # ── Extract images ────────────────────────────────────────────
            images = []
            seen_srcs = set()
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
                alt = img.get("alt", "").strip()
                if not src or src in seen_srcs:
                    continue

                # Normalize URL
                if src.startswith("//"):
                    src = f"https:{src}"
                elif src.startswith("/"):
                    src = f"{base_url}{src}"
                elif not src.startswith("http"):
                    continue

                # Skip tiny icons/tracking pixels
                try:
                    width = img.get("width", "")
                    height = img.get("height", "")
                    if width and int(str(width).replace("px", "")) < 50:
                        continue
                    if height and int(str(height).replace("px", "")) < 50:
                        continue
                except (ValueError, TypeError):
                    pass

                # Skip common icon/tracker patterns
                skip_patterns = ["icon", "logo", "pixel", "tracker", "1x1", "spacer", "badge"]
                if any(p in src.lower() for p in skip_patterns) and not alt:
                    continue

                seen_srcs.add(src)
                images.append({"src": src, "alt": alt or "Image"})

            yield sse("images", {
                "status": "images_found",
                "count": len(images),
                "images": images[:20],
                "message": f"Found {len(images)} images"
            })

            # ── Extract tables ────────────────────────────────────────────
            tables = []
            for table in soup.find_all("table"):
                headers_row = []
                rows = []

                # Get headers from thead
                thead = table.find("thead")
                if thead:
                    headers_row = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]

                # Get body rows
                tbody = table.find("tbody") or table
                for tr in tbody.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if cells and any(c for c in cells):
                        if not headers_row and not rows:
                            headers_row = cells
                            continue
                        rows.append(cells)

                if not rows and not headers_row:
                    continue

                # Build markdown table
                md_lines = []
                if headers_row:
                    md_lines.append("| " + " | ".join(headers_row) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers_row)) + " |")
                    for row in rows:
                        padded = row[:len(headers_row)]
                        while len(padded) < len(headers_row):
                            padded.append("")
                        md_lines.append("| " + " | ".join(padded) + " |")
                else:
                    for row in rows:
                        md_lines.append("| " + " | ".join(row) + " |")

                if md_lines:
                    tables.append("\n".join(md_lines))

            yield sse("tables", {
                "status": "tables_found",
                "count": len(tables),
                "tables": tables[:10],
                "message": f"Found {len(tables)} tables"
            })

            yield sse("done", {
                "status": "ok",
                "url": req.url,
                "images_count": len(images),
                "tables_count": len(tables),
                "message": f"✅ Deep processed {req.url} — {len(images)} images, {len(tables)} tables found"
            })

        except httpx.HTTPStatusError as e:
            yield sse("error", {
                "status": "error",
                "message": f"❌ HTTP {e.response.status_code} error fetching {req.url}"
            })
        except httpx.TimeoutException:
            yield sse("error", {
                "status": "error",
                "message": f"❌ Timeout fetching {req.url}"
            })
        except Exception as e:
            yield sse("error", {
                "status": "error",
                "message": f"❌ Failed: {str(e)}"
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


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
    global service
    saved = []
    for cat in req.categories:
        cat_id = await upsert_category_in_db(cat.name, cat.description, req.user_id, req.workspace_id)
        write_category_md(cat.name, cat.description)
        saved.append({"id": cat_id, "name": cat.name, "description": cat.description})

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