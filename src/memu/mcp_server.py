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
import frontmatter as fm

from memu.app.service import MemoryService
from memu.config.settings import (
    DatabaseConfig,
    MetadataStoreConfig,
    VectorIndexConfig,
    LLMProfilesConfig,
    MemorizeConfig,
    RetrieveConfig,
    CategoryConfig,
    CustomPrompt,
    PromptBlock,
)

app = FastAPI(title="Lumen MCP Bridge", version="0.2.0")

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "./openclaw_learning/workspace"))
DEFAULT_USER_ID = "default"
DEFAULT_WORKSPACE_ID = "default"


# ── Per-user path helpers ─────────────────────────────────────────────────────

def _user_dir(user_id: str, workspace_id: str) -> Path:
    path = WORKSPACE_ROOT / "users" / f"{user_id}__{workspace_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _supadense_path(user_id: str, workspace_id: str) -> Path:
    return _user_dir(user_id, workspace_id) / "supadense.md"

def _categories_dir(user_id: str, workspace_id: str) -> Path:
    d = _user_dir(user_id, workspace_id) / "categories"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _digests_dir(user_id: str, workspace_id: str) -> Path:
    d = _user_dir(user_id, workspace_id) / "digests"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _memory_path(user_id: str, workspace_id: str) -> Path:
    return _user_dir(user_id, workspace_id) / "MEMORY.md"

def _wiki_dir(user_id: str, workspace_id: str) -> Path:
    d = _user_dir(user_id, workspace_id) / "wiki"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _wiki_page_path(user_id: str, workspace_id: str, slug: str) -> Path:
    return _wiki_dir(user_id, workspace_id) / f"{slug}.md"

def _index_path(user_id: str, workspace_id: str) -> Path:
    return _user_dir(user_id, workspace_id) / "index.md"

def _log_path(user_id: str, workspace_id: str) -> Path:
    return _user_dir(user_id, workspace_id) / "log.md"


# ── Category summary prompt ───────────────────────────────────────────────────

CATEGORY_SUMMARY_PROMPT = CustomPrompt(
    objective=PromptBlock(
        ordinal=10,
        prompt="""
# Memory Organizer

Maintain a running chronological knowledge log by appending new memory items to the existing log.

## Format Requirements
- Preserve the existing log exactly as-is
- Append new memory items in date-wise order (newest first)
- If a date already exists in the log, append new memory_type sections under that date
- If a date is new, insert it in the correct chronological position
- Group entries by date, then by memory_type within each date
- Preserve the original index-style table structure for each memory item
- No merging, no summarizing, no rewriting of existing content
- Maximum {target_length} tokens
"""
    ),
    workflow=PromptBlock(
        ordinal=20,
        prompt="""
# Organization Process
1. Take the existing log from original_content as the base state — output it exactly as received
2. Parse all incoming new memory items
3. Extract date and memory_type for each new item
4. For each new item:
   a. If its date already exists in the log → append the new memory_type section under that date block
   b. If its date is new → insert a new date block in the correct chronological position (newest first)
5. Render each new item with its date header, memory_type label, and original index-style table
6. Do not alter, reorder, or rewrite any existing log content
"""
    ),
    output=PromptBlock(
        ordinal=50,
        prompt="""
# Output Format
```markdown
# {category} — Memory Log

## {today}  <- newest date first

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| 1     | ...   | -         | ...         |
| 1.1   | ...   | ...       | ...         |

---
```

Rules:
- ALWAYS start with the full existing log before appending anything
- New date blocks go in the correct chronological position relative to existing dates
- Never remove or modify any existing entry
Target length: {target_length} tokens
"""
    ),
    input=PromptBlock(
        ordinal=90,
        prompt="""
Category: {category}
Existing Log (preserve as base state): {original_content}
New Memory Items (append these): {new_memory_items_text}
"""
    )
)

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


async def get_source_urls_for_items(items: list[dict]) -> list[str]:
    resource_ids = list({
        item.get("resource_id") for item in items
        if item.get("resource_id")
    })
    if not resource_ids:
        return []
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        placeholders = ", ".join(f"${i+1}" for i in range(len(resource_ids)))
        rows = await conn.fetch(
            f"SELECT DISTINCT url FROM learning.resources "
            f"WHERE id IN ({placeholders}) AND url LIKE 'http%'",
            *resource_ids
        )
        return [r["url"] for r in rows if r["url"]]
    finally:
        await conn.close()


# ── Markdown helpers ──────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_category_md(name: str, description: str, user_id: str = DEFAULT_USER_ID,
                      workspace_id: str = DEFAULT_WORKSPACE_ID, summary: str = ""):
    cats_dir = _categories_dir(user_id, workspace_id)
    path = cats_dir / f"{slug(name)}.md"
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


def delete_category_md(name: str, user_id: str = DEFAULT_USER_ID,
                       workspace_id: str = DEFAULT_WORKSPACE_ID):
    cats_dir = _categories_dir(user_id, workspace_id)
    path = cats_dir / f"{slug(name)}.md"
    if path.exists():
        path.unlink()


def append_item_to_category_md(category_name: str, summary_line: str,
                                user_id: str = DEFAULT_USER_ID,
                                workspace_id: str = DEFAULT_WORKSPACE_ID):
    from datetime import date
    cats_dir = _categories_dir(user_id, workspace_id)
    path = cats_dir / f"{slug(category_name)}.md"
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


async def build_service_from_db(user_id: str = DEFAULT_USER_ID,
                                 workspace_id: str = DEFAULT_WORKSPACE_ID) -> MemoryService:
    rows = await get_categories_from_db(user_id, workspace_id)
    if rows:
        categories = [
            CategoryConfig(
                name=r["name"],
                description=r["description"],
                summary_prompt=CATEGORY_SUMMARY_PROMPT,
            )
            for r in rows
        ]
    else:
        categories = [
            CategoryConfig(
                name="general",
                description="General knowledge and everything else",
                summary_prompt=CATEGORY_SUMMARY_PROMPT,
            )
        ]

    return MemoryService(
        llm_profiles=build_llm_profiles(),
        database_config=build_database_config(),
        memorize_config=MemorizeConfig(
            memory_categories=categories,
            enable_item_reinforcement=True,
        ),
        retrieve_config=RetrieveConfig(
            method="llm",
            route_intention=False,
        ),
        category_md_output_dir=str(_categories_dir(user_id, workspace_id)),
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
        workspace_id = req.workspace_id or DEFAULT_WORKSPACE_ID
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            existing = await conn.fetchrow(
                "SELECT id FROM learning.resources WHERE url = $1 AND user_id = $2 AND workspace_id = $3",
                req.url, req.user_id, workspace_id
            )
        finally:
            await conn.close()
        if existing:
            return {
                "status": "skipped",
                "reason": "already_memorized",
                "url": req.url,
                "message": f"URL already in knowledge base. Resource id: {existing['id']}",
            }
        user_service = await build_service_from_db(req.user_id, workspace_id)
        result = await user_service.memorize(
            resource_url=req.url,
            user={"user_id": req.user_id, "workspace_id": workspace_id},
        )
        updated_categories = [c.get("name") for c in result.get("categories", [])]
        for cat in result.get("categories", []):
            summary = cat.get("summary") or cat.get("description", "")
            if summary:
                append_item_to_category_md(cat.get("name", ""), summary[:120],
                                           req.user_id, workspace_id)
        await _update_index(req.user_id, workspace_id)
        _append_log(req.user_id, workspace_id,
            f"memorize | {req.url} → {', '.join(updated_categories) or 'general'} | "
            f"items: {len(result.get('items', []))}")
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
            workspace_id = req.workspace_id or DEFAULT_WORKSPACE_ID
            conn = await asyncpg.connect(DATABASE_URL)
            try:
                existing = await conn.fetchrow(
                    "SELECT id FROM learning.resources WHERE url = $1 AND user_id = $2 AND workspace_id = $3",
                    req.url, req.user_id, workspace_id
                )
            finally:
                await conn.close()
            if existing:
                yield sse("done", {
                    "status": "skipped",
                    "reason": "already_memorized",
                    "url": req.url,
                    "message": "URL already in knowledge base.",
                })
                return

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
                        user_service = await build_service_from_db(req.user_id, workspace_id)
                        result = await user_service.memorize(
                            resource_url=req.url,
                            user={
                                "user_id": req.user_id,
                                "workspace_id": workspace_id
                            },
                        )
                    finally:
                        hb_task.cancel()
                        try:
                            await hb_task
                        except asyncio.CancelledError:
                            pass

                    updated_categories = [c.get("name") for c in result.get("categories", [])]
                    for cat in result.get("categories", []):
                        summary = cat.get("summary") or cat.get("description", "")
                        if summary:
                            append_item_to_category_md(cat.get("name", ""), summary[:120],
                                                       req.user_id, workspace_id)
                    await _update_index(req.user_id, workspace_id)
                    _append_log(req.user_id, workspace_id,
                        f"memorize | {req.url} → {', '.join(updated_categories) or 'general'} | "
                        f"items: {len(result.get('items', []))}")

                    await queue.put(("done", {
                        "status": "ok",
                        "user_id": req.user_id,
                        "url": req.url,
                        "items_extracted": len(result.get("items", [])),
                        "categories": updated_categories,
                        "message": f"Saved to: {', '.join(updated_categories)}" if updated_categories else "Saved to knowledge base"
                    }))

                except Exception as e:
                    await queue.put(("error", {
                        "status": "error",
                        "message": f"Failed: {str(e)}"
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
                        "message": "Timeout: memorize did not complete within 15 minutes"
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
        workspace_id = req.workspace_id or DEFAULT_WORKSPACE_ID
        user_service = await build_service_from_db(req.user_id, workspace_id)
        result = await user_service.retrieve(
            queries=[{"role": "user", "content": {"text": req.query}}],
            where={"user_id": req.user_id, "workspace_id": workspace_id},
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

        source_urls = await get_source_urls_for_items(result.get("items", []))

        return {
            "status": "ok",
            "query": req.query,
            "context": "\n\n".join(context_parts),
            "items": result.get("items", []),
            "categories": result.get("categories", []),
            "resources": result.get("resources", []),
            "source_urls": source_urls,
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
        workspace_id = req.workspace_id or DEFAULT_WORKSPACE_ID
        user_service = await build_service_from_db(req.user_id, workspace_id)
        result = await user_service.list_memory_items(
            where={"user_id": req.user_id, "workspace_id": workspace_id},
        )
        items = result.get("items", [])
        source_urls = await get_source_urls_for_items(items)
        return {
            "status": "ok",
            "total": len(items),
            "items": items[:req.limit],
            "source_urls": source_urls,
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
        workspace_id = req.workspace_id or DEFAULT_WORKSPACE_ID
        user_service = await build_service_from_db(req.user_id, workspace_id)
        result = await user_service.list_memory_categories(
            where={"user_id": req.user_id, "workspace_id": workspace_id},
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
        workspace_id = req.workspace_id or DEFAULT_WORKSPACE_ID
        user_service = await build_service_from_db(req.user_id, workspace_id)
        await user_service.clear_memory(
            where={"user_id": req.user_id, "workspace_id": workspace_id},
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

            images = []
            seen_srcs = set()
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
                alt = img.get("alt", "").strip()
                if not src or src in seen_srcs:
                    continue
                if src.startswith("//"):
                    src = f"https:{src}"
                elif src.startswith("/"):
                    src = f"{base_url}{src}"
                elif not src.startswith("http"):
                    continue
                try:
                    width = img.get("width", "")
                    height = img.get("height", "")
                    if width and int(str(width).replace("px", "")) < 50:
                        continue
                    if height and int(str(height).replace("px", "")) < 50:
                        continue
                except (ValueError, TypeError):
                    pass
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

            tables = []
            for table in soup.find_all("table"):
                headers_row = []
                rows = []
                thead = table.find("thead")
                if thead:
                    headers_row = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
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
                "message": f"Deep processed {req.url} — {len(images)} images, {len(tables)} tables found"
            })

        except httpx.HTTPStatusError as e:
            yield sse("error", {"status": "error", "message": f"HTTP {e.response.status_code} error fetching {req.url}"})
        except httpx.TimeoutException:
            yield sse("error", {"status": "error", "message": f"Timeout fetching {req.url}"})
        except Exception as e:
            yield sse("error", {"status": "error", "message": f"Failed: {str(e)}"})

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


DEFAULT_CATEGORIES = []


class OnboardRequest(BaseModel):
    categories: list[CategoryPayload] = []
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/admin/onboard")
async def onboard(req: OnboardRequest):
    existing_names = {c.name.lower() for c in req.categories}
    all_categories = req.categories + [c for c in DEFAULT_CATEGORIES if c.name.lower() not in existing_names]
    saved = []
    for cat in all_categories:
        cat_id = await upsert_category_in_db(cat.name, cat.description, req.user_id, req.workspace_id)
        write_category_md(cat.name, cat.description, req.user_id, req.workspace_id)
        saved.append({"id": cat_id, "name": cat.name, "description": cat.description})
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
    cat_id = await upsert_category_in_db(req.name, req.description, req.user_id, req.workspace_id)
    write_category_md(req.name, req.description, req.user_id, req.workspace_id)
    return {"status": "ok", "id": cat_id, "name": req.name, "slug": slug(req.name)}


# ── Admin: Update category ────────────────────────────────────────────────────

class UpdateCategoryRequest(BaseModel):
    description: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.put("/admin/categories/{category_name}")
async def update_category(category_name: str, req: UpdateCategoryRequest):
    cat_id = await upsert_category_in_db(category_name, req.description, req.user_id, req.workspace_id)
    write_category_md(category_name, req.description, req.user_id, req.workspace_id)
    return {"status": "ok", "id": cat_id, "name": category_name, "description": req.description}


# ── Admin: Delete category ────────────────────────────────────────────────────

class DeleteCategoryRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.delete("/admin/categories/{category_name}")
async def delete_category(category_name: str, req: DeleteCategoryRequest):
    if category_name.lower() in ("supadense", "supadense.md"):
        raise HTTPException(status_code=400, detail="supadense.md is protected and cannot be deleted.")
    deleted = await delete_category_from_db(category_name, req.user_id, req.workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
    delete_category_md(category_name, req.user_id, req.workspace_id)
    return {"status": "ok", "deleted": category_name}


# ── Admin: Reload service ─────────────────────────────────────────────────────

@app.post("/admin/reload")
async def reload_service(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID):
    rows = await get_categories_from_db(user_id, workspace_id)
    return {
        "status": "reloaded",
        "categories": [r["name"] for r in rows],
    }


# ── Admin: Regenerate category summaries ──────────────────────────────────────

@app.post("/admin/regenerate_summaries")
async def regenerate_summaries(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT mc.id AS cat_id, mc.name, mc.summary AS cat_summary,
                   mi.id AS item_id, mi.memory_type, mi.summary AS item_summary
            FROM learning.memory_categories mc
            JOIN learning.category_items ci ON ci.category_id = mc.id
            JOIN learning.memory_items mi ON mi.id = ci.item_id
            WHERE mc.user_id = $1 AND mc.workspace_id = $2
            ORDER BY mc.id, mi.created_at
            """,
            user_id, workspace_id
        )
    finally:
        await conn.close()

    from collections import defaultdict
    cat_items: dict = defaultdict(list)
    cat_meta: dict = {}
    for row in rows:
        cid = row["cat_id"]
        cat_meta[cid] = {"name": row["name"], "summary": row["cat_summary"] or ""}
        cat_items[cid].append((row["item_id"], row["memory_type"], row["item_summary"]))

    if not cat_items:
        return {"status": "ok", "message": "No category-item links found", "updated": []}

    user_service = await build_service_from_db(user_id, workspace_id)
    llm_client = user_service._get_llm_client()
    updated = []
    for cid, items in cat_items.items():
        meta = cat_meta[cid]

        class _Cat:
            name = meta["name"]
            summary = meta["summary"]

        try:
            prompt = user_service._build_category_summary_prompt(category=_Cat(), new_memories=items)
            summary_text = await llm_client.summarize(prompt, system_prompt=None)
            cleaned = summary_text.replace("```markdown", "").replace("```", "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error for category '{meta['name']}': {e}")

        conn2 = await asyncpg.connect(DATABASE_URL)
        try:
            await conn2.execute(
                "UPDATE learning.memory_categories SET summary=$1, updated_at=now() WHERE id=$2",
                cleaned, cid
            )
        finally:
            await conn2.close()

        updated.append({"category": meta["name"], "id": cid})

    return {"status": "ok", "updated": updated}


# ── supadense helpers ─────────────────────────────────────────────────────────

def _parse_bullet_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            items.append(line[2:].strip())
    return [i for i in items if i]


def _parse_key_value(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if ": " in line:
            k, v = line.split(": ", 1)
            result[k.strip()] = v.strip()
    return result


def _parse_supadense(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID) -> dict:
    path = _supadense_path(user_id, workspace_id)
    if not path.exists():
        return {"error": f"supadense.md not found for user {user_id}"}
    raw = path.read_text(encoding="utf-8")
    post = fm.loads(raw)
    sections: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []
    for line in post.content.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip().lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return {
        "meta": dict(post.metadata),
        "goals": _parse_bullet_list(sections.get("goals", "")),
        "gaps": _parse_bullet_list(sections.get("gaps", "")),
        "learning_intent": sections.get("learning_intent", "").strip(),
        "trusted_sources": _parse_bullet_list(sections.get("trusted_sources", "")),
        "depth_preferences": _parse_key_value(sections.get("depth_preferences", "")),
        "scout_config": post.metadata.get("scout_config", {}),
        "last_synthesis_at": post.metadata.get("last_synthesis_at"),
    }


def _update_supadense_meta(key: str, value, user_id: str = DEFAULT_USER_ID,
                            workspace_id: str = DEFAULT_WORKSPACE_ID) -> None:
    path = _supadense_path(user_id, workspace_id)
    raw = path.read_text(encoding="utf-8")
    post = fm.loads(raw)
    post.metadata[key] = value
    path.write_text(fm.dumps(post), encoding="utf-8")


def _append_to_supadense_section(section_title: str, new_items: list[str],
                                  user_id: str = DEFAULT_USER_ID,
                                  workspace_id: str = DEFAULT_WORKSPACE_ID) -> None:
    path = _supadense_path(user_id, workspace_id)
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    target = f"## {section_title}"
    section_start = None
    section_end = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == target:
            section_start = i
        elif section_start is not None and line.startswith("## "):
            section_end = i
            break
    if section_start is None:
        lines.append("")
        lines.append(target)
        for item in new_items:
            lines.append(f"- {item}")
    else:
        for item in reversed(new_items):
            lines.insert(section_end, f"- {item}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── supadense endpoints ───────────────────────────────────────────────────────

@app.get("/tools/supadense/read")
async def supadense_read(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID):
    try:
        return {"status": "ok", "data": _parse_supadense(user_id, workspace_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateGoalsRequest(BaseModel):
    goals: list[str]
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/supadense/update_goals")
async def supadense_update_goals(req: UpdateGoalsRequest):
    try:
        _append_to_supadense_section("Goals", req.goals, req.user_id, req.workspace_id)
        return {
            "status": "ok",
            "added": req.goals,
            "message": f"Added {len(req.goals)} goal(s) to supadense.md",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── MEMORY.md endpoints ───────────────────────────────────────────────────────

@app.get("/tools/memory/read")
async def memory_read(user_id: str = DEFAULT_USER_ID, workspace_id: str = DEFAULT_WORKSPACE_ID):
    try:
        path = _memory_path(user_id, workspace_id)
        if not path.exists():
            return {"status": "ok", "content": "", "exists": False}
        return {"status": "ok", "content": path.read_text(encoding="utf-8"), "exists": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MemoryWriteRequest(BaseModel):
    content: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/memory/write")
async def memory_write(req: MemoryWriteRequest):
    try:
        path = _memory_path(req.user_id, req.workspace_id)
        path.write_text(req.content, encoding="utf-8")
        return {"status": "ok", "message": "MEMORY.md written", "path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MemoryAppendRequest(BaseModel):
    content: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/memory/append")
async def memory_append(req: MemoryAppendRequest):
    try:
        path = _memory_path(req.user_id, req.workspace_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{req.content}")
        return {"status": "ok", "message": "Appended to MEMORY.md", "path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Onboarding ────────────────────────────────────────────────────────────────

async def get_or_create_learning_profile(user_id: str, workspace_id: str) -> dict:
    import uuid
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, workspace_id, onboarded_at,
                   last_synthesis_at, depth_prefs, spaced_rep_config
            FROM learning.learning_profiles
            WHERE user_id = $1 AND workspace_id = $2
            """,
            user_id, workspace_id
        )
        if row:
            return dict(row)
        new_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO learning.learning_profiles
              (id, user_id, workspace_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, workspace_id) DO NOTHING
            """,
            new_id, user_id, workspace_id
        )
        row = await conn.fetchrow(
            """
            SELECT id, user_id, workspace_id, onboarded_at,
                   last_synthesis_at, depth_prefs, spaced_rep_config
            FROM learning.learning_profiles
            WHERE user_id = $1 AND workspace_id = $2
            """,
            user_id, workspace_id
        )
        if row:
            return dict(row)
        return {"id": new_id, "user_id": user_id, "workspace_id": workspace_id,
                "onboarded_at": None, "depth_prefs": {}}
    finally:
        await conn.close()


@app.get("/tools/onboarding/status")
async def onboarding_status(
    user_id: str = DEFAULT_USER_ID,
    workspace_id: str = DEFAULT_WORKSPACE_ID
):
    try:
        profile = await get_or_create_learning_profile(user_id, workspace_id)
        return {
            "status": "ok",
            "onboarded": profile["onboarded_at"] is not None,
            "onboarded_at": str(profile["onboarded_at"]) if profile["onboarded_at"] else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class OnboardingCompleteRequest(BaseModel):
    goals: list[str]
    gaps: list[str]
    learning_intent: str
    depth_preferences: dict[str, str]
    trusted_sources: list[str]
    scout_platforms: list[str]
    categories: list[dict]
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/onboarding/complete")
async def onboarding_complete(req: OnboardingCompleteRequest):
    from datetime import datetime, timezone

    # Ensure user dir exists
    _user_dir(req.user_id, req.workspace_id)

    goals_lines = "\n".join(f"- {g}" for g in req.goals)
    gaps_lines = "\n".join(f"- {g}" for g in req.gaps)
    sources_lines = "\n".join(f"- {s}" for s in req.trusted_sources)
    depth_lines = "\n".join(f"{k}: {v}" for k, v in req.depth_preferences.items())

    supadense_content = f"""---
version: 1
last_synthesis_at: null
last_digest_item_ids: []
scout_config:
  platforms: {json.dumps(req.scout_platforms)}
  scroll_depth: 25
  relevance_threshold: 0.72
  auto_ingest: true
  notify_threshold: 0.5
---

## Goals
{goals_lines}

## Gaps
{gaps_lines}

## Learning intent
{req.learning_intent}

## Trusted sources
{sources_lines}

## Depth preferences
{depth_lines}
"""
    supadense_p = _supadense_path(req.user_id, req.workspace_id)
    supadense_p.write_text(supadense_content, encoding="utf-8")

    # Init MEMORY.md if it doesn't exist
    mem_path = _memory_path(req.user_id, req.workspace_id)
    if not mem_path.exists():
        mem_path.write_text(
            f"# Lumen Memory — {req.user_id}\n\n"
            f"*Initialised during onboarding*\n\n"
            f"## User profile\n- Learning intent: {req.learning_intent}\n",
            encoding="utf-8"
        )

    profile = await get_or_create_learning_profile(req.user_id, req.workspace_id)
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            """
            UPDATE learning.learning_profiles
            SET depth_prefs = $1, onboarded_at = $2, updated_at = now()
            WHERE id = $3
            """,
            json.dumps(req.depth_preferences),
            datetime.now(timezone.utc),
            profile["id"]
        )
    finally:
        await conn.close()

    existing = await get_categories_from_db(req.user_id, req.workspace_id)
    if not existing:
        cats_to_create = req.categories if req.categories else []
        for cat in cats_to_create:
            await upsert_category_in_db(
                cat["name"], cat.get("description", ""),
                req.user_id, req.workspace_id
            )
            write_category_md(cat["name"], cat.get("description", ""),
                              req.user_id, req.workspace_id)

    # Create skeleton index.md and first log entry immediately
    await _update_index(req.user_id, req.workspace_id)
    _append_log(req.user_id, req.workspace_id,
        f"onboarding | user: {req.user_id} | workspace: {req.workspace_id} | "
        f"categories: {', '.join(c['name'] for c in req.categories)} | "
        f"goals: {len(req.goals)} | gaps: {len(req.gaps)}")

    return {
        "status": "ok",
        "message": "Onboarding complete. Lumen is ready.",
        "user_dir": str(_user_dir(req.user_id, req.workspace_id)),
        "goals": req.goals,
        "gaps": req.gaps,
        "depth_preferences": req.depth_preferences,
        "scout_platforms": req.scout_platforms,
        "categories_created": [c["name"] for c in req.categories],
        "index_created": str(_index_path(req.user_id, req.workspace_id)),
        "log_created": str(_log_path(req.user_id, req.workspace_id)),
    }


# ── Synthesis helpers ─────────────────────────────────────────────────────────

async def _get_items_since(user_id: str, workspace_id: str, since=None) -> list[dict]:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if since:
            rows = await conn.fetch(
                """
                SELECT mi.id, mi.memory_type, mi.summary, mc.name as category, mi.created_at
                FROM learning.memory_items mi
                LEFT JOIN learning.category_items ci ON ci.item_id = mi.id
                LEFT JOIN learning.memory_categories mc ON mc.id = ci.category_id
                WHERE mi.user_id = $1 AND mi.workspace_id = $2
                AND mi.created_at > $3
                ORDER BY mi.created_at DESC
                """,
                user_id, workspace_id, since
            )
        else:
            rows = await conn.fetch(
                """
                SELECT mi.id, mi.memory_type, mi.summary, mc.name as category, mi.created_at
                FROM learning.memory_items mi
                LEFT JOIN learning.category_items ci ON ci.item_id = mi.id
                LEFT JOIN learning.memory_categories mc ON mc.id = ci.category_id
                WHERE mi.user_id = $1 AND mi.workspace_id = $2
                ORDER BY mi.created_at DESC
                LIMIT 50
                """,
                user_id, workspace_id
            )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _update_last_synthesis(user_id: str, workspace_id: str, item_ids: list[str]):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            """
            UPDATE learning.learning_profiles
            SET last_synthesis_at = $1, last_digest_item_ids = $2, updated_at = now()
            WHERE user_id = $3 AND workspace_id = $4
            """,
            now, json.dumps(item_ids), user_id, workspace_id
        )
    finally:
        await conn.close()
    supadense_p = _supadense_path(user_id, workspace_id)
    if supadense_p.exists():
        _update_supadense_meta("last_synthesis_at", now.isoformat(), user_id, workspace_id)


async def _llm_call(prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


async def _score_items_relevance(
    items: list[dict],
    goals: list[str],
    learning_intent: str,
    gaps: list[str],
    trusted_sources: list[str],
) -> list[dict]:
    if not items:
        return []

    goals_text = "\n".join(f"- {g}" for g in goals)
    gaps_text = "\n".join(f"- {g}" for g in gaps)
    sources_lower = [s.lower() for s in trusted_sources]

    items_for_scoring = []
    for i, item in enumerate(items):
        summary_snippet = (item.get("summary") or "")[:200]
        items_for_scoring.append(f"[{i}] category={item.get('category','?')} | {summary_snippet}")

    items_text = "\n".join(items_for_scoring)

    scoring_prompt = f"""You are a learning relevance scorer. Score each knowledge item for relevance to the learner's goals.

## Learner Goals
{goals_text}

## Learner Gaps
{gaps_text}

## Learning Intent
{learning_intent}

## Items to score (format: [index] category | summary)
{items_text}

## Instructions
For each item return a JSON array. Each element must have:
- "index": the item index number
- "score": float 0.0-1.0 (1.0 = directly advances a goal or fills a gap, 0.0 = completely irrelevant)
- "matched_goal": which goal it advances (exact string from goals list, or null)
- "fills_gap": which gap it addresses (exact string from gaps list, or null)

Return ONLY a valid JSON array, no other text. Example:
[{{"index": 0, "score": 0.85, "matched_goal": "Master agentic AI system design", "fills_gap": null}}]"""

    try:
        raw = await _llm_call(scoring_prompt, max_tokens=500, temperature=0.1)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        scores = json.loads(raw)
        score_map = {s["index"]: s for s in scores}
    except Exception:
        return [{**item, "relevance_score": 0.5, "matched_goal": None,
                 "fills_gap": None, "trusted_source_boost": False} for item in items]

    scored_items = []
    for i, item in enumerate(items):
        score_data = score_map.get(i, {"score": 0.3, "matched_goal": None, "fills_gap": None})
        base_score = float(score_data.get("score", 0.3))
        summary_lower = (item.get("summary") or "").lower()
        is_trusted = any(src in summary_lower for src in sources_lower)
        final_score = min(1.0, base_score + 0.15) if is_trusted else base_score
        scored_items.append({
            **item,
            "relevance_score": round(final_score, 3),
            "matched_goal": score_data.get("matched_goal"),
            "fills_gap": score_data.get("fills_gap"),
            "trusted_source_boost": is_trusted,
        })

    scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored_items


async def _get_existing_items_by_category(
    user_id: str, workspace_id: str, since, categories: list[str]
) -> dict[str, list[str]]:
    if not categories:
        return {}
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if since:
            rows = await conn.fetch(
                """
                SELECT mi.summary, mc.name as category
                FROM learning.memory_items mi
                JOIN learning.category_items ci ON ci.item_id = mi.id
                JOIN learning.memory_categories mc ON mc.id = ci.category_id
                WHERE mi.user_id = $1 AND mi.workspace_id = $2
                AND mi.created_at <= $3
                ORDER BY mc.name, mi.created_at DESC
                LIMIT 100
                """,
                user_id, workspace_id, since
            )
        else:
            return {}
        by_cat: dict[str, list[str]] = {}
        for row in rows:
            cat = row["category"] or "general"
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append((row["summary"] or "")[:300])
        return by_cat
    finally:
        await conn.close()


def _parse_topic_rows(summary: str) -> list[dict]:
    rows = []
    for line in summary.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            rows.append({
                "index": parts[0] if len(parts) > 0 else "",
                "topic": parts[1] if len(parts) > 1 else "",
                "subtopic": parts[2] if len(parts) > 2 else "",
                "description": parts[3] if len(parts) > 3 else "",
            })
    return rows


def _make_topic_key(topic: str, subtopic: str) -> str:
    t = re.sub(r"\s+", " ", topic.lower().strip())
    s = re.sub(r"\s+", " ", subtopic.lower().strip())
    return f"{t}||{s}"


def _diff_topics(
    new_items_by_category: dict[str, list[dict]],
    existing_by_category: dict[str, list[str]],
) -> dict[str, dict]:
    result = {}

    for cat, new_items in new_items_by_category.items():
        existing_summaries = existing_by_category.get(cat, [])

        existing_keys: set[str] = set()
        for summary in existing_summaries:
            for row in _parse_topic_rows(summary):
                if row["topic"] and row["topic"] != "-":
                    existing_keys.add(_make_topic_key(row["topic"], row["subtopic"]))

        novel_topics = []
        reinforced_topics = []
        all_new_rows = []

        for item in new_items:
            summary = item.get("summary") or ""
            for row in _parse_topic_rows(summary):
                if not row["topic"] or row["topic"] == "-":
                    continue
                all_new_rows.append(row)
                key = _make_topic_key(row["topic"], row["subtopic"])
                subtopic_display = f" → {row['subtopic']}" if row["subtopic"] and row["subtopic"] != "-" else ""
                display = f"{row['topic']}{subtopic_display}"
                entry = {
                    "topic": row["topic"],
                    "subtopic": row["subtopic"] if row["subtopic"] != "-" else "",
                    "description": row["description"],
                    "display": display,
                }
                if key not in existing_keys:
                    novel_topics.append(entry)
                else:
                    reinforced_topics.append(entry)

        seen: set[str] = set()
        deduped_novel = []
        for t in novel_topics:
            k = _make_topic_key(t["topic"], t["subtopic"])
            if k not in seen:
                seen.add(k)
                deduped_novel.append(t)

        seen = set()
        deduped_reinforced = []
        for t in reinforced_topics:
            k = _make_topic_key(t["topic"], t["subtopic"])
            if k not in seen:
                seen.add(k)
                deduped_reinforced.append(t)

        total = len(all_new_rows)
        novel_count = len(deduped_novel)
        novelty_score = round(novel_count / total, 2) if total > 0 else 0.0

        result[cat] = {
            "novel_topics": deduped_novel,
            "reinforced_topics": deduped_reinforced,
            "novelty_score": novelty_score,
            "total_new_rows": total,
            "novel_count": novel_count,
            "reinforced_count": len(deduped_reinforced),
            "summary": f"{novel_count} new topics, {len(deduped_reinforced)} reinforced out of {total} total rows",
        }

    return result


async def _score_item_quality(
    items: list[dict],
    goals: list[str],
    gaps: list[str],
) -> dict[int, dict]:
    """
    LLM-judged quality scoring for each item. Single batched call.
    Returns dict keyed by item index with depth, specificity, goal_match, gap_addressed scores.
    """
    if not items:
        return {}

    goals_text = "\n".join(f"- {g}" for g in goals)
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    items_text = ""
    for i, item in enumerate(items):
        summary_snippet = (item.get("summary") or "")[:300]
        items_text += f"\n[{i}] category={item.get('category','?')}\n{summary_snippet}\n"

    prompt = f"""You are a learning content quality assessor. Evaluate each knowledge item on 4 dimensions.

## Learner Goals
{goals_text}

## Learner Gaps
{gaps_text}

## Items to assess
{items_text}

## Instructions
Return a JSON array. Each element must have:
- "index": item index number
- "depth": "low" | "medium" | "high"
  low = overview/intro only, medium = explains concepts with some detail, high = internals/implementation/specific algorithms
- "depth_reason": one sentence explaining the depth rating
- "specificity": "low" | "medium" | "high"
  low = general concepts only, medium = named techniques/approaches, high = specific code/numbers/benchmarks/exact parameters
- "specificity_reason": one sentence explaining the specificity rating
- "goal_match": "none" | "weak" | "strong"
  none = unrelated, weak = tangentially related, strong = directly advances a goal
- "goal_match_reason": one sentence, which goal and how
- "gap_addressed": "none" | "partial" | "full"
  none = doesn't address any gap, partial = touches on a gap, full = substantially fills a gap
- "gap_addressed_reason": one sentence, which gap and how much

Return ONLY a valid JSON array, no other text."""

    try:
        raw = await _llm_call(prompt, max_tokens=1000, temperature=0.1)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        scores = json.loads(raw)
        return {s["index"]: s for s in scores}
    except Exception:
        return {
            i: {
                "depth": "medium", "depth_reason": "",
                "specificity": "medium", "specificity_reason": "",
                "goal_match": "weak", "goal_match_reason": "",
                "gap_addressed": "none", "gap_addressed_reason": "",
            }
            for i in range(len(items))
        }


def _find_similar_in_kb(
    new_items_by_category: dict[str, list[dict]],
    existing_by_category: dict[str, list[str]],
) -> dict[str, list[dict]]:
    """
    For each new item, find existing KB items that share topic keys.
    Returns per category a list of {new_item_memory_type, similar_topics} dicts.
    No LLM needed — pure topic key intersection.
    """
    result = {}

    for cat, new_items in new_items_by_category.items():
        existing_summaries = existing_by_category.get(cat, [])
        if not existing_summaries:
            result[cat] = []
            continue

        # Build existing topics with their source summary index
        existing_topic_map: dict[str, list[str]] = {}  # key -> list of topic displays
        for summary in existing_summaries:
            for row in _parse_topic_rows(summary):
                if not row["topic"] or row["topic"] == "-":
                    continue
                key = _make_topic_key(row["topic"], row["subtopic"])
                subtopic_display = f" → {row['subtopic']}" if row["subtopic"] and row["subtopic"] != "-" else ""
                display = f"{row['topic']}{subtopic_display}"
                if key not in existing_topic_map:
                    existing_topic_map[key] = []
                if display not in existing_topic_map[key]:
                    existing_topic_map[key].append(display)

        cat_similar = []
        for item in new_items:
            summary = item.get("summary") or ""
            memory_type = item.get("memory_type") or "unknown"
            shared_topics = []
            for row in _parse_topic_rows(summary):
                if not row["topic"] or row["topic"] == "-":
                    continue
                key = _make_topic_key(row["topic"], row["subtopic"])
                if key in existing_topic_map:
                    subtopic_display = f" → {row['subtopic']}" if row["subtopic"] and row["subtopic"] != "-" else ""
                    shared_topics.append(f"{row['topic']}{subtopic_display}")

            if shared_topics:
                cat_similar.append({
                    "memory_type": memory_type,
                    "shared_topics": list(dict.fromkeys(shared_topics)),  # dedup preserving order
                    "overlap_count": len(shared_topics),
                })

        result[cat] = cat_similar

    return result


async def _run_synthesis(user_id: str, workspace_id: str, trigger: str = "manual") -> dict:
    from datetime import datetime, timezone, date

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow(
            """
            SELECT last_synthesis_at, depth_prefs
            FROM learning.learning_profiles
            WHERE user_id = $1 AND workspace_id = $2
            """,
            user_id, workspace_id
        )
    finally:
        await conn.close()

    last_synthesis_at = row["last_synthesis_at"] if row else None
    depth_prefs_raw = row["depth_prefs"] if row else {}
    if isinstance(depth_prefs_raw, str):
        try:
            depth_prefs = json.loads(depth_prefs_raw)
        except Exception:
            depth_prefs = {}
    else:
        depth_prefs = depth_prefs_raw or {}

    items = await _get_items_since(user_id, workspace_id, since=last_synthesis_at)

    if not items:
        return {
            "status": "skipped",
            "reason": "no_new_items",
            "message": "No new items since last synthesis.",
            "trigger": trigger,
        }

    supadense = _parse_supadense(user_id, workspace_id)
    goals = supadense.get("goals", [])
    learning_intent = supadense.get("learning_intent", "")
    gaps = supadense.get("gaps", [])
    trusted_sources = supadense.get("trusted_sources", [])
    supadense_depth_prefs = supadense.get("depth_preferences", {})
    merged_depth_prefs = {**supadense_depth_prefs, **depth_prefs}

    scored_items = await _score_items_relevance(
        items, goals, learning_intent, gaps, trusted_sources
    )

    relevant_items = [i for i in scored_items if i["relevance_score"] >= 0.4]
    if len(relevant_items) < 3:
        relevant_items = scored_items[:3]

    by_category: dict = {}
    for item in relevant_items:
        cat = item.get("category") or "general"
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    existing_by_category = await _get_existing_items_by_category(
        user_id, workspace_id, since=last_synthesis_at, categories=list(by_category.keys())
    )

    novelty = _diff_topics(by_category, existing_by_category)

    # Quality scoring — LLM judged
    quality_map = await _score_item_quality(relevant_items, goals, gaps)

    # Similar links already in KB
    similar_in_kb = _find_similar_in_kb(by_category, existing_by_category)

    # Attach quality scores to relevant_items
    for i, item in enumerate(relevant_items):
        q = quality_map.get(i, {})
        item["depth"] = q.get("depth", "medium")
        item["depth_reason"] = q.get("depth_reason", "")
        item["specificity"] = q.get("specificity", "medium")
        item["specificity_reason"] = q.get("specificity_reason", "")
        item["goal_match"] = q.get("goal_match", "weak")
        item["goal_match_reason"] = q.get("goal_match_reason", "")
        item["gap_addressed"] = q.get("gap_addressed", "none")
        item["gap_addressed_reason"] = q.get("gap_addressed_reason", "")

    items_text = ""
    for cat, cat_items in by_category.items():
        depth_target = merged_depth_prefs.get(cat, "working")
        cat_novelty = novelty.get(cat, {})
        novelty_score = cat_novelty.get("novelty_score", 0.0)
        novelty_summary = cat_novelty.get("summary", "")
        cat_similar = similar_in_kb.get(cat, [])
        items_text += f"\n### {cat} (depth target: {depth_target} | novelty: {novelty_score:.0%})\n"
        if novelty_summary:
            items_text += f"*{novelty_summary}*\n"
        for item in cat_items[:8]:
            score = item["relevance_score"]
            depth = item.get("depth", "medium")
            specificity = item.get("specificity", "medium")
            goal_match = item.get("goal_match", "weak")
            gap_addressed = item.get("gap_addressed", "none")
            matched_goal = item.get("matched_goal") or "general learning"
            fills_gap = item.get("fills_gap")
            trusted = "TRUSTED SOURCE " if item.get("trusted_source_boost") else ""
            summary_snippet = (item.get("summary") or "")[:300]
            gap_note = f" | fills gap: {fills_gap}" if fills_gap else ""
            items_text += (
                f"- [relevance={score} | depth={depth} | specificity={specificity} | "
                f"goal_match={goal_match} | gap_addressed={gap_addressed}] {trusted}\n"
                f"  goal: {matched_goal}{gap_note}\n"
                f"  {item.get('depth_reason','')}\n"
                f"  {summary_snippet}\n"
            )
        if cat_similar:
            items_text += "Similar already in KB:\n"
            for s in cat_similar[:3]:
                shared = ", ".join(s["shared_topics"][:5])
                items_text += f"  ~ {s['memory_type']}: shares [{shared}]\n"

    novelty_text = ""
    for cat, n in novelty.items():
        novel_topics = n.get("novel_topics", [])
        reinforced = n.get("reinforced_topics", [])
        novelty_score = n.get("novelty_score", 0.0)
        novelty_text += f"\n**{cat}** (novelty: {novelty_score:.0%} — {n.get('novel_count', 0)} new, {n.get('reinforced_count', 0)} reinforced)\n"
        if novel_topics:
            novelty_text += "Genuinely new topics not previously in KB:\n"
            for t in novel_topics[:20]:
                desc = f" — {t['description'][:80]}" if t.get("description") else ""
                novelty_text += f"  + {t['display']}{desc}\n"
        else:
            novelty_text += "  (no novel topics — all content already in KB)\n"
        if reinforced:
            novelty_text += "Already in KB (reinforced by this source):\n"
            for t in reinforced[:10]:
                novelty_text += f"  = {t['display']}\n"

    depth_instructions = ""
    for cat, depth in merged_depth_prefs.items():
        if depth == "deep":
            depth_instructions += f"- {cat}: go deep — explain internals, surface open questions, connect to other concepts\n"
        elif depth == "working":
            depth_instructions += f"- {cat}: practical focus — key takeaways and how to apply\n"
        else:
            depth_instructions += f"- {cat}: surface awareness only — one-liner summary\n"

    goals_text = "\n".join(f"- {g}" for g in goals)
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    prompt = f"""You are Lumen, a personal learning agent. Generate a targeted, personalised learning digest.

## Learner Profile
**Goals:**
{goals_text}

**Gaps:**
{gaps_text}

**Learning Intent:**
{learning_intent}

**Depth instructions per category:**
{depth_instructions or "- default: working depth for all categories"}

## Relevant new knowledge ({len(relevant_items)} items with quality scores)
{items_text}

## Knowledge delta (what's new vs already in KB)
{novelty_text or "First synthesis — no existing KB to compare against."}

## Instructions
Write a targeted digest with these sections:

**What's genuinely new in your KB**
Only cover topics flagged as novel. For each:
- Name the exact topic/subtopic
- Explain at correct depth (use the depth score as guide — high = go deep)
- State which goal it advances
- If specificity is high, call out the specific detail (code, number, benchmark)
- If similar content already exists in KB, note what's different about this new item

**What this reinforces**
Brief — list topics this confirms you already knew. 2-3 bullets max.

**What this updates**
If new content corrects or deepens prior understanding, call it out explicitly.

**Connections**
How does this connect to your goals or other knowledge?

**Gaps this surfaces**
What do you still not know? Be specific.

**Suggested next**
1-2 specific topics to seek out next, tied to your goals and remaining gaps.

Be direct, dense, and personal. Talk to the learner as "you". Max 600 words."""

    digest_text = await _llm_call(prompt, max_tokens=1000, temperature=0.7)

    item_ids = [item["id"] for item in items]
    await _update_last_synthesis(user_id, workspace_id, item_ids)

    resources_table = "\n\n---\n\n## Knowledge sources\n\n"
    resources_table += "| # | Category | Relevance | Depth | Specificity | Goal match | Gap addressed | Novelty | Trusted |\n"
    resources_table += "|---|----------|-----------|-------|-------------|------------|---------------|---------|--------|\n"
    for idx, item in enumerate(relevant_items, 1):
        cat = item.get("category") or "general"
        score = item["relevance_score"]
        depth = item.get("depth", "-")
        specificity = item.get("specificity", "-")
        goal_match = item.get("goal_match", "-")
        gap_addressed = item.get("gap_addressed", "-")
        cat_novelty = novelty.get(cat, {})
        novelty_score = cat_novelty.get("novelty_score", "-")
        novelty_pct = f"{novelty_score:.0%}" if isinstance(novelty_score, float) else "-"
        trusted = "yes" if item.get("trusted_source_boost") else "-"
        resources_table += f"| {idx} | {cat} | {score} | {depth} | {specificity} | {goal_match} | {gap_addressed} | {novelty_pct} | {trusted} |\n"

    # Similar links section
    similar_text = "\n\n## Similar already in your KB\n\n"
    has_similar = False
    for cat, similar_list in similar_in_kb.items():
        if similar_list:
            has_similar = True
            similar_text += f"**{cat}**\n"
            for s in similar_list:
                shared = ", ".join(s["shared_topics"][:8])
                similar_text += f"- *{s['memory_type']}* — shares topics: {shared}\n"
    if not has_similar:
        similar_text += "*No similar content found in existing KB — this is all new territory.*\n"

    digests_dir = _digests_dir(user_id, workspace_id)
    digest_file = digests_dir / f"{date.today().isoformat()}.md"
    digest_file.write_text(
        f"# Lumen Digest — {date.today().isoformat()}\n\n"
        f"*Trigger: {trigger} | Items synthesized: {len(relevant_items)} / {len(items)} total*\n\n"
        f"{digest_text}"
        f"{resources_table}"
        f"{similar_text}\n",
        encoding="utf-8"
    )

    _append_log(user_id, workspace_id,
        f"digest | {trigger} | items: {len(relevant_items)}/{len(items)} | "
        f"categories: {', '.join(by_category.keys())} | file: {digest_file.name}")

    return {
        "status": "ok",
        "trigger": trigger,
        "items_total": len(items),
        "items_relevant": len(relevant_items),
        "items_filtered_out": len(items) - len(relevant_items),
        "categories": list(by_category.keys()),
        "digest": digest_text + resources_table + similar_text,
        "digest_file": str(digest_file),
        "last_synthesis_at": datetime.now(timezone.utc).isoformat(),
        "novelty": {
            cat: {
                "novelty_score": n.get("novelty_score", 0.0),
                "novel_topics": n.get("novel_topics", []),
                "reinforced_topics": n.get("reinforced_topics", []),
                "novel_count": n.get("novel_count", 0),
                "reinforced_count": n.get("reinforced_count", 0),
                "total_new_rows": n.get("total_new_rows", 0),
                "summary": n.get("summary", ""),
            }
            for cat, n in novelty.items()
        },
        "quality_scores": [
            {
                "category": i.get("category"),
                "memory_type": i.get("memory_type"),
                "relevance": i["relevance_score"],
                "depth": i.get("depth"),
                "depth_reason": i.get("depth_reason"),
                "specificity": i.get("specificity"),
                "specificity_reason": i.get("specificity_reason"),
                "goal_match": i.get("goal_match"),
                "goal_match_reason": i.get("goal_match_reason"),
                "gap_addressed": i.get("gap_addressed"),
                "gap_addressed_reason": i.get("gap_addressed_reason"),
            }
            for i in relevant_items
        ],
        "similar_in_kb": similar_in_kb,
    }


# ── Synthesis endpoints ───────────────────────────────────────────────────────

class SynthesisRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/synthesis/digest")
async def synthesis_digest(req: SynthesisRequest):
    """Daily digest — called by openclaw cron at 07:00."""
    try:
        result = await _run_synthesis(req.user_id, req.workspace_id, trigger="daily_digest")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Wiki helpers ──────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


async def _update_index(user_id: str, workspace_id: str) -> None:
    """Regenerate index.md for user from DB categories + wiki/ pages."""
    from datetime import datetime, timezone

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT name, description, parent_category,
                   COUNT(ci.item_id) as item_count
            FROM learning.memory_categories mc
            LEFT JOIN learning.category_items ci ON ci.category_id = mc.id
            WHERE mc.user_id = $1 AND mc.workspace_id = $2
            GROUP BY mc.name, mc.description, mc.parent_category
            ORDER BY mc.parent_category NULLS FIRST, mc.name
            """,
            user_id, workspace_id
        )
    finally:
        await conn.close()

    wiki_d = _wiki_dir(user_id, workspace_id)

    # Separate categories, subcategories, concepts
    categories: dict[str, dict] = {}
    subcategories: dict[str, list] = {}

    for row in rows:
        name = row["name"]
        parent = row["parent_category"]
        item_count = row["item_count"] or 0
        if parent:
            if parent not in subcategories:
                subcategories[parent] = []
            subcategories[parent].append({
                "name": name,
                "slug": _slug(name),
                "description": row["description"] or "",
                "items": item_count,
            })
        else:
            categories[name] = {
                "slug": _slug(name),
                "description": row["description"] or "",
                "items": item_count,
            }

    # Find concept pages — wiki pages with type: concept in frontmatter
    concept_pages = []
    if wiki_d.exists():
        for page_file in wiki_d.glob("*.md"):
            try:
                raw = page_file.read_text(encoding="utf-8")
                post = fm.loads(raw)
                if post.metadata.get("type") == "concept":
                    used_by = post.metadata.get("used_by", [])
                    concept_pages.append({
                        "name": post.metadata.get("title", page_file.stem),
                        "slug": page_file.stem,
                        "used_by": used_by if isinstance(used_by, list) else [used_by],
                    })
            except Exception:
                pass

    # Count totals
    total_sources_row = None
    try:
        conn2 = await asyncpg.connect(DATABASE_URL)
        try:
            total_sources_row = await conn2.fetchval(
                "SELECT COUNT(*) FROM learning.resources WHERE user_id = $1 AND workspace_id = $2",
                user_id, workspace_id
            )
        finally:
            await conn2.close()
    except Exception:
        pass

    total_pages = len(categories) + sum(len(v) for v in subcategories.values()) + len(concept_pages)
    total_sources = total_sources_row or 0
    now = datetime.now(timezone.utc).isoformat()

    # Build index.md content
    lines = [
        f"---",
        f"updated: {now}",
        f"total_pages: {total_pages}",
        f"total_sources: {total_sources}",
        f"total_concepts: {len(concept_pages)}",
        f"---",
        f"",
        f"# Techapedia — Knowledge Index",
        f"",
        f"*{total_pages} pages · {total_sources} sources · last updated {now[:10]}*",
        f"",
    ]

    supadense = _parse_supadense(user_id, workspace_id)
    depth_prefs = supadense.get("depth_preferences", {})

    for cat_name, cat_info in sorted(categories.items()):
        subs = subcategories.get(cat_name, [])
        depth = depth_prefs.get(cat_name, "working")
        sources_note = f"{cat_info['items']} items"
        lines.append(f"## {cat_name} ({sources_note} · depth: {depth})")
        lines.append(f"- [[{cat_info['slug']}]] — {cat_info['description']}")
        for sub in subs:
            lines.append(f"  - [[{sub['slug']}]] — {sub['description']} ({sub['items']} items)")
        lines.append("")

    if concept_pages:
        lines.append("## Concept pages (cross-category)")
        for cp in sorted(concept_pages, key=lambda x: x["name"]):
            used = ", ".join(cp["used_by"]) if cp["used_by"] else "—"
            lines.append(f"- [[{cp['slug']}]] → used by: {used}")
        lines.append("")

    _index_path(user_id, workspace_id).write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _append_log(user_id: str, workspace_id: str, entry: str) -> None:
    """Append a timestamped entry to log.md."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    log_p = _log_path(user_id, workspace_id)
    if not log_p.exists():
        log_p.write_text("# Techapedia — Activity Log\n\n", encoding="utf-8")
    with open(log_p, "a", encoding="utf-8") as f:
        f.write(f"\n## [{now}] {entry}\n")


def _parse_wiki_page(content: str) -> dict:
    """
    Parse a wiki page markdown into structured JSON.
    Extracts frontmatter + all ## sections.
    Returns dict with metadata + sections dict.
    """
    post = fm.loads(content)
    sections: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []

    for line in post.content.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip().lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    # Parse key_concepts into list of {display, slug, description}
    key_concepts = []
    for line in sections.get("key_concepts", "").splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        # Extract [[slug]] or [[Title]] patterns
        wiki_links = re.findall(r"\[\[([^\]]+)\]\]", line)
        desc = re.sub(r"\[\[([^\]]+)\]\]", r"\1", line)
        key_concepts.append({
            "display": line,
            "links": wiki_links,
            "description": desc,
        })

    # Parse connections into list
    connections = []
    for line in sections.get("connections", "").splitlines():
        line = line.strip().lstrip("- →").strip()
        if not line:
            continue
        wiki_links = re.findall(r"\[\[([^\]]+)\]\]", line)
        connections.append({
            "display": line,
            "links": wiki_links,
        })

    # Parse sources into list
    sources = []
    for line in sections.get("sources", "").splitlines():
        line = line.strip().lstrip("0123456789.-) ").strip()
        if not line:
            continue
        url_match = re.search(r"\(([^)]+)\)", line)
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        sources.append({
            "title": title.strip(),
            "url": url_match.group(1) if url_match else None,
        })

    # Parse open_questions
    questions = [
        line.strip().lstrip("- ?").strip()
        for line in sections.get("open_questions", "").splitlines()
        if line.strip().lstrip("- ?").strip()
    ]

    # Parse gaps
    gaps = [
        line.strip().lstrip("- ").strip()
        for line in sections.get("gaps", "").splitlines()
        if line.strip().lstrip("- ").strip()
    ]

    # Parse subcategories
    subcats = []
    for line in sections.get("subcategories", "").splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        wiki_links = re.findall(r"\[\[([^\]]+)\]\]", line)
        subcats.append({
            "display": line,
            "links": wiki_links,
        })

    return {
        "title": post.metadata.get("title", ""),
        "type": post.metadata.get("type", "category"),
        "parent": post.metadata.get("parent"),
        "created": str(post.metadata.get("created", "")),
        "updated": str(post.metadata.get("updated", "")),
        "sources": post.metadata.get("sources", 0),
        "depth": post.metadata.get("depth", "working"),
        "subcategories_meta": post.metadata.get("subcategories", []),
        "used_by": post.metadata.get("used_by", []),
        "sections": {
            "overview": sections.get("overview", ""),
            "key_concepts_raw": sections.get("key_concepts", ""),
            "key_concepts": key_concepts,
            "current_understanding": sections.get("current_understanding", ""),
            "open_questions": questions,
            "connections": connections,
            "sources_raw": sections.get("sources", ""),
            "sources": sources,
            "gaps": gaps,
            "subcategories": subcats,
        },
        "raw_content": post.content,
    }


# ── Wiki endpoints ────────────────────────────────────────────────────────────

@app.get("/tools/wiki/index")
async def wiki_index(
    user_id: str = DEFAULT_USER_ID,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    refresh: bool = False,
):
    """
    Return parsed index.md as structured JSON.
    If index.md doesn't exist or refresh=True, regenerate it first.
    """
    try:
        index_p = _index_path(user_id, workspace_id)
        if not index_p.exists() or refresh:
            await _update_index(user_id, workspace_id)

        raw = index_p.read_text(encoding="utf-8")
        post = fm.loads(raw)

        # Parse categories + subcategories from content
        categories = []
        current_cat = None
        for line in post.content.splitlines():
            if line.startswith("## ") and "Concept pages" not in line:
                current_cat = {
                    "name": re.sub(r"\s*\(.*\)", "", line[3:]).strip(),
                    "subcategories": [],
                }
                categories.append(current_cat)
            elif line.startswith("- [[") and current_cat is not None and not line.startswith("  "):
                m = re.search(r"\[\[([^\]]+)\]\]", line)
                desc_m = re.search(r"\]\] — (.+)$", line)
                if m:
                    current_cat["slug"] = m.group(1)
                    current_cat["description"] = desc_m.group(1) if desc_m else ""
            elif line.startswith("  - [[") and current_cat is not None:
                m = re.search(r"\[\[([^\]]+)\]\]", line)
                desc_m = re.search(r"\]\] — (.+?)(?:\s*\(|$)", line)
                if m:
                    current_cat["subcategories"].append({
                        "slug": m.group(1),
                        "name": m.group(1),
                        "description": desc_m.group(1).strip() if desc_m else "",
                    })

        # Parse concept pages
        concepts = []
        in_concepts = False
        for line in post.content.splitlines():
            if "Concept pages" in line:
                in_concepts = True
                continue
            if in_concepts and line.startswith("## "):
                in_concepts = False
            if in_concepts and line.startswith("- [["):
                m = re.search(r"\[\[([^\]]+)\]\]", line)
                used_m = re.search(r"used by: (.+)$", line)
                if m:
                    concepts.append({
                        "slug": m.group(1),
                        "name": m.group(1),
                        "used_by": [u.strip() for u in used_m.group(1).split(",")] if used_m else [],
                    })

        return {
            "status": "ok",
            "meta": dict(post.metadata),
            "categories": categories,
            "concepts": concepts,
            "total_pages": post.metadata.get("total_pages", 0),
            "total_sources": post.metadata.get("total_sources", 0),
            "total_concepts": post.metadata.get("total_concepts", 0),
            "updated": post.metadata.get("updated", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/wiki/page")
async def wiki_page(
    name: str,
    user_id: str = DEFAULT_USER_ID,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
):
    """
    Return a specific wiki page parsed into structured JSON.
    name = slug e.g. 'agents', 'agents--evaluation', 'mips-retrieval'
    """
    try:
        page_p = _wiki_page_path(user_id, workspace_id, name)
        if not page_p.exists():
            raise HTTPException(status_code=404, detail=f"Wiki page '{name}' not found.")
        raw = page_p.read_text(encoding="utf-8")
        parsed = _parse_wiki_page(raw)
        parsed["slug"] = name
        return {"status": "ok", "page": parsed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WikiUpdateIndexRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/wiki/update_index")
async def wiki_update_index(req: WikiUpdateIndexRequest):
    """Manually regenerate index.md."""
    try:
        await _update_index(req.user_id, req.workspace_id)
        index_p = _index_path(req.user_id, req.workspace_id)
        return {
            "status": "ok",
            "message": "index.md regenerated",
            "path": str(index_p),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WikiAppendLogRequest(BaseModel):
    entry: str
    user_id: str = DEFAULT_USER_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID


@app.post("/tools/wiki/append_log")
async def wiki_append_log(req: WikiAppendLogRequest):
    """Manually append an entry to log.md."""
    try:
        _append_log(req.user_id, req.workspace_id, req.entry)
        return {"status": "ok", "message": "Appended to log.md"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))