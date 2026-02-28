from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import create_engine, inspect, text

from memu.database.postgres.schema import get_metadata

try:  # Optional dependency for Postgres backend
    from alembic import command
    from alembic.config import Config as AlembicConfig
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = "alembic is required for Postgres migrations"
    raise ImportError(msg) from exc

logger = logging.getLogger(__name__)

DDLMode = Literal["create", "validate"]


def make_alembic_config(*, dsn: str, scope_model: type[Any]) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
    cfg.set_main_option("sqlalchemy.url", dsn)
    cfg.attributes["scope_model"] = scope_model
    return cfg


def _ensure_prerequisites(engine: Any) -> None:
    """
    Ensure pgvector extension and learning schema exist.
    Runs in its own connection so failures don't affect the migration transaction.
    """
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension enabled")

        # Ensure learning schema exists
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS learning"))
        logger.info("learning schema ensured")

        conn.commit()


def run_migrations(*, dsn: str, scope_model: type[Any], ddl_mode: DDLMode = "create") -> None:
    """
    Run database migrations based on the ddl_mode setting.

    Args:
        dsn: Database connection string
        scope_model: User scope model for scoped tables
        ddl_mode: "create" to create missing tables, "validate" to only check schema
    """
    metadata = get_metadata(scope_model)
    engine = create_engine(dsn)

    if ddl_mode == "create":
        _ensure_prerequisites(engine)

        # Create all tables that don't exist
        metadata.create_all(engine)
        logger.info("Database tables created/verified")

    elif ddl_mode == "validate":
        # Validate that all expected tables exist
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names(schema="learning"))  # ← fixed: check learning schema
        expected_tables = {
            "resources",
            "memory_items",
            "memory_categories",
            "category_items",
        }
        missing_tables = expected_tables - existing_tables

        if missing_tables:
            msg = f"Database schema validation failed. Missing tables in learning schema: {sorted(missing_tables)}"
            raise RuntimeError(msg)
        logger.info("Database schema validated successfully")

    # Run any pending Alembic migrations
    cfg = make_alembic_config(dsn=dsn, scope_model=scope_model)
    command.upgrade(cfg, "head")


__all__ = ["DDLMode", "make_alembic_config", "run_migrations"]