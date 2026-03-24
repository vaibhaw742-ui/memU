from __future__ import annotations

from logging.config import fileConfig
from typing import Any

from alembic import context
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy import MetaData, engine_from_config, pool, text

from memu.database.postgres.schema import get_metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


SCHEMA = "learning"


class DefaultScope(PydanticBaseModel):
    user_id: str
    workspace_id: str


def get_target_metadata() -> MetaData | None:
    scope_model = config.attributes.get("scope_model")
    if scope_model is None:
        scope_model = DefaultScope
    return get_metadata(scope_model)


target_metadata: MetaData | None = get_target_metadata()


def include_object(object: Any, name: str, type_: str, reflected: bool, compare_to: Any) -> bool:
    """Only manage tables in the learning schema. Ignore everything else."""
    if type_ == "table":
        schema = getattr(object, "schema", None)
        if schema != SCHEMA:
            return False
    return True


def run_migrations_offline() -> None:
    import os 
    url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    import os 
    url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    configuration = {"sqlalchemy.url": url}
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.execute(text(f"SET search_path TO {SCHEMA}"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            version_table_schema=SCHEMA,
            include_schemas=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()