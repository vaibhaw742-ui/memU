"""007_parent_category

Add parent_category column to learning.memory_categories.
Enables subcategory hierarchy — agents--evaluation has parent_category = 'agents'.
"""

from alembic import op
import sqlalchemy as sa

revision = "007_parent_category"
down_revision = "c3d4e5f6a1b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memory_categories",
        sa.Column("parent_category", sa.String(), nullable=True),
        schema="learning",
    )
    op.create_index(
        "ix_memory_categories_parent",
        "memory_categories",
        ["parent_category"],
        schema="learning",
        postgresql_where=sa.text("parent_category IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_categories_parent",
        table_name="memory_categories",
        schema="learning",
    )
    op.drop_column("memory_categories", "parent_category", schema="learning")