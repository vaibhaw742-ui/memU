"""review_due_at column on memory_items

Revision ID: b2c3d4e5f6a1
Revises: a1b2c3d4e5f6
Create Date: 2026-04-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'b2c3d4e5f6a1'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'memory_items',
        sa.Column('review_due_at',
                  sa.DateTime(timezone=True), nullable=True),
        schema='learning'
    )
    op.create_index('ix_memory_items_review_due_at', 'memory_items',
                    ['review_due_at'], schema='learning')


def downgrade() -> None:
    op.drop_index('ix_memory_items_review_due_at',
                  table_name='memory_items', schema='learning')
    op.drop_column('memory_items', 'review_due_at', schema='learning')
