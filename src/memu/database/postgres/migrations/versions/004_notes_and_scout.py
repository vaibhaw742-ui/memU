"""notes_index and scout_log tables

Revision ID: a1b2c3d4e5f6
Revises: 299a8f5f0acd
Create Date: 2026-04-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '299a8f5f0acd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("SET search_path TO learning, public")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")

    op.create_table(
        'notes_index',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('notion_page_id', sa.String(), nullable=False),
        sa.Column('notion_url', sa.Text(), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('content_hash', sa.String(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(), nullable=True),
        sa.Column('last_synced_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('workspace_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('notion_page_id',
                            name='uq_notes_index_page_id'),
        schema='learning'
    )
    op.create_index('ix_notes_index__scope', 'notes_index',
                    ['user_id', 'workspace_id'], schema='learning')

    op.create_table(
        'scout_log',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('platform', sa.String(), nullable=False),
        sa.Column('post_url', sa.Text(), nullable=True),
        sa.Column('author', sa.String(), nullable=True),
        sa.Column('post_text', sa.Text(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('relevance_reason', sa.Text(), nullable=True),
        sa.Column('matched_goal', sa.Text(), nullable=True),
        sa.Column('action_taken', sa.String(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('workspace_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        schema='learning'
    )
    op.create_index('ix_scout_log__scope', 'scout_log',
                    ['user_id', 'workspace_id'], schema='learning')


def downgrade() -> None:
    op.drop_index('ix_scout_log__scope',
                  table_name='scout_log', schema='learning')
    op.drop_table('scout_log', schema='learning')
    op.drop_index('ix_notes_index__scope',
                  table_name='notes_index', schema='learning')
    op.drop_table('notes_index', schema='learning')