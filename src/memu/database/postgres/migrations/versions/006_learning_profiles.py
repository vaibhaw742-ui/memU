"""learning_profiles table

Revision ID: c3d4e5f6a1b2
Revises: b2c3d4e5f6a1
Create Date: 2026-04-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = 'c3d4e5f6a1b2'
down_revision: Union[str, None] = 'b2c3d4e5f6a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'learning_profiles',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('workspace_id', sa.String(), nullable=False),
        sa.Column('onboarded_at',
                  sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_synthesis_at',
                  sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_digest_item_ids', JSONB(), nullable=True,
                  server_default='[]'),
        sa.Column('depth_prefs', JSONB(), nullable=True,
                  server_default='{}'),
        sa.Column('spaced_rep_config', JSONB(), nullable=True,
                  server_default='{"desired_retention": 0.9}'),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        schema='learning'
    )
    op.create_index('ix_learning_profiles__scope', 'learning_profiles',
                    ['user_id', 'workspace_id'], unique=True,
                    schema='learning')


def downgrade() -> None:
    op.drop_index('ix_learning_profiles__scope',
                  table_name='learning_profiles', schema='learning')
    op.drop_table('learning_profiles', schema='learning')
