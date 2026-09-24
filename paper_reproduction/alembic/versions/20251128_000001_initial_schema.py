"""Initial schema for papers, sections, chunks tables.

Revision ID: 20251128_000001
Revises:
Create Date: 2025-11-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20251128_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create papers table
    op.create_table(
        "papers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pmcid", sa.String(20), nullable=False),
        sa.Column("pmid", sa.String(20), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("journal", sa.String(500), nullable=True),
        sa.Column("doi", sa.String(100), nullable=True),
        sa.Column("authors", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("subject_categories", postgresql.ARRAY(sa.String(100)), nullable=True),
        sa.Column("keywords", postgresql.ARRAY(sa.String(100)), nullable=True),
        sa.Column("license", sa.String(50), nullable=True),
        sa.Column("storage_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("pmcid"),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("doi"),
    )

    # Papers indexes
    op.create_index("idx_papers_id", "papers", ["id"])
    op.create_index("idx_papers_pmid", "papers", ["pmid"])
    op.create_index("idx_papers_pub_date", "papers", ["publication_date"])
    op.create_index("idx_papers_journal", "papers", ["journal"])
    op.create_index("idx_papers_doi", "papers", ["doi"])
    op.create_index("idx_papers_license", "papers", ["license"])
    op.create_index(
        "idx_papers_authors_gin",
        "papers",
        ["authors"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_papers_categories_gin",
        "papers",
        ["subject_categories"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_papers_keywords_gin",
        "papers",
        ["keywords"],
        postgresql_using="gin",
    )

    # Create sections table
    op.create_table(
        "sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("section_type", sa.String(50), nullable=True),
        sa.Column("hierarchy_level", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("sequence_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("hierarchy_level >= 1", name="check_hierarchy_level_positive"),
        sa.CheckConstraint("sequence_order >= 0", name="check_sequence_order_non_negative"),
    )

    # Sections indexes
    op.create_index("idx_sections_paper_id", "sections", ["paper_id"])
    op.create_index("idx_sections_type", "sections", ["section_type"])
    op.create_index("idx_sections_order", "sections", ["paper_id", "sequence_order"])

    # Create chunks table
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("sequence_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("start_char_offset", sa.Integer(), nullable=True),
        sa.Column("end_char_offset", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["sections.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("token_count > 0", name="check_token_count_positive"),
        sa.CheckConstraint("token_count <= 512", name="check_token_count_max"),
        sa.CheckConstraint("sequence_order >= 0", name="check_chunk_sequence_non_negative"),
        sa.CheckConstraint(
            "(start_char_offset IS NULL AND end_char_offset IS NULL) OR "
            "(start_char_offset IS NOT NULL AND end_char_offset IS NOT NULL AND end_char_offset > start_char_offset)",
            name="check_char_offsets_consistency",
        ),
    )

    # Chunks indexes
    op.create_index("idx_chunks_section_id", "chunks", ["section_id"])
    op.create_index("idx_chunks_order", "chunks", ["section_id", "sequence_order"])

    # Create section_parent_rels table for hierarchical sections
    op.create_table(
        "section_parent_rels",
        sa.Column("child_section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("child_section_id", "parent_section_id"),
        sa.ForeignKeyConstraint(
            ["child_section_id"],
            ["sections.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_section_id"],
            ["sections.id"],
            ondelete="CASCADE",
        ),
    )

    # Section parent relationship indexes
    op.create_index("idx_section_parent_child", "section_parent_rels", ["child_section_id"])
    op.create_index("idx_section_parent_parent", "section_parent_rels", ["parent_section_id"])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("section_parent_rels")
    op.drop_table("chunks")
    op.drop_table("sections")
    op.drop_table("papers")
