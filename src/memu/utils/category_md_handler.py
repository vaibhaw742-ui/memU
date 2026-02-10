"""Handler for saving category data to local markdown files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# Protocol for MemoryCategory - allows duck typing
class MemoryCategory(Protocol):
    """Protocol for MemoryCategory objects."""
    id: str
    name: str
    description: str
    summary: str | None
    created_at: Any
    updated_at: Any
    embedding: list[float] | None


class CategoryMarkdownHandler:
    """Manages saving and updating category data to markdown files."""

    def __init__(self, output_dir: str | Path = "./categories"):
        """Initialize the markdown handler.
        
        Args:
            output_dir: Directory where category markdown files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CategoryMarkdownHandler initialized with output_dir: {self.output_dir}")

    def _sanitize_filename(self, name: str) -> str:
        """Convert category name to safe filename.
        
        Args:
            name: Category name
            
        Returns:
            Sanitized filename safe for filesystem
        """
        # Replace spaces with underscores and remove special characters
        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name)
        safe_name = safe_name.replace(" ", "_").lower()
        return f"{safe_name}.md"

    def _build_markdown_content(self, category: MemoryCategory) -> str:
        """Build markdown content from category data.
        
        Args:
            category: MemoryCategory object
            
        Returns:
            Formatted markdown string
        """
        lines = [
            f"# {category.name}",
            "",
            "## Description",
            "",
            category.description or "No description provided.",
            "",
            "## Summary",
            "",
            category.summary or "*No summary yet*",
            "",
            "## Metadata",
            "",
            f"- **Category ID**: `{category.id}`",
            f"- **Created**: {category.created_at.isoformat() if category.created_at else 'N/A'}",
            f"- **Last Updated**: {category.updated_at.isoformat() if category.updated_at else 'N/A'}",
            f"- **Has Embedding**: {'Yes' if category.embedding else 'No'}",
            "",
        ]
        
        return "\n".join(lines)

    def save_category(self, category: MemoryCategory) -> Path:
        """Save category to a markdown file.
        
        Args:
            category: MemoryCategory object to save
            
        Returns:
            Path to the saved file
        """
        filename = self._sanitize_filename(category.name)
        filepath = self.output_dir / filename
        
        content = self._build_markdown_content(category)
        
        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Saved category '{category.name}' to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save category '{category.name}': {e}")
            raise
        
        return filepath

    def save_categories(self, categories: dict[str, MemoryCategory]) -> list[Path]:
        """Save multiple categories to markdown files.
        
        Args:
            categories: Dict of category_id -> MemoryCategory
            
        Returns:
            List of paths to saved files
        """
        saved_paths = []
        for category in categories.values():
            try:
                path = self.save_category(category)
                saved_paths.append(path)
            except Exception as e:
                logger.warning(f"Skipping category '{category.name}' due to error: {e}")
                continue
        
        logger.info(f"Saved {len(saved_paths)} category markdown files")
        return saved_paths

    def update_category(self, category: MemoryCategory) -> Path:
        """Update an existing category markdown file (same as save).
        
        Args:
            category: MemoryCategory object to update
            
        Returns:
            Path to the updated file
        """
        return self.save_category(category)

    def delete_category(self, category_name: str) -> bool:
        """Delete a category markdown file.
        
        Args:
            category_name: Name of the category to delete
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        filename = self._sanitize_filename(category_name)
        filepath = self.output_dir / filename
        
        if filepath.exists():
            try:
                filepath.unlink()
                logger.info(f"Deleted category file: {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete category file {filepath}: {e}")
                raise
        else:
            logger.warning(f"Category file not found: {filepath}")
            return False

    def get_category_path(self, category_name: str) -> Path:
        """Get the path where a category's markdown file would be saved.
        
        Args:
            category_name: Name of the category
            
        Returns:
            Path to the category markdown file
        """
        filename = self._sanitize_filename(category_name)
        return self.output_dir / filename


__all__ = ["CategoryMarkdownHandler"]