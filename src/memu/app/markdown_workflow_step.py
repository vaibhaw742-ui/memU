"""Workflow step to save category markdown files after memorization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from category_md_handler import CategoryMarkdownHandler

logger = logging.getLogger(__name__)

# Type alias for workflow state
WorkflowState = Dict[str, Any]


def create_save_markdown_step_handler(md_handler: CategoryMarkdownHandler):
    """Factory to create a step handler with access to the markdown handler.
    
    Args:
        md_handler: CategoryMarkdownHandler instance
        
    Returns:
        Step handler function
    """
    
    def _save_categories_markdown(state: WorkflowState, step_context: Any) -> WorkflowState:
        """Save updated categories to markdown files.
        
        This step runs after category summaries are updated and saves
        the updated categories to local .md files.
        
        Args:
            state: Current workflow state
            step_context: Step execution context
            
        Returns:
            Updated workflow state
        """
        store = state.get("store")
        if not store:
            logger.warning("No store in state, skipping markdown save")
            return state
        
        # Get category updates from state
        category_updates = state.get("category_updates", {})
        if not category_updates:
            logger.debug("No category updates to save")
            return state
        
        # Get updated category IDs
        updated_category_ids = list(category_updates.keys())
        
        # Save each updated category to markdown
        saved_count = 0
        for cat_id in updated_category_ids:
            category = store.memory_category_repo.categories.get(cat_id)
            if category:
                try:
                    filepath = md_handler.save_category(category)
                    logger.info(f"Saved category '{category.name}' to {filepath}")
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Failed to save category {cat_id} to markdown: {e}")
        
        logger.info(f"Saved {saved_count} category markdown files")
        
        # Add saved paths to state for reference
        state["markdown_files_saved"] = saved_count
        
        return state
    
    return _save_categories_markdown


__all__ = ["create_save_markdown_step_handler"]