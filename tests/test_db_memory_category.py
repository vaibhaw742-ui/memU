"""
PostgreSQL Memory Category Repository - Usage Guide (Fixed for workspace_id)
====================================================

This guide demonstrates how to use three key functions:
1. get_or_create_category() - Create or retrieve categories
2. list_categories() - List categories with optional filtering
3. clear_categories() - Delete categories with optional filtering

Fixed to include workspace_id which is required in your schema.
"""

from pydantic import BaseModel
from memu.config.settings import DatabaseConfig
from memu.database.postgres import build_postgres_database


# ============================================================================
# Step 1: Define Your User Model (matching your actual schema)
# ============================================================================
class UserModel(BaseModel):
    """
    User scope model matching your database schema.
    Both user_id and workspace_id are required.
    """
    user_id: str | None = None
    workspace_id: str | None = None  # This is required in your schema!


# ============================================================================
# Step 2: Initialize Database
# ============================================================================
def initialize_database(dsn: str):
    """
    Set up the PostgreSQL database with pgvector support.
    
    Args:
        dsn: PostgreSQL connection string
        
    Returns:
        Database instance with initialized repositories
    """
    config = DatabaseConfig(
        metadata_store={
            "provider": "postgres",
            "ddl_mode": "create",  # or "validate" for production
            "dsn": dsn,
        },
        vector_index={
            "provider": "pgvector",
            "dsn": dsn,
        },
    )
    
    db = build_postgres_database(config=config, user_model=UserModel)
    return db


# ============================================================================
# Function 1: get_or_create_category()
# ============================================================================
def demo_get_or_create_category(repo):
    """
    Demonstrates get_or_create_category() function.
    
    This function either:
    - Creates a new category if it doesn't exist
    - Returns the existing category if it already exists (idempotent)
    
    Signature:
        get_or_create_category(
            name: str,              # Category name (must be unique per user)
            description: str,       # Human-readable description
            embedding: list[float], # Vector embedding (typically 1536 dims)
            user_data: dict         # User scope fields (user_id, workspace_id)
        ) -> MemoryCategory
    """
    
    print("\n" + "=" * 80)
    print("FUNCTION 1: get_or_create_category()")
    print("=" * 80)
    
    # Example 1: Create a new category
    print("\nExample 1: Creating a new category")
    print("-" * 80)
    
    embedding = [0.1] * 1536  # Mock embedding vector
    
    category = repo.get_or_create_category(
        name="Personal Projects",
        description="Personal side projects and hobbies",
        embedding=embedding,
        user_data={
            "user_id": "user123",
            "workspace_id": "workspace-alpha"  # Required!
        }
    )
    
    print(f"Created category:")
    print(f"  ID: {category.id}")
    print(f"  Name: {category.name}")
    print(f"  Description: {category.description}")
    print(f"  User ID: {category.user_id}")
    print(f"  Workspace ID: {category.workspace_id}")
    print(f"  Embedding dimensions: {len(category.embedding) if category.embedding is not None else 0}")
    
    # Example 2: Idempotent behavior (get existing)
    print("\nExample 2: Getting existing category (idempotent)")
    print("-" * 80)
    
    same_category = repo.get_or_create_category(
        name="Personal Projects",  # Same name
        description="Personal side projects and hobbies",
        embedding=embedding,
        user_data={
            "user_id": "user123",
            "workspace_id": "workspace-alpha"  # Same user + workspace
        }
    )
    
    print(f"Retrieved category:")
    print(f"  ID: {same_category.id}")
    print(f"  Same as before: {same_category.id == category.id}")
    
    # Example 3: Different workspace
    print("\nExample 3: Same name, different workspace (separate category)")
    print("-" * 80)
    
    different_workspace_category = repo.get_or_create_category(
        name="Personal Projects",  # Same name
        description="Personal side projects and hobbies",
        embedding=embedding,
        user_data={
            "user_id": "user123",
            "workspace_id": "workspace-beta"  # Different workspace!
        }
    )
    
    print(f"Created separate category:")
    print(f"  ID: {different_workspace_category.id}")
    print(f"  Workspace ID: {different_workspace_category.workspace_id}")
    print(f"  Different from first: {different_workspace_category.id != category.id}")
    
    # Example 4: Different user, same workspace
    print("\nExample 4: Different user, same workspace (separate category)")
    print("-" * 80)
    
    different_user_category = repo.get_or_create_category(
        name="Personal Projects",  # Same name
        description="Personal side projects and hobbies",
        embedding=embedding,
        user_data={
            "user_id": "user456",  # Different user
            "workspace_id": "workspace-alpha"  # Same workspace
        }
    )
    
    print(f"Created separate category:")
    print(f"  ID: {different_user_category.id}")
    print(f"  User ID: {different_user_category.user_id}")
    print(f"  Different from first: {different_user_category.id != category.id}")
    
    return category, same_category, different_workspace_category, different_user_category


# ============================================================================
# Function 2: list_categories()
# ============================================================================
def demo_list_categories(repo):
    """
    Demonstrates list_categories() function.
    
    This function retrieves categories with optional filtering.
    
    Signature:
        list_categories(
            where: dict | None = None  # Optional filter conditions
        ) -> dict[str, MemoryCategory]
        
    Returns:
        Dictionary mapping category_id -> MemoryCategory object
    """
    
    print("\n" + "=" * 80)
    print("FUNCTION 2: list_categories()")
    print("=" * 80)
    
    # Example 1: List all categories
    print("\nExample 1: List all categories (no filter)")
    print("-" * 80)
    
    all_categories = repo.list_categories()
    
    print(f"Found {len(all_categories)} total categories:")
    for cat_id, cat in all_categories.items():
        print(f"  - {cat.name}")
        print(f"    User: {cat.user_id}, Workspace: {cat.workspace_id}")
        print(f"    ID: {cat_id[:8]}...")
    
    # Example 2: Filter by user_id
    print("\nExample 2: Filter by user_id")
    print("-" * 80)
    
    user123_categories = repo.list_categories(where={"user_id": "user123"})
    
    print(f"Found {len(user123_categories)} categories for user123:")
    for cat_id, cat in user123_categories.items():
        print(f"  - {cat.name} (Workspace: {cat.workspace_id})")
    
    # Example 3: Filter by workspace_id
    print("\nExample 3: Filter by workspace_id")
    print("-" * 80)
    
    workspace_alpha_categories = repo.list_categories(
        where={"workspace_id": "workspace-alpha"}
    )
    
    print(f"Found {len(workspace_alpha_categories)} categories in workspace-alpha:")
    for cat_id, cat in workspace_alpha_categories.items():
        print(f"  - {cat.name} (User: {cat.user_id})")
    
    # Example 4: Filter by both user_id AND workspace_id
    print("\nExample 4: Filter by user_id AND workspace_id")
    print("-" * 80)
    
    specific_categories = repo.list_categories(
        where={
            "user_id": "user123",
            "workspace_id": "workspace-alpha"
        }
    )
    
    print(f"Found {len(specific_categories)} categories for user123 in workspace-alpha:")
    for cat_id, cat in specific_categories.items():
        print(f"  - {cat.name}")
    
    # Example 5: Filter with __in operator
    print("\nExample 5: Filter by multiple workspaces (using __in)")
    print("-" * 80)
    
    multi_workspace_categories = repo.list_categories(
        where={"workspace_id__in": ["workspace-alpha", "workspace-beta"]}
    )
    
    print(f"Found {len(multi_workspace_categories)} categories in alpha or beta:")
    for cat_id, cat in multi_workspace_categories.items():
        print(f"  - {cat.name} (Workspace: {cat.workspace_id})")
    
    return all_categories, user123_categories


# ============================================================================
# Function 3: clear_categories()
# ============================================================================
def demo_clear_categories(repo):
    """
    Demonstrates clear_categories() function.
    
    This function deletes categories with optional filtering.
    
    Signature:
        clear_categories(
            where: dict | None = None  # Optional filter conditions
        ) -> dict[str, MemoryCategory]
        
    Returns:
        Dictionary of deleted categories (category_id -> MemoryCategory)
    """
    
    print("\n" + "=" * 80)
    print("FUNCTION 3: clear_categories()")
    print("=" * 80)
    
    # Example 1: Clear categories for specific workspace
    print("\nExample 1: Clear categories for workspace-beta")
    print("-" * 80)
    
    deleted = repo.clear_categories(where={"workspace_id": "workspace-beta"})
    
    print(f"Deleted {len(deleted)} categories:")
    for cat_id, cat in deleted.items():
        print(f"  - {cat.name} (User: {cat.user_id}, ID: {cat_id[:8]}...)")
    
    # Verify deletion
    remaining_workspace_beta = repo.list_categories(where={"workspace_id": "workspace-beta"})
    print(f"\nVerification: {len(remaining_workspace_beta)} categories remain in workspace-beta")
    
    # Example 2: Clear categories for specific user in specific workspace
    print("\nExample 2: Clear categories for user456 in workspace-alpha")
    print("-" * 80)
    
    deleted_specific = repo.clear_categories(
        where={
            "user_id": "user456",
            "workspace_id": "workspace-alpha"
        }
    )
    
    print(f"Deleted {len(deleted_specific)} categories:")
    for cat_id, cat in deleted_specific.items():
        print(f"  - {cat.name}")
    
    # Example 3: Clear all remaining categories
    print("\nExample 3: Clear all remaining categories (no filter)")
    print("-" * 80)
    
    all_deleted = repo.clear_categories()
    
    print(f"Deleted {len(all_deleted)} categories:")
    for cat_id, cat in all_deleted.items():
        print(f"  - {cat.name} (User: {cat.user_id}, Workspace: {cat.workspace_id})")
    
    # Verify all cleared
    final_count = repo.list_categories()
    print(f"\nFinal verification: {len(final_count)} total categories")
    
    return deleted, all_deleted


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Run all demonstrations"""
    
    import os
    
    print("=" * 80)
    print("PostgreSQL Memory Category Repository - Three Functions Usage Guide")
    print("=" * 80)
    
    # Initialize database
    dsn = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memu")
    print(f"\nConnecting to: {dsn}")
    
    db = initialize_database(dsn)
    repo = db.memory_category_repo
    
    print("✓ Database initialized")
    
    # Run demonstrations
    demo_get_or_create_category(repo)
    demo_list_categories(repo)
    demo_clear_categories(repo)
    
    # Cleanup
    db.close()
    
    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. get_or_create_category() - Idempotent category creation")
    print("  2. list_categories() - Flexible querying with 'where' filters")
    print("  3. clear_categories() - Safe deletion with optional filtering")
    print("\nImportant:")
    print("  - Always include both user_id AND workspace_id in user_data")
    print("  - Categories are scoped by BOTH user_id and workspace_id")
    print()


if __name__ == "__main__":
    main()