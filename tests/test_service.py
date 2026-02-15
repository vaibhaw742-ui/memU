#!/usr/bin/env python3
"""Quick test script for MemoryService"""
# Add the src directory to the Python path
from pathlib import Path
import sys

from anyio import Path


project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import asyncio
import os
from memu.app import service
from memu.config.settings import MemorizeConfig
from memu.config.settings import CategoryConfig
from memu.app import MemoryService
from memu.app import create_save_markdown_step_handler
from memu.workflow.step import WorkflowStep

async def main():
    # Initialize the service
    service = MemoryService(
        database_config={
            "metadata_store": {
                "provider": "postgres",
                "dsn": os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
            }
        },
        llm_profiles={
            "default": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "chat_model": "gpt-4o-mini",
                "embed_model": "text-embedding-3-small"
            }
        },
        memorize_config=MemorizeConfig(
                                # Choose which memory types to extract (optional, defaults to all 5)
                               # memory_types=["profile", "events", "knowledge"],
                                
                                # Define your categories (required, or categories won't be used)
                                memory_categories=[
                                    CategoryConfig(name="Agents", description="Agent-related content"),
                                    CategoryConfig(name="RAG", description="RAG-related content"),
                                    CategoryConfig(name="LLM Training", description="Training insights"),
                                    CategoryConfig(name="LLM Inference", description="Inference techniques")
                                ]
                            ) 
        ,category_md_output_dir="./categories"
    )
    
    print("✅ MemoryService initialized successfully!")
    print(f"Provider info: {service._provider_summary()}")
    

    # Add markdown save step to workflow
    # save_markdown_handler = create_save_markdown_step_handler(service.category_md_handler)
    # save_markdown_step = WorkflowStep(
    #     step_id="save_categories_markdown",
    #     role="save_markdown",
    #     handler=save_markdown_handler,
    #     requires={"category_updates", "ctx", "store"},
    #     produces={"markdown_files_saved"},
    #     capabilities=set(),
    # )

    # service.insert_step_after(
    #     target_step_id="persist_index",
    #     new_step=save_markdown_step,
    #     pipeline="memorize"
    # )


    result = await service.memorize(
                    resource_url="https://www.linkedin.com/posts/zainhas_inference-ai-llm-activity-7427597857324494849-JxVE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACmrL44B-pNi9lNjFQtuPtX_ODwJk7-cC-0",
                    user={"user_id": "user123", "workspace_id": "workspace-alpha"}
                )


    # Test memorize function
    # print("\n📝 Testing memorize function...")
    # result = await service.memorize(
    #     resource_url="tests/test_convers3.txt",
    #     modality="conversation",
    #     user={
    #         "user_id": "user4568",  # Different user
    #         "workspace_id": "workspace-alphahbj"  # Same workspace
    #     }
    # )
    #print(f"Memorize result: {result}")
    
    #Test retrieve function
    # print("\n🔍 Testing retrieve function...")
    # retrieve_result = await service.retrieve(
    #     queries=[{"role": "user", "content": {"text": "What do you know about me?"}}],
    #     where={"user_id": "user4568", "workspace_id": "workspace-alphahbj"}
    # )
    # print(f"Retrieve result: {retrieve_result}")
    
    # List memory items
    print("\n📋 Listing memory items...")
    items = await service.list_memory_items(where={"user_id": "user123", "workspace_id": "workspace-alpha"})
    print(f"Found {len(items.get('items', []))} memory items")
    
    # # List categories
    print("\n🏷️  Listing categories...")
    categories = await service.list_memory_categories(where={"user_id": "user123", "workspace_id": "workspace-alpha"})
    print(f"Found {len(categories.get('categories', []))} categories")
    
    print("\n✨ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())