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
from memu.config.settings import (
    MemorizeConfig, 
    CategoryConfig,
    CustomPrompt,
    PromptBlock
)

DEFAULT_EXTRACTION_TEMPLATE = """Analyze the following document and extract it as a structured table.

                                        YOUR TASK:
                                        1. Determine a MEMORY_TYPE (2-3 words max) that best describes this content (e.g., "AI Knowledge", "Quantum Tech", "Business Strategy", etc.)

                                        2. Create a SINGLE table representation of this document with rows for different topics/concepts. The table should have this format:
                                        - Each row: "topic | sub_topic | description"
                                        - Capture all main knowledge areas from the document
                                        - Keep descriptions concise (1-2 sentences per row)

                                        3. Categorize the entire document based on its overall content into the available categories.

                                        RESPONSE FORMAT (JSON):
                                        {{
                                            "memory_type": "2-3 word type",
                                            "entries": [
                                                {{
                                                    "table": "Topic 1 | Sub-topic 1 | Description of topic 1\\nTopic 2 | Sub-topic 2 | Description of topic 2\\nTopic 3 | Sub-topic 3 | Description of topic 3",
                                                    "categories": ["Category1", "Category2"]
                                                }}
                                            ]
                                        }}

                                        GUIDELINES:
                                        - memory_type: Short, descriptive (2-3 words)
                                        - entries: Usually contains just ONE entry representing the whole document
                                        - table: Multiple rows separated by \\n, each row is "topic | sub_topic | description"
                                        - Capture 3-6 key topics from the document
                                        - categories: Assign based on overall document content
                                        - Ensure JSON is valid

                                        Now analyze the document and provide the JSON response:"""

custom_agent_category_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# Agent Knowledge Consolidator

Merge and organize agent-related concepts into a comprehensive, structured knowledge base.

## Format Requirements
- Use markdown headers (##) for main topics
- Consolidate similar concepts and subtopics
- Maintain the knowledge table schema throughout
- Include all technical details, frameworks, patterns, and examples
- Eliminate redundancy while preserving unique insights
- Maximum {target_length} tokens
"""
            ),
            workflow=PromptBlock(
                ordinal=20,
                prompt="""
# Consolidation Process
1. Analyze original content and new memory items
2. Group by similar high-level topics and themes (architecture, evaluation, tools, frameworks, patterns, etc.)
3. Merge related subtopics and eliminate duplicates
4. Combine descriptions, enriching with complementary details
5. Aggregate resource references
6. Create unified knowledge tables per topic area
7. Update overview to reflect consolidated knowledge
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# Output Format
```markdown
# {category}

## Overview
[Updated comprehensive introduction incorporating all agent knowledge]

## [High-Level Topic 1]

### Knowledge Table

| High-Level Topic | Subtopic | Description | Resource |
|-----------------|----------|-------------|----------|
| [Consolidated topic] | [Merged subtopic] | [Combined description with all relevant details, patterns, code examples, metrics] | [All source references] |
| [Consolidated topic] | [Merged subtopic] | [Enhanced description from multiple sources] | [All source references] |

### Key Insights
- [Synthesized best practices and patterns]
- [Common approaches and techniques]
- [Critical considerations and trade-offs]

## [High-Level Topic 2]

### Knowledge Table

| High-Level Topic | Subtopic | Description | Resource |
|-----------------|----------|-------------|----------|
| [Consolidated topic] | [Merged subtopic] | [Combined description] | [All source references] |

### Key Insights
- [Synthesized insights]

## Cross-Topic Connections
- [How agent concepts relate to each other]
- [Workflows and patterns spanning multiple areas]
- [Integration points between frameworks and tools]
```

Target length: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="""
Category: {category}
Original Content: {original_content}
New Memory Items: {new_memory_items_text}
"""
            )
        )

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
                               extraction_prompt_template= DEFAULT_EXTRACTION_TEMPLATE,
                               categories_prompt_str = """ - Agents: Agent-related content
                                                           - RAG: RAG-related content
                                                           - LLM Training: Training insights
                                                           - LLM Inference: Inference techniques""",
                                
                                # Define your categories (required, or categories won't be used)
                                memory_categories=[
                                    CategoryConfig(name="Agents", description="Agent-related content",summary_prompt = custom_agent_category_summary),
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
                    resource_url="https://www.linkedin.com/posts/pauliusztin_last-year-i-shipped-the-first-version-of-activity-7429155120677998592-3Hgm/?utm_source=share&utm_medium=member_desktop&rcm=ACoAACmrL44B-pNi9lNjFQtuPtX_ODwJk7-cC-0",
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