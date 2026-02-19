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

# DEFAULT_EXTRACTION_TEMPLATE = """Analyze the following document and extract it as a structured table.

#                                         YOUR TASK:
#                                         1. Determine a MEMORY_TYPE (2-3 words max) that best describes this content (e.g., "AI Knowledge", "Quantum Tech", "Business Strategy", etc.)

#                                         2. Create a SINGLE table representation of this document with rows for different topics/concepts. The table should have this format:
#                                         - Each row: "topic | sub_topic | description"
#                                         - Capture all main knowledge areas from the document
#                                         - Keep descriptions concise (1-2 sentences per row)

#                                         3. Categorize the entire document based on its overall content into the available categories.

#                                         RESPONSE FORMAT (JSON):
#                                         {{
#                                             "memory_type": "2-3 word type",
#                                             "entries": [
#                                                 {{
#                                                     "table": "Topic 1 | Sub-topic 1 | Description of topic 1\\nTopic 2 | Sub-topic 2 | Description of topic 2\\nTopic 3 | Sub-topic 3 | Description of topic 3",
#                                                     "categories": ["Category1", "Category2"]
#                                                 }}
#                                             ]
#                                         }}

#                                         GUIDELINES:
#                                         - memory_type: Short, descriptive (2-3 words)
#                                         - entries: Usually contains just ONE entry representing the whole document
#                                         - table: Multiple rows separated by \\n, each row is "topic | sub_topic | description"
#                                         - Capture 3-6 key topics from the document
#                                         - categories: Assign based on overall document content
#                                         - Ensure JSON is valid

#                                         Now analyze the document and provide the JSON response:"""

DEFAULT_EXTRACTION_TEMPLATE = """Analyze the following document and extract ML/AI design-related concepts as an index-style structured table.

                                        YOUR TASK:
                                        1. Determine a MEMORY_TYPE (2-3 words max) that best describes the ML/AI design focus of this content (e.g., "Model Architecture", "AI System Design", "ML Pipeline", etc.)

                                        2. Create a SINGLE index-style table that acts as a Table of Contents for the ML/AI design concepts in this document:
                                        - Think of it like a BOOK INDEX: top-level topics as chapters, sub-topics as sections within each chapter
                                        - Each row format: "index | topic | sub_topic | description"
                                        - index follows a hierarchical numbering: 1, 1.1, 1.2, 2, 2.1, 2.2, 2.3, 3, 3.1 ...
                                        - Topic rows (1, 2, 3 ...) = broad concept family, description gives the overall theme
                                        - Sub-topic rows (1.1, 1.2 ...) = specific concept within that family, description answers "why read this?"
                                        - Focus ONLY on ML/AI design concepts, patterns, methodologies, and architectural decisions
                                        - Skip anything unrelated to ML/AI design

                                        INDEX STRUCTURE RULES:
                                        - Top-level index (1, 2, 3): Represents a distinct concept family or domain (e.g., "Attention Mechanism", "Training Strategy", "Deployment")
                                        - Second-level index (1.1, 1.2): Specific sub-concepts or techniques within that family
                                        - A topic with only ONE sub-concept should still follow the same structure for consistency
                                        - Keep top-level topics genuinely distinct — group related ideas, split only when clearly different

                                        3. Categorize the document based on its ML/AI design content into the available categories.

                                        RESPONSE FORMAT (JSON):
                                        {{
                                            "memory_type": "2-3 word ML/AI design type",
                                            "entries": [
                                                {{
                                                    "table": "1 | Topic A | - | High-level theme of this concept family\\n1.1 | Topic A | Sub-topic 1 | Why this specific concept is worth reading\\n1.2 | Topic A | Sub-topic 2 | Why this specific concept is worth reading\\n2 | Topic B | - | High-level theme of this concept family\\n2.1 | Topic B | Sub-topic 1 | Why this specific concept is worth reading",
                                                    "categories": ["Category1", "Category2"]
                                                }}
                                            ]
                                        }}

                                        GUIDELINES:
                                        - memory_type: Short, descriptive (2-3 words), ML/AI design focused
                                        - entries: ONE entry representing the whole document
                                        - index: Hierarchical numbering (1, 1.1, 1.2, 2, 2.1 ...) — mirrors a book index or table of contents
                                        - Top-level topic rows use "-" as sub_topic placeholder and give an overview description
                                        - Sub-topic rows give specific, compelling read-worthy descriptions
                                        - Only create a new top-level topic when the concept domain is genuinely distinct
                                        - Capture ALL relevant ML/AI design concepts — do not arbitrarily limit rows
                                        - categories: Reflect the ML/AI design domains covered
                                        - Ensure JSON is valid

                                        Now analyze the document and produce an index-style table of ML/AI design concepts structured like a book's table of contents:"""


custom_agent_category_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# Agent Memory Organizer

Maintain a running chronological knowledge log by appending new memory items to the existing log.

## Format Requirements
- Preserve the existing log exactly as-is
- Append new memory items in date-wise order (newest first)
- If a date already exists in the log, append new memory_type sections under that date
- If a date is new, insert it in the correct chronological position
- Group entries by date, then by memory_type within each date
- Preserve the original index-style table structure for each memory item
- No merging, no summarizing, no rewriting of existing content
- Maximum {target_length} tokens
"""
            ),
            workflow=PromptBlock(
                ordinal=20,
                prompt="""
# Organization Process
1. Take the existing log from original_content as the base state — output it exactly as received
2. Parse all incoming new memory items
3. Extract date and memory_type for each new item
4. For each new item:
   a. If its date already exists in the log → append the new memory_type section under that date block
   b. If its date is new → insert a new date block in the correct chronological position (newest first)
5. Render each new item with its date header, memory_type label, and original index-style table
6. Do not alter, reorder, or rewrite any existing log content
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# Output Format
```markdown
# {category} — Memory Log

## [Date: YYYY-MM-DD]  ← newest date first

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| 1     | ...   | -         | ...         |
| 1.1   | ...   | ...       | ...         |
| 1.2   | ...   | ...       | ...         |
| 2     | ...   | -         | ...         |
| 2.1   | ...   | ...       | ...         |

---

### [Memory Type]  ← another memory_type under same date if applicable

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---

## [Date: YYYY-MM-DD]  ← older date

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---
```

Rules:
- ALWAYS start with the full existing log before appending anything
- New date blocks go in the correct chronological position relative to existing dates
- New memory_type sections under an existing date go AFTER existing sections for that date
- Never remove or modify any existing entry
Target length: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="""
Category: {category}
Existing Log (preserve as base state): {original_content}
New Memory Items (append these): {new_memory_items_text}
"""
            )
        )

custom_rag_category_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# RAG Memory Organizer

Maintain a running chronological knowledge log by appending new RAG-related memory items to the existing log.

## Format Requirements
- Preserve the existing log exactly as-is
- Append new memory items in date-wise order (newest first)
- If a date already exists in the log, append new memory_type sections under that date
- If a date is new, insert it in the correct chronological position
- Group entries by date, then by memory_type within each date
- Preserve the original index-style table structure for each memory item
- No merging, no summarizing, no rewriting of existing content
- Maximum {target_length} tokens
"""
            ),
            workflow=PromptBlock(
                ordinal=20,
                prompt="""
# Organization Process
1. Take the existing log from original_content as the base state — output it exactly as received
2. Parse all incoming new memory items related to RAG (Retrieval-Augmented Generation)
3. Extract date and memory_type for each new item
4. For each new item:
   a. If its date already exists in the log → append the new memory_type section under that date block
   b. If its date is new → insert a new date block in the correct chronological position (newest first)
5. Render each new item with its date header, memory_type label, and original index-style table
6. Do not alter, reorder, or rewrite any existing log content
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# Output Format
```markdown
# {category} — Memory Log

## [Date: YYYY-MM-DD]  ← newest date first

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| 1     | ...   | -         | ...         |
| 1.1   | ...   | ...       | ...         |
| 1.2   | ...   | ...       | ...         |
| 2     | ...   | -         | ...         |
| 2.1   | ...   | ...       | ...         |

---

### [Memory Type]  ← another memory_type under same date if applicable

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---

## [Date: YYYY-MM-DD]  ← older date

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---
```

Rules:
- ALWAYS start with the full existing log before appending anything
- New date blocks go in the correct chronological position relative to existing dates
- New memory_type sections under an existing date go AFTER existing sections for that date
- Never remove or modify any existing entry
Target length: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="""
Category: {category}
Existing Log (preserve as base state): {original_content}
New Memory Items (append these): {new_memory_items_text}
"""
            )
        )

custom_llm_inference_category_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# LLM Inference Memory Organizer

Maintain a running chronological knowledge log by appending new LLM Inference-related memory items to the existing log.

## Format Requirements
- Preserve the existing log exactly as-is
- Append new memory items in date-wise order (newest first)
- If a date already exists in the log, append new memory_type sections under that date
- If a date is new, insert it in the correct chronological position
- Group entries by date, then by memory_type within each date
- Preserve the original index-style table structure for each memory item
- No merging, no summarizing, no rewriting of existing content
- Maximum {target_length} tokens
"""
            ),
            workflow=PromptBlock(
                ordinal=20,
                prompt="""
# Organization Process
1. Take the existing log from original_content as the base state — output it exactly as received
2. Parse all incoming new memory items related to LLM Inference (optimization, serving, deployment, performance)
3. Extract date and memory_type for each new item
4. For each new item:
   a. If its date already exists in the log → append the new memory_type section under that date block
   b. If its date is new → insert a new date block in the correct chronological position (newest first)
5. Render each new item with its date header, memory_type label, and original index-style table
6. Do not alter, reorder, or rewrite any existing log content
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# Output Format
```markdown
# {category} — Memory Log

## [Date: YYYY-MM-DD]  ← newest date first

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| 1     | ...   | -         | ...         |
| 1.1   | ...   | ...       | ...         |
| 1.2   | ...   | ...       | ...         |
| 2     | ...   | -         | ...         |
| 2.1   | ...   | ...       | ...         |

---

### [Memory Type]  ← another memory_type under same date if applicable

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---

## [Date: YYYY-MM-DD]  ← older date

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---
```

Rules:
- ALWAYS start with the full existing log before appending anything
- New date blocks go in the correct chronological position relative to existing dates
- New memory_type sections under an existing date go AFTER existing sections for that date
- Never remove or modify any existing entry
Target length: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="""
Category: {category}
Existing Log (preserve as base state): {original_content}
New Memory Items (append these): {new_memory_items_text}
"""
            )
        )

custom_llm_training_category_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# LLM Training Memory Organizer

Maintain a running chronological knowledge log by appending new LLM Training-related memory items to the existing log.

## Format Requirements
- Preserve the existing log exactly as-is
- Append new memory items in date-wise order (newest first)
- If a date already exists in the log, append new memory_type sections under that date
- If a date is new, insert it in the correct chronological position
- Group entries by date, then by memory_type within each date
- Preserve the original index-style table structure for each memory item
- No merging, no summarizing, no rewriting of existing content
- Maximum {target_length} tokens
"""
            ),
            workflow=PromptBlock(
                ordinal=20,
                prompt="""
# Organization Process
1. Take the existing log from original_content as the base state — output it exactly as received
2. Parse all incoming new memory items related to LLM Training (pretraining, fine-tuning, RLHF, optimization techniques, training infrastructure)
3. Extract date and memory_type for each new item
4. For each new item:
   a. If its date already exists in the log → append the new memory_type section under that date block
   b. If its date is new → insert a new date block in the correct chronological position (newest first)
5. Render each new item with its date header, memory_type label, and original index-style table
6. Do not alter, reorder, or rewrite any existing log content
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# Output Format
```markdown
# {category} — Memory Log

## [Date: YYYY-MM-DD]  ← newest date first

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| 1     | ...   | -         | ...         |
| 1.1   | ...   | ...       | ...         |
| 1.2   | ...   | ...       | ...         |
| 2     | ...   | -         | ...         |
| 2.1   | ...   | ...       | ...         |

---

### [Memory Type]  ← another memory_type under same date if applicable

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---

## [Date: YYYY-MM-DD]  ← older date

### [Memory Type]

| Index | Topic | Sub-Topic | Description |
|-------|-------|-----------|-------------|
| ...   | ...   | ...       | ...         |

---
```

Rules:
- ALWAYS start with the full existing log before appending anything
- New date blocks go in the correct chronological position relative to existing dates
- New memory_type sections under an existing date go AFTER existing sections for that date
- Never remove or modify any existing entry
Target length: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="""
Category: {category}
Existing Log (preserve as base state): {original_content}
New Memory Items (append these): {new_memory_items_text}
"""
            )
        )


# custom_agent_category_summary = CustomPrompt(
#             objective=PromptBlock(
#                 ordinal=10,
#                 prompt="""
# # Agent Memory Organizer

# Maintain a running chronological knowledge log by appending new memory items to the existing log.

# ## Format Requirements
# - Preserve the existing log exactly as-is
# - Append new memory items in date-wise order (newest first)
# - If a date already exists in the log, append new memory_type sections under that date
# - If a date is new, insert it in the correct chronological position
# - Use the exact memory_type value from each memory item as the section header (### )
# - Group entries by date, then by memory_type within each date
# - Preserve the original index-style table structure for each memory item
# - No merging, no summarizing, no rewriting of existing content
# - Maximum {target_length} tokens
# """
#             ),
#             workflow=PromptBlock(
#                 ordinal=20,
#                 prompt="""
# # Organization Process
# 1. Take the existing log from original_content as the base state — output it exactly as received
# 2. Parse all incoming new memory items
# 3. For each new memory item extract:
#    - date → used as the ## Date header
#    - memory_type → used verbatim as the ### section header
#    - table → rendered as-is in the index-style table under that section
# 4. For each new item:
#    a. If its date already exists in the log → append the new ### memory_type section under that date block
#    b. If its date is new → insert a new ## date block in the correct chronological position (newest first)
# 5. Do not alter, reorder, or rewrite any existing log content
# """
#             ),
#             output=PromptBlock(
#                 ordinal=50,
#                 prompt="""
# # Output Format

# Each memory item maps to the output like this:
# - memory item date      → ## [Date: YYYY-MM-DD]
# - memory item memory_type → ### {memory_type}  ← use exact value from the memory item
# - memory item table     → index-style table rows below the header
# ```markdown
# # {category} — Memory Log

# ## [Date: YYYY-MM-DD]  ← newest date first

# ### {memory_type}  ← exact memory_type from memory item (e.g., "Model Architecture", "Training Strategy")

# | Index | Topic | Sub-Topic | Description |
# |-------|-------|-----------|-------------|
# | 1     | ...   | -         | ...         |
# | 1.1   | ...   | ...       | ...         |
# | 1.2   | ...   | ...       | ...         |
# | 2     | ...   | -         | ...         |
# | 2.1   | ...   | ...       | ...         |

# ---

# ### {memory_type}  ← another memory_type under same date if applicable

# | Index | Topic | Sub-Topic | Description |
# |-------|-------|-----------|-------------|
# | ...   | ...   | ...       | ...         |

# ---

# ## [Date: YYYY-MM-DD]  ← older date

# ### {memory_type}

# | Index | Topic | Sub-Topic | Description |
# |-------|-------|-----------|-------------|
# | ...   | ...   | ...       | ...         |

# ---
# ```

# Rules:
# - ALWAYS start with the full existing log before appending anything
# - ### header must be the exact memory_type string from the memory item — do not rename, generalize, or infer
# - New date blocks go in the correct chronological position relative to existing dates
# - New memory_type sections under an existing date go AFTER existing sections for that date
# - Never remove or modify any existing entry

# Target length: {target_length} tokens
# """
#             ),
#             input=PromptBlock(
#                 ordinal=90,
#                 prompt="""
# Category: {category}
# Existing Log (preserve as base state): {original_content}
# New Memory Items (append these): {new_memory_items_text}
# """
#             )
#         )

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
                                    CategoryConfig(name="RAG", description="RAG-related content",summary_prompt = custom_rag_category_summary),
                                    CategoryConfig(name="LLM Training", description="Training insights",summary_prompt = custom_llm_training_category_summary),
                                    CategoryConfig(name="LLM Inference", description="Inference techniques",summary_prompt = custom_llm_inference_category_summary)
                                ],
                                enable_item_references=True
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
                    resource_url="https://www.linkedin.com/posts/pauliusztin_i-created-an-ai-agent-to-write-a-substack-activity-7420095430807691266-fQ1U?utm_source=share&utm_medium=member_desktop&rcm=ACoAACmrL44B-pNi9lNjFQtuPtX_ODwJk7-cC-0",
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
    #     queries=[{"role": "user", "content": {"text": "Tell me about agents observability?"}}],
    #     where={"user_id": "user123", "workspace_id": "workspace-alphah"}
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