#!/usr/bin/env python3
"""
Comprehensive test for DOCUMENT modality memorize() flow with custom prompts
Tests: preprocessing → extraction → categorization → summary
"""

import asyncio
import os
from pathlib import Path
from memu.config.settings import (
    MemorizeConfig, 
    CategoryConfig,
    CustomPrompt,
    PromptBlock
)
from memu.app import MemoryService

# Import existing prompts for reference
from memu.prompts.preprocess import PROMPTS as PREPROCESS_PROMPTS
from memu.prompts.memory_type import CUSTOM_PROMPTS as MEMORY_TYPE_CUSTOM_PROMPTS
from memu.prompts.category_summary import CUSTOM_PROMPT as CATEGORY_SUMMARY_CUSTOM_PROMPT


class DocumentFlowTester:
    def __init__(self):
        self.llm_calls = []
        self.db_config = {
            "metadata_store": {
                "provider": "postgres",
                "dsn": os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
            }
        }
        self.llm_config = {
            "default": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "chat_model": "gpt-4o-mini",
                "embed_model": "text-embedding-3-small"
            }
        }
    
    def create_test_document(self):
        """Create a comprehensive test document"""
        document = """
        PROFESSIONAL PROFILE: Dr. Sarah Chen
        
        ============================================
        EXECUTIVE SUMMARY
        ============================================
        
        Dr. Sarah Chen is a Senior Research Scientist at DeepMind with over 10 years of experience 
        in artificial intelligence and machine learning. She specializes in reinforcement learning, 
        natural language processing, and neural architecture search.
        
        ============================================
        CURRENT POSITION
        ============================================
        
        Position: Senior Research Scientist
        Company: DeepMind (Google)
        Location: London, United Kingdom
        Start Date: January 2020
        Team: Language Understanding & Generation
        
        Responsibilities:
        - Leading a team of 8 researchers on large language model development
        - Developing novel training techniques for more efficient models
        - Publishing research in top-tier conferences (NeurIPS, ICML, ACL)
        - Collaborating with product teams to deploy models in production
        
        ============================================
        EDUCATION
        ============================================
        
        Ph.D. in Computer Science
        Stanford University, 2015-2019
        Thesis: "Efficient Neural Architecture Search for Language Models"
        Advisor: Professor Andrew Ng
        
        M.S. in Computer Science
        Stanford University, 2013-2015
        
        B.S. in Mathematics and Computer Science
        MIT, 2009-2013
        Graduated Summa Cum Laude
        
        ============================================
        PROFESSIONAL EXPERIENCE
        ============================================
        
        Research Scientist | Google AI | 2019-2020
        - Developed BERT-based models for question answering
        - Improved model efficiency by 40% through pruning techniques
        - Filed 3 patents on neural architecture optimization
        
        Research Intern | Facebook AI Research | Summer 2018
        - Worked on few-shot learning for NLP tasks
        - Published paper at EMNLP 2018
        
        Software Engineer | Microsoft | 2013-2015
        - Built recommendation systems for Bing Search
        - Implemented A/B testing infrastructure
        
        ============================================
        PUBLICATIONS (Selected)
        ============================================
        
        1. "Efficient Transformers: A Survey" (2023)
           NeurIPS 2023 - Best Paper Award
           Citations: 1,200+
           
        2. "Neural Architecture Search for Language Understanding" (2021)
           ICML 2021 - Oral Presentation
           Citations: 800+
           
        3. "Sparse Attention Mechanisms for Long Documents" (2020)
           ACL 2020
           Citations: 650+
           
        Total Publications: 24 papers
        H-Index: 18
        Total Citations: 5,400+
        
        ============================================
        TECHNICAL SKILLS
        ============================================
        
        Programming Languages:
        - Python (Expert - 10 years)
        - C++ (Advanced - 7 years)
        - Julia (Intermediate - 2 years)
        
        ML Frameworks:
        - PyTorch (Expert)
        - TensorFlow (Advanced)
        - JAX (Advanced)
        
        Specializations:
        - Large Language Models
        - Reinforcement Learning
        - Neural Architecture Search
        - Model Compression & Efficiency
        - Transfer Learning
        
        ============================================
        AWARDS & RECOGNITION
        ============================================
        
        - NeurIPS 2023 Best Paper Award
        - Google Founders' Award (2022)
        - MIT Technology Review 35 Under 35 (2021)
        - ICML Outstanding Paper Award (2021)
        - Stanford Graduate Fellowship (2015-2019)
        
        ============================================
        LEADERSHIP & SERVICE
        ============================================
        
        Conference Organization:
        - Area Chair: NeurIPS 2023, ICLR 2024
        - Program Committee: ACL, EMNLP, ICML, NeurIPS (2020-present)
        - Workshop Organizer: "Efficient NLP" at EMNLP 2022
        
        Mentorship:
        - Supervised 5 Ph.D. students (visiting researchers)
        - Mentored 12 interns at Google/DeepMind
        - Guest lecturer at UCL and Oxford
        
        Community:
        - Co-founder: Women in ML London chapter
        - Speaker at 15+ AI conferences and meetups
        - AI Ethics Advisory Board member at Oxford
        
        ============================================
        PERSONAL INTERESTS
        ============================================
        
        Outside of work, Sarah is passionate about:
        - Rock climbing (climbs 5.12a, member of London Climbing Club)
        - Classical piano (Grade 8 certified, performs at charity events)
        - Science communication (writes blog on AI for general audience)
        - Hiking and outdoor photography
        - Cooking international cuisines
        
        Languages:
        - English (Native)
        - Mandarin Chinese (Native)
        - Spanish (Conversational)
        - French (Basic)
        
        ============================================
        CURRENT PROJECTS (2024)
        ============================================
        
        1. "MiniLLM" - Developing a 7B parameter model that matches GPT-3.5 performance
        2. "EcoNLP" - Research on reducing carbon footprint of language model training
        3. "MultilingualBERT 2.0" - Improving cross-lingual transfer learning
        
        ============================================
        FUTURE GOALS
        ============================================
        
        Short-term (1-2 years):
        - Publish breakthrough research on model efficiency
        - Transition into research management role
        - Launch open-source initiative for efficient NLP
        
        Long-term (5+ years):
        - Establish AI research lab focused on sustainable AI
        - Contribute to AI policy and governance
        - Write a book on practical deep learning
        """
        
        Path("tests").mkdir(exist_ok=True)
        doc_path = Path("tests/sarah_chen_profile.txt")
        doc_path.write_text(document)
        
        return str(doc_path)
    
    def track_llm_call(self, step_name: str):
        """Decorator to track LLM calls"""
        def interceptor(metadata, **kwargs):
            prompt = kwargs.get('prompt', '')
            system_prompt = kwargs.get('system_prompt', '')
            
            call_info = {
                'step': step_name,
                'operation': metadata.operation,
                'step_id': metadata.step_id,
                'profile': metadata.profile,
                'prompt_length': len(prompt),
                'has_system_prompt': bool(system_prompt),
                'prompt_preview': prompt[:300] + "..." if len(prompt) > 300 else prompt
            }
            
            self.llm_calls.append(call_info)
            
            print(f"\n{'='*80}")
            print(f"📝 LLM Call #{len(self.llm_calls)}: {step_name}")
            print(f"{'='*80}")
            print(f"Operation: {metadata.operation}")
            print(f"Step ID: {metadata.step_id}")
            print(f"Profile: {metadata.profile}")
            print(f"Prompt Length: {len(prompt)} chars")
            print(f"\nPrompt Preview:")
            print(f"{'-'*80}")
            print(prompt[:500])
            print(f"{'-'*80}")
            
            return call_info
        
        return interceptor
    
    async def test_full_flow_with_default_prompts(self):
        """Test 1: Full document flow with DEFAULT prompts"""
        print("\n" + "="*80)
        print("TEST 1: Document Flow with DEFAULT Prompts")
        print("="*80)
        
        service = MemoryService(
            database_config=self.db_config,
            llm_profiles=self.llm_config,
            memorize_config=MemorizeConfig(
                memory_types=["profile", "knowledge"],
                memory_categories=[
                    CategoryConfig(
                        name="professional_background",
                        description="Career, education, and work experience"
                    ),
                    CategoryConfig(
                        name="technical_skills",
                        description="Programming languages, frameworks, and technical expertise"
                    ),
                    CategoryConfig(
                        name="achievements",
                        description="Awards, publications, and notable accomplishments"
                    ),
                    CategoryConfig(
                        name="personal_interests",
                        description="Hobbies, interests, and personal activities"
                    ),
                ]
            )
        )
        
        # Track all LLM calls
        service.intercept_before_llm_call(self.track_llm_call("default_flow"))
        
        # Create and process document
        doc_path = self.create_test_document()
        
        print("\n📄 Processing document: Sarah Chen's Professional Profile")
        print(f"   File: {doc_path}")
        print(f"   Size: {Path(doc_path).stat().st_size} bytes")
        
        result = await service.memorize(
            resource_url=doc_path,
            modality="document",
            user={"user_id": "sarah_chen", "workspace_id": "workspace-research"}
        )
        
        # Analyze results
        self.analyze_results("DEFAULT PROMPTS", result)
        
        return result
    
    async def test_full_flow_with_custom_prompts(self):
        """Test 2: Full document flow with CUSTOM prompts"""
        print("\n" + "="*80)
        print("TEST 2: Document Flow with CUSTOM Prompts")
        print("="*80)
        
        # Reset tracking
        self.llm_calls = []
        
        # ========== CUSTOM PROMPT 1: Document Preprocessing ==========
        custom_doc_preprocess = """
# System Design Blog Analyzer

Extract and structure system design concepts, patterns, and technical insights from the blog post.

## Analysis Requirements
1. Identify core system design concepts and patterns discussed
2. Extract architectural components and their relationships
3. Capture trade-offs, scalability considerations, and design decisions
4. Note specific technologies, databases, and infrastructure mentioned
5. Identify performance metrics, bottlenecks, and optimization techniques

## Output Format
<processed_content>
**Topic:** [Main system design topic]
**Key Concepts:** [Primary system design patterns and principles]
**Architecture Components:** [Main components like load balancers, caches, databases, etc.]
**Scalability Patterns:** [Horizontal/vertical scaling, sharding, replication, etc.]
**Trade-offs Discussed:** [CAP theorem considerations, consistency vs availability, etc.]
**Technologies Mentioned:** [Specific databases, frameworks, tools, protocols]
**Performance Considerations:** [Latency, throughput, bottlenecks discussed]
**Data Flow:** [How data moves through the system]
**Real-world Examples:** [Companies or use cases referenced]
**Best Practices:** [Key recommendations and anti-patterns to avoid]
</processed_content>

<caption>
[One sentence summary of the system design topic, main architecture pattern, and primary use case]
</caption>

## Input Document
{document_text}
"""
        
        # ========== CUSTOM PROMPT 2: Profile Memory Type ==========
#         custom_profile_extraction = CustomPrompt(
#             objective=PromptBlock(
#                 ordinal=10,
#                 prompt="""
# # Professional Profile Memory Extractor

# Extract ONLY factual, stable information about the person's professional identity.

# ## What to Extract
# - Current position and employer
# - Educational credentials (degrees, institutions)
# - Core technical skills and expertise areas
# - Years of experience in key areas
# - Professional specializations

# ## What NOT to Extract
# - Temporary projects or ongoing work
# - Future goals or aspirations
# - Time-bound achievements (save for events)
# - Opinions or preferences
# """
#             ),
#             workflow=PromptBlock(
#                 ordinal=20,
#                 prompt="""
# # Extraction Process
# 1. Identify factual statements about professional identity
# 2. Separate stable facts from temporary situations
# 3. Create self-contained memory items
# 4. Assign to appropriate categories
# """
#             ),
#             rules=PromptBlock(
#                 ordinal=30,
#                 prompt="""
# # Strict Rules
# - Each item must be a complete, standalone statement
# - Maximum 20 words per item
# - Use present tense for current facts
# - No speculation or inference
# - No time-sensitive information
# """
#             ),
#             category=PromptBlock(
#                 ordinal=40,
#                 prompt="## Categories:\n{categories_str}"
#             ),
#             output=PromptBlock(
#                 ordinal=50,
#                 prompt="""
# <item>
#     <memory>
#         <content>Factual professional information</content>
#         <categories>
#             <category>Category Name</category>
#         </categories>
#     </memory>
# </item>
# """
#             ),
#             input=PromptBlock(
#                 ordinal=90,
#                 prompt="<resource>{resource}</resource>"
#             )
#         )
        
        # ========== CUSTOM PROMPT 3: Knowledge Memory Type ==========
#         custom_knowledge_extraction = CustomPrompt(
#             objective=PromptBlock(
#                 ordinal=10,
#                 prompt="""
# # Technical Knowledge Extractor

# Extract factual knowledge, research findings, and technical information.

# ## What to Extract
# - Published research papers (titles, venues, metrics)
# - Awards and recognition with specific details
# - Patents and innovations
# - Technical methodologies and approaches
# - Quantifiable achievements with numbers
# """
#             ),
#             workflow=PromptBlock(
#                 ordinal=20,
#                 prompt="""
# # Process
# 1. Find concrete facts and figures
# 2. Extract research contributions
# 3. Identify technical innovations
# 4. List awards with context
# """
#             ),
#             category=PromptBlock(
#                 ordinal=40,
#                 prompt="## Categories:\n{categories_str}"
#             ),
#             output=PromptBlock(
#                 ordinal=50,
#                 prompt="<item><memory><content>Knowledge fact</content></memory></item>"
#             ),
#             input=PromptBlock(
#                 ordinal=90,
#                 prompt="<resource>{resource}</resource>"
#             )
#         )
        

        custom_agent_evaluation_extraction = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# Agent Evaluation Knowledge Extractor

Extract comprehensive information about agent evaluation concepts, methodologies, frameworks, and best practices.

## What to Extract
- Evaluation frameworks and methodologies
- Metrics and measurement approaches
- Benchmarking techniques and datasets
- Testing strategies and patterns
- Tools and platforms for evaluation
- Common challenges and solutions
- Best practices and anti-patterns
- Specific examples and case studies
"""
            ),
            workflow=PromptBlock(
                ordinal=20,
                prompt="""
# Extraction Process
1. Identify all agent evaluation concepts and frameworks
2. Extract metrics, methodologies, and technical approaches
3. Capture evaluation patterns and best practices
4. Organize into high-level topics and subtopics
5. Create structured memory items with detailed tables
6. Link related concepts and cross-references
"""
            ),
            rules=PromptBlock(
                ordinal=30,
                prompt="""
# Strict Rules
- Extract ALL relevant agent evaluation information
- Include technical details, formulas, and specifications
- Preserve examples, code snippets, and data formats
- Maintain relationships between concepts
- Each memory item should cover a coherent topic area
- Tables must include: high-level topic, subtopics, descriptions, and resource references
"""
            ),
            category=PromptBlock(
                ordinal=40,
                prompt="## Categories:\n{categories_str}"
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
<agent_evaluation>
    <memory>
        <content>
## [High-Level Topic Name]

[Brief overview of the topic - 2-3 sentences]

### Evaluation Table

| High-Level Topic | Subtopic | Description | Resource |
|-----------------|----------|-------------|----------|
| [Main category] | [Specific aspect] | [Detailed explanation of concept, methodology, or approach] | [Source/reference] |
| [Main category] | [Specific aspect] | [Detailed explanation including metrics, formulas, or techniques] | [Source/reference] |

### Key Points
- [Important insight or best practice]
- [Critical consideration or common pitfall]
- [Practical application or example]

### Related Concepts
- [Cross-reference to related evaluation topics]
</content>
        <categories>
            <category>Category Name</category>
        </categories>
    </memory>
</agent_evaluation>

Example:
<agent_evaluation>
    <memory>
        <content>
## Benchmark-Based Evaluation

Standardized benchmarks provide reproducible ways to compare agent performance across different implementations and approaches.

### Evaluation Table

| High-Level Topic | Subtopic | Description | Resource |
|-----------------|----------|-------------|----------|
| Benchmark-Based Evaluation | Standard Datasets | Using established datasets like HotPotQA, MMLU, or custom domain benchmarks to test reasoning and knowledge retrieval | Research papers, benchmark documentation |
| Benchmark-Based Evaluation | Performance Metrics | Accuracy, F1 score, precision/recall for task completion; latency and token efficiency for production readiness | Evaluation frameworks |
| Benchmark-Based Evaluation | Leaderboard Tracking | Comparing agent implementations against public leaderboards to identify state-of-the-art approaches | Community benchmarks |

### Key Points
- Benchmarks enable objective comparison but may not reflect real-world performance
- Choose benchmarks aligned with your agent's specific use case
- Combine multiple benchmarks to avoid overfitting to single metrics

### Related Concepts
- Human evaluation for subjective quality assessment
- A/B testing for production validation
</content>
        <categories>
            <category>Agent Evaluation</category>
        </categories>
    </memory>
</agent_evaluation>
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="<resource>{resource}</resource>"
            )
        )

        # ========== CUSTOM PROMPT 4: Category Summary ==========
#         custom_category_summary = CustomPrompt(
#             objective=PromptBlock(
#                 ordinal=10,
#                 prompt="""
# # Structured Professional Summary Generator

# Create a well-organized summary using a clear structure.

# ## Format Requirements
# - Use markdown headers (##) for sections
# - Use bullet points for lists
# - Include specific numbers and metrics
# - Keep it concise but comprehensive
# - Maximum {target_length} tokens
# """
#             ),
#             workflow=PromptBlock(
#                 ordinal=20,
#                 prompt="""
# # Summarization Process
# 1. Analyze original content and new items
# 2. Organize by themes (experience, skills, achievements)
# 3. Prioritize recent and quantifiable information
# 4. Create clear hierarchy with headers
# """
#             ),
#             output=PromptBlock(
#                 ordinal=50,
#                 prompt="""
# # Output Format
# ```markdown
# # {category}

# ## Overview
# [Brief introduction]

# ## Key Points
# - Point 1 with specific details
# - Point 2 with metrics
# - Point 3 with context

# ## Highlights
# [Most notable items]
# ```

# Target length: {target_length} tokens
# """
#             ),
#             input=PromptBlock(
#                 ordinal=90,
#                 prompt="""
# Category: {category}
# Original Content: {original_content}
# New Memory Items: {new_memory_items_text}
# """
#             )
#         )
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
        
        # ========== Create Service with Custom Prompts ==========
        service = MemoryService(
            database_config=self.db_config,
            llm_profiles=self.llm_config,
            memorize_config=MemorizeConfig(
                memory_types=["agent_evaluation"],  # Only testing custom prompt for this type
                
                # Custom preprocessing prompt
                multimodal_preprocess_prompts={
                    "document": custom_doc_preprocess
                },
                
                # Custom memory type prompts
                memory_type_prompts={
                    "agent_evaluation": custom_agent_evaluation_extraction,
                },
                
                # Custom category summary prompt
                # default_category_summary_prompt=custom_category_summary,
                # default_category_summary_target_length=400,
                
                # Categories
                memory_categories=[
                    CategoryConfig(
                        name="Agents",
                        description="All things related to agents, including evaluation frameworks, methodologies, tools, and best practices",
                        target_length=3000,  ### can add per category prompt too
                        summary_prompt = custom_agent_category_summary

                    ),
                    CategoryConfig(
                        name="RAG",
                        description="All things related to Retrieval-Augmented Generation, including techniques, frameworks, and best practices",
                        target_length=2500,
                        #summary_prompt = custom_rag_category_summary
                    ),
                ]
            )
        )
        
        # Track all LLM calls
        service.intercept_before_llm_call(self.track_llm_call("custom_flow"))
        
        # Process document
        #doc_path = self.create_test_document()
        doc_path = str(Path("tests/agent_evaluation.txt"))
        
        print("\n📄 Processing document with CUSTOM prompts")
        print(f"   Custom preprocessing: ✓")
        print(f"   Custom profile extraction: ✓")
        print(f"   Custom knowledge extraction: ✓")
        print(f"   Custom category summary: ✓")
        
        result = await service.memorize(
            resource_url=doc_path,
            modality="document",
            user={"user_id": "vk456", "workspace_id": "workspace-research"}
        )
        
        # Analyze results
        self.analyze_results("CUSTOM PROMPTS", result)
        
        return result
    
    async def test_per_category_custom_prompts(self):
        """Test 3: Different custom prompts for each category"""
        print("\n" + "="*80)
        print("TEST 3: Per-Category Custom Summary Prompts")
        print("="*80)
        
        # Reset tracking
        self.llm_calls = []
        
        # Different summary style for each category
        technical_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# Technical Skills Summary - Bullet List Format
Create a concise bullet-point summary of technical skills.
Each bullet should mention specific technology and proficiency level.
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# {category}
- Skill 1 (Proficiency)
- Skill 2 (Proficiency)
- Skill 3 (Proficiency)

Max: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="Category: {category}\nItems: {new_memory_items_text}"
            )
        )
        
        achievements_summary = CustomPrompt(
            objective=PromptBlock(
                ordinal=10,
                prompt="""
# Achievements Summary - Metric-Focused Format
Highlight achievements with specific numbers, dates, and impact.
Prioritize quantifiable results.
"""
            ),
            output=PromptBlock(
                ordinal=50,
                prompt="""
# {category}

## Publications
[Number] papers, [citations] citations, h-index [number]

## Awards
- Award 1 (Year)
- Award 2 (Year)

## Impact
[Key metrics and outcomes]

Max: {target_length} tokens
"""
            ),
            input=PromptBlock(
                ordinal=90,
                prompt="Category: {category}\nItems: {new_memory_items_text}"
            )
        )
        
        service = MemoryService(
            database_config=self.db_config,
            llm_profiles=self.llm_config,
            memorize_config=MemorizeConfig(
                memory_types=["profile", "knowledge"],
                memory_categories=[
                    CategoryConfig(
                        name="technical_skills",
                        description="Technical expertise",
                        summary_prompt=technical_summary,  # Custom for this category
                        target_length=200
                    ),
                    CategoryConfig(
                        name="achievements",
                        description="Awards and accomplishments",
                        summary_prompt=achievements_summary,  # Custom for this category
                        target_length=300
                    ),
                    CategoryConfig(
                        name="professional_background",
                        description="Career and education"
                        # Uses default summary prompt
                    ),
                ]
            )
        )
        
        service.intercept_before_llm_call(self.track_llm_call("per_category"))
        
        doc_path = self.create_test_document()
        result = await service.memorize(
            resource_url=doc_path,
            modality="document",
            user={"user_id": "sarah_chen_percategory", "workspace_id": "workspace-research"}
        )
        
        self.analyze_results("PER-CATEGORY PROMPTS", result)
        
        return result
    
    def analyze_results(self, test_name: str, result: dict):
        """Analyze and display results"""
        print(f"\n{'='*80}")
        print(f"📊 RESULTS ANALYSIS - {test_name}")
        print(f"{'='*80}")
        
        # Resources
        resources = result.get('resources', []) or [result.get('resource')]
        print(f"\n📁 Resources Created: {len(resources)}")
        for i, res in enumerate(resources, 1):
            if res:
                print(f"   {i}. {res.get('url', 'N/A')}")
                print(f"      Caption: {res.get('caption', 'No caption')[:100]}...")
                print(f"      Modality: {res.get('modality', 'N/A')}")
        
        # Memory Items
        items = result.get('items', [])
        print(f"\n💾 Memory Items Created: {len(items)}")
        
        items_by_type = {}
        for item in items:
            mtype = item.get('memory_type', 'unknown')
            items_by_type.setdefault(mtype, []).append(item)
        
        for mtype, type_items in items_by_type.items():
            print(f"\n   {mtype.upper()} ({len(type_items)} items):")
            for i, item in enumerate(type_items[:5], 1):  # Show first 5
                summary = item.get('summary', 'No summary')
                print(f"      {i}. {summary[:80]}...")
            if len(type_items) > 5:
                print(f"      ... and {len(type_items) - 5} more")
        
        # Categories
        categories = result.get('categories', [])
        print(f"\n🏷️  Categories Updated: {len(categories)}")
        for cat in categories:
            print(f"\n   📂 {cat.get('name', 'Unknown')}")
            summary = cat.get('summary', 'No summary yet')
            if summary and summary != 'No summary yet':
                print(f"      Summary ({len(summary)} chars):")
                print(f"      {'-'*70}")
                # Print first 300 chars of summary
                print(f"      {summary[:300]}")
                if len(summary) > 300:
                    print(f"      ... ({len(summary) - 300} more chars)")
                print(f"      {'-'*70}")
        
        # LLM Calls Summary
        print(f"\n🤖 LLM Calls Made: {len(self.llm_calls)}")
        for i, call in enumerate(self.llm_calls, 1):
            print(f"   {i}. {call['step_id']:<30} ({call['prompt_length']} chars)")
        
        print(f"\n{'='*80}\n")
    
    def print_final_comparison(self, default_result, custom_result, percategory_result):
        """Compare all three test results"""
        print("\n" + "="*80)
        print("🔍 FINAL COMPARISON - DEFAULT vs CUSTOM vs PER-CATEGORY")
        print("="*80)
        
        tests = [
            ("DEFAULT", default_result),
            ("CUSTOM", custom_result),
            ("PER-CATEGORY", percategory_result)
        ]
        
        print(f"\n{'Metric':<30} {'Default':<15} {'Custom':<15} {'Per-Category':<15}")
        print("-" * 80)
        
        # Items created
        items_counts = [len(r.get('items', [])) for _, r in tests]
        print(f"{'Memory Items':<30} {items_counts[0]:<15} {items_counts[1]:<15} {items_counts[2]:<15}")
        
        # Categories updated
        cat_counts = [len(r.get('categories', [])) for _, r in tests]
        print(f"{'Categories Updated':<30} {cat_counts[0]:<15} {cat_counts[1]:<15} {cat_counts[2]:<15}")
        
        # Profile items
        profile_counts = [
            len([i for i in r.get('items', []) if i.get('memory_type') == 'profile'])
            for _, r in tests
        ]
        print(f"{'Profile Items':<30} {profile_counts[0]:<15} {profile_counts[1]:<15} {profile_counts[2]:<15}")
        
        # Knowledge items
        knowledge_counts = [
            len([i for i in r.get('items', []) if i.get('memory_type') == 'knowledge'])
            for _, r in tests
        ]
        print(f"{'Knowledge Items':<30} {knowledge_counts[0]:<15} {knowledge_counts[1]:<15} {knowledge_counts[2]:<15}")
        
        print("\n" + "="*80)
        
        # Category summary lengths
        print("\n📏 Category Summary Lengths:")
        print("-" * 80)
        
        for test_name, result in tests:
            print(f"\n{test_name}:")
            for cat in result.get('categories', []):
                name = cat.get('name', 'Unknown')
                summary = cat.get('summary', '')
                length = len(summary) if summary else 0
                print(f"   {name:<35} {length:>6} chars")
        
        print("\n" + "="*80)


async def main():
    print("="*80)
    print("COMPREHENSIVE DOCUMENT MODALITY TEST")
    print("Testing Full Memorize Flow with Custom Prompts")
    print("="*80)
    
    tester = DocumentFlowTester()
    
    # Run all three tests
    print("\n🧪 Running 3 comprehensive tests...\n")
    
    # default_result = await tester.test_full_flow_with_default_prompts()
    custom_result = await tester.test_full_flow_with_custom_prompts()
    # percategory_result = await tester.test_per_category_custom_prompts()
    
    # Final comparison
    #tester.print_final_comparison(default_result, custom_result, percategory_result)
    #tester.print_final_comparison(None, custom_result, None)
    
    print("\n✅ All tests completed!")
    print("\nKey Takeaways:")
    print("1. Default prompts: Uses built-in prompts from prompts/ folder")
    print("2. Custom prompts: Demonstrates fully customized extraction and summarization")
    print("3. Per-category: Shows how different categories can have different summary styles")
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())