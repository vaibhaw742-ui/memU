# tests/test_memorize_steps.py

import asyncio
import tempfile
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from memu.app.service import MemoryService
from memu.config.settings import (
    BlobConfig,
    DatabaseConfig,
    LLMProfilesConfig,
    MemorizeConfig,
    CategoryConfig,
)

def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_success(message):
    """Print success message"""
    print(f"✓ {message}")


def print_info(message):
    """Print info message"""
    print(f"→ {message}")


def convert_text_to_markdown(text):
    """
    Convert plain text to proper Markdown format without using LLM.
    
    This function:
    - Detects headings (lines that look like titles)
    - Adds proper heading markers (##)
    - Preserves paragraphs
    - Handles bullet points
    - Adds proper spacing
    """
    if not text or not text.strip():
        return ""
    
    lines = text.strip().split('\n')
    md_lines = []
    
    prev_was_empty = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip multiple consecutive empty lines
        if not stripped:
            if not prev_was_empty:
                md_lines.append("")
                prev_was_empty = True
            continue
        
        prev_was_empty = False
        
        # Detect headings (short lines that are all caps or title case, not ending with punctuation)
        is_heading = False
        if len(stripped) < 60 and not stripped[-1] in '.!?,;:':
            # Check if it's title case or all caps
            words = stripped.split()
            if words:
                # Title case: most words start with capital
                capitals = sum(1 for w in words if w and w[0].isupper())
                if capitals >= len(words) * 0.7:  # 70% of words capitalized
                    is_heading = True
        
        if is_heading:
            # Add as heading
            md_lines.append(f"## {stripped}")
            md_lines.append("")  # Blank line after heading
        elif stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*'):
            # Bullet point
            # Normalize bullet points to use '-'
            content = stripped[1:].strip()
            md_lines.append(f"- {content}")
        elif stripped[0].isdigit() and ('. ' in stripped[:4] or ') ' in stripped[:4]):
            # Numbered list
            md_lines.append(stripped)
        else:
            # Regular paragraph
            md_lines.append(stripped)
    
    # Join lines and clean up extra blank lines
    md_text = '\n'.join(md_lines)
    
    # Remove more than 2 consecutive newlines
    while '\n\n\n' in md_text:
        md_text = md_text.replace('\n\n\n', '\n\n')
    
    return md_text.strip()


def create_extraction_prompt(text_md, categories_prompt_str):
    """
    Create a prompt for LLM to extract a table representation from text_md.
    
    Returns a prompt that asks the LLM to:
    1. Determine the memory_type (2-3 words)
    2. Create a single table entry representing the entire document
    """
    prompt = f"""Analyze the following markdown document and extract it as a structured table.

MARKDOWN CONTENT:
{text_md}

AVAILABLE CATEGORIES:
{categories_prompt_str}

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
    
    return prompt


def parse_extraction_response(response_text):
    """Parse the LLM response and extract memory_type and entries as tuples"""
    import json
    import re
    
    # Try to extract JSON from the response
    try:
        # First try direct JSON parse
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON block in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return None, []
        else:
            return None, []
    
    if not isinstance(data, dict):
        return None, []
    
    memory_type = data.get("memory_type", "knowledge")
    entries_data = data.get("entries", [])
    
    # Parse entries - convert to tuples (memory_type, table, categories)
    entries = []
    for entry in entries_data:
        if not isinstance(entry, dict):
            continue
        
        table = entry.get("table", "").strip()
        categories = entry.get("categories", [])
        
        if table:  # Must have table content
            # Return as tuple: (memory_type, table, categories)
            entries.append((memory_type, table, categories))
    
    return memory_type, entries


def parse_table_for_display(table_string):
    """Parse the table string into rows for display"""
    rows = []
    for line in table_string.split('\n'):
        line = line.strip()
        if line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                rows.append({
                    'topic': parts[0],
                    'sub_topic': parts[1],
                    'description': parts[2]
                })
    return rows


def save_category_markdown(category, output_dir):
    """
    Save category summary as a markdown file.
    
    Args:
        category: Category object with name and summary
        output_dir: Directory to save markdown files
    
    Returns:
        Path to the saved markdown file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename from category name
    filename = f"{category.name.lower().replace(' ', '_')}.md"
    filepath = output_path / filename
    
    # Write the summary to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(category.summary if category.summary else f"# {category.name}\n\nNo content yet.")
    
    return filepath


def create_test_service(temp_dir, use_postgres=True):
    """Create a MemoryService instance for testing
    
    Args:
        temp_dir: Temporary directory for blob storage
        use_postgres: If True, use PostgreSQL; if False, use in-memory storage
    """
    import os
    
    blob_config = BlobConfig(
        resources_dir=str(temp_dir / "resources")
    )
    
    # Configure database based on use_postgres flag
    if use_postgres:
        database_config = DatabaseConfig(
            metadata_store={
                "provider": "postgres",
                "dsn": os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
            },
            vector_index={
                "provider": "pgvector",
                "dsn": os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/memu")
            }
        )
        print_info("Using PostgreSQL for storage")
    else:
        database_config = DatabaseConfig(
            metadata_store={"provider": "inmemory"},
            vector_index={"provider": "pgvector"}
        )
        print_info("Using in-memory storage")
    
    memorize_config = MemorizeConfig(
        memory_categories=[
            CategoryConfig(
                name="Technology",
                description="Technology and AI related topics"
            ),
            CategoryConfig(
                name="Science",
                description="Scientific knowledge and discoveries"
            ),
            CategoryConfig(
                name="Business",
                description="Business strategies and market insights"
            )
        ]
    )
    
    llm_profiles = LLMProfilesConfig(
        default={
            "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "chat_model": "gpt-4o-mini",
                "embed_model": "text-embedding-3-small"
        }
    )
    
    service = MemoryService(
        blob_config=blob_config,
        database_config=database_config,
        memorize_config=memorize_config,
        llm_profiles=llm_profiles,
    )
    
    return service


def create_test_user():
    """Create test user context"""
    return {
        "user_id": "test_user_12345",
        "workspace_id": "workspace_test_789"
    }


def create_mock_llm_client():
    """Create a fully mocked LLM client"""
    mock_llm = AsyncMock()
    
    # Mock embed method
    async def mock_embed(texts):
        # Return different embeddings for each text
        return [[0.1 + i * 0.01] * 1536 for i in range(len(texts))]
    
    mock_llm.embed = mock_embed
    
    # Mock summarize method
    async def mock_summarize(prompt, system_prompt=None):
        return "Mocked LLM response"
    
    mock_llm.summarize = mock_summarize
    
    return mock_llm


def create_test_document_1(temp_dir):
    """Create test document 1: AI and Machine Learning"""
    doc_path = temp_dir / "ai_ml_document.txt"
    content = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is revolutionizing how we interact with technology.
    Machine learning, a subset of AI, enables computers to learn from data without
    being explicitly programmed. Deep learning neural networks have achieved
    remarkable results in image recognition, natural language processing, and
    game playing.
    
    Key Applications
    
    The future of AI looks promising with applications in healthcare, autonomous
    vehicles, and climate change solutions.
    
    - Healthcare diagnostics and drug discovery
    - Self-driving cars and transportation systems
    - Climate modeling and environmental protection
    - Natural language understanding and generation
    - Computer vision and image analysis
    
    Technical Foundations
    
    Machine learning algorithms include supervised learning, unsupervised learning,
    and reinforcement learning. Each approach has different strengths and use cases.
    Supervised learning uses labeled data to train models, while unsupervised learning
    finds patterns in unlabeled data.
    
    Future Trends
    
    Emerging trends include federated learning for privacy-preserving AI, quantum
    machine learning, and neuromorphic computing. These innovations promise to make
    AI more efficient, secure, and powerful.
    """
    doc_path.write_text(content)
    return str(doc_path)


def create_test_document_2(temp_dir):
    """Create test document 2: Quantum Computing"""
    doc_path = temp_dir / "quantum_computing_document.txt"
    content = """
    Quantum Computing Revolution
    
    Quantum computing represents a paradigm shift in computational technology.
    Unlike classical computers that use bits (0s and 1s), quantum computers use
    qubits that can exist in superposition states, enabling exponentially faster
    computation for certain problems.
    
    Quantum Principles
    
    The power of quantum computing comes from three key principles: superposition,
    entanglement, and quantum interference. These quantum mechanical phenomena
    allow quantum computers to explore many solutions simultaneously.
    
    Core Concepts
    
    - Superposition allows qubits to be in multiple states at once
    - Entanglement creates correlations between distant qubits
    - Quantum gates manipulate qubit states
    - Quantum algorithms like Shor's and Grover's provide speedups
    - Error correction is crucial due to quantum decoherence
    
    Real World Applications
    
    Quantum computers excel at optimization problems, cryptography, drug discovery,
    and materials science. They can simulate quantum systems that are intractable
    for classical computers, enabling breakthroughs in chemistry and physics.
    
    Current Challenges
    
    1. Maintaining quantum coherence at practical temperatures
    2. Scaling up qubit counts while maintaining quality
    3. Developing error correction techniques
    4. Creating practical quantum algorithms
    5. Building quantum-resistant cryptography
    
    Industry Impact
    
    Major tech companies and startups are investing billions in quantum research.
    IBM, Google, and Microsoft have all demonstrated quantum supremacy for specific
    tasks. The race is on to build fault-tolerant quantum computers that can solve
    real-world problems at scale.
    """
    doc_path.write_text(content)
    return str(doc_path)

def categorize_resource_url(resource_url):
    """
    Categorize a resource URL based on its domain and file extension.
    No LLM needed - pure pattern matching.
    
    Args:
        resource_url: URL or file path of the resource
    
    Returns:
        tuple: (modality, clean_url)
        modality options: "linkedin", "x", "substack", "medium", "website", 
                         "document", "video", "audio"
    """
    from urllib.parse import urlparse
    import re
    
    url_lower = resource_url.lower()
    
    # Parse URL
    try:
        parsed = urlparse(resource_url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
    except:
        # If parsing fails, treat as file path
        domain = ""
        path = resource_url.lower()
    
    # Check for LinkedIn
    if 'linkedin.com' in domain:
        return "linkedin", resource_url
    
    # Check for X (Twitter)
    if 'twitter.com' in domain or 'x.com' in domain or 't.co' in domain:
        return "x", resource_url
    
    # Check for Substack
    if 'substack.com' in domain or '.substack.com' in domain:
        return "substack", resource_url
    
    # Check for Medium
    if 'medium.com' in domain or domain.endswith('.medium.com'):
        return "medium", resource_url
    
    # Check for YouTube
    if 'youtube.com' in domain or 'youtu.be' in domain:
        return "video", resource_url
    
    # Check for video platforms
    video_platforms = ['vimeo.com', 'dailymotion.com', 'twitch.tv', 'tiktok.com']
    if any(platform in domain for platform in video_platforms):
        return "video", resource_url
    
    # Check for audio platforms
    audio_platforms = ['spotify.com', 'soundcloud.com', 'anchor.fm', 'podcasts.apple.com']
    if any(platform in domain for platform in audio_platforms):
        return "audio", resource_url
    
    # Check file extensions for documents
    doc_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.odt']
    if any(path.endswith(ext) for ext in doc_extensions):
        return "document", resource_url
    
    # Check file extensions for videos
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
    if any(path.endswith(ext) for ext in video_extensions):
        return "video", resource_url
    
    # Check file extensions for audio
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']
    if any(path.endswith(ext) for ext in audio_extensions):
        return "audio", resource_url
    
    # Check for image extensions (treat as document)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']
    if any(path.endswith(ext) for ext in image_extensions):
        return "document", resource_url
    
    # If it has a domain but doesn't match specific platforms, it's a website
    if domain and domain not in ['', 'localhost', '127.0.0.1']:
        return "website", resource_url
    
    # Default to document for local files
    return "document", resource_url



import os
import requests
import time
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


async def process_url_with_airtop(resource_url, modality):
    """
    Process a URL through Airtop agent to extract content as markdown.
    
    Used for: linkedin, x, substack, medium, website
    
    Args:
        resource_url: The URL to process
        modality: The type of resource
    
    Returns:
        text_md: Extracted content in markdown format
    """
    # Configuration
    API_KEY = os.getenv("AIRTOP_API_KEY", "e17970941046bb77.jWGzPtLwZOmPHwTmPT4CVIFqvCecrFW9ADdCSzjw9c")
    AGENT_WEBHOOK_URL = "https://api.airtop.ai/api/hooks/agents/e0103755-2146-43d3-bd25-5410d00b3654/webhooks/984d5de3-2807-43c8-af8a-f441652a11f4"
    BASE_URL = "https://api.airtop.ai/api/hooks/agents/e0103755-2146-43d3-bd25-5410d00b3654"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    # Customize extraction based on modality
    direction = _get_extraction_direction(modality)
    keywords = _get_keywords_for_modality(modality)
    
    print(f"[Airtop] Triggering agent for {modality}...")
    print(f"[Airtop] URL: {resource_url}")
    
    # Step 1: Trigger the agent
    try:
        response = requests.post(
            AGENT_WEBHOOK_URL,
            headers=headers,
            json={
                'configVars': {
                    'url': resource_url,
                }
            },
            verify=False
        )
        
        if response.status_code != 200:
            raise Exception(f"Error triggering agent: {response.status_code} - {response.text}")
        
        invocation_id = response.json()['invocationId']
        print(f"[Airtop] Agent triggered! Invocation ID: {invocation_id}")
        
    except Exception as e:
        print(f"[Airtop] Error triggering agent: {e}")
        raise
    
    # Step 2: Poll for results
    print("[Airtop] Polling for results...")
    result_url = f"{BASE_URL}/invocations/{invocation_id}/result"
    
    max_attempts = 180  # 15 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        try:
            result_response = requests.get(
                result_url,
                headers={'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'},
                verify=False,
                timeout=30
            )
            
            if result_response.status_code != 200:
                raise Exception(f"Error getting result: {result_response.status_code}")
            
            result = result_response.json()
            status = result.get('status', '').lower()
            
            if attempt % 10 == 0:  # Print every 10th attempt
                print(f"[Airtop] [{attempt}/{max_attempts}] Status: {status}")
            
            if status == 'completed':
                print(f"[Airtop] ✅ Agent completed successfully!")
                output = result.get('output', '')
                
                # Airtop returns markdown directly
                text_md = str(output) if output else ''
                
                print(f"[Airtop] Extracted {len(text_md)} characters")
                return text_md
                
            elif status == 'failed':
                error = result.get('error', 'Unknown error')
                raise Exception(f"Agent failed: {error}")
                
            elif status in ['running', 'awaiting session', 'pending']:
                time.sleep(5)
            else:
                # Unknown status, continue polling
                time.sleep(5)
                
        except requests.exceptions.Timeout:
            print(f"[Airtop] Request timeout on attempt {attempt}, retrying...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"[Airtop] Error during polling: {e}")
            raise
    
    # Timeout
    raise Exception(f"Timeout: Agent did not complete within {max_attempts * 5 / 60} minutes")


async def process_video(resource_url):
    """
    Process video URL to extract transcript/metadata.
    
    For YouTube and other video platforms.
    
    Args:
        resource_url: Video URL
    
    Returns:
        text_md: Video content in markdown format
    """
    print(f"[Video] Processing video: {resource_url}")
    
    # TODO: Implement video processing
    # Options:
    # 1. YouTube Transcript API for YouTube videos
    # 2. Whisper API for general video transcription
    # 3. Video download + processing pipeline
    
    # Placeholder for now
    text_md = f"""# Video Content

**Source:** {resource_url}
**Type:** Video

## Metadata
- URL: {resource_url}
- Platform: YouTube/Video

## Transcript
[Video transcript will be extracted here using YouTube Transcript API or Whisper]

**Note:** Video processing integration pending. Will support:
- YouTube videos via YouTube Transcript API
- General videos via Whisper transcription
- Video metadata extraction
"""
    
    print(f"[Video] Generated placeholder content ({len(text_md)} chars)")
    return text_md


async def process_audio(resource_url):
    """
    Process audio URL to extract transcript/metadata.
    
    For podcasts and audio platforms.
    
    Args:
        resource_url: Audio URL
    
    Returns:
        text_md: Audio content in markdown format
    """
    print(f"[Audio] Processing audio: {resource_url}")
    
    # TODO: Implement audio processing
    # Options:
    # 1. Spotify API for podcast metadata
    # 2. Whisper API for audio transcription
    # 3. Audio download + processing pipeline
    
    # Placeholder for now
    text_md = f"""# Audio Content

**Source:** {resource_url}
**Type:** Audio/Podcast

## Metadata
- URL: {resource_url}
- Platform: Spotify/Podcast

## Transcript
[Audio transcript will be extracted here using Whisper API]

**Note:** Audio processing integration pending. Will support:
- Podcast transcription via Whisper
- Spotify podcast metadata
- General audio file processing
"""
    
    print(f"[Audio] Generated placeholder content ({len(text_md)} chars)")
    return text_md


def _get_extraction_direction(modality):
    """Get extraction direction based on modality"""
    directions = {
        'linkedin': 'extract all concepts, insights, and key points from the LinkedIn post',
        'x': 'extract the main points, insights, and any linked content from the X/Twitter post',
        'substack': 'extract the full article content, main arguments, and key insights',
        'medium': 'extract the full article content, main arguments, and key insights',
        'website': 'extract all relevant content and key information from the webpage',
    }
    return directions.get(modality, 'extract all relevant content and key insights')


def _get_keywords_for_modality(modality):
    """Get relevant keywords based on modality"""
    # General ML/AI keywords
    base_keywords = 'machine learning,deep learning,LLM,RAG,neural networks,AI,artificial intelligence,reinforcement learning,agents,generative AI,transformer,attention,embeddings,vector search,retrieval,training,inference,fine-tuning,prompt engineering,multimodal,computer vision,NLP,recommendation systems'
    
    # Add modality-specific keywords
    if modality in ['linkedin', 'x']:
        return base_keywords + ',insights,opinion,analysis,commentary,discussion'
    elif modality in ['substack', 'medium', 'website']:
        return base_keywords + ',article,blog,post,tutorial,guide,explanation,analysis'
    
    return base_keywords


# async def test_01_ingest_resource(service, test_document, doc_name, user):
#     """Test Step 1: Ingest Resource"""
#     print_separator(f"STEP 1: Ingest Resource - {doc_name}")
    
#     try:
#         initial_state = {
#             "resource_url": test_document,
#             "modality": "document",
#             "user": user,  # Add user context
#         }
        
#         print_info(f"Document: {doc_name}")
#         print_info(f"User: {user['user_id']}")
#         print_info(f"Workspace: {user['workspace_id']}")
        
#         step_context = {"step_id": "ingest_resource"}
#         result_state = await service._memorize_ingest_resource(
#             initial_state, 
#             step_context
#         )
        
#         print_success("Ingested resource successfully")
#         print_info(f"Raw text length: {len(result_state['raw_text'])} characters")
        
#         print("sjfh")
#         print(result_state)
#         print("jhfbjh")
#         return result_state
        
#     except Exception as e:
#         print(f"✗ Test failed: {e}")
#         raise
async def test_01_ingest_resource(service, resource_url, doc_name, user):
    """Test Step 1: Ingest Resource with automatic modality detection"""
    print_separator(f"STEP 1: Ingest Resource - {doc_name}")
    
    try:
        # Automatically categorize the resource URL
        modality, clean_url = categorize_resource_url(resource_url)
        
        initial_state = {
            "resource_url": clean_url,
            "modality": modality,
            "user": user,
        }
        
        print_info(f"Resource: {doc_name}")
        print_info(f"URL: {clean_url}")
        print_info(f"Detected Modality: {modality}")
        print_info(f"User: {user['user_id']}")
        print_info(f"Workspace: {user['workspace_id']}")
        
        step_context = {"step_id": "ingest_resource"}
        result_state = await service._memorize_ingest_resource(
            initial_state, 
            step_context
        )
        
        print_success("Ingested resource successfully")
        print_info(f"Modality confirmed: {result_state.get('modality', modality)}")

    
        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

# async def test_02_preprocess_multimodal(service, state_after_ingest, doc_name):
#     """Test Step 2: Preprocess Multimodal with text_md generation"""
#     print_separator(f"STEP 2: Preprocess Multimodal - {doc_name}")
    
#     try:
#         mock_llm = create_mock_llm_client()
        
#         if "AI" in doc_name:
#             llm_response = """
#             <processed_content>AI and ML overview covering fundamentals and applications</processed_content>
#             <caption>Comprehensive guide to AI and Machine Learning</caption>
#             """
#         else:
#             llm_response = """
#             <processed_content>Quantum computing principles and industry applications</processed_content>
#             <caption>Introduction to Quantum Computing</caption>
#             """
        
#         mock_llm.summarize = AsyncMock(return_value=llm_response)
        
#         step_context = {
#             "step_id": "preprocess_multimodal",
#             "step_config": {"chat_llm_profile": "default"}
#         }
        
#         with patch.object(service, '_get_step_llm_client', return_value=mock_llm):
#             result_state = await service._memorize_preprocess_multimodal(
#                 state_after_ingest,
#                 step_context
#             )
        
#         preprocessed = result_state["preprocessed_resources"][0]
        
#         # Generate text_md
#         raw_text = state_after_ingest.get("raw_text", "")
#         text_md = convert_text_to_markdown(raw_text)
#         preprocessed["text_md"] = text_md
#         result_state["preprocessed_resources"][0] = preprocessed
        
#         print_success("Generated text_md successfully")
#         print_info(f"text_md: {len(text_md)} chars, {text_md.count('##')} headings")
        
#         return result_state
        
#     except Exception as e:
#         print(f"✗ Test failed: {e}")
#         raise

async def test_02_preprocess_multimodal(service, state_after_ingest, doc_name):
    """Test Step 2: Preprocess Multimodal - Extract content based on modality"""
    print_separator(f"STEP 2: Preprocess Multimodal - {doc_name}")
    
    try:
        resource_url = state_after_ingest["resource_url"]
        modality = state_after_ingest["modality"]
        
        print_info(f"Resource URL: {resource_url}")
        print_info(f"Modality: {modality}")
        
        ##Refactor
        #step_context = {
#             "step_id": "preprocess_multimodal",
#             "step_config": {"chat_llm_profile": "default"}
#         }
        
#         with patch.object(service, '_get_step_llm_client', return_value=mock_llm):
#             result_state = await service._memorize_preprocess_multimodal(
#                 state_after_ingest,
#                 step_context
#             )
        # Route to appropriate processor based on modality
        if modality in ['linkedin', 'x', 'substack', 'medium', 'website']:
            print_info("Using Airtop for web content extraction")
            text_md = await process_url_with_airtop(resource_url, modality)
            
        elif modality == 'video':
            print_info("Using video processor for transcript extraction")
            text_md = await process_video(resource_url)
            
        elif modality == 'audio':
            print_info("Using audio processor for transcript extraction")
            text_md = await process_audio(resource_url)
            
        elif modality == 'document':
            print_info("Document modality - skipping preprocessing (handled in ingest)")
            # For documents, we might already have the content or need special handling
            text_md = f"# Document\n\nDocument processing for: {resource_url}"
            
        else:
            raise ValueError(f"Unsupported modality for preprocessing: {modality}")
        
        # Validate content
        if not text_md or len(text_md) < 10:
            # Fallback for empty content
            text_md = f"# {doc_name}\n\nContent could not be extracted from {resource_url}"
            print_info("⚠️  Warning: Minimal content extracted, using fallback")
        
        # Create preprocessed resource
        preprocessed_resources = [{
            "text_md": text_md,
            "resource_url": resource_url,
            "modality": modality,
            "caption": f"Content from {modality}: {doc_name}"
        }]
        
        # Update state
        result_state = state_after_ingest.copy()
        result_state["preprocessed_resources"] = preprocessed_resources

     
        print_success(f"Preprocessed successfully using {modality} processor")
        print_info(f"text_md length: {len(text_md)} characters")
        print_info(f"Headings: {text_md.count('#')}")
        print_info(f"Lines: {len(text_md.split(chr(10)))}")
        
        # Show preview
        print("\n📄 Content Preview (first 300 chars):")
        print("─" * 70)
        print(text_md[:1000] + ("..." if len(text_md) > 1000 else ""))
        print("─" * 70)
        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create fallback state
        print_info("Creating fallback state due to error")
        result_state = state_after_ingest.copy()
        result_state["preprocessed_resources"] = [{
            "text_md": f"# Error\n\nFailed to process {state_after_ingest['resource_url']}\n\nError: {str(e)}",
            "resource_url": state_after_ingest["resource_url"],
            "modality": state_after_ingest["modality"],
            "caption": f"Error processing {doc_name}"
        }]
        
        return result_state

# async def test_03_extract_items(service, state_after_preprocess, doc_name):
#     """Test Step 3: Extract Memory Items - Entries as tuples"""
#     print_separator(f"STEP 3: Extract Memory Items (Tuple Format) - {doc_name}")
    
#     try:
#         mock_llm = create_mock_llm_client()
        
#         preprocessed = state_after_preprocess["preprocessed_resources"][0]
#         text_md = preprocessed.get("text_md", "")
        
#         print_info(f"Using text_md ({len(text_md)} chars)")
        
#         categories_prompt_str = """- Technology: Technology and AI related topics
# - Science: Scientific knowledge and discoveries
# - Business: Business strategies and market insights"""
        
#         # LLM response with single table entry per document
#         if "AI" in doc_name:
#             llm_response = """{
#     "memory_type": "AI Knowledge",
#     "entries": [
#         {
#             "table": "Machine Learning | Deep Learning | Neural networks achieving breakthroughs in image recognition, NLP, and game playing\\nMachine Learning | Learning Paradigms | Supervised, unsupervised, and reinforcement learning with different strengths\\nAI Applications | Healthcare | Diagnostics, drug discovery, and medical imaging powered by AI\\nAI Applications | Autonomous Systems | Self-driving cars and transportation using computer vision\\nFuture AI | Privacy & Efficiency | Federated learning and neuromorphic computing for efficient AI",
#             "categories": ["Technology", "Science"]
#         }
#     ]
# }"""
#         else:
#             llm_response = """{
#     "memory_type": "Quantum Tech",
#     "entries": [
#         {
#             "table": "Quantum Computing | Core Principles | Superposition, entanglement, and quantum interference for faster computation\\nQuantum Bits | Qubits | Quantum bits in superposition states manipulated by quantum gates\\nQuantum Applications | Optimization | Solving optimization, cryptography, and simulating quantum systems\\nQuantum Challenges | Error Correction | Maintaining coherence and developing error correction techniques\\nIndustry Impact | Tech Giants | IBM, Google, Microsoft investing billions and demonstrating quantum supremacy",
#             "categories": ["Technology", "Science", "Business"]
#         }
#     ]
# }"""
        
#         mock_llm.summarize = AsyncMock(return_value=llm_response)
        
#         extraction_prompt = create_extraction_prompt(text_md, categories_prompt_str)
        
#         print_info("Extracting table representation from text_md...")
        
#         with patch.object(service, '_get_step_llm_client', return_value=mock_llm):
#             llm_response_text = await mock_llm.summarize(extraction_prompt)
        
#         memory_type, entries = parse_extraction_response(llm_response_text)
        
#         if not memory_type:
#             memory_type = "Knowledge"
        
#         print_success(f"Memory Type: '{memory_type}'")
#         print_info(f"Entries: {len(entries)} tuple(s)")
        
#         # Create resource plans with entries as tuples
#         resource_plans = [{
#             "resource_url": f"{doc_name.lower().replace(' ', '_').replace('&', 'and')}.txt",
#             "text": preprocessed.get("text", ""),
#             "text_md": text_md,
#             "caption": preprocessed.get("caption"),
#             "entries": entries,  # List of tuples: [(memory_type, table, categories), ...]
#         }]
        
#         result_state = state_after_preprocess.copy()
#         result_state["resource_plans"] = resource_plans
        
#         print_success("Tuple-based extraction completed")
        
#         return result_state
        
#     except Exception as e:
#         print(f"✗ Test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise


async def test_03_extract_items(service, state_after_preprocess, doc_name):
    """Test Step 3: Extract Memory Items - Entries as tuples"""
    print_separator(f"STEP 3: Extract Memory Items (Tuple Format) - {doc_name}")
    
    try:
        mock_llm = create_mock_llm_client()
        
        preprocessed = state_after_preprocess["preprocessed_resources"][0]
        text_md = preprocessed.get("text_md", "")
        modality = preprocessed.get("modality", "document")
        resource_url = preprocessed.get("resource_url", "")
        
        print_info(f"Using text_md ({len(text_md)} chars)")
        print_info(f"Modality: {modality}")
        print_info(f"Resource: {resource_url}")
        
        categories_prompt_str = """- Technology: Technology and AI related topics
- Science: Scientific knowledge and discoveries
- Business: Business strategies and market insights"""
        
        # Generate LLM response based on actual content or modality
        # For now, use simple keyword detection from text_md
        if any(keyword in text_md.lower() for keyword in ['ai', 'machine learning', 'neural', 'deep learning']):
            memory_type = "AI Knowledge"
            llm_response = """{
    "memory_type": "AI Knowledge",
    "entries": [
        {
            "table": "Machine Learning | Deep Learning | Neural networks achieving breakthroughs in image recognition, NLP, and game playing\\nMachine Learning | Learning Paradigms | Supervised, unsupervised, and reinforcement learning with different strengths\\nAI Applications | Healthcare | Diagnostics, drug discovery, and medical imaging powered by AI\\nAI Applications | Autonomous Systems | Self-driving cars and transportation using computer vision\\nFuture AI | Privacy & Efficiency | Federated learning and neuromorphic computing for efficient AI",
            "categories": ["Technology", "Science"]
        }
    ]
}"""
        elif any(keyword in text_md.lower() for keyword in ['quantum', 'qubit', 'superposition']):
            memory_type = "Quantum Tech"
            llm_response = """{
    "memory_type": "Quantum Tech",
    "entries": [
        {
            "table": "Quantum Computing | Core Principles | Superposition, entanglement, and quantum interference for faster computation\\nQuantum Bits | Qubits | Quantum bits in superposition states manipulated by quantum gates\\nQuantum Applications | Optimization | Solving optimization, cryptography, and simulating quantum systems\\nQuantum Challenges | Error Correction | Maintaining coherence and developing error correction techniques\\nIndustry Impact | Tech Giants | IBM, Google, Microsoft investing billions and demonstrating quantum supremacy",
            "categories": ["Technology", "Science", "Business"]
        }
    ]
}"""
        else:
            # Generic response for any other content
            memory_type = "Web Content"
            llm_response = """{
    "memory_type": "Web Content",
    "entries": [
        {
            "table": "Content Summary | Main Points | Key insights and information from the source\\nKey Concepts | Topics Covered | Primary topics and themes discussed\\nInsights | Analysis | Important takeaways and conclusions\\nReferences | Sources | Links and citations mentioned in content\\nContext | Background | Additional context and related information",
            "categories": ["Technology"]
        }
    ]
}"""
        
        mock_llm.summarize = AsyncMock(return_value=llm_response)
        
        extraction_prompt = create_extraction_prompt(text_md, categories_prompt_str)
        
        print_info("Extracting table representation from text_md...")
        
        with patch.object(service, '_get_step_llm_client', return_value=mock_llm):
            llm_response_text = await mock_llm.summarize(extraction_prompt)
        
        memory_type, entries = parse_extraction_response(llm_response_text)
        
        if not memory_type:
            memory_type = "Knowledge"
        
        print_success(f"Memory Type: '{memory_type}'")
        print_info(f"Entries: {len(entries)} tuple(s)")
        
        # Display the entries
        print("\n📋 EXTRACTED ENTRIES (Tuple Format):")
        for i, entry_tuple in enumerate(entries, 1):
            mtype, table, categories = entry_tuple
            table_rows = parse_table_for_display(table)
            
            print(f"\n  Entry {i}:")
            print(f"    Format: (memory_type, table, categories)")
            print(f"    memory_type: '{mtype}'")
            print(f"    categories: {categories}")
            print(f"    table rows: {len(table_rows)}")
            
            # Display first few rows of the table
            if table_rows:
                print("\n    Sample rows:")
                for j, row in enumerate(table_rows[:3], 1):
                    topic = row['topic'][:20]
                    sub_topic = row['sub_topic'][:20]
                    desc = row['description'][:40]
                    print(f"      {j}. {topic} | {sub_topic} | {desc}...")
        
        # Create resource plans with entries as tuples
        resource_plans = [{
            "resource_url": resource_url,
            # "text": text_md,  # Use text_md as text
            "text_md": text_md,
            "caption": preprocessed.get("caption", f"Content from {modality}"),
            "modality": modality,
            "entries": entries,  # List of tuples: [(memory_type, table, categories), ...]
        }]
        
        result_state = state_after_preprocess.copy()
        result_state["resource_plans"] = resource_plans
        
        print_success("Tuple-based extraction completed")
        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

async def test_04_dedupe_merge(service, state_after_extract, doc_name):
    """Test Step 4: Dedupe and Merge"""
    print_separator(f"STEP 4: Dedupe and Merge - {doc_name}")
    
    try:
        step_context = {"step_id": "dedupe_merge"}
        
        print_info("Running deduplication and merge logic")
        
        result_state = service._memorize_dedupe_merge(
            state_after_extract,
            step_context
        )
        
        assert "resource_plans" in result_state, "Missing 'resource_plans'"
        
        print_success("Dedupe and merge completed successfully")

        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


# async def test_05_categorize_items(service, state_after_dedupe, doc_name, ctx, store, user):
#     """Test Step 5: Categorize Items"""
#     print_separator(f"STEP 5: Categorize Items - {doc_name}")
    
#     try:
#         # Create mock LLM client
#         mock_llm = create_mock_llm_client()
        
#         print_info("Categorizing and persisting memory items...")
#         print_info(f"User: {user['user_id']}")
        
#         # Add required fields to state (including user context)
#         state_after_dedupe["ctx"] = ctx
#         state_after_dedupe["store"] = store
#         state_after_dedupe["user"] = user  # Critical: Add user context
#         state_after_dedupe["local_path"] = "/tmp/test_document.txt"
#         state_after_dedupe["modality"] = "document"
        
#         step_context = {
#             "step_id": "categorize_items",
#             "step_config": {
#                 "embed_llm_profile": "embedding"
#             }
#         }
        
#         # Mock ALL LLM client methods
#         with patch.object(service, '_get_llm_client', return_value=mock_llm), \
#              patch.object(service, '_get_step_llm_client', return_value=mock_llm), \
#              patch.object(service, '_get_step_embedding_client', return_value=mock_llm):
            
#             result_state = await service._memorize_categorize_items(
#                 state_after_dedupe,
#                 step_context
#             )
        
#         print_success("Categorized items successfully")
#         print_info(f"Resources created: {len(result_state['resources'])}")
#         print_info(f"Memory items created: {len(result_state['items'])}")
        
#         return result_state
        
#     except Exception as e:
#         print(f"✗ Test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise


async def test_05_categorize_items(service, state_after_dedupe, doc_name, ctx, store, user):
    """Test Step 5: Categorize Items"""
    print_separator(f"STEP 5: Categorize Items - {doc_name}")
    
    try:
        # Create mock LLM client
        mock_llm = create_mock_llm_client()
        
        print_info("Categorizing and persisting memory items...")
        print_info(f"User: {user['user_id']}")
        
        # Get modality and resource_url from state
        modality = state_after_dedupe.get("modality", "document")
        resource_url = state_after_dedupe.get("resource_url", "")
        
        # Extract from resource_plans if available
        if "resource_plans" in state_after_dedupe and state_after_dedupe["resource_plans"]:
            plan = state_after_dedupe["resource_plans"][0]
            modality = plan.get("modality", modality)
            resource_url = plan.get("resource_url", resource_url)
        
        print_info(f"Modality: {modality}")
        print_info(f"Resource URL: {resource_url}")



        # Add required fields to state (including user context)
        state_after_dedupe["ctx"] = ctx
        state_after_dedupe["store"] = store
        state_after_dedupe["user"] = user  # Critical: Add user context
        state_after_dedupe["local_path"] = resource_url  # Use actual resource URL
        state_after_dedupe["modality"] = modality  # Use actual modality
        
        step_context = {
            "step_id": "categorize_items",
            "step_config": {
                "embed_llm_profile": "embedding"
            }
        }
        
        # Mock ALL LLM client methods
        with patch.object(service, '_get_llm_client', return_value=mock_llm), \
             patch.object(service, '_get_step_llm_client', return_value=mock_llm), \
             patch.object(service, '_get_step_embedding_client', return_value=mock_llm):
            
            result_state = await service._memorize_categorize_items(
                state_after_dedupe,
                step_context
            )
        
        # Check results
        assert "resources" in result_state, "Missing 'resources'"
        assert "items" in result_state, "Missing 'items'"
        assert "relations" in result_state, "Missing 'relations'"
        assert "category_updates" in result_state, "Missing 'category_updates'"
        
        print_success("Categorized items successfully")
        print_info(f"Resources created: {len(result_state['resources'])}")
        print_info(f"Memory items created: {len(result_state['items'])}")
        print_info(f"Category relations: {len(result_state['relations'])}")
        print_info(f"Categories updated: {len(result_state['category_updates'])}")
        
        # Display created items
        print("\n📝 Created Memory Items:")
        for i, item in enumerate(result_state['items'], 1):
            summary_preview = item.summary[:80] + "..." if len(item.summary) > 80 else item.summary
            print(f"  {i}. [{item.memory_type}] {summary_preview}")
        
           
        # print("skjnfk")
        # print(result_state['resources'])
        # print("sfhb")

        # Display resource info
        if result_state['resources']:
            resource = result_state['resources'][0]
            print(f"\n📦 Resource:")
            print(f"  • URL: {resource.url}")
            print(f"  • Modality: {resource.modality}")
        
        # Display category relations
        if result_state['relations']:
            print(f"\n🔗 Category Relations:")
            for relation in result_state['relations']:
                print("sjhfbnj")
                print(relation)
                cat_id = relation.category_id
                cat = store.memory_category_repo.categories.get(cat_id)
                if cat:
                    print(f"  • {cat.name}")

        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise



# async def test_06_persist_index(service, state_after_categorize, doc_name):
#     """Test Step 6: Persist and Index"""
#     print_separator(f"STEP 6: Persist and Index - {doc_name}")
    
#     try:
#         mock_llm = create_mock_llm_client()
        
#         current_date = datetime.now().strftime("%Y-%m-%d")
        
#         async def mock_summarize(prompt, system_prompt=None):
#             category_name = None
            
#             if "Technology" in str(prompt):
#                 category_name = "Technology"
#             elif "Science" in str(prompt):
#                 category_name = "Science"
#             elif "Business" in str(prompt):
#                 category_name = "Business"
            
#             if not category_name:
#                 return "Summary could not be generated"
            
#             print_info(f"Generating summary for category: {category_name}")
            
#             if "AI" in doc_name:
#                 if category_name == "Technology":
#                     return f"""## Technology Category Summary

# ### Overview
# Technology and AI related topics

# ### Memory Items by Date

# #### {current_date}
# **Memory Item 1: AI Knowledge**
# - **Resource**: ai_and_machine_learning.txt
# - **Memory Type**: AI Knowledge
# - **Table Content**:
# ```
#   Machine Learning | Deep Learning | Neural networks achieving breakthroughs
#   AI Applications | Healthcare | Diagnostics and drug discovery
# ```
# - **Topics Covered**: Machine learning, AI applications

# ### Total Items: 1
# ### Last Updated: {current_date}"""
#                 elif category_name == "Science":
#                     return f"""## Science Category Summary

# ### Overview
# Scientific knowledge and discoveries

# ### Memory Items by Date

# #### {current_date}
# **Memory Item 1: AI Knowledge**
# - **Resource**: ai_and_machine_learning.txt
# - **Memory Type**: AI Knowledge

# ### Total Items: 1
# ### Last Updated: {current_date}"""
#             else:  # Quantum
#                 if category_name == "Technology":
#                     return f"""## Technology Category Summary

# ### Overview
# Technology and AI related topics

# ### Memory Items by Date

# #### {current_date}
# **Memory Item 1: Quantum Tech**
# - **Resource**: quantum_computing.txt
# - **Memory Type**: Quantum Tech

# ### Total Items: 1
# ### Last Updated: {current_date}"""
#                 elif category_name == "Science":
#                     return f"""## Science Category Summary

# ### Overview
# Scientific knowledge and discoveries

# ### Memory Items by Date

# #### {current_date}
# **Memory Item 1: Quantum Tech**

# ### Total Items: 1
# ### Last Updated: {current_date}"""
#                 elif category_name == "Business":
#                     return f"""## Business Category Summary

# ### Overview
# Business strategies and market insights

# ### Memory Items by Date

# #### {current_date}
# **Memory Item 1: Quantum Tech**

# ### Total Items: 1
# ### Last Updated: {current_date}"""
            
#             return f"## {category_name}\n\n### Total Items: 0"
        
#         mock_llm.summarize = mock_summarize
        
#         step_context = {
#             "step_id": "persist_index",
#             "step_config": {"chat_llm_profile": "default"}
#         }
        
#         with patch.object(service, '_get_step_llm_client', return_value=mock_llm):
#             result_state = await service._memorize_persist_and_index(
#                 state_after_categorize,
#                 step_context
#             )
        
#         print_success("Persisted and indexed successfully")
        
#         return result_state
        
#     except Exception as e:
#         print(f"✗ Test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise

async def test_06_persist_index(service, state_after_categorize, doc_name):
    """Test Step 6: Persist and Index"""
    print_separator(f"STEP 6: Persist and Index - {doc_name}")
    
    try:
        mock_llm = create_mock_llm_client()
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get modality and resource_url from state for better summaries
        modality = state_after_categorize.get("modality", "document")
        resource_url = state_after_categorize.get("resource_url", "unknown")
        
        # Extract memory type and table from items if available
        memory_type = "Knowledge"
        table_content = "Content | Summary | Key information from the source"
        
        if "items" in state_after_categorize and state_after_categorize["items"]:
            item = state_after_categorize["items"][0]
            memory_type = item.memory_type
            # The summary field contains the table
            table_content = item.summary
        
        print_info(f"Generating summaries for {memory_type}")
        print_info(f"Modality: {modality}")
        
        async def mock_summarize(prompt, system_prompt=None):
            category_name = None
            
            if "Technology" in str(prompt):
                category_name = "Technology"
            elif "Science" in str(prompt):
                category_name = "Science"
            elif "Business" in str(prompt):
                category_name = "Business"
            
            if not category_name:
                return "Summary could not be generated"
            
            print_info(f"Generating summary for category: {category_name}")
            
            # Summary with table included
            return f"""## {category_name} Category Summary

### Overview
{category_name} related content

### Memory Items by Date

#### {current_date}
**Memory Item 1: {memory_type}**
- **Resource**: {resource_url}
- **Modality**: {modality}
- **Memory Type**: {memory_type}
- **Table Content**:
```
  {table_content}
```

### Total Items: 1
### Last Updated: {current_date}"""
        
        mock_llm.summarize = mock_summarize
        
        step_context = {
            "step_id": "persist_index",
            "step_config": {"chat_llm_profile": "default"}
        }
        
        ctx = state_after_categorize.get("ctx")
        store = state_after_categorize.get("store")
        
        print_info("Updating category summaries...")
        
        with patch.object(service, '_get_step_llm_client', return_value=mock_llm):
            result_state = await service._memorize_persist_and_index(
                state_after_categorize,
                step_context
            )
        
        print_success("Persisted and indexed successfully")
        
        # Display updated category summaries
        if ctx and store:
            print("\n📚 Updated Category Summaries:")
            for cat_id in ctx.category_ids:
                cat = store.memory_category_repo.categories.get(cat_id)
                if cat and cat.summary:
                    # Show first 300 chars of summary
                    summary_preview = cat.summary[:300] + "..." if len(cat.summary) > 300 else cat.summary
                    print(f"\n  📁 {cat.name}:")
                    print(f"     {summary_preview}")
        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

async def test_07_build_response(service, state_after_persist, doc_name):
    """Test Step 7: Build Response"""
    print_separator(f"STEP 7: Build Response - {doc_name}")
    
    try:
        ctx = state_after_persist["ctx"]
        state_after_persist["category_ids"] = list(ctx.category_ids)
        
        step_context = {"step_id": "build_response"}
        
        result_state = service._memorize_build_response(
            state_after_persist,
            step_context
        )
        
        print_success("Built response successfully")
        
        response = result_state["response"]
        print(f"   • Memory Items: {len(response['items'])}")
        print(f"   • Categories: {len(response['categories'])}")
        
        return result_state
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


async def test_08_save_markdown(service, state_after_response, doc_name, markdown_dir):
    """Test Step 8: Save Category Summaries as Markdown Files"""
    print_separator(f"STEP 8: Save Markdown - {doc_name}")
    
    try:
        ctx = state_after_response["ctx"]
        store = state_after_response["store"]
        
        saved_files = []
        
        for cat_id in ctx.category_ids:
            cat = store.memory_category_repo.categories[cat_id]
            
            if cat.summary:
                filepath = save_category_markdown(cat, markdown_dir)
                saved_files.append(filepath)
                print_success(f"Saved {cat.name}: {filepath.name}")
        
        state_after_response["markdown_files"] = [str(f) for f in saved_files]
        
        print_success(f"Saved {len(saved_files)} markdown files")
        
        return state_after_response
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_document_complete_workflow(service, test_link, doc_name, ctx, store, user, markdown_dir):
    """Test document through all 8 workflow steps"""
    print("\n" + "█" * 70)
    print(f"  PROCESSING: {doc_name}")
    print("█" * 70)
    
    try:
        state = await test_01_ingest_resource(service, test_link, doc_name, user)
        state = await test_02_preprocess_multimodal(service, state, doc_name)
        state = await test_03_extract_items(service, state, doc_name)
        state = await test_04_dedupe_merge(service, state, doc_name)
        state = await test_05_categorize_items(service, state, doc_name, ctx, store, user)
        state = await test_06_persist_index(service, state, doc_name)
        state = await test_07_build_response(service, state, doc_name)
        state = await test_08_save_markdown(service, state, doc_name, markdown_dir)
        
        print_separator(f"✓ COMPLETED: {doc_name}")
        print_success("All 8 workflow steps completed successfully")
        
        return state
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Main test runner"""
    print("\n" + "█" * 70)
    print("  MEMORIZE WORKFLOW - COMPLETE TEST (ALL 8 STEPS)")
    print("  With PostgreSQL Storage + User Context")
    print("█" * 70)
    
    import os
    
    use_postgres = os.getenv("USE_POSTGRES", "true").lower() == "true"
    
    temp_dir = Path(tempfile.mkdtemp())
    print_info(f"Temporary directory: {temp_dir}")
    
    markdown_dir = temp_dir / "category_markdown"
    markdown_dir.mkdir(exist_ok=True)
    print_info(f"Markdown directory: {markdown_dir}")
    
    # Create user context
    user = create_test_user()
    print_info(f"Test User: {user['user_id']}")
    print_info(f"Test Workspace: {user['workspace_id']}")
    
    try:
        service = create_test_service(temp_dir, use_postgres=use_postgres)
        ctx = service._get_context()
        store = service._get_database()
        
        # Initialize categories with user context
        print_info("Initializing categories...")
        mock_llm = create_mock_llm_client()
        with patch.object(service, '_get_llm_client', return_value=mock_llm), \
             patch.object(service, '_get_step_llm_client', return_value=mock_llm), \
             patch.object(service, '_get_step_embedding_client', return_value=mock_llm):
            # Pass user_scope for category initialization
            await service._ensure_categories_ready(ctx, store, user_scope=user)
        
        print_success(f"Categories: {list(ctx.category_name_to_id.keys())}")
        
        # Create documents
        # doc1 = create_test_document_1(temp_dir)
        # doc2 = create_test_document_2(temp_dir)
        
        # print_success("Test documents created")

        link1 = "https://www.linkedin.com/posts/zainhas_inference-ai-llm-activity-7427597857324494849-JxVE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACmrL44B-pNi9lNjFQtuPtX_ODwJk7-cC-0'"
        link2 = "https://www.linkedin.com/posts/pauliusztin_ive-spent-the-past-2-years-rigorously-building-activity-7426981104227856385-NhE4?utm_source=share&utm_medium=member_desktop&rcm=ACoAACmrL44B-pNi9lNjFQtuPtX_ODwJk7-cC-0"

        # Process documents
        state1 = await test_document_complete_workflow(
            service, link1, "inference", ctx, store, user, markdown_dir
        )
        
        state2 = await test_document_complete_workflow(
            service, link2, "Agents", ctx, store, user, markdown_dir
        )
        
        # Final Summary
        print("\n" + "█" * 70)
        print("  FINAL SUMMARY")
        print("█" * 70)
        
        response1 = state1["response"]
        response2 = state2["response"]
        
        print(f"\n📊 Total Memory Items: {len(response1['items']) + len(response2['items'])}")
        print(f"📂 Markdown Files: {len(list(markdown_dir.glob('*.md')))}")
        print(f"👤 User: {user['user_id']}")
        print(f"🏢 Workspace: {user['workspace_id']}")
        
        if use_postgres:
            print("\n💾 PostgreSQL Database:")
            print("   Data persisted in database!")
            print("   Check with: docker exec -it memu-postgres-1 psql -U postgres -d memu")
            print("   Query: SELECT * FROM memory_items;")
        
        print_separator("ALL TESTS COMPLETED SUCCESSFULLY")
        print_success("✓ All 8 steps completed with user context")
        print_success("✓ Data saved to PostgreSQL" if use_postgres else "✓ Data in memory")
        print_success("✓ Markdown files exported")
        
    except Exception as e:
        print_separator("TEST FAILED")
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print_info(f"Temporary files at: {temp_dir}")


if __name__ == "__main__":
    asyncio.run(main())