import requests
import time
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
API_KEY = "e17970941046bb77.jWGzPtLwZOmPHwTmPT4CVIFqvCecrFW9ADdCSzjw9c"  # Replace with your actual API key
AGENT_WEBHOOK_URL = "https://api.airtop.ai/api/hooks/agents/8c576110-03ce-4d68-9191-37a4db94686f/webhooks/6319c508-1373-4996-9260-5b023c4f5a4c"
BASE_URL = "https://api.airtop.ai/api/hooks/agents/8c576110-03ce-4d68-9191-37a4db94686f"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Step 1: Trigger the agent
print("Triggering ML content scraper agent...")
response = requests.post(
    AGENT_WEBHOOK_URL,
    headers=headers,
    json={
        'configVars': {
            'direction': 'extract all concepts related to prompt',
            'main_link': 'https://www.linkedin.com/posts/zainhas_inference-ai-llm-activity-7427597857324494849-JxVE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACmrL44B-pNi9lNjFQtuPtX_ODwJk7-cC-0',  # Replace with your target URL
            'max_links_to_follow': 5,
            'ml_related_keywords': 'machine learning,deep learning,LLM,RAG,neural networks,AI,artificial intelligence,reinforcement learning,agents,generative AI,transformer,attention,embeddings,vector search,retrieval,training,inference,fine-tuning,prompt engineering,multimodal,computer vision,NLP,recommendation systems'
        }
    },
    verify=False
)

if response.status_code != 200:
    print(f"Error triggering agent: {response.status_code}")
    print(response.text)
    exit(1)

invocation_id = response.json()['invocationId']
print(f"✓ Agent triggered! Invocation ID: {invocation_id}")

# Step 2: Poll for results
print("\nPolling for results...")
result_url = f"{BASE_URL}/invocations/{invocation_id}/result"

max_attempts = 180  # Poll for up to 15 minutes (scraping can take longer)
attempt = 0

while attempt < max_attempts:
    attempt += 1
    
    result_response = requests.get(
        result_url,
        headers={'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'},
        verify=False
    )
    
    if result_response.status_code != 200:
        print(f"Error getting result: {result_response.status_code}")
        print(result_response.text)
        break
    
    result = result_response.json()
    status = result.get('status', '').lower()
    
    print(f"[{attempt}/{max_attempts}] Status: {result.get('status')}")
    
    if status == 'completed':
        print("\n" + "="*60)
        print("✅ Agent completed successfully!")
        print("="*60)
        print("\nOutput:")
        output = result.get('output')
        if isinstance(output, dict):
            import json
            print(json.dumps(output, indent=2))
        else:
            print(output)
        break
    elif status == 'failed':
        print("\n" + "="*60)
        print("❌ Agent failed!")
        print("="*60)
        error = result.get('error')
        print(f"Error: {error}")
        break
    elif status in ['running', 'awaiting session', 'pending']:
        time.sleep(5)
    else:
        # Continue polling for unknown statuses
        print(f"   (Unknown status, continuing to poll...)")
        time.sleep(5)
else:
    print("\n⏱️ Timeout: Agent did not complete within 15 minutes")
    print("Final status:", result.get('status'))