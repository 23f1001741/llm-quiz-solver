import sys
import asyncio
import os
import json
import traceback
import requests
import re
import pandas as pd
import numpy as np
import bs4 
import io
import base64
import hashlib
import mimetypes
import tempfile
import csv
import random
import time
from urllib.parse import urljoin, urlparse 
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
import google.generativeai as genai

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL")
STUDENT_SECRET = os.environ.get("STUDENT_SECRET")

if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not found via Env Vars", flush=True)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash')

class TaskPayload(BaseModel):
    email: str
    secret: str
    url: str

# ==============================================================================
# 1. THE RECURSIVE CRAWLER
# ==============================================================================

visited_urls = set()
context_log = []

def transcribe_media(content: bytes, mime_type: str) -> str:
    """Uploads media to Gemini and gets a description."""
    try:
        suffix = ".mp3" if "audio" in mime_type else ".png"
        if "video" in mime_type: suffix = ".mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        myfile = genai.upload_file(tmp_path)
        prompt = "Transcribe this file VERBATIM. Write down every single word. Do not summarize. If there are visual/text instructions in the video/image, write them down exactly."
        result = model.generate_content([myfile, prompt])
        
        os.unlink(tmp_path)
        transcript = result.text.strip()
        
        print(f"\n🎤 [MEDIA TRANSCRIPT FOUND]:\n{transcript}\n", flush=True)
        return f"[TRANSCRIPT OF {mime_type}]: {transcript}"
    except Exception as e:
        return f"[ERROR TRANSCRIBING MEDIA]: {e}"

def inspect_csv(content: bytes) -> str:
    """
    NUCLEAR OPTION: Hide data to prevent LLM hardcoding.
    """
    try:
        text_val = content.decode('utf-8', errors='ignore')
        lines = text_val.splitlines()
        if not lines: return "[CSV ERROR]: Empty file"
        
        return f"""
[CSV STRUCTURE DETECTED]
- Total Rows: {len(lines)}
- Header Candidate: '{lines[0]}'
[INSTRUCTION]: Data is hidden. YOU MUST WRITE CODE TO DOWNLOAD AND READ THE FULL FILE using the URL.
"""
    except Exception as e:
        return f"[CSV ERROR]: {e}"

def process_puzzle_piece(url: str, mime_type: str, content: bytes) -> str:
    print(f"   🧩 Found Puzzle Piece ({mime_type}): {url}", flush=True)
    
    extracted_info = ""
    if "audio" in mime_type or "ogg" in mime_type:
        extracted_info = f"--- AUDIO FILE ({url}) ---\n" + transcribe_media(content, "audio/mp3")
    elif "video" in mime_type or "mp4" in mime_type:
        extracted_info = f"--- VIDEO FILE ({url}) ---\n" + transcribe_media(content, "video/mp4")
    elif "image" in mime_type:
        extracted_info = f"--- IMAGE FILE ({url}) ---\n" + transcribe_media(content, mime_type)
    elif "csv" in mime_type or url.endswith(".csv"):
        extracted_info = f"--- CSV DATA ({url}) ---\n" + inspect_csv(content)
    elif "json" in mime_type or "text" in mime_type:
        text_val = content.decode('utf-8', errors='ignore')
        extracted_info = f"--- TEXT DATA ({url}) ---\nCONTENT: {text_val[:1500]}"
    
    return extracted_info

async def recursive_crawl(url: str, depth: int, browser, max_depth: int = 2):
    if url in visited_urls or depth > max_depth:
        return
    
    visited_urls.add(url)
    prefix = "  " * depth
    print(f"{prefix}👉 Crawling: {url} (Depth {depth})", flush=True)

    try:
        try:
            head = requests.head(url, timeout=5, allow_redirects=True)
            content_type = head.headers.get("Content-Type", "").lower()
        except:
            content_type = ""

        is_asset = any(x in content_type for x in ['audio', 'image', 'video', 'csv', 'json', 'text/plain', 'ogg'])
        is_html = 'html' in content_type

        # --- CASE A: ASSET ---
        if is_asset or (not is_html and depth > 0): 
            resp = requests.get(url, timeout=10)
            piece_info = process_puzzle_piece(url, content_type, resp.content)
            if piece_info:
                context_log.append(piece_info)
            return

        # --- CASE B: HTML PAGE ---
        if is_html:
            page = await browser.new_page()
            try:
                await page.goto(url)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except:
                    pass
                
                visible_text = await page.inner_text("body")
                print(f"\n📄 [PAGE TEXT] ({url}):\n{visible_text[:200]}...\n", flush=True)
                context_log.append(f"=== PAGE TEXT ({url}) ===\n{visible_text[:4000]}")
                
                if depth >= max_depth:
                    await page.close()
                    return

                html = await page.content()
                soup = bs4.BeautifulSoup(html, 'html.parser')
                links = []
                for tag in soup.find_all(['a', 'script', 'img', 'source', 'audio', 'video']):
                    href = tag.get('href') or tag.get('src')
                    if href:
                        full_link = urljoin(url, href)
                        if "s-anand.net" in full_link or "localhost" in full_link:
                            if full_link not in visited_urls:
                                links.append(full_link)
                
                await page.close()
                print(f"{prefix}   ↳ Found {len(links)} links. Recursing...", flush=True)
                
                for link in links:
                    await recursive_crawl(link, depth + 1, browser, max_depth)

            except Exception as e:
                print(f"{prefix}⚠️ Error rendering page {url}: {e}", flush=True)
                await page.close()

    except Exception as e:
        print(f"{prefix}⚠️ Error crawling {url}: {e}", flush=True)

async def build_complete_context(start_url):
    print("🕵️ STARTING RECURSIVE CRAWL (Full Browser)...", flush=True)
    visited_urls.clear()
    context_log.clear()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        await recursive_crawl(start_url, 0, browser, max_depth=2)
        await browser.close()

    return "\n\n".join(context_log)

# ==============================================================================
# 2. EXECUTION ENGINE
# ==============================================================================
def execute_generated_code(code_str: str):
    print("⚡ Executing Python code...", flush=True)
    print(f"--- GENERATED CODE ---\n{code_str}\n----------------------", flush=True)
    
    code_str = code_str.replace("```python", "").replace("```", "").strip()
    
    allowed_globals = {
        "pd": pd, "np": np, "requests": requests, "json": json, "re": re,
        "bs4": bs4, "BeautifulSoup": bs4.BeautifulSoup, "urljoin": urljoin,
        "io": io, "base64": base64, "hashlib": hashlib, "print": print,
        "genai": genai, "os": os,
        "csv": csv, "random": random, "time": time
    }
    local_vars = {}

    try:
        exec(code_str, allowed_globals, local_vars)
        if 'solution' in local_vars:
            sol = local_vars['solution']
            # Serialization Fix
            if isinstance(sol, (np.integer, np.floating)):
                return sol.item()
            if isinstance(sol, np.ndarray):
                return sol.tolist()
            return sol
        else:
            return "Error: Code executed but 'solution' variable was never defined."
    except Exception as e:
        return f"Execution Error: {traceback.format_exc()}"

# ==============================================================================
# 3. GEMINI ANALYST (STRICT PROMPT)
# ==============================================================================
async def analyze_task(deep_context: str, current_url: str):
    print("🧠 Gemini is analyzing the gathered context...", flush=True)
    
    prompt = r"""
    You are an intelligent Data Extraction Agent.
    
    FULL CONTEXT DUMP (Read Carefully):
    =========================================
    {deep_context}
    =========================================
    
    CURRENT URL: {current_url}
    
    TASK: Write a Python script to calculate the `solution`.
    
    CRITICAL RULES:
    1. **NO RE-FETCHING HTML (Step 2)**:
       - The logs ALREADY contain the JS-rendered text (e.g., "Secret code is 39529").
       - **NEVER** write code to `requests.get()` the current URL. It will return raw HTML and FAIL.
       - Instead, simply extract the string: `solution = "39529"`.
    
    2. **FORCE CSV DOWNLOAD (Step 3)**:
       - The CSV Preview HIDES the data.
       - **YOU MUST** write code to download the file: `df = pd.read_csv(io.StringIO(requests.get(csv_url).text))`
    
    3. **AUDIO/VIDEO LOGIC**:
       - Use the transcript logic EXACTLY.
       - "Sum numbers > 50": `df[df[0] > 50].sum()`.
    
    4. **Output**: Return ONLY Python code.
    """
    
    safe_context = deep_context.replace('{', '{{').replace('}', '}}').replace("'", "")
    final_prompt = prompt.format(deep_context=safe_context, current_url=current_url, email=STUDENT_EMAIL)

    print("\n📝 [FULL PROMPT SENT TO GEMINI]:", flush=True)
    print("="*60)
    print(final_prompt[:3000] + "... (truncated)")
    print("="*60 + "\n", flush=True)

    try:
        response = await asyncio.to_thread(model.generate_content, final_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}", flush=True)
        return ""

def extract_submit_url(text: str, base_url: str):
    match = re.search(r'(?:POST|submit).*?to\s+((?:https?://|/)\S+)', text, re.IGNORECASE)
    if match:
        raw_url = match.group(1).strip(".,;\"'")
        return urljoin(base_url, raw_url)
    return None

# ==============================================================================
# 4. AGENT LOOP
# ==============================================================================
async def run_quiz_chain(start_url: str):
    current_url = start_url
    steps = 0
    max_steps = 15 
    
    print(f"🚀 STARTING QUIZ CHAIN at {start_url}", flush=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        while current_url and steps < max_steps:
            steps += 1
            print(f"\n--- STEP {steps}: Visiting {current_url} ---", flush=True)

            try:
                await page.goto(current_url)
                try:
                    await page.wait_for_load_state("networkidle", timeout=3000)
                except:
                    pass
                
                deep_context = await build_complete_context(current_url)
                
                code = await analyze_task(deep_context, current_url)
                answer = execute_generated_code(code)
                
                retries = 0
                while "Error" in str(answer) and retries < 2:
                    print(f"⚠️ Code failed. Retrying... ({retries+1}/2)", flush=True)
                    fix_prompt = f"Previous code failed with: {answer}\n\nCode:\n{code}\n\nFIX IT. Initialize all variables."
                    retry_resp = await asyncio.to_thread(model.generate_content, fix_prompt)
                    code = retry_resp.text
                    answer = execute_generated_code(code)
                    retries += 1
                
                print(f"💡 Final Answer: {answer}", flush=True)
                
                visible_text = await page.inner_text("body")
                submit_url = extract_submit_url(visible_text, current_url)
                
                if not submit_url:
                    if "demo" in current_url:
                         submit_url = "https://tds-llm-analysis.s-anand.net/demo/submit"
                    else:
                        print("⚠️ FATAL: No submit URL found.", flush=True)
                        break

                payload = {
                    "email": STUDENT_EMAIL,
                    "secret": STUDENT_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                print(f"📤 Submitting to {submit_url}...", flush=True)
                response = requests.post(submit_url, json=payload)
                
                try:
                    resp_data = response.json()
                    print(f"✅ Response: {resp_data}", flush=True)
                    
                    if resp_data.get("url"):
                        current_url = resp_data.get("url")
                    else:
                        print("🏆 QUIZ COMPLETED.", flush=True)
                        break
                except:
                    print(f"❌ Submission Error: {response.text}", flush=True)
                    break
                    
            except Exception as e:
                print(f"💥 Error: {e}", flush=True)
                traceback.print_exc()
                break
        
        await browser.close()
        print("🏁 Agent finished.", flush=True)

@app.post("/llm-agent")
async def start_task(payload: TaskPayload, background_tasks: BackgroundTasks):
    if payload.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(run_quiz_chain, payload.url)
    return {"message": "Agent started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)